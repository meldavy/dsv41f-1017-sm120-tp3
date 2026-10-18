# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 (trellis, mul1 codebook) routed experts for DeepSeek V4.1 on SM120.

Only the routed experts of the target layers use EXL3; every other projection
keeps the checkpoint's native block32 FP8 / MXFP4 path. The expert GEMMs run on
ExLlamaV3's kernels (``exllamav3_ext``): the cooperative decode kernel for up to
``_COOP_MAX_ROWS`` tokens and the fused grouped kernel above that. Eager calls
with at least ``_DO_MIN_TOKENS`` tokens (prefill) use the decode-once grouped
GEMM (``ds41_exl3_do``, from diffbot's xmoe_do) over a job list built on the
device, so prefill never waits on the host for routing counts.

Tensor parallelism splits every expert's intermediate axis: w1/w3 keep output
tile columns ``[r*I/16, (r+1)*I/16)`` and w2 keeps the matching input tile rows.
EXL3's Hadamard rotations act on 128-wide blocks, so a slice on a 128 boundary
is exact.
"""

from __future__ import annotations

import os

import regex as re
import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

logger = init_logger(__name__)

EXL3_SUFFIXES = (".trellis", ".suh", ".svh", ".mul1", ".mcg")
_EXPERT_NAME = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(\w+)$")

_ACT_SILU = 0
# The cooperative decode kernel holds at most 256 (token, expert) slots and
# ExLlamaV3 sizes its row counters for MAX_BSZN = 8 tokens.
_COOP_MAX_ROWS = 8
# Rows per expert the fused kernel's scratch holds (ExLlamaV3 caps it at 512).
_FUSED_ROWS = int(os.environ.get("DS41_EXL3_FUSED_ROWS", "512"))
_MTILE_T1, _MTILE_T2 = 16, 32
# Decode-once prefill path: tokens per sub-batch bound its scratch
# (2 x rows x hidden + 2 x rows x intermediate fp16, rows = tokens x top-k).
_DO_ENABLED = os.environ.get("DS41_EXL3_DO", "1") == "1"
# Directory holding the built ds41_exl3_do extension, if it is not on PYTHONPATH.
_DO_DIR = os.environ.get("DS41_EXL3_DO_DIR", "")
_DO_MIN_TOKENS = int(os.environ.get("DS41_EXL3_DO_MIN_TOKENS", "257"))
_DO_TOKENS = int(os.environ.get("DS41_EXL3_DO_TOKENS", "4096"))
_DO_JOB_ROWS = 128

_ext = None
_do = None


def exllamav3_ext():
    global _ext
    if _ext is None:
        import exllamav3_ext as ext

        _ext = ext
    return _ext


def ds41_exl3_do():
    global _do
    if _do is None:
        import sys

        if _DO_DIR and _DO_DIR not in sys.path:
            sys.path.insert(0, _DO_DIR)
        import ds41_exl3_do as do

        _do = do
    return _do


def parse_expert_name(name: str):
    """``layers.L.ffn.experts.E.wN.kind`` -> (L, E, wN, kind), else None."""
    if not name.endswith(EXL3_SUFFIXES):
        return None
    m = _EXPERT_NAME.search(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)


class _Scratch:
    """Per-device buffers shared by every EXL3 layer (layers run sequentially)."""

    _by_key: dict = {}

    @classmethod
    def get(cls, device, hidden: int, inter: int, topk: int):
        key = (str(device), hidden, inter, topk)
        s = cls._by_key.get(key)
        if s is None:
            s = cls(device, hidden, inter, topk)
            cls._by_key[key] = s
        return s

    def __init__(self, device, hidden, inter, topk):
        ext = exllamav3_ext()
        slots = _COOP_MAX_ROWS * topk
        half = dict(dtype=torch.float16, device=device)
        self.coop_had_g = torch.empty((slots, hidden), **half)
        self.coop_had_u = torch.empty((slots, hidden), **half)
        self.coop_gu_g = torch.empty((slots, inter), **half)
        self.coop_gu_u = torch.empty((slots, inter), **half)
        self.coop_act = torch.empty((slots, inter), **half)
        self.coop_d_out = torch.empty((slots, hidden), dtype=torch.float32, device=device)
        self.coop_out = torch.empty((_COOP_MAX_ROWS, hidden), dtype=torch.float32, device=device)
        ctr_len = (
            slots * (inter // 128)
            + _COOP_MAX_ROWS * (hidden // 128)
            + 2
            + (slots + 1)
            + slots
        )
        self.coop_ctr = torch.zeros(ctr_len, dtype=torch.int32, device=device)
        idx = torch.device(device).index or 0
        self.concurrency = max(1, int(ext.exl3_moe_max_concurrency(idx)))
        c = self.concurrency
        self.temp_state_g = torch.empty((c, _FUSED_ROWS, hidden), **half)
        self.temp_state_u = torch.empty((c, _FUSED_ROWS, hidden), **half)
        self.temp_interm_g = torch.empty((c, _FUSED_ROWS, inter), **half)
        self.temp_interm_u = torch.empty((c, _FUSED_ROWS, inter), **half)
        self.device, self.hidden, self.inter, self.topk = device, hidden, inter, topk
        self.do_rows = 0

    def ensure_do(self, num_experts: int):
        """Allocate the decode-once scratch on first prefill (the profile run)."""
        rows = _DO_TOKENS * self.topk
        if self.do_rows >= rows:
            return
        half = dict(dtype=torch.float16, device=self.device)
        self.do_xg = torch.empty((rows, self.hidden), **half)
        self.do_xu = torch.empty((rows, self.hidden), **half)
        self.do_cg = torch.empty((rows, self.inter), **half)
        self.do_cu = torch.empty((rows, self.inter), **half)
        jobs = -(-rows // _DO_JOB_ROWS) + num_experts
        self.do_jobs = torch.arange(jobs, dtype=torch.int64, device=self.device)
        self.do_rows = rows


class DeepseekV41Exl3ExpertsMethod(FusedMoEMethodBase):
    """Routed experts stored as EXL3 trellis tensors, one bitrate per layer."""

    def __init__(self, moe: FusedMoEConfig, layer_index: int, act_limit: float):
        super().__init__(moe)
        parallel = moe.moe_parallel_config
        if parallel.use_ep or parallel.dp_size != 1:
            raise NotImplementedError("EXL3 experts support tensor parallelism only")
        self.layer_index = layer_index
        self.act_limit = float(act_limit or 0.0)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.bits: int | None = None
        self.mcg = False
        self.mul1 = True
        self._seen = 0
        self._codebook_seen: set[str] = set()

    # ---- weights ---------------------------------------------------------------

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if hidden_size % 128 or intermediate_size_per_partition % 128:
            raise ValueError(
                "EXL3 experts need 128-aligned hidden/intermediate shards, got "
                f"{hidden_size}/{intermediate_size_per_partition}"
            )
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.inter = intermediate_size_per_partition
        self.full_inter = self.inter * self.tp_size
        # The modular runner and EPLB bookkeeping look these handles up by name;
        # the trellis storage lives in plain tensors allocated on first load.
        for name in ("w13_weight", "w2_weight"):
            layer.register_parameter(
                name,
                torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False),
            )
        layer._exl3 = None

    def get_fused_moe_quant_config(self, layer):
        return None

    def _allocate(self, layer, bits: int, device) -> dict:
        E, H, I, w = self.num_experts, self.hidden_size, self.inter, 16 * bits
        i16 = dict(dtype=torch.int16, device=device)
        f16 = dict(dtype=torch.float16, device=device)
        st = {
            "w1.trellis": torch.empty((E, H // 16, I // 16, w), **i16),
            "w3.trellis": torch.empty((E, H // 16, I // 16, w), **i16),
            "w2.trellis": torch.empty((E, I // 16, H // 16, w), **i16),
            "w1.suh": torch.empty((E, H), **f16),
            "w3.suh": torch.empty((E, H), **f16),
            "w2.suh": torch.empty((E, I), **f16),
            "w1.svh": torch.empty((E, I), **f16),
            "w3.svh": torch.empty((E, I), **f16),
            "w2.svh": torch.empty((E, H), **f16),
        }
        layer._exl3 = st
        self.bits = bits
        return st

    def load_exl3(self, layer, expert: int, proj: str, kind: str, tensor: torch.Tensor) -> None:
        if kind in ("mul1", "mcg"):
            self._codebook_seen.add(kind)
            return
        st = layer._exl3
        device = layer.w13_weight.device
        if st is None:
            if kind != "trellis":
                # Bitrate is only known from a trellis; suh/svh can arrive first.
                bits = int(os.environ.get("DS41_EXL3_BITS_HINT", "0")) or None
                if bits is None:
                    pend = layer.__dict__.setdefault("_exl3_pending", [])
                    pend.append((expert, proj, kind, tensor.clone()))
                    return
            else:
                bits = tensor.shape[-1] // 16
            st = self._allocate(layer, bits, device)
            for args in layer.__dict__.pop("_exl3_pending", []):
                self.load_exl3(layer, *args)
        r, I = self.tp_rank, self.inter
        if kind == "trellis":
            if tensor.shape[-1] != 16 * self.bits:
                raise ValueError(
                    f"layer {self.layer_index}: mixed EXL3 bitrates within one layer "
                    f"({tensor.shape[-1] // 16} vs {self.bits}) are unsupported"
                )
            t16 = I // 16
            if proj in ("w1", "w3"):
                src = tensor[:, r * t16 : (r + 1) * t16, :]
            else:
                src = tensor[r * t16 : (r + 1) * t16]
        elif kind == "suh":
            src = tensor if proj in ("w1", "w3") else tensor[r * I : (r + 1) * I]
        elif kind == "svh":
            src = tensor[r * I : (r + 1) * I] if proj in ("w1", "w3") else tensor
        else:
            raise ValueError(f"unexpected EXL3 tensor kind {kind!r}")
        st[f"{proj}.{kind}"][expert].copy_(src, non_blocking=False)
        self._seen += 1

    def process_weights_after_loading(self, layer) -> None:
        st = layer._exl3
        expected = 9 * self.num_experts
        if st is None or self._seen != expected:
            raise ValueError(
                f"EXL3 layer {self.layer_index}: loaded {self._seen} of {expected} "
                "expert tensors"
            )
        if self._codebook_seen != {"mul1"}:
            raise ValueError(
                f"EXL3 layer {self.layer_index}: expected the mul1 codebook, "
                f"got markers {sorted(self._codebook_seen)}"
            )

        def ptrs(key):
            t = st[key]
            stride = t[0].numel() * t.element_size() if t.shape[0] > 1 else 0
            base = t.data_ptr()
            return torch.tensor(
                [base + e * stride for e in range(t.shape[0])],
                dtype=torch.int64,
                device=t.device,
            )

        layer._exl3_ptrs = {k: ptrs(k) for k in st}
        self.scratch = _Scratch.get(
            layer.w13_weight.device,
            self.hidden_size,
            self.inter,
            self.moe.experts_per_token,
        )
        self.flat_token_cache: dict[int, torch.Tensor] = {}
        logger.info_once(
            "EXL3 routed experts: K=%d mul1, TP%d slice %d of %d channels, "
            "coop<=%d tokens, fused rows %d",
            self.bits,
            self.tp_size,
            self.inter,
            self.full_inter,
            _COOP_MAX_ROWS,
            _FUSED_ROWS,
        )

    # ---- forward ---------------------------------------------------------------

    def prepare_workspace(self, hidden_states, shared_workspace_size):
        # Shared experts fall back to the workspace manager; the EXL3 kernels
        # use their own scratch.
        return None, None

    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
        workspace=None,
    ) -> torch.Tensor:
        tokens = x.shape[0]
        if tokens == 0:
            return torch.empty_like(x)
        xh = x.to(torch.float16)
        if tokens <= _COOP_MAX_ROWS:
            out = self._coop(layer, xh, topk_weights, topk_ids)
        elif (
            _DO_ENABLED
            and tokens >= _DO_MIN_TOKENS
            and not torch.cuda.is_current_stream_capturing()
        ):
            out = self._decode_once(layer, xh, topk_weights, topk_ids)
        else:
            out = self._fused(layer, xh, topk_weights, topk_ids)
        return out.to(x.dtype)

    def _coop(self, layer, xh, topk_weights, topk_ids):
        ext, s, p = exllamav3_ext(), self.scratch, layer._exl3_ptrs
        tokens = xh.shape[0]
        k = float(self.bits)
        out = s.coop_out[:tokens]
        # The [0, E) range filter turns padded-token picks (id -1, weight 0)
        # into exact zeros instead of pointer-table reads out of bounds.
        ext.exl3_moe_coop(
            xh,
            topk_ids.to(torch.int64).contiguous(),
            topk_weights.to(torch.float16).contiguous(),
            0,
            self.num_experts,
            self.hidden_size,
            p["w1.trellis"], p["w1.suh"], p["w1.svh"],
            p["w3.trellis"], p["w3.suh"], p["w3.svh"],
            p["w2.trellis"], p["w2.suh"], p["w2.svh"],
            None, None, None,
            k, k, k,
            self.mcg, self.mul1,
            _ACT_SILU,
            self.act_limit,
            True,
            s.coop_had_g, s.coop_had_u, s.coop_gu_g, s.coop_gu_u, s.coop_act,
            s.coop_d_out, s.coop_ctr, out,
            None, None,
        )
        return out

    def _decode_once(self, layer, xh, topk_weights, topk_ids):
        do, s, p = ds41_exl3_do(), self.scratch, layer._exl3_ptrs
        tokens, topk = topk_ids.shape
        E, dev = self.num_experts, xh.device
        s.ensure_do(E)
        out = torch.zeros((tokens, self.hidden_size), dtype=torch.float32, device=dev)
        jobs = s.do_jobs
        for t0 in range(0, tokens, _DO_TOKENS):
            t1 = min(t0 + _DO_TOKENS, tokens)
            ids = topk_ids[t0:t1].reshape(-1).to(torch.int64)
            # Padded tokens carry id -1: sort them last as the skipped expert E.
            ids = torch.where(ids < 0, E, ids)
            order = torch.argsort(ids, stable=True)
            row_exp = ids.index_select(0, order).to(torch.int32)
            tok = torch.div(order, topk, rounding_mode="floor") + t0
            wts = topk_weights[t0:t1].reshape(-1).index_select(0, order).to(torch.float16)
            count = torch.zeros(E + 1, dtype=torch.int64, device=dev)
            count.scatter_add_(0, ids, torch.ones_like(ids))
            count = count[:E]
            first_row = torch.cumsum(count, 0) - count
            n_jobs = torch.div(count + _DO_JOB_ROWS - 1, _DO_JOB_ROWS, rounding_mode="floor")
            job_end = torch.cumsum(n_jobs, 0)
            job_exp = torch.searchsorted(job_end, jobs, right=True).clamp_(max=E - 1)
            k = jobs - (job_end - n_jobs).index_select(0, job_exp)
            job_row0 = (first_row.index_select(0, job_exp) + _DO_JOB_ROWS * k).to(torch.int32)
            job_rows = (count.index_select(0, job_exp) - _DO_JOB_ROWS * k).clamp_(0, _DO_JOB_ROWS)
            do.do_moe(
                xh, out, tok, wts, row_exp,
                job_row0, job_rows.to(torch.int32), job_exp,
                p["w1.trellis"], p["w1.suh"], p["w1.svh"],
                p["w3.trellis"], p["w3.suh"], p["w3.svh"],
                p["w2.trellis"], p["w2.suh"], p["w2.svh"],
                s.do_xg, s.do_xu, s.do_cg, s.do_cu,
                self.bits, 2 if self.mul1 else 1, self.act_limit, _ACT_SILU, E,
            )
        return out

    def _flat_token(self, tokens: int, topk: int, device) -> torch.Tensor:
        t = self.flat_token_cache.get(tokens * 64 + topk)
        if t is None:
            t = torch.arange(tokens, device=device, dtype=torch.int64).repeat_interleave(topk)
            self.flat_token_cache[tokens * 64 + topk] = t
        return t

    def _fused(self, layer, xh, topk_weights, topk_ids):
        ext, s, p = exllamav3_ext(), self.scratch, layer._exl3_ptrs
        tokens, topk = topk_ids.shape
        E = self.num_experts
        local = topk_ids.reshape(-1).to(torch.int64)
        # Padded tokens carry id -1: route them to the kernel's skipped bucket E.
        local = torch.where(local < 0, E, local)
        flat_w = topk_weights.reshape(-1).to(torch.float16)
        flat_t = self._flat_token(tokens, topk, xh.device)
        order = torch.argsort(local, stable=True)
        token_sorted = flat_t.index_select(0, order)
        weight_sorted = flat_w.index_select(0, order)
        count = torch.zeros(E + 1, dtype=torch.int64, device=xh.device)
        count.scatter_add_(0, local, torch.ones_like(local))
        out = torch.zeros((tokens, self.hidden_size), dtype=torch.float32, device=xh.device)
        k = float(self.bits)

        def run(cnt, tok, wts, num_active, lo, hi, m_tile):
            ext.exl3_moe(
                xh, out, cnt, tok, wts,
                s.temp_state_g, s.temp_state_u, s.temp_interm_g, s.temp_interm_u,
                _ACT_SILU, k, k, k,
                p["w1.trellis"], p["w1.suh"], p["w1.svh"],
                p["w3.trellis"], p["w3.suh"], p["w3.svh"],
                p["w2.trellis"], p["w2.suh"], p["w2.svh"],
                self.mcg, self.mul1, self.mcg, self.mul1, self.mcg, self.mul1,
                self.act_limit, num_active, None, None, lo, hi, m_tile,
            )

        if tokens <= _FUSED_ROWS or torch.cuda.is_current_stream_capturing():
            # No expert can hold more rows than there are tokens: one launch at
            # full concurrency, no host synchronization (graph safe).
            run(count, token_sorted, weight_sorted, -1, 1, _FUSED_ROWS, 16)
            return out

        # Prefill: read the per-expert row counts once, launch one row-tile
        # instance per occupancy band, and hand experts with more rows than the
        # scratch holds to further passes over their remaining rows.
        counts = count[:E].tolist()
        max_rows = max(counts)
        if max_rows <= _FUSED_ROWS:
            passes = [(count, token_sorted, weight_sorted, counts)]
        else:
            start = torch.cumsum(count, 0) - count
            pos = torch.arange(local.numel(), device=xh.device) - start.index_select(
                0, local.index_select(0, order)
            )
            passes = []
            for lo_row in range(0, max_rows, _FUSED_ROWS):
                keep = (pos >= lo_row) & (pos < lo_row + _FUSED_ROWS)
                cnt = (count - lo_row).clamp(0, _FUSED_ROWS)
                cnt[E] = 0
                cl = [min(max(c - lo_row, 0), _FUSED_ROWS) for c in counts]
                passes.append((cnt, token_sorted[keep], weight_sorted[keep], cl))
        for cnt, tok, wts, cl in passes:
            bands = (
                (sum(1 for c in cl if c > _MTILE_T2), _MTILE_T2 + 1, _FUSED_ROWS, 64),
                (sum(1 for c in cl if _MTILE_T1 < c <= _MTILE_T2), _MTILE_T1 + 1, _MTILE_T2, 32),
                (sum(1 for c in cl if 0 < c <= _MTILE_T1), 1, _MTILE_T1, 16),
            )
            for n, lo, hi, m_tile in bands:
                if n:
                    run(cnt, tok, wts, n, lo, hi, m_tile)
        return out
