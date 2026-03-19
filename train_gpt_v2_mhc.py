from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange, repeat

def l2_norm_fn(x: Tensor) -> Tensor:
    return F.normalize(x, dim=-1)

ACT2FN = {
    "swish": F.silu,
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
}

class FLARMSNorm(nn.Module):
    def __init__(self, hidden_size: int, elementwise_affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        x = F.rms_norm(x, (x.size(-1),), eps=self.eps)
        if self.weight is not None:
            x = x * self.weight.to(x.dtype)
        return x

class FusedRMSNormSwishGate(nn.Module):
    """RMSNorm on x followed by element-wise swish(g) gating."""
    def __init__(self, hidden_size: int, elementwise_affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        x = F.rms_norm(x, (x.size(-1),), eps=self.eps)
        if self.weight is not None:
            x = x * self.weight.to(x.dtype)
        return x * F.silu(g)

class ShortConvolution(nn.Module):
    """Causal depthwise Conv1d with optional activation."""
    def __init__(self, channels: int, kernel_size: int, activation: str | None = None, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=bias)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        B, L, C = x.shape
        x = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))  # [B, C, L+pad]
        x = self.conv(x).transpose(1, 2)                          # [B, L, C]
        if self.activation == "silu":
            x = F.silu(x)
        return x

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False
    def chunk_gated_delta_rule(q, k, v, beta, g, scale=None, output_final_state=False, initial_state=None, BT=32, **kwargs):
        # Pure-PyTorch fallback — install flash-linear-attention for the Triton kernel.
        B, H, T, d_k = q.shape; d_v = v.shape[-1]; dtype = v.dtype
        q, k, v, beta, g = (x.float() for x in (q, k, v, beta, g))
        if scale is not None: q = q * scale
        S = initial_state.float() if initial_state is not None else q.new_zeros(B, H, d_k, d_v)
        out = q.new_zeros(B, H, T, d_v)
        for t in range(T):
            k_t = k[..., t, :]; v_t = v[..., t, :]; b_t = beta[..., t]; g_t = g[..., t].exp()
            kS = torch.einsum('bhi,bhij->bhj', k_t, S)
            S = S * g_t[..., None, None] + b_t[..., None, None] * torch.einsum('bhi,bhj->bhij', k_t, v_t - kS)
            out[..., t, :] = torch.einsum('bhi,bhij->bhj', q[..., t, :], S)
        return out.to(dtype), (S if output_final_state else None)

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 1))  # legacy; prefer MLP_HIDDEN
    mlp_hidden = int(os.environ.get("MLP_HIDDEN", 512))  # explicit hidden dim
    attn_every = int(os.environ.get("ATTN_EVERY", 4))  # 1=all-attn, 2=1:1, 4=3:1 GDN
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    nope_ratio = float(os.environ.get("NOPE_RATIO", 0.0))  # fraction of attn heads that skip RoPE (NoPE)
    hc_streams = int(os.environ.get("HC_STREAMS", 0))  # MHC parallel streams N (0=disabled, uses resid_mix)
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.050))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    prune_ratio = float(os.environ.get("PRUNE_RATIO", 0.0))
    # Mixed-precision quantization tiers: step=4 (~int6), step=16 (~int4)
    # Set INT6_LAYERS/INT4_LAYERS or QUANT_AUTO_BUDGET_MB for auto-assignment.
    int4_layers = os.environ.get("INT4_LAYERS", "")
    int4_step   = int(os.environ.get("INT4_STEP", 16))
    int6_layers = os.environ.get("INT6_LAYERS", "")
    int6_step   = int(os.environ.get("INT6_STEP", 4))
    quant_auto_budget_mb = float(os.environ.get("QUANT_AUTO_BUDGET_MB", 0.0))

    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon: Newton-Schulz5 orthogonalized gradient descent for matrix params.
    Scales update by sqrt(max(m,n)/min(m,n)) so RMS matches Adam at the same lr.
    Ref: https://kellerjordan.github.io/posts/muon/
    """
    def __init__(self, params, lr: float, momentum: float, backend_steps: int = 5, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "moment1" not in state:
                        state["moment1"] = torch.zeros_like(g)
                    buf = state["moment1"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g + momentum * buf
                    else:
                        g = buf
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                p.add_(updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype), alpha=-lr)
                curr += p.numel()

        return loss


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    if args.val_batch_size // args.train_seq_len < 1:
        raise ValueError(f"VAL_BATCH_SIZE too small for seq_len={args.train_seq_len}")
    # VAL_BATCH_SIZE is the TOTAL token budget across all ranks.
    # Shard it evenly across ranks so each rank evaluates a different slice.
    max_val_seqs = args.val_batch_size // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    max_val_seqs = min(max_val_seqs, total_seqs)
    seq_start = (max_val_seqs * rank) // world_size
    seq_end = (max_val_seqs * (rank + 1)) // world_size
    # Process in chunks to avoid OOM — cap at 64 seqs per forward pass
    local_batch_seqs = max(1, min(seq_end - seq_start, 64))
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    """Sliding window eval: every token scored with maximum context. Only last `stride` tokens per window scored."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride]
    total_windows = len(window_starts)

    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_batch)  # [bsz, seq_len, vocab]

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else wlen - stride
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            if rank == 0 and (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(my_windows))
                rl = (loss_sum / token_count).item() if token_count.item() > 0 else 0.0
                running_bpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1))
                print(f"  sliding_eval [{pct:5.1f}%] {done}/{len(my_windows)} windows running_bpb={running_bpb:.6f}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Per-row scale for matrices (tracks output-channel ranges better)
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # per-row int8 for 2D floats, per-tensor int8 for others, fp16 passthrough for small tensors
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def _load_cpu(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        """CPU-only: read tokens from the shard stream. Safe to call from a background thread."""
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x, y  # CPU tensors

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        x, y = self._load_cpu(global_tokens, seq_len, grad_accum_steps)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 0.75,
        expand_v: float = 1.5,
        num_heads: int = 9,
        num_kv_heads: Optional[int] = None,
        qk_norm: str = 'l2',
        conv_size: int = 4,
        conv_bias: bool = False,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        fuse_norm: bool = True,
        layer_idx: int = None,
        use_mamba_gate: bool = True,
        use_mva: bool = False,
        use_residual: bool = False, # residual as in Mamba2,
        use_input_gate: bool = False,
    ) -> GatedDeltaNet:
        super().__init__()
        self.qk_norm = qk_norm
        assert self.qk_norm in ['l2', 'longhorn', 'softmax']

        self.use_mva = use_mva

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.layer_idx = layer_idx

        assert mode in ['chunk'], f"Not suppoerted mode `{mode}`."

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)

        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu' if self.qk_norm != 'softmax' else None)
        self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu' if self.qk_norm != 'softmax' else None)
        self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')
        self.gk_proj = nn.Linear(hidden_size, self.num_heads, bias= not use_mamba_gate)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=True)

        if gate_fn == 'swish' and fuse_norm:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = FLARMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
            self.gate_fn = ACT2FN[gate_fn]
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.gate_logit_normalizer = gate_logit_normalizer

        self.use_mamba_gate = use_mamba_gate
        if use_mamba_gate:
            A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
            A_log = torch.log(A)
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
            self.D = nn.Parameter(torch.ones(self.num_heads))
            self.D._no_weight_decay = True
            dt_min=0.001
            dt_max=0.1
            dt_init_floor=1e-4
            dt = torch.exp(
                torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True

        self.use_residual = use_residual            
        if self.use_residual:
            self.D = nn.Parameter(torch.ones(self.num_heads))
            self.D._no_weight_decay = True
        self.use_input_gate = use_input_gate


    def forward(self, hidden_states: Tensor) -> Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = self.q_conv1d(q)
        k = self.k_conv1d(k)
        v = self.v_conv1d(v)

        gk = self.gk_proj(hidden_states).float()
        if self.use_mamba_gate:
            gk = -self.A_log.float().exp() * F.softplus(gk + self.dt_bias)
        else:
            gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        # gk stays as [B, T, H] — FLA API is time-first, no transpose needed
        gk = gk.contiguous()

        beta = self.b_proj(hidden_states).float().sigmoid()
        # beta stays as [B, T, H] — time-first, no transpose needed
        beta = beta.contiguous()
        # Rearrange to time-first [B, T, H, D] as required by the FLA API
        q = rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads).contiguous()
        if self.num_kv_groups > 1:
            k, v = (repeat(x, 'b l (h d) -> b l (h g) d', h=self.num_kv_heads, g=self.num_kv_groups).contiguous() for x in (k, v))
        else:
            k, v = (rearrange(x, 'b l (h d) -> b l h d', h=self.num_kv_heads).contiguous() for x in (k, v))

        if self.qk_norm == 'l2':
            q = l2_norm_fn(q).to(v).contiguous()
            k = l2_norm_fn(k).to(v).contiguous()
        elif self.qk_norm == 'softmax':
            k = k.softmax(dim=-1).to(v)
            q = q.softmax(dim=-1).to(v)
        elif self.qk_norm == 'longhorn':
            beta = beta / (1 + beta * (k * k).sum(-1))

        if self.use_input_gate:
            original_v_dtype = v.dtype
            v = (v * (1 - gk.float().exp())[..., None]).to(original_v_dtype)

        # Correct argument order: (q, k, v, g, beta) — was swapped before
        # Output is time-first [B, T, H, D]
        o, _ = chunk_gated_delta_rule(q, k, v, gk, beta, scale=1.0, output_final_state=False)

        if self.use_residual:
            # D is [H], broadcast over [B, T, H, D]
            o = o + self.D[None, None, :, None] * v
        # o is already [B, T, H, D], no rearrange needed

        g = self.g_proj(hidden_states)
        if self.fuse_norm_and_gate:
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.g_norm_swish_gate(o, g)
            o = rearrange(o, 'b l h d -> b l (h d)')
        else:
            o = rearrange(self.g_norm(o), 'b l h d -> b l (h d)')
            o = o * self.gate_fn(g)
        o = self.o_proj(o)
        return o


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        nope_heads: int = 0,  # number of heads that skip RoPE (NoPE)
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        # NoPE: last nope_heads Q/K heads receive no positional encoding.
        # The first rope_heads heads keep RoPE as usual.
        self.nope_heads = min(nope_heads, num_heads)
        self.rope_heads = num_heads - self.nope_heads
        gqa_ratio = num_heads // num_kv_heads
        self.nope_kv_heads = self.nope_heads // gqa_ratio
        self.rope_kv_heads = num_kv_heads - self.nope_kv_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base) if self.rope_heads > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        if self.rotary is not None:  # at least some heads use RoPE
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            if self.nope_heads == 0:
                # All heads use RoPE (fast path, no cat overhead)
                q = apply_rotary_emb(q, cos, sin)
                k = apply_rotary_emb(k, cos, sin)
            else:
                # Split: first rope_heads get RoPE, last nope_heads are untouched
                q = torch.cat([apply_rotary_emb(q[:, :self.rope_heads], cos, sin),
                               q[:, self.rope_heads:]], dim=1)
                k = torch.cat([apply_rotary_emb(k[:, :self.rope_kv_heads], cos, sin),
                               k[:, self.rope_kv_heads:]], dim=1)
        # else: all heads are NoPE — q and k keep their RMSNorm'd values as-is
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_hidden: int):
        super().__init__()
        self.fc = CastedLinear(dim, mlp_hidden, bias=False)
        self.proj = CastedLinear(mlp_hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    """Transformer block with optional Multi-Head Hyper-Connections (MHC).

    MHC (hc_streams=N>0): the residual stream is N parallel copies [B,L,N,D].
    Each block mixes them via learned [N,N] matrices:
      x_in  = hc_in  @ streams        (select which blend enters the sublayer)
      x_out  stored back via hc_out @ streams + sublayer_out broadcast
    Both matrices are 2-D → routed to Muon.
    hc_streams=0 → standard resid_mix scalar path (unchanged).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden: int,
        rope_base: float,
        qk_gain_init: float,
        block_idx: int = 0,
        attn_every: int = 4,
        nope_heads: int = 0,
        hc_streams: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        if attn_every <= 1 or block_idx % attn_every == (attn_every - 1):
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, nope_heads=nope_heads)
        else:
            self.attn = GatedDeltaNet(hidden_size=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, expand_k=0.5, expand_v=0.5)
        self.mlp = MLP(dim, mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.hc_streams = hc_streams
        if hc_streams > 0:
            N = hc_streams
            # hc_in/hc_out: [N,N] mixing matrices, init as identity → Muon-updated
            self.hc_in  = nn.Parameter(torch.eye(N, dtype=torch.float32))
            self.hc_out = nn.Parameter(torch.eye(N, dtype=torch.float32))
        else:
            self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, streams: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Returns (x_out, updated_streams). streams is [B, L, N, D] or None."""
        if streams is not None:
            w_in = self.hc_in.to(x.dtype)  # [N, N]
            # Collapse N streams → 1 input via first row of hc_in output
            x = torch.einsum('n m, b l m d -> b l n d', w_in, streams)[:, :, 0, :]
        else:
            mix = self.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))

        if streams is not None:
            w_out = self.hc_out.to(x.dtype)
            streams = torch.einsum('n m, b l m d -> b l n d', w_out, streams) + x.unsqueeze(2)
            return x, streams
        return x, None


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        attn_every: int = 4,
        nope_heads: int = 0,
        hc_streams: int = 0,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.hc_streams = hc_streams
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim, num_heads, num_kv_heads, mlp_hidden,
                    rope_base, qk_gain_init,
                    block_idx=i, attn_every=attn_every,
                    nope_heads=nope_heads, hc_streams=hc_streams,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _run_blocks(self, x: Tensor) -> Tensor:
        x0 = x
        N = self.hc_streams
        streams = x0.unsqueeze(2).expand(-1, -1, N, -1).clone() if N > 0 else None
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x, streams = self.blocks[i](x, x0, streams)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if streams is not None:
                    streams = streams + x.unsqueeze(2)  # keep streams in sync after skip
            x, streams = self.blocks[self.num_encoder_layers + i](x, x0, streams)
        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x = self.final_norm(self._run_blocks(x)).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits [bsz, seq_len, vocab] without loss (for sliding-window eval)."""
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x = self.final_norm(self._run_blocks(x))
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.fx_graph_remote_cache = False
    torch._inductor.config.autotune_local_cache = True
    torch._inductor.config.triton.persistent_reductions = False  # laptop GPU has insufficient registers
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg, flush=True)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # NoPE: snap to GQA-group boundary so KV-head split is always clean
    _gqa_ratio = max(1, args.num_heads // args.num_kv_heads)
    nope_heads = (int(round(args.nope_ratio * args.num_heads)) // _gqa_ratio) * _gqa_ratio
    nope_heads = max(0, min(nope_heads, args.num_heads))
    log0(
        f"nope:nope_ratio={args.nope_ratio} nope_heads={nope_heads}/{args.num_heads} "
        f"rope_heads={args.num_heads - nope_heads}/{args.num_heads}"
    )
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_hidden=args.mlp_hidden,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        attn_every=args.attn_every,
        nope_heads=nope_heads,
        hc_streams=args.hc_streams,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    # hc_in/hc_out [N,N] are 2-D → go to Muon alongside regular weight matrices
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    n_attn_blocks = sum(1 for i in range(args.num_layers) if args.attn_every <= 1 or i % args.attn_every == (args.attn_every - 1))
    n_gdn_blocks = args.num_layers - n_attn_blocks
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"arch:hybrid_gdn_attn layers:{args.num_layers} "
        f"gated_delta_net:{n_gdn_blocks} causal_attn:{n_attn_blocks} "
        f"(attn_every={args.attn_every} → ratio {n_gdn_blocks}:{n_attn_blocks})"
    )
    log0(
        f"model_dim:{args.model_dim} num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} "
        f"mlp_hidden:{args.mlp_hidden} expand_k:0.5 expand_v:0.5"
    )
    log0(
        f"unet:encoder_layers:{base_model.num_encoder_layers} "
        f"decoder_layers:{base_model.num_decoder_layers} "
        f"skip_weights:{base_model.num_skip_weights}"
    )
    hc_desc = f"mhc[N={args.hc_streams},hc_in[N×N]+hc_out[N×N]→Muon]" if args.hc_streams > 0 else "resid_mix"
    log0(f"block_extras:{hc_desc} attn_scale mlp_scale hc_streams={args.hc_streams}")
    log0(f"gdn_backend:{'fla-triton' if _FLA_AVAILABLE else 'pure-pytorch-fallback'} fullgraph=False")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_beta1 = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_beta1

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # sync wallclock cap across ranks
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())

    def _get_layer_num(param_name: str) -> int:
        if "blocks." in param_name:
            try:
                return int(param_name.split("blocks.")[1].split(".")[0])
            except (ValueError, IndexError):
                pass
        return -1

    def _apply_quant_step(qobj: dict, layer_set: set, step: int) -> None:
        """Round int8 weights in `layer_set` to multiples of `step`."""
        for nm in list(qobj.get("quantized", {}).keys()):
            if _get_layer_num(nm) in layer_set:
                t = qobj["quantized"][nm]
                qobj["quantized"][nm] = (
                    (t.float() / step).round() * step
                ).clamp(-127, 127).to(torch.int8)

    def _estimate_compressed_mb(qobj: dict, extra_bytes: int = 0) -> float:
        """Rough size estimate: serialize to BytesIO and zlib-compress."""
        buf = io.BytesIO()
        torch.save(qobj, buf)
        raw = buf.getvalue()
        compressed = zlib.compress(raw, level=9)
        return (len(compressed) + extra_bytes) / (1024 * 1024)

    def auto_assign_quant_tiers(num_layers: int, budget_mb: float, code_size_bytes: int) -> tuple[set, set]:
        """Greedy tier assignment: boundary layers int8, ring int6, middle int4."""
        boundary = max(1, round(0.17 * num_layers))  # ~17% from each end → int8
        ring     = max(0, round(0.13 * num_layers))  # ~13% from each side → int6

        int8_layers  = set(range(boundary)) | set(range(num_layers - boundary, num_layers))
        int6_set_auto = set(range(boundary, boundary + ring)) | set(range(num_layers - boundary - ring, num_layers - boundary))
        int4_set_auto = set(range(num_layers)) - int8_layers - int6_set_auto

        return int4_set_auto, int6_set_auto

    # Parse user-specified tiers
    int4_set: set[int] = {int(x) for x in args.int4_layers.split(",") if x.strip()} if args.int4_layers else set()
    int6_set: set[int] = {int(x) for x in args.int6_layers.split(",") if x.strip()} if args.int6_layers else set()

    # Auto-budget mode: override with auto-assigned tiers
    if args.quant_auto_budget_mb > 0:
        code_bytes_est = len(code.encode("utf-8"))
        int4_set_auto, int6_set_auto = auto_assign_quant_tiers(
            args.num_layers, args.quant_auto_budget_mb, code_bytes_est
        )
        # Only use auto sets if user didn't provide explicit ones
        if not int4_set:
            int4_set = int4_set_auto
        if not int6_set:
            int6_set = int6_set_auto
        log0(
            f"quant_auto_budget_mb:{args.quant_auto_budget_mb:.1f} "
            f"int8_layers:{sorted(set(range(args.num_layers)) - int4_set - int6_set)} "
            f"int6_layers:{sorted(int6_set)} "
            f"int4_layers:{sorted(int4_set)}"
        )

    # Optional: prune near-zero int8 values for better compressibility
    if args.prune_ratio > 0:
        threshold = int(127 * args.prune_ratio)
        for name in list(quant_obj.get("quantized", {}).keys()):
            t = quant_obj["quantized"][name]
            t[t.abs() <= threshold] = 0

    # Apply int6 first (less aggressive), then int4 (int4 wins if both sets overlap)
    if int6_set:
        _apply_quant_step(quant_obj, int6_set, args.int6_step)
        log0(f"quant_int6 step:{args.int6_step} layers:{sorted(int6_set)}")
    if int4_set:
        _apply_quant_step(quant_obj, int4_set, args.int4_step)
        log0(f"quant_int4 step:{args.int4_step} layers:{sorted(int4_set)}")

    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        log0(f"final_eval_mode:sliding_window stride:{args.eval_stride} batch_seqs:{args.eval_batch_seqs}")
        q_val_loss, q_val_bpb = eval_val_sliding(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            batch_seqs=args.eval_batch_seqs,
        )
    else:
        q_val_loss, q_val_bpb = eval_val(
            args,
            model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
