from __future__ import annotations
import copy, glob, io, math, os, random, subprocess, sys, time, uuid, zlib
from pathlib import Path
try:
    import zstandard as _zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
_USE_FLASH_ATTN = os.environ.get("FLASH_ATTN", "1") == "1"
try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func
except ImportError:
    _USE_FLASH_ATTN = False
    _flash_attn_func = None

def _attn_forward(q, k, v):
    if _USE_FLASH_ATTN and _flash_attn_func is not None:
        return _flash_attn_func(q, k, v, causal=True)
    q_ = q.transpose(1, 2); k_ = k.transpose(1, 2); v_ = v.transpose(1, 2)
    if k_.size(1) != q_.size(1):
        rep = q_.size(1) // k_.size(1)
        k_ = k_.repeat_interleave(rep, dim=1); v_ = v_.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q_, k_, v_, is_causal=True).transpose(1, 2)


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    lr_warmup_steps = int(os.environ.get("LR_WARMUP_STEPS", 20))
    lr_switch_step = int(os.environ.get("LR_SWITCH_STEP", "0"))
    lr_switch_factor = float(os.environ.get("LR_SWITCH_FACTOR", "0.1"))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_hidden = int(os.environ.get("MLP_HIDDEN", 1536))
    attn_every = int(os.environ.get("ATTN_EVERY", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    nope_ratio = float(os.environ.get("NOPE_RATIO", 0.25))
    nope_mode = os.environ.get("NOPE_MODE", "block")
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", 0.0))
    use_focal = bool(int(os.environ.get("USE_FOCAL", "1")))
    focal_gamma = float(os.environ.get("FOCAL_GAMMA", 0.45))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.050))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.2))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.01))
    prune_pct = float(os.environ.get("PRUNE_PCT", 0.03))
    int6_last_n = int(os.environ.get("INT6_LAST_N", "0"))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    share_weights = bool(int(os.environ.get("SHARE_WEIGHTS", "0")))
    use_swiglu = bool(int(os.environ.get("USE_SWIGLU", "1")))
    swiglu_hidden = int(os.environ.get("SWIGLU_HIDDEN", "0"))
    use_power_mlp_swiglu = bool(int(os.environ.get("USE_POWER_MLP_SWIGLU", "1")))
    power_mlp_repu_order = int(os.environ.get("POWER_MLP_REPU_ORDER", "2"))
    use_adamuon = bool(int(os.environ.get("USE_ADAMUON", "1")))
    adamuon_beta2 = float(os.environ.get("ADAMUON_BETA2", "0.92175"))
    adamuon_eps = float(os.environ.get("ADAMUON_EPS", "1e-8"))
    use_olion = bool(int(os.environ.get("USE_OLION", "0")))
    olion_beta1 = float(os.environ.get("OLION_BETA1", "0.1"))   # Nesterov mix weight (current grad fraction)
    olion_beta2 = float(os.environ.get("OLION_BETA2", "0.99"))  # slow momentum EMA
    use_lion = bool(int(os.environ.get("USE_LION", "0")))        # use Lion for tok/scalar/head
    lion_beta1 = float(os.environ.get("LION_BETA1", "0.9"))     # update interpolation (momentum fraction)
    lion_beta2 = float(os.environ.get("LION_BETA2", "0.99"))    # slow momentum EMA
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", "0.997"))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    backout_enabled = bool(int(os.environ.get("BACKOUT_ENABLED", "1")))
    backout_lambda_init = float(os.environ.get("BACKOUT_LAMBDA_INIT", "0.24"))
    backout_layer = int(os.environ.get("BACKOUT_LAYER", "-1"))
    use_lora = bool(int(os.environ.get("USE_LORA", "1")))
    lora_rank = int(os.environ.get("LORA_RANK", "16"))
    large_gpu = bool(int(os.environ.get("LARGE_GPU", "0")))
    grad_accum_target = int(os.environ.get("GRAD_ACCUM_TARGET", "8"))
    late_qat = bool(int(os.environ.get("LATE_QAT", "1")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", "0.50"))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", "10"))
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    norm = X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    X = X / norm
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + b * A @ X + c * A @ A @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, backend_steps=5, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]; mo = group["momentum"]; wd = group["weight_decay"]; ns = group["backend_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - mo)
                g = buf
                if g.ndim == 2:
                    g = zeropower_via_newtonschulz5(g, steps=ns)
                    g *= max(1, g.size(-2) / g.size(-1)) ** 0.5
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr)


class AdaMuon(torch.optim.Optimizer):
    """AdaMuon: Muon + per-element Adam second moment in orthogonalized space + RMS-aligned rescaling.
    Algorithm:
      1. EMA momentum:        m = beta*m + (1-beta)*g
      2. Orthogonalize:       o = NS5(m)
      3. Flatten + EMA v²:   v = beta2*v + (1-beta2)*o_flat²
      4. Bias-correct + scale: o_hat = o_flat / (sqrt(v/bc) + eps)
      5. RMS-align:           o_hat *= 0.2 / RMS(o_hat)
      6. Update:              w -= lr * o_hat
    """
    def __init__(self, params, lr=0.02, beta=0.95, beta2=0.92175, eps=1e-8, backend_steps=5, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, beta=beta, beta2=beta2, eps=eps, backend_steps=backend_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]; beta = group["beta"]; beta2 = group["beta2"]
            eps = group["eps"]; ns = group["backend_steps"]; wd = group["weight_decay"]

            total_numel = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_numel, device=params[0].device, dtype=torch.float32)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad.float()
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                    state["step"] += 1
                    t = state["step"]
                    # Step 1: EMA momentum
                    if "moment1" not in state:
                        state["moment1"] = torch.zeros_like(g)
                    m = state["moment1"]
                    m.mul_(beta).add_(g, alpha=1.0 - beta)
                    # Step 2: orthogonalize (2D) or normalize (1D)
                    if g.ndim == 2:
                        o = zeropower_via_newtonschulz5(m, steps=ns).float()
                    else:
                        o = m / (m.norm() + eps)
                    # Step 3-5: per-element adaptive scaling + RMS-align (2D only)
                    if g.ndim == 2:
                        o_flat = o.reshape(-1)
                        if "moment2" not in state:
                            state["moment2"] = torch.zeros(o_flat.numel(), device=p.device, dtype=torch.float32)
                        v = state["moment2"]
                        v.mul_(beta2).addcmul_(o_flat, o_flat, value=1.0 - beta2)
                        bc = 1.0 - beta2 ** t
                        o_hat_flat = o_flat / ((v / bc).sqrt() + eps)
                        o_hat = o_hat_flat.view_as(o)
                        rms = o_hat.norm() / (o_hat.numel() ** 0.5)
                        o = o_hat * (0.2 / rms.clamp_min(eps))
                    updates_flat[curr: curr + p.numel()] = o.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                upd = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(upd, alpha=-lr)
                curr += p.numel()


class OLion(torch.optim.Optimizer):
    """OLion: Orthogonalize → Sign → RMS-align optimizer.
    Algorithm (from OLion paper Algorithm 1):
      1. Slow momentum:   M_t = β2*M_{t-1} + (1-β2)*g_t
      2. Nesterov mix:   G̃_t = (1-β1)*g_t + β1*M_t
      3. Orthogonalize:  Q_t = NS5(G̃_t)
      4. Sign:           S_t = sign(Q_t)
      5. RMS-align:      D_t = 0.2/RMS(S_t) * S_t  (≈ 0.2*S_t since RMS(sign)=1)
      6. Update:         X_{t+1} = X_t - η*D_t - λη*X_t
    Advantage over Muon: sign-after-orthogonalization gives coordinate-wise
    normalization on top of spectral structure, matching both operator-norm
    and ℓ∞ implicit biases simultaneously.
    """
    def __init__(self, params, lr=0.02, beta1=0.1, beta2=0.99,
                 backend_steps=5, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2,
                                      backend_steps=backend_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]; beta1 = group["beta1"]; beta2 = group["beta2"]
            ns = group["backend_steps"]; wd = group["weight_decay"]
            eps = 1e-7

            total_numel = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_numel, device=params[0].device, dtype=torch.float32)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad.float()
                    state = self.state[p]
                    # Step 1: slow momentum buffer
                    if "moment" not in state:
                        state["moment"] = torch.zeros_like(g)
                    m = state["moment"]
                    m.mul_(beta2).add_(g, alpha=1.0 - beta2)
                    # Step 2: Nesterov mix
                    g_mix = g.mul(1.0 - beta1).add_(m, alpha=beta1)
                    # Step 3-5: orthogonalize → sign → RMS-align (2D only)
                    if g.ndim == 2:
                        q = zeropower_via_newtonschulz5(g_mix, steps=ns).float()
                        s = q.sign()
                        rms = s.norm() / (s.numel() ** 0.5)
                        upd = s * (0.2 / rms.clamp_min(eps))
                    else:
                        # 1D params: just normalize
                        upd = g_mix / (g_mix.norm() + eps)
                    updates_flat[curr: curr + p.numel()] = upd.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                upd = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(upd, alpha=-lr)
                curr += p.numel()


class Lion(torch.optim.Optimizer):
    """Lion optimizer for non-matrix params (tok_emb, scalars, head).
    Algorithm:
      c_t = β1 * m_{t-1} + (1 - β1) * g_t   (update direction)
      w_t = w_{t-1} - lr * (sign(c_t) + wd * w_{t-1})
      m_t = β2 * m_{t-1} + (1 - β2) * g_t   (slow momentum)
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.99, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]; beta1 = group["beta1"]; beta2 = group["beta2"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if "moment" not in state:
                    state["moment"] = torch.zeros_like(g)
                m = state["moment"]
                c = beta1 * m + (1.0 - beta1) * g
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(c.sign().to(dtype=p.dtype), alpha=-lr)
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)


def build_sentencepiece_luts(sp, vocab_size: int, device: torch.device):
    base_bytes = torch.zeros(vocab_size, dtype=torch.int32)
    has_leading_space = torch.zeros(vocab_size, dtype=torch.bool)
    is_boundary_token = torch.zeros(vocab_size, dtype=torch.bool)
    for tok_id in range(vocab_size):
        piece = sp.id_to_piece(tok_id)
        raw = sp.decode([tok_id])
        base_bytes[tok_id] = len(piece.replace("▁", "").encode("utf-8")) or 1
        has_leading_space[tok_id] = piece.startswith("▁")
        is_boundary_token[tok_id] = (len(raw.strip()) == 0)
    return base_bytes.to(device), has_leading_space.to(device), is_boundary_token.to(device)


def load_validation_tokens(val_files: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(val_files))
    if not files:
        raise FileNotFoundError(f"No val files: {val_files}")
    shards = [load_data_shard(Path(f)) for f in files]
    tokens = torch.cat(shards, dim=0)
    extra = tokens.numel() % seq_len
    if extra:
        tokens = tokens[:-extra]
    return tokens


def eval_val(args, model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    max_val_seqs = args.val_batch_size // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    max_val_seqs = min(max_val_seqs, total_seqs)
    seq_start = (max_val_seqs * rank) // world_size
    seq_end = (max_val_seqs * (rank + 1)) // world_size
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
                logits = model.forward_logits(x)
                batch_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="mean").detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1); tgt_ids = y.reshape(-1)
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


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file(); continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank; self.world_size = world_size; self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__(); self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    _use_soft_round: bool = False
    _soft_round_alpha: float = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clip_range: int = 15

    @staticmethod
    def soft_round(y: Tensor, alpha: float) -> Tensor:
        fl = torch.floor(y); r = y - fl - 0.5
        return fl + 0.5 * torch.tanh(alpha * r) / (math.tanh(alpha / 2) + 1e-10) + 0.5

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        cr = self._clip_range
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            if CastedLinear._use_soft_round:
                w32 = self.weight.float()
                row_clip = torch.quantile(w32.abs(), 0.9995, dim=1)
                scale = (row_clip / float(cr)).clamp_min(1.0 / float(cr))
                w_scaled = w32 / scale[:, None]
                w_rounded = CastedLinear.soft_round(w_scaled, CastedLinear._soft_round_alpha)
                w = (torch.clamp(w_rounded, -(cr + 1), cr) * scale[:, None]).to(x.dtype)
            else:
                with torch.no_grad():
                    w32 = self.weight.float()
                    row_clip = torch.quantile(w32.abs(), 0.9995, dim=1)
                    scale = (row_clip / float(cr)).clamp_min(1.0 / float(cr))
                    w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -(cr + 1), cr) * scale[:, None]).to(x.dtype)
                w = w + (w_q - w.to(x.dtype)).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",") if p
)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
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
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[:, None, :].unsqueeze(0)
            self._sin_cached = freqs.sin()[:, None, :].unsqueeze(0)
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))
        x_prev = F.pad(x, (0, 0, 1, 0))[:, :-1, :]
        return x + g[None, None, :] * (x_prev - x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, nope_heads: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.nope_heads = min(nope_heads, num_heads)
        self.rope_heads = num_heads - self.nope_heads
        gqa_ratio = num_heads // num_kv_heads
        self.nope_kv_heads = self.nope_heads // gqa_ratio
        self.rope_kv_heads = num_kv_heads - self.nope_kv_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base) if self.rope_heads > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if self.rotary is not None:
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            if self.nope_heads == 0:
                q = apply_rotary_emb(q, cos, sin)
                k = apply_rotary_emb(k, cos, sin)
            else:
                q = torch.cat([apply_rotary_emb(q[:, :, :self.rope_heads], cos, sin),
                               q[:, :, self.rope_heads:]], dim=2)
                k = torch.cat([apply_rotary_emb(k[:, :, :self.rope_kv_heads], cos, sin),
                               k[:, :, self.rope_kv_heads:]], dim=2)
        y = _attn_forward(q, k, v)
        return self.proj(y.reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_hidden: int):
        super().__init__()
        self.fc = CastedLinear(dim, mlp_hidden, bias=False)
        self.proj = CastedLinear(mlp_hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = CastedLinear(dim, hidden, bias=False)
        self.val_proj  = CastedLinear(dim, hidden, bias=False)
        self.down_proj = CastedLinear(hidden, dim, bias=False)
        self.down_proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.val_proj(x))


class PowerMLPSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, repu_order: int = 2):
        super().__init__()
        self.gate_proj = CastedLinear(dim, hidden, bias=False)
        self.val_proj  = CastedLinear(dim, hidden, bias=False)
        self.down_proj = CastedLinear(hidden, dim, bias=False)
        self.down_proj._zero_init = True
        self.repu_order = repu_order

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        return self.down_proj((F.silu(gate).pow(2) * self.val_proj(x)))


# Bayesian + PowerMLP WIP
class BPMLPSwiglu(nn.Module):
    def __init__(self, dim: int, hidden: int, repu_order: int = 2):
        super().__init__()
        self.gate_proj = CastedLinear(dim, hidden, bias=False)
        self.val_proj  = CastedLinear(dim, hidden, bias=False)
        self.down_proj = CastedLinear(hidden, dim, bias=False)
        self.down_proj._zero_init = True
        self.repu_order = repu_order

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        return self.down_proj((F.relu(gate).pow(self.repu_order) + F.silu(gate)) * self.val_proj(x))




class BwdLoRADelta(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        self.B._zero_init = True
        self.switch = nn.Parameter(torch.zeros((), dtype=torch.float32))
        nn.init.orthogonal_(self.A.weight)

    def forward(self, x: Tensor) -> Tensor:
        x_prev = F.pad(x, (0, 0, 1, 0))[:, :-1, :]
        return torch.tanh(self.switch) * self.B(self.A(x_prev))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_hidden: int,
                 rope_base: float, qk_gain_init: float, block_idx: int = 0,
                 nope_heads: int = 0, use_swiglu: bool = False, swiglu_hidden: int = 0,
                 ln_scale: bool = False, use_lora: bool = False, lora_rank: int = 16,
                 use_power_mlp_swiglu: bool = False, power_mlp_repu_order: int = 2):
        super().__init__()

        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, nope_heads=nope_heads)
        _sh = swiglu_hidden if swiglu_hidden > 0 else max(1, round(mlp_hidden * 2 / 3))
        if use_power_mlp_swiglu and use_swiglu:
            self.mlp = PowerMLPSwiGLU(dim, _sh, power_mlp_repu_order)
        elif use_swiglu:
            self.mlp = SwiGLUMLP(dim, _sh)
        else:
            self.mlp = MLP(dim, mlp_hidden)

        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.lora: BwdLoRADelta | None = BwdLoRADelta(dim, lora_rank) if use_lora else None
        self.ln_scale_factor = 1.0 / math.sqrt(block_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        dt = x.dtype
        mix = self.resid_mix.to(dtype=dt)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s = self.ln_scale_factor

        

        # main math function inside
        x = x + self.attn_scale.to(dtype=dt)[None, None, :] * self.attn(self.attn_norm(x) * s)
        mlp_out = self.mlp_scale.to(dtype=dt)[None, None, :] * self.mlp(self.mlp_norm(x) * s)
        
        
        if self.lora is not None:
            mlp_out = mlp_out + self.lora(x)
        return x + mlp_out


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_hidden: int, tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 nope_heads: int = 0, nope_ratio: float = 0.0, nope_mode: str = "head",
                 share_weights: bool = False, use_swiglu: bool = False, swiglu_hidden: int = 0,
                 use_power_mlp_swiglu: bool = False, power_mlp_repu_order: int = 3,
                 ln_scale: bool = False, backout_enabled: bool = False,
                 backout_lambda_init: float = 0.2, backout_layer: int = -1,
                 label_smoothing: float = 0.0, use_focal: bool = True, focal_gamma: float = 0.45,
                 use_lora: bool = False, lora_rank: int = 16):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.share_weights = share_weights
        self.backout_lambda = nn.Parameter(torch.tensor(backout_lambda_init, dtype=torch.float32)) if backout_enabled else None
        self.backout_layer = backout_layer if backout_layer >= 0 else self.num_encoder_layers // 2
        num_unique_blocks = self.num_encoder_layers if share_weights else num_layers
        _nope_period = max(1, round(1.0 / nope_ratio)) if (nope_mode == "block" and nope_ratio > 0) else 0
        def _block_nope(i: int) -> int:
            if _nope_period > 0:
                return num_heads if (i % _nope_period == 0) else 0
            return nope_heads
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_hidden, rope_base, qk_gain_init,
                  block_idx=i, nope_heads=_block_nope(i), use_swiglu=use_swiglu,
                  swiglu_hidden=swiglu_hidden, ln_scale=ln_scale, use_lora=use_lora,
                  lora_rank=lora_rank, use_power_mlp_swiglu=use_power_mlp_swiglu,
                  power_mlp_repu_order=power_mlp_repu_order)
            for i in range(num_unique_blocks)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and "lora" not in name:
                    nn.init.orthogonal_(module.weight)
                    if any(s in name for s in (".proj", "down_proj", ".c_o")):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def _run_blocks(self, x: Tensor) -> Tensor:
        x0 = x
        skips: list[Tensor] = []
        x_backout: Tensor | None = None
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
            if self.backout_lambda is not None and i == self.backout_layer:
                x_backout = x
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            blk_idx = (self.num_encoder_layers - 1 - i) if self.share_weights else (self.num_encoder_layers + i)
            x = self.blocks[blk_idx](x, x0)
            if self.backout_lambda is not None and (self.num_encoder_layers + i) == self.backout_layer:
                x_backout = x
        if x_backout is not None and self.backout_lambda is not None:
            x = x - self.backout_lambda.to(dtype=x.dtype) * x_backout
        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x = self.smear(x)
        hidden = self.final_norm(self._run_blocks(x))
        h_flat = hidden.reshape(-1, hidden.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(h_flat, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(h_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if self.use_focal:
            per_token_ce = F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=self.label_smoothing)
            focal_weight = (1.0 - (-per_token_ce.detach()).exp()).pow(self.focal_gamma)
            return (focal_weight * per_token_ce).mean()
        return F.cross_entropy(logits.float(), targets, reduction="mean", label_smoothing=self.label_smoothing)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x = self.smear(x)
        x = self.final_norm(self._run_blocks(x))
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_int6_per_row(t: Tensor, clip_range: int = 15) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    amax = t.abs().max().clamp(min=1e-12)
    scale = amax / 127.0
    q = (t / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.float()


def mixed_quantize_int6(state_dict: dict, int6_cats: set, int6_last_n: int = 0, num_layers: int = 0) -> tuple[dict, dict]:
    import re
    result: dict = {}; meta: dict = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"; continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float(); meta[name] = "passthrough_ctrl"; continue
        clip_range = 15
        if int6_last_n > 0 and num_layers > 0:
            m = re.search(r'blocks\.(\d+)\.', name)
            if m and int(m.group(1)) >= num_layers - int6_last_n:
                clip_range = 31
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t, clip_range=clip_range)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = {"type": "int5" if clip_range == 15 else "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def dequantize_mixed_int6(result: dict, meta: dict, template_sd: dict) -> dict:
    out: dict = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t; continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, args.grad_accum_target // world_size)
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
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_mem_efficient_sdp, enable_math_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
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
    log0(f"Running Python {sys.version}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_dir = Path(args.data_path).resolve()
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_loader:tokens:{val_tokens.numel() - 1}")
    _gqa_ratio = max(1, args.num_heads // args.num_kv_heads)
    nope_heads = (int(round(args.nope_ratio * args.num_heads)) // _gqa_ratio) * _gqa_ratio
    nope_heads = max(0, min(nope_heads, args.num_heads))
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_hidden=args.mlp_hidden,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        nope_heads=nope_heads, nope_ratio=args.nope_ratio, nope_mode=args.nope_mode,
        share_weights=args.share_weights, use_swiglu=args.use_swiglu, swiglu_hidden=args.swiglu_hidden,
        use_power_mlp_swiglu=args.use_power_mlp_swiglu, power_mlp_repu_order=args.power_mlp_repu_order,
        ln_scale=args.ln_scale, backout_enabled=args.backout_enabled,
        backout_lambda_init=args.backout_lambda_init, backout_layer=args.backout_layer,
        label_smoothing=args.label_smoothing, use_focal=args.use_focal, focal_gamma=args.focal_gamma,
        use_lora=args.use_lora, lora_rank=args.lora_rank,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    with torch.no_grad():
        for module in base_model.modules():
            if isinstance(module, Rotary):
                module(args.train_seq_len, device, torch.bfloat16)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False) if args.large_gpu else base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    CastedLinear._qat_enabled = args.qat_enabled
    int6_start = args.num_layers - args.int6_last_n
    for i, block in enumerate(base_model.blocks):
        if i >= int6_start and args.int6_last_n > 0:
            for m in block.modules():
                if isinstance(m, CastedLinear):
                    m._clip_range = 31
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} world_size:{world_size} grad_accum:{grad_accum_steps}")
    log0(f"layers:{args.num_layers} dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} mlp:{args.mlp_hidden}")
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params
                     if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for name, p in block_named_params
                     if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.backout_lambda is not None:
        scalar_params.append(base_model.backout_lambda)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = (
        Lion([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
             beta1=args.lion_beta1, beta2=args.lion_beta2, weight_decay=args.weight_decay)
        if args.use_lion else
        torch.optim.Adam(
            [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    )
    optimizer_muon = (
        OLion(matrix_params, lr=args.matrix_lr, beta1=args.olion_beta1, beta2=args.olion_beta2,
              backend_steps=args.muon_backend_steps, weight_decay=args.weight_decay)
        if args.use_olion else
        AdaMuon(matrix_params, lr=args.matrix_lr, beta=args.muon_momentum, beta2=args.adamuon_beta2,
                eps=args.adamuon_eps, backend_steps=args.muon_backend_steps, weight_decay=args.weight_decay)
        if args.use_adamuon else
        Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
             backend_steps=args.muon_backend_steps, weight_decay=args.weight_decay)
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = (
        Lion([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
             beta1=args.lion_beta1, beta2=args.lion_beta2, weight_decay=args.weight_decay)
        if args.use_lion else
        torch.optim.Adam(
            [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = (
            Lion([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
                 beta1=args.lion_beta1, beta2=args.lion_beta2, weight_decay=args.weight_decay)
            if args.use_lion else
            torch.optim.Adam(
                [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
                betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        )
        optimizers.insert(1, optimizer_head)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        warmup_scale = min(step / args.lr_warmup_steps, 1.0) if args.lr_warmup_steps > 0 else 1.0
        if args.warmdown_iters <= 0:
            return warmup_scale
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            decay = max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
            return warmup_scale * decay
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        decay = remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
        return warmup_scale * decay

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

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
            log0(f"warmup:{warmup_step + 1}/{args.warmup_steps} loss:{warmup_loss.item():.4f}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
    training_time_ms = 0.0
    stop_after_step: int | None = None
    ema_state: dict[str, Tensor] | None = None
    if args.ema_enabled:
        ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    swa_state: dict[str, Tensor] | None = None
    swa_count: int = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, val_tokens,
                                         base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early:wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        _past_warmup = step >= max(args.lr_warmup_steps, 1)
        if args.late_qat and not CastedLinear._qat_enabled and _past_warmup and scale < args.late_qat_threshold:
            CastedLinear._qat_enabled = True
            CastedLinear._use_soft_round = os.environ.get("SOFT_ROUND_QAT", "0") == "1"
            for m in base_model.modules():
                if isinstance(m, CastedLinear):
                    m._clip_range = 31
            log0(f"qat:enabling int6 STE at step:{step} lr_scale:{scale:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            crownq_lambda = float(os.environ.get("CROWN_Q_LAMBDA", "0.01"))
            if CastedLinear._qat_enabled and crownq_lambda > 0:
                cq_loss = torch.zeros((), device=device)
                for m in base_model.modules():
                    if isinstance(m, CastedLinear) and m.weight.ndim == 2:
                        w = m.weight.float(); cr = float(m._clip_range)
                        row_max = w.detach().abs().amax(dim=1)
                        cq_loss = cq_loss + (w.pow(2) * row_max.pow(2).unsqueeze(1) / cr ** 2).mean()
                loss = loss + crownq_lambda * cq_loss / 12.0
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_beta1 = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            if args.use_adamuon:
                group["beta"] = muon_beta1
            else:
                group["momentum"] = muon_beta1
        if args.lr_switch_step > 0 and step == args.lr_switch_step:
            for opt in optimizers:
                for group in opt.param_groups:
                    group["base_lr"] *= args.lr_switch_factor
            log0(f"lr_switch:step:{step} factor:{args.lr_switch_factor}")
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        step += 1
        if ema_state is not None:
            d = args.ema_decay
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            with torch.no_grad():
                if swa_state is None:
                    swa_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
                    swa_count = 1
                else:
                    for n, t in base_model.state_dict().items():
                        swa_state[n].mul_(swa_count / (swa_count + 1)).add_(t.detach().float(), alpha=1.0 / (swa_count + 1))
                    swa_count += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak_mem:{torch.cuda.max_memory_allocated() // 1024 // 1024}MiB")
    if swa_state is not None:
        log0(f"swa:applying {swa_count} snapshots")
        base_model.load_state_dict({n: t.to(dtype=base_model.state_dict()[n].dtype) for n, t in swa_state.items()}, strict=True)
        del swa_state
    if ema_state is not None:
        log0("ema:applying EMA weights")
        base_model.load_state_dict({n: t.to(dtype=base_model.state_dict()[n].dtype) for n, t in ema_state.items()}, strict=True)
        del ema_state
    export_sd = base_model.state_dict()
    if master_process:
        torch.save(export_sd, "final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"model_bytes:{os.path.getsize('final_model.pt')} code_bytes:{code_bytes}")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    if args.prune_pct > 0:
        for k, v in sd_cpu.items():
            if v.ndim == 2 and v.numel() > 65536:
                thresh = torch.quantile(v.abs().float(), args.prune_pct)
                v[v.abs() < thresh] = 0.0
        log0(f"pruning:{args.prune_pct * 100:.1f}% applied")
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"}, int6_last_n=args.int6_last_n, num_layers=args.num_layers)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    if _HAS_ZSTD:
        quant_blob = _zstd.ZstdCompressor(level=22).compress(quant_raw); _compress_tag = "zstd-22"
    else:
        quant_blob = zlib.compress(quant_raw, level=9); _compress_tag = "zlib-9"
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        log0(f"quant:{_compress_tag} bytes:{len(quant_blob)} total:{len(quant_blob) + len(code.encode('utf-8'))}")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_raw_disk = _zstd.ZstdDecompressor().decompress(quant_blob_disk) if _HAS_ZSTD else zlib.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(quant_raw_disk), map_location="cpu")
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    base_model.load_state_dict(deq_state, strict=True)
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, base_model, rank, world_size, device,
                                     val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0*(time.perf_counter()-t_qeval):.0f}ms")
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
