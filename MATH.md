# Training Pipeline — Mathematical Reference

> Derived from `train_gpt_smear_attn_hopper.py` + `big_scale_script_hopper.sh`.
> All variable names match the `Hyperparameters` env vars exactly.

---

## 0. Current Run Values

| Symbol | Env var | Value |
|--------|---------|-------|
| $T$ | `ITERATIONS` | 20 000 |
| $T_{wd}$ | `WARMDOWN_ITERS` | 3 000 |
| $T_{wu}^{LR}$ | `LR_WARMUP_STEPS` | 20 |
| $t_{sw}$ | `LR_SWITCH_STEP` | 1 000 |
| $f_{sw}$ | `LR_SWITCH_FACTOR` | 0.1 |
| $W$ | `MAX_WALLCLOCK_SECONDS` | 600 s |
| $\eta_0^{mat}$ | `MATRIX_LR` | 0.04 |
| $\eta_0^{tok}$ | `TIED_EMBED_LR` | 0.05 |
| $\eta_0^{sc}$ | `SCALAR_LR` | 0.04 |
| $\beta_{mu,0}$ | `MUON_MOMENTUM_WARMUP_START` | 0.92 |
| $\beta_{mu}$ | `MUON_MOMENTUM` | 0.99 |
| $T_{mu}$ | `MUON_MOMENTUM_WARMUP_STEPS` | 1 500 |
| $\beta_2^{AM}$ | `ADAMUON_BETA2` | 0.921 75 |
| $\gamma$ | `FOCAL_GAMMA` | 0.45 |
| $c$ | `LOGIT_SOFTCAP` | 30.0 |
| $\tau_{qat}$ | `LATE_QAT_THRESHOLD` | 0.10 |
| $N$ | `NUM_LAYERS` | 11 |

---

## 1. Learning-Rate Schedule

The **effective LR** for every optimizer group at step $t$ is:

$$\eta(t) = \eta_{\text{base}}(t) \cdot \text{scale}(t)$$

### 1.1 Base-LR (with LR switch)

$$
\eta_{\text{base}}(t) =
\begin{cases}
\eta_0 & t < t_{sw} \\
\eta_0 \cdot f_{sw} & t \ge t_{sw}
\end{cases}
$$

> Current: $t_{sw}=1000$, $f_{sw}=0.1$ → base LR drops **10×** at step 1 000.

### 1.2 Scale factor (warmup × warmdown)

**Warmup ramp** (linear, over $T_{wu}^{LR}$ main training steps):

$$
s_{wu}(t) = \min\!\left(\frac{t}{T_{wu}^{LR}},\; 1\right), \quad T_{wu}^{LR} > 0
$$

**Warmdown decay** (wall-clock driven, because `MAX_WALLCLOCK_SECONDS` is set):

Let $\bar{t}_{step}$ be the rolling average ms per step, then:

$$
t_{step} = \frac{\text{elapsed\_ms}}{\max(t, 1)}, \qquad
T_{wd}^{ms} = T_{wd} \cdot t_{step}
$$

$$
r_{ms} = \max(W_{ms} - \text{elapsed\_ms},\; 0)
$$

$$
s_{wd}(t) =
\begin{cases}
\dfrac{r_{ms}}{T_{wd}^{ms}} & r_{ms} \le T_{wd}^{ms} \\
1 & \text{otherwise}
\end{cases}
$$

**Combined scale** (both multiply):

$$
\boxed{\text{scale}(t) = s_{wu}(t) \cdot s_{wd}(t)}
$$

### 1.3 Timeline sketch (wall-clock 600 s, ~69 ms/step)

```
Step 0          20        1000                 ~5700         ~8690
  |─ warmup ─|─ lr-ramp ─|── base drop 10x ──|── warmdown ──|  stop
   s_wu ramps 0→1        η_base *= 0.1        scale(t) ramps down
```

---

## 2. Muon / AdaMuon Momentum Warmup

The first-moment decay $\beta_1(t)$ is linearly warmed from $\beta_{\mu,0}$ to $\beta_\mu$ over $T_{mu}$ steps:

$$
\text{frac}(t) = \min\!\left(\frac{t}{T_{mu}},\; 1\right)
$$

$$
\boxed{\beta_1(t) = (1 - \text{frac})\,\beta_{\mu,0} + \text{frac}\,\beta_\mu}
$$

> Current: $0.92 \to 0.99$ over 1 500 steps.

---

## 3. Optimizers

### 3.1 Muon (plain, reference)

Newton-Schulz 5th-order iteration to orthogonalize gradient $G$:

$$
X_0 = \frac{G}{\|G\|}, \qquad
A_k = X_k X_k^\top, \qquad
X_{k+1} = a\,X_k + (b\,A_k + c\,A_k^2)\,X_k
$$

with $(a,b,c) = (3.4445,\,-4.7750,\,2.0315)$, run for `backend_steps` = 5 iterations.

Weight update with weight decay:

$$
W_{t+1} = W_t\,(1 - \eta\,\lambda) - \eta \cdot \underbrace{\sqrt{\frac{\max(m,n)}{\min(m,n)}}}_{\text{RMS align}} \cdot \text{NS}(m_t)
$$

where $m_t = \beta_1 m_{t-1} + g_t$ (EMA momentum with optional Nesterov).

### 3.2 AdaMuon (active: `USE_ADAMUON=1`)

Six steps per parameter matrix:

| Step | Formula |
|------|---------|
| 1. EMA momentum | $M_t = \beta_1 M_{t-1} + (1-\beta_1)\,G_t$ |
| 2. Orthogonalize | $O_t = \text{NS}(M_t)$ |
| 3. Second moment | $v_t = \beta_2 v_{t-1} + (1-\beta_2)\,o_t^2$ (element-wise, flattened) |
| 4. Bias correction | $\hat v_t = v_t \big/ (1 - \beta_2^t)$ |
| 5. Adaptive scale | $\hat o_t = o_t \big/ (\sqrt{\hat v_t} + \varepsilon)$ |
| 6. RMS align + update | $W_{t+1} = W_t(1-\eta\lambda) - \eta \cdot \dfrac{0.2}{\text{RMS}(\hat O_t)+\varepsilon}\,\hat O_t$ |

> Current: $\beta_1(t)$ as above, $\beta_2=0.92175$, $\varepsilon=10^{-8}$, $\lambda=0.01$.

### 3.3 Adam (embeddings, scalars, head)

Standard fused Adam — no surprises.

---

## 4. Gradient Clipping

Applied **before** optimizer steps, every step:

$$
g_{\theta} \leftarrow g_{\theta} \cdot \min\!\left(1,\; \frac{C}{\|g_\theta\|_2}\right), \quad C = 0.2
$$

---

## 5. Forward Pass

### 5.1 Input embedding

$$
x = \text{RMSNorm}(\text{TokEmb}(\text{ids}))
$$

Then passed through a **SmearGate** (learned interpolation between `x` and a shifted/smeared version).

### 5.2 Depth scale (LN_SCALE=1)

Each block $\ell$ scales its normed input by:

$$s_\ell = \frac{1}{\sqrt{\ell + 1}}$$

### 5.3 U-Net block forward

Each block receives both the **current state** $x$ and the **initial embedding** $x_0$. The residual mix blends them:

$$
\tilde x = \alpha_\ell \cdot x + \beta_\ell \cdot x_0, \quad (\alpha_\ell, \beta_\ell) = \text{resid\_mix}_\ell
$$

Then:

$$
x \leftarrow \tilde x + \gamma_\ell^{attn} \cdot \text{Attn}(\text{RMSNorm}(\tilde x) \cdot s_\ell)
$$

$$
x \leftarrow x + \gamma_\ell^{mlp} \cdot \text{MLP}(\text{RMSNorm}(x) \cdot s_\ell)
$$

where $\gamma_\ell^{attn}$, $\gamma_\ell^{mlp}$ are learned per-channel scale vectors (init = 1).

**U-Net skip connections** (encoder → decoder):

$$
x_{\text{dec}, i} \leftarrow x_{\text{dec}, i} + w_i \cdot x_{\text{enc}, N_e - i}
$$

where $N_e = \lfloor N/2 \rfloor = 5$ and $w_i$ are learned per-channel vectors.

**Backout** (BACKOUT_ENABLED=1, layer=2, $\lambda_0=0.24$):

Save $x_{\text{bo}}$ at encoder layer 2. After all decoder layers:

$$
x_{\text{final}} \leftarrow x - \lambda \cdot x_{\text{bo}}
$$

$\lambda$ is a learned scalar initialized to 0.24.

### 5.4 RoPE positional encoding

$$
\theta_k = \frac{1}{b^{2k/d}}, \quad b = 10000, \quad k = 0,\ldots,d/2-1
$$

$$
\text{cos\_cache}[t,k] = \cos(t\,\theta_k), \quad \text{sin\_cache}[t,k] = \sin(t\,\theta_k)
$$

Applied to each query/key head via rotation of half-dimension pairs:

$$
(x_1, x_2) \mapsto (x_1 \cos - x_2 \sin,\; x_1 \sin + x_2 \cos)
$$

---

## 6. MLP Variants (current run: USE_POWER_MLP_SWIGLU + USE_BAYES_MLP + USE_SWIGLU)

| Block type | MLP |
|---|---|
| RoPE blocks | PowerMLPSwiGLU |
| NoPE blocks (NOPE_RATIO=0 → none in this run) | BayesianSwiGLUMLP |

### 6.1 PowerMLPSwiGLU (RoPE blocks)

$$
g = W_{\text{gate}}\,x, \quad v = W_{\text{val}}\,x
$$

$$
\text{act}(g) = \underbrace{\text{ReLU}(g)^n}_{\text{RePU}} + \underbrace{\text{SiLU}(g)}_{\text{ResSiLU}}, \quad n=2
$$

$$
\text{out} = W_{\text{down}}\,(\text{act}(g) \odot v)
$$

Hidden dim: $H' = \lfloor 2/3 \cdot 1536 \rceil = 1024$.

### 6.2 BayesianSwiGLUMLP (NoPE blocks — inactive in this config)

Same SwiGLU structure but with reparameterized weight noise at train time:

$$
\text{pre\_gate} \sim \mathcal{N}(W_{\text{gate}}\,x,\; \text{softplus}(\sigma_{\text{gate}})^2)
$$

At eval: noise term drops out → deterministic forward.

---

## 7. Attention

Grouped Query Attention with $H=8$ heads, $H_{kv}=4$ KV-heads, head dim $d_h = D/H = 64$.

**Learned QK gain** (init $= 1.5$, scalar per head):

$$
\text{scores} = \frac{(Q\,g) \cdot K^\top}{\sqrt{d_h}}
$$

**Logit softcap** (applied to final logits, not attention scores).

---

## 8. Output & Loss

### 8.1 Logit softcap

$$
\hat\ell = c \cdot \tanh\!\left(\frac{W_e^\top h}{c}\right), \quad c = 30
$$

Tied embeddings ($W_e$ is the same as tok\_emb weight).

### 8.2 Focal loss (FOCAL_GAMMA=0.45)

Per-token cross-entropy: $\ell_i = \text{CE}(\hat\ell_i, y_i)$

Focal weight: $f_i = (1 - e^{-\ell_i})^{\gamma}$

$$
\mathcal{L} = \frac{1}{|B|} \sum_i f_i \cdot \ell_i, \quad \gamma = 0.45
$$

> At $\gamma=0$ this is plain CE. At $\gamma=0.45$ the weight curve is mild — easy tokens (low CE) get ~0.7× the gradient, hard tokens stay at ~1×.

### 8.3 Validation metric

$$
\text{bpb} = \frac{\mathcal{L}_{\text{val}}}{\ln 2} \cdot \frac{\text{tokens}}{\text{bytes}}
$$

---

## 9. Post-Training Averaging

### 9.1 SWA (every 10 steps)

Arithmetic average of all snapshots collected at multiples of 10:

$$
\theta_{\text{SWA}}^{(k)} = \frac{k-1}{k}\,\theta_{\text{SWA}}^{(k-1)} + \frac{1}{k}\,\theta_k
$$

~869 snapshots at step 8 690.

### 9.2 EMA (disabled in this run)

$$
\theta_{\text{EMA}} \leftarrow d\,\theta_{\text{EMA}} + (1-d)\,\theta
$$

---

## 10. Quantization

### 10.1 Base: per-row int8

For each 2-D weight matrix $W \in \mathbb{R}^{m \times n}$:

$$
s_i = \frac{\text{clip}_{99.99984\%}(|W_{i,:}|)}{127}, \quad
W^q_{i,j} = \text{round}\!\left(\text{clip}\!\left(\frac{W_{i,j}}{s_i},\,-127,\,127\right)\right)
$$

Dequantize: $\hat W_{i,j} = W^q_{i,j} \cdot s_i$

Small tensors ($\le 65536$ elements): stored as fp16 passthrough.  
Control scalars (`attn_scale`, `mlp_scale`, `resid_mix`, …): stored as fp32 passthrough.

### 10.2 Tier assignment (QUANT_AUTO_BUDGET_MB=15, DISABLE_INT4=1)

Auto-budget with int4 disabled → only int8 / int6 tiers:

```
boundary = round(0.17 × 11) = 2   →  layers {0, 1, 9, 10}  → int8
ring     = round(0.13 × 11) = 1   →  layers {2, 8}          → int6
middle                              →  layers {3,4,5,6,7}    → int6  (promoted from int4)
```

### 10.3 int6 re-rounding (step=4)

Applied to the already-quantized int8 values:

$$
W^{q6}_{i,j} = \text{clip}\!\left(\text{round}\!\left(\frac{W^q_{i,j}}{4}\right) \cdot 4,\,-127,\,127\right)
$$

Effective precision: $\pm 31$ distinct levels (127/4 = 31.75 → 32 levels per side = ~6 bits).

### 10.4 MLP_INT6 / ATTN_INT6 overrides

After tier assignment, **all** `.mlp.*` and `.attn.*` tensors in every layer are re-rounded to step=4, regardless of their auto-assigned tier. This means int8-assigned boundary layers also get int6 for their MLP and attention projections.

### 10.5 QAT alignment (LATE_QAT=1)

During training, once `scale(t) < τ_{qat} = 0.10` **and** `t ≥ T_{wu}^{LR} = 20`:

$$
w_{\text{fwd}} = w + \text{sg}\!\left(\text{round}\!\left(\frac{w}{s}\right) \cdot s - w\right), \quad s_i = \frac{\max_j |w_{i,j}|}{31}
$$

(Straight-Through Estimator — gradient flows through $w$, forward uses the int6 grid.)

---

## 11. LR Switch — Effect Summary

```
Before step 1000:   η(t) = η₀ · s_wu(t) · s_wd(t)        (s_wd ≈ 1.0, still early)
At step 1000:       η_base *= 0.1                           (one-time, permanent)
After step 1000:    η(t) = (η₀ × 0.1) · s_wu(t) · s_wd(t)
```

At step 1000 with $T_{wu}^{LR}=20$ already complete, $s_{wu}=1.0$, $s_{wd} \approx 1.0$:

$$
\eta_{\text{matrix}}: 0.04 \to 0.004
\quad
\eta_{\text{embed}}: 0.05 \to 0.005
\quad
\eta_{\text{scalar}}: 0.04 \to 0.004
$$

QAT will trigger when the combined scale drops below 0.10 — with the base already halved by the switch, warmdown needs to push scale below ~0.10 of the **new** base, which happens roughly when $r_{ms} \le 0.10 \cdot T_{wd}^{ms}$, i.e. in the last ~300 s of wall time.
