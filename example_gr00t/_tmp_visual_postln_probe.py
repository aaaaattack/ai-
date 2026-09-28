import numpy as np
from pathlib import Path

base = Path("/data01/home/wei.xie/v1.4.0_test/workspace/917_embodied_ai/embodied_ai/gr00t")
vis = base / "output/xh2/hmquant/visual"
gold = base / "work_dirs/golden/visual/post_quant/hmquant_gr00t_with_act"

fx = np.load(vis / "four_way/fx/hmquant_gr00t_output_latent_output.npy").astype(np.float32)[0]
hm = np.load(vis / "post_quant/hmquant_gr00t_output_latent_output.npy").astype(np.float32)[0]
wrap = np.load(vis / "pre_quant/hmquant_gr00t_output_latent_output.npy").astype(np.float32)[0]
inp = np.load(vis / "post_quant/hmquant_gr00t_windows_tensor_input.npy").astype(np.float32)
print("input shape", inp.shape, "fx/hm", fx.shape, hm.shape)


def cos(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def token_stats(name, x):
    mean = x.mean(-1)
    std = x.std(-1)
    print(
        name,
        "token-mean min/mean/max",
        float(mean.min()),
        float(mean.mean()),
        float(mean.max()),
        "token-std min/mean/max",
        float(std.min()),
        float(std.mean()),
        float(std.max()),
    )


token_stats("fx", fx)
token_stats("hm", hm)
token_stats("wrap", wrap)

# If both are proper LN outputs, token mean ~ 0. If one skipped LN, mean may be large.
print("fx vs hm token-mean cosine", cos(fx.mean(-1), hm.mean(-1)))
print("fx vs hm token-std cosine", cos(fx.std(-1), hm.std(-1)))

# per-channel affine
scale = (fx * hm).sum(0) / np.maximum((fx ** 2).sum(0), 1e-12)
bias = hm.mean(0) - scale * fx.mean(0)
print("per-ch scale min/mean/max", float(scale.min()), float(scale.mean()), float(scale.max()))
print("per-ch bias min/mean/max", float(bias.min()), float(bias.mean()), float(bias.max()))
print("cos after scale", cos(fx * scale, hm))
print("cos after affine", cos(fx * scale + bias, hm))

# Would last LN over dim=1 (tokens) explain it?
# Apply token-axis LN to fx and compare to hm
fx_m = fx.mean(0, keepdims=True)
fx_s = fx.std(0, keepdims=True) + 1e-5
fx_tokln = (fx - fx_m) / fx_s
print("cos fx token-LN vs hm", cos(fx_tokln, hm))
print("cos fx token-LN vs fx", cos(fx_tokln, fx))
hm_m = hm.mean(0, keepdims=True)
hm_s = hm.std(0, keepdims=True) + 1e-5
hm_tokln = (hm - hm_m) / hm_s
print("cos hm token-LN vs fx", cos(hm_tokln, fx))

# channel-axis LN (standard) on hm vs fx
def ln_last(x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)

print("cos ln(fx) vs ln(hm)", cos(ln_last(fx), ln_last(hm)))
print("cos ln(fx) vs hm", cos(ln_last(fx), hm))
print("cos fx vs ln(hm)", cos(fx, ln_last(hm)))

# intermediates
def load_gold(name):
    p = gold / name
    if not p.exists():
        return None
    arr = np.load(p)
    print("gold", name, arr.shape, arr.dtype, "min/max", float(arr.min()), float(arr.max()))
    return arr


add53 = load_gold("add_53.npy")
out_g = load_gold("output_latent.npy")
ln0 = load_gold("layer_norm.npy")
win = load_gold("windows_tensor.npy")
add0 = load_gold("add.npy")  # maybe first residual
linear0 = load_gold("linear.npy")
reshape0 = load_gold("reshape.npy")
matmul0 = load_gold("mat_mul.npy")
softmax0 = load_gold("softmax.npy")
mul0 = load_gold("mul.npy")

if out_g is not None:
    og = np.squeeze(out_g).astype(np.float32)
    if og.ndim == 3:
        og = og[0]
    print("gold output vs staged hm", cos(og, hm), "shape", og.shape)

if add53 is not None:
    a = np.squeeze(add53).astype(np.float32)
    if a.ndim == 3:
        a = a[0]
    print("add53 shape", a.shape)
    token_stats("add53", a)
    print("cos add53 vs hm", cos(a, hm))
    print("cos add53 vs fx", cos(a, fx))
    print("cos ln(add53) vs hm", cos(ln_last(a), hm))
    print("cos ln(add53) vs fx", cos(ln_last(a), fx))
    print("cos ln(add53) vs ln(fx)", cos(ln_last(a), ln_last(fx)))
    print("cos ln(add53) vs ln(hm)", cos(ln_last(a), ln_last(hm)))
    # per-ch relation add53 -> hm (this is post LN)
    sc = (a * hm).sum(0) / np.maximum((a ** 2).sum(0), 1e-12)
    print("add53->hm per-ch scale min/mean/max", float(sc.min()), float(sc.mean()), float(sc.max()))
    print("cos add53*scale vs hm", cos(a * sc, hm))
    # standard LN then unknown gamma: correlate gamma with scale_from_fx
    ln_a = ln_last(a)
    # gamma_hat such that gamma * ln_a + b ~ hm
    # For each channel, fit
    g = (ln_a * hm).sum(0) / np.maximum((ln_a ** 2).sum(0), 1e-12)
    b = hm.mean(0) - g * ln_a.mean(0)
    print("fitted gamma from add53-LN vs hm min/mean/max", float(g.min()), float(g.mean()), float(g.max()))
    print("cos gamma*ln(add53)+b vs hm", cos(ln_a * g + b, hm))
    # fitted gamma from add53-LN vs fx
    g2 = (ln_a * fx).sum(0) / np.maximum((ln_a ** 2).sum(0), 1e-12)
    b2 = fx.mean(0) - g2 * ln_a.mean(0)
    print("fitted gamma from add53-LN vs fx min/mean/max", float(g2.min()), float(g2.mean()), float(g2.max()))
    print("cos gamma2*ln(add53)+b2 vs fx", cos(ln_a * g2 + b2, fx))
    print("cos fitted_gamma_hm vs fitted_gamma_fx", cos(g, g2))
    print("sign agree gamma", float((np.sign(g) == np.sign(g2)).mean()))

if ln0 is not None and win is not None:
    x = np.squeeze(win).astype(np.float32)
    y = np.squeeze(ln0).astype(np.float32)
    if x.ndim == 3:
        x = x[0]
    if y.ndim == 3:
        y = y[0]
    print("cos ln(input) vs first layer_norm gold", cos(ln_last(x), y))
    token_stats("input", x)
    token_stats("ln0", y)

if reshape0 is not None:
    r = np.array(reshape0)
    print("reshape0 full shape", r.shape)

if matmul0 is not None:
    m = np.array(matmul0)
    print("matmul0 full shape", m.shape, "min/max", float(m.min()), float(m.max()))

if softmax0 is not None:
    s = np.array(softmax0).astype(np.float32)
    print("softmax0 shape", s.shape, "sum last", s.reshape(-1, s.shape[-1]).sum(-1)[:4])

# 64-group scale stats
sc64 = scale.reshape(18, 64)
print("scale 64-group stds", np.round(sc64.std(1), 3).tolist())
print("scale overall std", float(scale.std()), "mean abs", float(np.abs(scale).mean()))
print("neg scales", int((scale < 0).sum()), "near0 |s|<0.1", int((np.abs(scale) < 0.1).sum()), "|s|>2", int((np.abs(scale) > 2).sum()))
