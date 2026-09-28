import numpy as np
from collections import Counter
from pathlib import Path

try:
    import onnx
except ImportError:
    onnx = None

base = Path("/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/visual")
if not base.exists():
    base = Path("/data01/home/wei.xie/v1.4.0_test/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/visual")

fx = np.load(base / "four_way/fx/hmquant_gr00t_output_latent_output.npy").astype(np.float32)[0]
hm = np.load(base / "post_quant/hmquant_gr00t_output_latent_output.npy").astype(np.float32)[0]


def cos(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


ch = np.array(
    [
        np.dot(fx[:, c], hm[:, c])
        / (np.linalg.norm(fx[:, c]) * np.linalg.norm(hm[:, c]) + 1e-12)
        for c in range(fx.shape[1])
    ]
)
print("channel cosine min/mean/max", ch.min(), ch.mean(), ch.max())
print("neg", int((ch < 0).sum()), "pos", int((ch > 0).sum()), "|cos|>0.9", int((np.abs(ch) > 0.9).sum()))
print("even mean", ch[0::2].mean(), "odd mean", ch[1::2].mean())
print("first 96 channel cos", np.round(ch[:96], 3).tolist())

hm_flip = hm.copy()
hm_flip[:, ch < 0] *= -1
print("cos after flipping all negative channels", cos(fx, hm_flip))

scale = (fx * hm).sum(0) / np.maximum((fx ** 2).sum(0), 1e-12)
print("per-ch scale min/mean/max", scale.min(), scale.mean(), scale.max())
print("cos after per-ch scale", cos(fx * scale, hm))

neg = np.where(ch < 0)[0]
print("neg first 48", neg[:48].tolist())
if len(neg) > 1:
    d = np.diff(neg)
    print("neg stride unique", np.unique(d)[:30].tolist(), "mean", float(d.mean()))

for g in [16, 32, 36, 64, 72, 96, 128]:
    if 1152 % g:
        continue
    blocks = ch.reshape(-1, g)
    print(
        f"group{g} block0-3 mean",
        np.round(blocks.mean(1)[:4], 3).tolist(),
        "neg-count0-3",
        (blocks < 0).sum(1)[:4].tolist(),
        "neg-blocks",
        int((blocks.mean(1) < 0).sum()),
        "/",
        blocks.shape[0],
    )

for h in [12, 16, 18, 24, 32, 36]:
    dim = 1152 // h
    arr = ch.reshape(h, dim)
    print(
        f"heads{h}x{dim} per-head mean min/max",
        float(arr.mean(1).min()),
        float(arr.mean(1).max()),
        "neg heads",
        int((arr.mean(1) < 0).sum()),
    )

# regular sign pattern search: period p
print("period search for sign(ch)")
sgn = np.sign(ch)
for p in range(2, 129):
    if 1152 % p:
        continue
    pat = sgn[:p]
    tiled = np.tile(pat, 1152 // p)
    agree = float((tiled == sgn).mean())
    if agree > 0.8:
        print("  period", p, "agree", agree, "pat", pat[:16].tolist())

onnx_path = base / "hmquant_gr00t_with_act.onnx"
prefill = base.parent / "prefill/hmquant_gr00t_with_act.onnx"
if onnx is not None:
    for p in [onnx_path, prefill]:
        m = onnx.load(str(p), load_external_data=False)
        g = m.graph
        print("\n====", p.name, p.parent.name, "====")
        print("opsets", [(o.domain, o.version) for o in m.opset_import])
        for kind, arr in [("in", g.input), ("out", g.output)]:
            for i in arr:
                t = i.type.tensor_type
                sh = [d.dim_value for d in t.shape.dim]
                print(kind, i.name, "elem", t.elem_type, sh)
        ops = Counter((n.domain or "ai.onnx", n.op_type) for n in g.node)
        print("nodes", len(g.node))
        for k, v in ops.most_common(25):
            print(f"  {v:4d} {k[0]}::{k[1]}")
        print("first 6:")
        for n in g.node[:6]:
            print(" ", n.op_type, n.name, "in", list(n.input)[:3], "out", list(n.output)[:2], n.domain)
        print("last 6:")
        for n in g.node[-6:]:
            print(" ", n.op_type, n.name, "in", list(n.input)[:3], "out", list(n.output)[:2], n.domain)
else:
    print("onnx not importable")
