import numpy as np
import onnx
from pathlib import Path

p = Path("/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/visual/hmquant_gr00t_with_act.onnx")
m = onnx.load(str(p), load_external_data=True)
g = m.graph
init = {i.name: i for i in g.initializer}
print("n init", len(init))

keys = [
    "vision_model.post_layernorm.weight",
    "vision_model.post_layernorm.bias",
    "vision_model.encoder.layers.0.layer_norm1.weight",
    "vision_model.encoder.layers.0.layer_norm1.bias",
    "vision_model.encoder.layers.0.layer_norm2.weight",
    "vision_model.encoder.layers.26.layer_norm2.weight",
    "vision_model.encoder.layers.26.layer_norm2.bias",
]
for k in keys:
    if k not in init:
        print("MISSING", k)
        continue
    arr = onnx.numpy_helper.to_array(init[k]).astype(np.float32)
    print(
        k,
        "shape",
        arr.shape,
        "min/mean/max",
        float(arr.min()),
        float(arr.mean()),
        float(arr.max()),
        "std",
        float(arr.std()),
        "absmean",
        float(np.abs(arr).mean()),
        "close1",
        float(np.mean(np.abs(arr - 1.0))),
        "close0",
        float(np.mean(np.abs(arr))),
    )

# all layernorm weights
ln_w = [n for n in init if "layernorm" in n.lower() or "layer_norm" in n.lower()]
print("ln-related inits", len(ln_w))
for n in sorted(ln_w):
    arr = onnx.numpy_helper.to_array(init[n]).astype(np.float32)
    print(
        " ",
        n,
        arr.shape,
        "mean",
        float(arr.mean()),
        "std",
        float(arr.std()),
        "min",
        float(arr.min()),
        "max",
        float(arr.max()),
        "mae_to_1",
        float(np.mean(np.abs(arr - 1.0))) if "weight" in n else "",
        "mae_to_0",
        float(np.mean(np.abs(arr))) if "bias" in n else "",
    )

# LayerNorm node attrs
ln_nodes = [n for n in g.node if n.op_type == "LayerNorm"]
print("LayerNorm nodes", len(ln_nodes))
n0 = ln_nodes[0]
print("first LN name", n0.name, "inputs", list(n0.input), "outputs", list(n0.output))
for a in n0.attribute:
    print(" attr", a.name, "i", a.i, "f", a.f, "s", a.s, "ints", list(a.ints)[:8])
nlast = ln_nodes[-1]
print("last LN name", nlast.name, "inputs", list(nlast.input), "outputs", list(nlast.output))
for a in nlast.attribute:
    print(" attr", a.name, "i", a.i, "f", a.f, "s", a.s, "ints", list(a.ints)[:8])
