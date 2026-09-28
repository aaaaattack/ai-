#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from safetensors import safe_open

CKPT = Path("/workspace/917_embodied_ai/embodied_ai/gr00t/GR00T-N1.6-3B")

want = []
all_ln = []
with_prefix = []
for shard in sorted(CKPT.glob("model-*.safetensors")):
    with safe_open(str(shard), framework="pt") as f:
        for k in f.keys():
            kl = k.lower()
            if "action_head" in k or "transformer_blocks" in k or k.startswith("model."):
                if "norm" in kl and (k.endswith("weight") or k.endswith("bias")):
                    t = f.get_tensor(k).float().numpy()
                    all_ln.append((k, t))
            if "vision_model" in k and "post_layernorm" in k:
                with_prefix.append(k)

print("action/dit norm tensors", len(all_ln))
# group by suffix
from collections import Counter
c = Counter()
for k, t in all_ln:
    print(
        k,
        t.shape,
        "mean",
        float(t.mean()),
        "std",
        float(t.std()),
        "mae1",
        float(np.mean(np.abs(t - 1.0))),
        "mae0",
        float(np.mean(np.abs(t))),
        "min",
        float(t.min()),
        "max",
        float(t.max()),
    )

print("sample vision post keys", with_prefix[:5])
