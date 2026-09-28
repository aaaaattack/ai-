#!/usr/bin/env python3
"""Confirm visual HMONNX LayerNorm affine vs safetensors, and whether Linear absorbed gamma."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from safetensors import safe_open

CKPT = Path("/workspace/917_embodied_ai/embodied_ai/gr00t/GR00T-N1.6-3B")
ONNX = Path("/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/visual/hmquant_gr00t_with_act.onnx")
GOLD = Path("/workspace/917_embodied_ai/embodied_ai/gr00t/work_dirs/golden/visual/post_quant/hmquant_gr00t_with_act")
FX = Path("/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/visual/four_way/fx")
HM = Path("/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/visual/post_quant")


def cosine(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def load_ckpt_keys():
    keys = {}
    for shard in sorted(CKPT.glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if "vision_model" in k and (
                    "layer_norm" in k or "post_layernorm" in k or ".self_attn.q_proj" in k or ".mlp.fc1" in k
                ):
                    t = f.get_tensor(k)
                    keys[k] = t.float().numpy()
    return keys


def main():
    ckpt = load_ckpt_keys()
    ln_keys = sorted(k for k in ckpt if "layer_norm" in k or "post_layernorm" in k)
    print("ckpt vision LN-related", len(ln_keys))
    for k in [
        "backbone.eagle_model.model.vision_model.post_layernorm.weight",
        "backbone.eagle_model.model.vision_model.post_layernorm.bias",
        "backbone.eagle_model.model.vision_model.encoder.layers.0.layer_norm1.weight",
        "backbone.eagle_model.model.vision_model.encoder.layers.0.layer_norm1.bias",
        "backbone.eagle_model.model.vision_model.encoder.layers.0.layer_norm2.weight",
        "backbone.eagle_model.model.vision_model.encoder.layers.26.layer_norm1.weight",
    ]:
        hits = [x for x in ckpt if x.endswith(k.split("vision_model.")[-1]) and "vision_model" in x]
        if not hits:
            # print candidates
            continue
        arr = ckpt[hits[0]].astype(np.float32)
        print(
            hits[0],
            arr.shape,
            "min/mean/max",
            float(arr.min()),
            float(arr.mean()),
            float(arr.max()),
            "std",
            float(arr.std()),
            "mae_to_1",
            float(np.mean(np.abs(arr - 1.0))),
            "mae_to_0",
            float(np.mean(np.abs(arr))),
        )

    # dump a few LN stats
    w_stats = []
    b_stats = []
    for k, arr in ckpt.items():
        if "vision_model" not in k:
            continue
        if k.endswith("layer_norm1.weight") or k.endswith("layer_norm2.weight") or k.endswith("post_layernorm.weight"):
            a = arr.astype(np.float32)
            w_stats.append((k, float(a.mean()), float(a.std()), float(np.mean(np.abs(a - 1.0)))))
        if k.endswith("layer_norm1.bias") or k.endswith("layer_norm2.bias") or k.endswith("post_layernorm.bias"):
            a = arr.astype(np.float32)
            b_stats.append((k, float(a.mean()), float(a.std()), float(np.mean(np.abs(a)))))
    print("LN weights count", len(w_stats), "bias count", len(b_stats))
    if w_stats:
        mae = [x[3] for x in w_stats]
        print("LN weight mae_to_1 min/mean/max", min(mae), sum(mae) / len(mae), max(mae))
        print("example largest mae_to_1", max(w_stats, key=lambda x: x[3]))
        print("example smallest mae_to_1", min(w_stats, key=lambda x: x[3]))
    if b_stats:
        mae0 = [x[3] for x in b_stats]
        print("LN bias mae_to_0 min/mean/max", min(mae0), sum(mae0) / len(mae0), max(mae0))

    m = onnx.load(str(ONNX), load_external_data=True)
    g = m.graph
    init = {i.name: onnx.numpy_helper.to_array(i) for i in g.initializer}

    ln_nodes = [n for n in g.node if n.op_type == "LayerNorm"]
    print("\nONNX LayerNorm nodes", len(ln_nodes))
    print("first LN", ln_nodes[0].name, "inputs", list(ln_nodes[0].input))
    print("last LN", ln_nodes[-1].name, "inputs", list(ln_nodes[-1].input))

    # Linear names around first LN
    linear_nodes = [n for n in g.node if n.op_type == "Linear"]
    print("Linear nodes", len(linear_nodes), "first 6:")
    for n in linear_nodes[:6]:
        print(" ", n.name, "inputs", list(n.input)[:4], "outputs", list(n.output))

    # Compare first q_proj-like Linear vs ckpt
    q_keys = sorted(k for k in ckpt if "vision_model" in k and k.endswith("self_attn.q_proj.weight"))
    fc1_keys = sorted(k for k in ckpt if "vision_model" in k and k.endswith("mlp.fc1.weight"))
    print("ckpt q_proj", len(q_keys), "fc1", len(fc1_keys))
    if q_keys:
        q0 = ckpt[q_keys[0]].astype(np.float32)
        print("ckpt", q_keys[0], q0.shape, "mean", float(q0.mean()), "std", float(q0.std()), "absmean", float(np.abs(q0).mean()))
    if fc1_keys:
        f0 = ckpt[fc1_keys[0]].astype(np.float32)
        print("ckpt", fc1_keys[0], f0.shape, "mean", float(f0.mean()), "std", float(f0.std()), "absmean", float(np.abs(f0).mean()))

    ln1_w_keys = [k for k in ckpt if "vision_model" in k and k.endswith("encoder.layers.0.layer_norm1.weight")]
    ln1_b_keys = [k for k in ckpt if "vision_model" in k and k.endswith("encoder.layers.0.layer_norm1.bias")]
    post_w_keys = [k for k in ckpt if "vision_model" in k and k.endswith("post_layernorm.weight")]
    post_b_keys = [k for k in ckpt if "vision_model" in k and k.endswith("post_layernorm.bias")]
    print("matched ln1", ln1_w_keys, ln1_b_keys)
    print("matched post", post_w_keys, post_b_keys)

    # first Linear weight initializer
    lin0 = linear_nodes[0]
    w_name = lin0.input[1] if len(lin0.input) > 1 else None
    b_name = lin0.input[2] if len(lin0.input) > 2 else None
    print("first Linear weight/bias names", w_name, b_name)
    if w_name in init:
        w = init[w_name].astype(np.float32)
        print("onnx first Linear W", w.shape, "mean", float(w.mean()), "std", float(w.std()), "absmean", float(np.abs(w).mean()))
        if q_keys:
            q0 = ckpt[q_keys[0]].astype(np.float32)
            # Linear ONNX is often (out, in) same as pytorch
            if w.shape == q0.shape:
                print("cosine onnx_lin0 vs ckpt_q0", cosine(w, q0))
                print("max_abs", float(np.max(np.abs(w - q0))))
            elif w.T.shape == q0.shape:
                print("cosine onnx_lin0.T vs ckpt_q0", cosine(w.T, q0))
                print("max_abs", float(np.max(np.abs(w.T - q0))))
            if ln1_w_keys and w.shape == q0.shape:
                gamma = ckpt[ln1_w_keys[0]].astype(np.float32)
                folded = q0 * gamma[None, :]
                print("cosine onnx_lin0 vs q0*gamma", cosine(w, folded), "max_abs", float(np.max(np.abs(w - folded))))
            elif ln1_w_keys and w.T.shape == q0.shape:
                gamma = ckpt[ln1_w_keys[0]].astype(np.float32)
                folded = q0 * gamma[None, :]
                print("cosine onnx_lin0.T vs q0*gamma", cosine(w.T, folded), "max_abs", float(np.max(np.abs(w.T - folded))))

    # last Linear vs last mlp.fc2 or whatever precedes post_ln — skip
    # Compare post_layernorm onnx vs ckpt
    last = ln_nodes[-1]
    print("\nlast LN inputs", list(last.input))
    for inp in last.input[1:3]:
        if inp in init:
            arr = init[inp].astype(np.float32)
            print(" onnx", inp, arr.shape, "mean", float(arr.mean()), "std", float(arr.std()), "unique_approx", np.unique(np.round(arr, 5))[:6])
            if post_w_keys and "weight" in inp.lower() or inp.endswith(".weight") or True:
                pass
    if post_w_keys:
        pw = ckpt[post_w_keys[0]].astype(np.float32)
        print("ckpt post_ln.weight mae_to_1", float(np.mean(np.abs(pw - 1.0))), "std", float(pw.std()), "min/max", float(pw.min()), float(pw.max()))
    if post_b_keys:
        pb = ckpt[post_b_keys[0]].astype(np.float32)
        print("ckpt post_ln.bias mae_to_0", float(np.mean(np.abs(pb))), "std", float(pb.std()), "min/max", float(pb.min()), float(pb.max()))

    # apply ckpt post_ln affine onto HMONNX add_53 and compare to FX output
    add53 = GOLD / "add_53.npy"
    fx_out = None
    for cand in [
        FX / "hmquant_gr00t_output_latent_output.npy",
        Path("/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/visual/four_way/fx/hmquant_gr00t_output_latent_output.npy"),
        Path("/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/visual/pre_quant/hmquant_gr00t_output_latent_output.npy"),
    ]:
        if cand.exists():
            fx_out = np.load(cand)
            print("loaded fx/pre", cand, fx_out.shape, fx_out.dtype)
            break
    hm_out = None
    for cand in [
        HM / "hmquant_gr00t_output_latent_output.npy",
        GOLD / "output_latent.npy",
    ]:
        if cand.exists():
            hm_out = np.load(cand)
            print("loaded hm", cand, hm_out.shape)
            break
    if add53.exists() and post_w_keys and post_b_keys:
        a = np.load(add53).astype(np.float32)
        gma = ckpt[post_w_keys[0]].astype(np.float32)
        beta = ckpt[post_b_keys[0]].astype(np.float32)
        mu = a.mean(axis=-1, keepdims=True)
        var = a.var(axis=-1, keepdims=True)
        hat = (a - mu) / np.sqrt(var + 1e-6)
        restored = hat * gma + beta
        print("add_53 vs hm_out", cosine(a, hm_out) if hm_out is not None else None)
        print("identity_ln(add_53) vs hm_out", cosine(hat, hm_out) if hm_out is not None else None)
        print("ckpt_affine_ln(add_53) vs hm_out", cosine(restored, hm_out) if hm_out is not None else None)
        if fx_out is not None:
            print("ckpt_affine_ln(add_53) vs fx_out", cosine(restored, fx_out))
            print("identity_ln(add_53) vs fx_out", cosine(hat, fx_out))
            print("add_53 vs fx_out", cosine(a, fx_out))
            print("hm vs fx", cosine(hm_out, fx_out) if hm_out is not None else None)

    # layer-wise residual cosine vs fx is not available; instead report residual std growth
    print("\nresidual add_* vs later add, std:")
    for i in [0, 1, 2, 5, 10, 20, 40, 52, 53]:
        p = GOLD / (f"add_{i}.npy" if i else "add.npy")
        if not p.exists() and i == 0:
            p = GOLD / "add.npy"
        if p.exists():
            arr = np.load(p).astype(np.float32)
            print(p.name, arr.shape, "mean", float(arr.mean()), "std", float(arr.std()), "min", float(arr.min()), "max", float(arr.max()))

    # print a few vision key prefixes
    prefixes = set()
    for k in list(ckpt)[:]:
        if "vision_model" in k:
            prefixes.add(".".join(k.split(".")[:6]))
    print("\nvision prefixes", sorted(prefixes)[:20])
    print("sample keys", sorted(ckpt)[:15])


if __name__ == "__main__":
    main()
