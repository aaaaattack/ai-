#!/usr/bin/env python3
"""Check whether GR00T visual LayerNorm weights survive from_pretrained."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open


CKPT_LN = "backbone.model.vision_model.vision_model.encoder.layers.0.layer_norm1.weight"
CKPT_FC = "backbone.model.vision_model.vision_model.encoder.layers.0.mlp.fc1.weight"
CKPT_PATCH = "backbone.model.vision_model.vision_model.embeddings.patch_embedding.weight"


def tensor_stats(x: torch.Tensor) -> str:
    y = x.detach().float().cpu()
    mae1 = float(torch.mean(torch.abs(y - 1)))
    return (
        f"shape={tuple(y.shape)} dtype={x.dtype} mean={float(y.mean()):.4f} "
        f"min={float(y.min()):.4f} max={float(y.max()):.4f} mae_to_1={mae1:.6f}"
    )


def load_ckpt_tensor(model_dir: str, key: str) -> torch.Tensor:
    files = sorted(Path(model_dir).glob("*.safetensors"))
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            if key in handle.keys():
                return handle.get_tensor(key)
    raise KeyError(f"{key} not found")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().cpu().flatten()
    y = b.detach().float().cpu().flatten()
    denom = float(x.norm() * y.norm())
    return float(torch.dot(x, y) / denom) if denom else 0.0


def compare_key(named: dict, ckpt_dir: str, key: str, label: str) -> None:
    ckpt = load_ckpt_tensor(ckpt_dir, key)
    print(f"{label} ckpt", tensor_stats(ckpt))
    if key not in named:
        print(f"{label} MISSING in model")
        nearby = [k for k in named if k.endswith(key.split(".", 1)[-1])][:5]
        for k in nearby:
            print("  nearby", k)
        return
    loaded = named[key]
    print(f"{label} loaded", tensor_stats(loaded))
    print(f"{label} ckpt vs loaded cosine", cosine(ckpt, loaded))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/gr00t/GR00T-N1.6-3B/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-policy", action="store_true")
    args = parser.parse_args()

    import transformers
    from transformers import AutoModel
    import gr00t.model  # noqa: F401  register Gr00tN1d6

    print("transformers", transformers.__version__, transformers.__file__)
    print("ckpt", CKPT_LN, tensor_stats(load_ckpt_tensor(args.model, CKPT_LN)))

    kwargs = dict(
        attn_implementation="eager",
        output_loading_info=True,
    )
    if transformers.__version__.startswith("5"):
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    print("from_pretrained output_loading_info ...")
    loaded = AutoModel.from_pretrained(Path(args.model), **kwargs)
    if isinstance(loaded, tuple):
        model, info = loaded
    else:
        model, info = loaded, {}
    missing = list(info.get("missing_keys", []) or [])
    unexpected = list(info.get("unexpected_keys", []) or [])
    mismatched = list(info.get("mismatched_keys", []) or [])
    vis_missing = [k for k in missing if "vision_model" in k]
    vis_ln_missing = [k for k in vis_missing if "layer_norm" in k or "layernorm" in k]
    print("missing", len(missing), "unexpected", len(unexpected), "mismatched", len(mismatched))
    print("vision missing", len(vis_missing), "vision LN missing", len(vis_ln_missing))
    for k in vis_ln_missing[:20]:
        print("  missing LN", k)
    print("vision missing sample")
    for k in vis_missing[:20]:
        print(" ", k)
    print("missing sample")
    for k in missing[:20]:
        print(" ", k)
    print("unexpected sample")
    for k in unexpected[:20]:
        print(" ", k)

    named = dict(model.named_parameters())
    print("n_params", len(named))
    print("has", CKPT_LN, CKPT_LN in named)
    vis_ln = [
        k for k in named if "vision_model" in k and "layer_norm" in k and k.endswith(".weight")
    ]
    ident = 0
    for k in vis_ln:
        mae1 = float(torch.mean(torch.abs(named[k].detach().float() - 1)))
        if mae1 < 1e-5:
            ident += 1
    print("model vision LN weights", len(vis_ln), "identity", ident)
    compare_key(named, args.model, CKPT_LN, "LN")
    compare_key(named, args.model, CKPT_FC, "FC1")
    compare_key(named, args.model, CKPT_PATCH, "PATCH")

    if args.skip_policy:
        return

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        model_path=args.model,
        embodiment_tag=EmbodimentTag("gr1"),
        device=args.device,
    )
    named = dict(policy.model.named_parameters())
    print("policy has ckpt key", CKPT_LN in named)
    vis_ln = [
        k for k in named if "vision_model" in k and "layer_norm" in k and k.endswith(".weight")
    ]
    ident = 0
    for k in vis_ln:
        mae1 = float(torch.mean(torch.abs(named[k].detach().float() - 1)))
        if mae1 < 1e-5:
            ident += 1
    print("policy vision LN weights", len(vis_ln), "identity", ident)
    compare_key(named, args.model, CKPT_LN, "policy LN")
    compare_key(named, args.model, CKPT_FC, "policy FC1")


if __name__ == "__main__":
    main()
