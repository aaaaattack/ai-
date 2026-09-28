#!/usr/bin/env python3
"""Probe where visual LayerNorm gamma becomes identity during FX -> HMONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from onnx import numpy_helper


def mae_to_ones(t: torch.Tensor) -> float:
    x = t.detach().float().cpu().numpy()
    return float(np.mean(np.abs(x - 1.0)))


def mae_to_zeros(t: torch.Tensor | None) -> float | None:
    if t is None:
        return None
    x = t.detach().float().cpu().numpy()
    return float(np.mean(np.abs(x)))


def report_ln(model: torch.nn.Module, tag: str, limit: int = 6) -> None:
    rows = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        if "LayerNorm" not in cls and "layernorm" not in name.lower():
            continue
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if not torch.is_tensor(weight):
            continue
        rows.append(
            (
                name,
                cls,
                tuple(weight.shape),
                str(weight.dtype),
                mae_to_ones(weight),
                mae_to_zeros(bias) if torch.is_tensor(bias) else None,
                float(weight.detach().float().mean()),
            )
        )
    ident = sum(1 for r in rows if r[4] < 1e-5 and (r[5] is None or r[5] < 1e-5))
    print(f"\n==== {tag} LayerNorm-like={len(rows)} identity={ident}")
    for row in rows[:limit]:
        print(
            f"  {row[0]} cls={row[1]} shape={row[2]} dtype={row[3]} "
            f"mae_to_1={row[4]:.6f} mae_to_0={row[5]} mean={row[6]:.4f}"
        )
    if rows:
        last = rows[-1]
        print(
            f"  LAST {last[0]} cls={last[1]} mae_to_1={last[4]:.6f} "
            f"mae_to_0={last[5]} mean={last[6]:.4f}"
        )


def report_fx_get_attr(graph_module, tag: str) -> None:
    nodes = list(graph_module.graph.nodes)
    print(f"\n==== {tag} graph nodes={len(nodes)}")
    shown = 0
    for n in nodes:
        text = f"{n.op} {n.name} target={n.target} args={n.args[:4] if n.args else ()}"
        if "layer_norm" in text.lower() or "layernorm" in text.lower():
            print(" ", text[:240])
            shown += 1
            if shown >= 20:
                break
    get_attrs = [n for n in nodes if n.op == "get_attr" and "layer_norm" in str(n.target).lower()]
    print(f"  get_attr layer_norm count={len(get_attrs)}")
    for n in get_attrs[:8]:
        print("   ", n.target)


def report_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    init = {i.name: i for i in model.graph.initializer}
    ln = [n for n in model.graph.node if n.op_type.endswith("LayerNorm")]
    ident = 0
    checked = 0
    print(f"\n==== ONNX {path} LN={len(ln)}")
    for idx, node in enumerate(ln):
        w = b = None
        if len(node.input) >= 2 and node.input[1] in init:
            w = numpy_helper.to_array(init[node.input[1]]).astype(np.float32)
        if len(node.input) >= 3 and node.input[2] in init:
            b = numpy_helper.to_array(init[node.input[2]]).astype(np.float32)
        if w is None:
            continue
        checked += 1
        mae1 = float(np.mean(np.abs(w - 1)))
        mae0 = float(np.mean(np.abs(b))) if b is not None else None
        if mae1 < 1e-5 and (mae0 is None or mae0 < 1e-5):
            ident += 1
        if idx < 3 or idx == len(ln) - 1:
            print(
                f"  {node.name} w={node.input[1] if len(node.input)>1 else None} "
                f"mean={float(w.mean()):.4f} mae_to_1={mae1:.6f} mae_to_0={mae0}"
            )
    print(f"  identity={ident}/{checked}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/gr00t/GR00T-N1.6-3B/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--export-onnx", default="")
    args = parser.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from xhquant.api import (
        Config,
        ConfigDict,
        DeviceType,
        QuantScheme,
        create_quant_config,
        convert_fx_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
    )
    from xhquant.export.convert import to_export_graph
    from xh_model_zoo.xh_llm.models.builder import wrap_llm_model

    print("loading policy", args.model)
    policy = Gr00tPolicy(
        model_path=args.model,
        embodiment_tag=EmbodimentTag("gr1"),
        device=args.device,
    )
    vision = policy.model.backbone.model.vision_model
    report_ln(vision, "native vision")

    from xh_model_zoo.xh_llm.models.groot._vision_model import (
        register_wrap_modules as vision_register_wrap_modules,
    )

    vision_register_wrap_modules()
    wrap_cfg = Config(
        dict(
            batch_size=1,
            token_len=256,
            num_logits_to_keep=0,
            input_sequence_length=324,
            use_cache=False,
            max_sequence_length=888888,
        )
    )
    wrap = wrap_llm_model(vision, wrap_cfg).to(args.device).to(torch.bfloat16).eval()
    report_ln(wrap, "after wrap")

    dummy = torch.rand(1, 324, 1152, device=args.device, dtype=torch.float16)
    quant_config = ConfigDict(
        create_quant_config(
            QuantScheme(target_device=DeviceType.XH2a, quant_type="w8a8h1_sefp")
        )
    )
    print("convert_fx_model_to_quanted_model ...")
    with torch.no_grad():
        quanted = convert_fx_model_to_quanted_model(
            wrap, [dummy], DeviceType.XH2a, quant_config
        )
    report_ln(quanted, "after FX quant")
    report_fx_get_attr(quanted, "FX quant graph")

    if args.export_onnx:
        out = Path(args.export_onnx)
        out.parent.mkdir(parents=True, exist_ok=True)
        print("convert_quanted_model_to_hmonnx ...", out)
        convert_quanted_model_to_hmonnx(
            quanted,
            [dummy],
            str(out),
            ["windows_tensor"],
            ["output_latent"],
        )
        report_ln(quanted, "after hmonnx export (quanted module)")
        report_onnx(out)
    else:
        print("to_export_graph ...")
        with torch.no_grad():
            exported = to_export_graph(quanted, [dummy])
        report_ln(exported, "after to_export_graph")
        report_fx_get_attr(exported, "export graph")


if __name__ == "__main__":
    main()
