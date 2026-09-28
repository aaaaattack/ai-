"""Prefill wrap vs FX ablation: frontend graph / Q-graph no-PTQ / PTQ ALIGNED."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from quant_pipeline import (
    EMBODIMENT_CONFIGS,
    _move_module_tensors,
    _register_plain_tensors_as_buffers,
    build_fp16_module,
    clone_inputs_as,
    converter_dummy_inputs,
    cosine_similarity,
    floating_param_dtype,
    load_gr00t_policy,
    max_abs_diff,
    mse,
    unwrap_model_output,
)


def to_numpy(value):
    value = unwrap_model_output(value)
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def stats(name, lhs, rhs):
    a = to_numpy(lhs)
    b = to_numpy(rhs)
    print(
        f"{name}: cosine={cosine_similarity(a, b):.8f} "
        f"mse={mse(a, b):.6e} max_abs={max_abs_diff(a, b):.6e} "
        f"shape={a.shape}->{b.shape}"
    )
    return a, b


def feed_for(model, inputs, label):
    feed = clone_inputs_as(inputs, floating_param_dtype(model), label=label)
    return [value.to(next(model.parameters()).device) if torch.is_tensor(value) else value for value in feed]


def run_model(model, inputs, label):
    feed = feed_for(model, inputs, label)
    with torch.no_grad():
        return model(*feed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GR00T-N1.6-3B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embodiment", default="gr1")
    parser.add_argument("--quant-type", default="w8a8h1_sefp")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--input-sequence-length", type=int, default=256)
    parser.add_argument(
        "--onnx",
        default="output/xh2/hmquant/prefill/hmquant_gr00t_with_act.onnx",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    args = SimpleNamespace(
        model=cli.model,
        device=cli.device,
        embodiment=cli.embodiment,
        quant_type=cli.quant_type,
        context_length=cli.context_length,
        input_sequence_length=cli.input_sequence_length,
        quant_weight=None,
        output_path="work_dirs",
    )
    embodiment_cfg = EMBODIMENT_CONFIGS[args.embodiment]
    onnx_file = Path(cli.onnx)
    print(f"device={args.device} model={args.model} onnx={onnx_file}")

    policy = load_gr00t_policy(args, embodiment_cfg)
    inputs = converter_dummy_inputs("prefill", policy, args, embodiment_cfg, onnx_file)
    wrap = build_fp16_module("prefill", policy, args, embodiment_cfg)
    _register_plain_tensors_as_buffers(wrap)
    wrap = _move_module_tensors(wrap, args.device).eval()

    wrap_out = run_model(wrap, inputs, "wrap")
    print("wrap dtype", floating_param_dtype(wrap), "out", unwrap_model_output(wrap_out).shape)

    from xhquant import PrecisionMode
    from xhquant.api import (
        ConfigDict,
        DeviceType,
        FrontendType,
        QuantScheme,
        create_quant_config,
        to_frontend_graph,
    )
    from xhquant.api.ptq_export_hmonnx import _convert_model_to_quanted_model
    from xhquant.quantization.chain_api.ptq_quantize import ptq_quantize

    quant_config = ConfigDict(create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)))
    fx_feed = feed_for(wrap, inputs, "fx.trace")

    print("\n=== 1) wrap -> frontend graph (no Q modules, no PTQ) ===")
    frontend = to_frontend_graph(
        wrap,
        FrontendType.TorchFX,
        fx_feed,
        quant_config=quant_config,
    )
    frontend = _move_module_tensors(frontend, args.device).eval()
    fe_out = run_model(frontend, inputs, "frontend")
    stats("wrap vs frontend", wrap_out, fe_out)

    print("\n=== 2) wrap -> Q-graph, use_ptq=False + disable_quant ===")
    qgraph = _convert_model_to_quanted_model(
        wrap,
        FrontendType.TorchFX,
        fx_feed,
        DeviceType.XH2a,
        quant_config,
        use_ptq=False,
    )
    if hasattr(qgraph, "disable_quant"):
        qgraph.disable_quant()
        print("disable_quant() applied")
    qgraph = _move_module_tensors(qgraph, args.device).eval()
    qfp_out = run_model(qgraph, inputs, "qgraph.fp")
    stats("wrap vs Q-graph(no PTQ)", wrap_out, qfp_out)
    stats("frontend vs Q-graph(no PTQ)", fe_out, qfp_out)

    print("\n=== 3) same Q-graph after ptq_quantize ALIGNED ===")
    exec_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_feed = feed_for(qgraph, inputs, "ptq.calibrate")
    ptq_quantize(qgraph, [run_feed], PrecisionMode.ALIGNED, exec_device)
    qptq_out = run_model(qgraph, inputs, "qgraph.ptq")
    stats("wrap vs Q-graph(PTQ)", wrap_out, qptq_out)
    stats("Q-graph(no PTQ) vs PTQ", qfp_out, qptq_out)

    wrap_np = to_numpy(wrap_out)[0]
    ptq_np = to_numpy(qptq_out)[0]
    tok = np.array(
        [
            np.dot(wrap_np[i], ptq_np[i])
            / (np.linalg.norm(wrap_np[i]) * np.linalg.norm(ptq_np[i]) + 1e-12)
            for i in range(wrap_np.shape[0])
        ]
    )
    print(
        "PTQ per-token cosine min/p50/mean/max",
        float(tok.min()),
        float(np.median(tok)),
        float(tok.mean()),
        float(tok.max()),
    )


if __name__ == "__main__":
    main()
