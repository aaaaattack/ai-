#!/usr/bin/env python3
import numpy as np
import onnx
from pathlib import Path

for name, p in [
    ("visual", "/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/visual/hmquant_gr00t_with_act.onnx"),
    ("head", "/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/head/hmquant_gr00t_with_act.onnx"),
    ("prefill", "/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/prefill/hmquant_gr00t_with_act.onnx"),
]:
    m = onnx.load(p, load_external_data=True)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    ops = {}
    for n in g.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("\n====", name, "====")
    print("ops", dict(sorted(ops.items(), key=lambda x: -x[1])[:15]))
    ln_nodes = [n for n in g.node if n.op_type == "LayerNorm"]
    rms_nodes = [n for n in g.node if n.op_type == "RMSNorm"]
    print("LayerNorm", len(ln_nodes), "RMSNorm", len(rms_nodes))
    if ln_nodes:
        n0 = ln_nodes[0]
        print(" first LN", n0.name, list(n0.input))
        for inp in n0.input[1:3]:
            if inp in init:
                arr = onnx.numpy_helper.to_array(init[inp]).astype(np.float32)
                print(" ", inp, arr.shape, "mean", float(arr.mean()), "std", float(arr.std()), "mae1", float(np.mean(np.abs(arr - 1))), "mae0", float(np.mean(np.abs(arr))))
        # how many LN weights are identity
        ident_w = ident_b = total_w = total_b = 0
        for n in ln_nodes:
            if len(n.input) > 1 and n.input[1] in init:
                arr = onnx.numpy_helper.to_array(init[n.input[1]]).astype(np.float32)
                total_w += 1
                if float(np.max(np.abs(arr - 1))) < 1e-6:
                    ident_w += 1
            if len(n.input) > 2 and n.input[2] in init:
                arr = onnx.numpy_helper.to_array(init[n.input[2]]).astype(np.float32)
                total_b += 1
                if float(np.max(np.abs(arr))) < 1e-6:
                    ident_b += 1
        print(" LN identity weight", ident_w, "/", total_w, "identity bias", ident_b, "/", total_b)
    if rms_nodes:
        n0 = rms_nodes[0]
        print(" first RMS", n0.name, list(n0.input))
        for inp in n0.input[1:2]:
            if inp in init:
                arr = onnx.numpy_helper.to_array(init[inp]).astype(np.float32)
                print(" ", inp, arr.shape, "mean", float(arr.mean()), "std", float(arr.std()), "mae1", float(np.mean(np.abs(arr - 1))))
        ident_w = total_w = 0
        for n in rms_nodes:
            if len(n.input) > 1 and n.input[1] in init:
                arr = onnx.numpy_helper.to_array(init[n.input[1]]).astype(np.float32)
                total_w += 1
                if float(np.max(np.abs(arr - 1))) < 1e-6:
                    ident_w += 1
        print(" RMS identity weight", ident_w, "/", total_w)
