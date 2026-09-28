import onnx
from collections import Counter
from pathlib import Path


def summarize(p: Path) -> None:
    m = onnx.load(str(p), load_external_data=False)
    g = m.graph
    print("====", p, "====")
    print("ir", m.ir_version, "opsets", [(o.domain, o.version) for o in m.opset_import])
    for kind, arr in [("IN", g.input), ("OUT", g.output)]:
        for i in arr:
            t = i.type.tensor_type
            sh = [d.dim_value if d.dim_value else d.dim_param for d in t.shape.dim]
            print(kind, i.name, "elem", t.elem_type, sh)
    ops = Counter((n.domain or "ai.onnx", n.op_type) for n in g.node)
    print("nodes", len(g.node))
    for k, v in ops.most_common(40):
        print("  {:4d} {}::{}".format(v, k[0], k[1]))
    print("FIRST 12:")
    for n in g.node[:12]:
        print(" ", n.op_type, n.name, "in", list(n.input)[:4], "out", list(n.output)[:2], "dom", n.domain)
    print("LAST 12:")
    for n in g.node[-12:]:
        print(" ", n.op_type, n.name, "in", list(n.input)[:4], "out", list(n.output)[:2], "dom", n.domain)

    for op in ["Pad", "Concat", "Reshape", "Transpose", "LayerNormalization", "Softmax", "MatMul", "Gemm"]:
        ns = [n for n in g.node if n.op_type == op or n.op_type.endswith(op)]
        if not ns:
            continue
        print("-- {} count {} sample --".format(op, len(ns)))
        sample = ns[:3] + ns[-2:] if len(ns) > 5 else ns
        for n in sample:
            attrs = []
            for a in n.attribute:
                if a.type == 7:
                    attrs.append("{}={}".format(a.name, list(a.ints)[:8]))
                elif a.type == 2:
                    attrs.append("{}={}".format(a.name, a.i))
            print(" ", n.op_type, n.name, "in", list(n.input)[:4], "out", list(n.output)[:2], attrs[:6])

    init = {i.name: i for i in g.initializer}
    print("initializers", len(init))
    reshapes = [n for n in g.node if n.op_type == "Reshape"]
    print("reshape nodes", len(reshapes))
    shown = 0
    unique_shapes = Counter()
    for n in reshapes:
        shape_name = n.input[1] if len(n.input) > 1 else None
        if shape_name in init:
            arr = onnx.numpy_helper.to_array(init[shape_name])
            vals = tuple(arr.tolist())
            unique_shapes[vals] += 1
            if shown < 40:
                print(" reshape", n.name, "shape", list(vals), "out", n.output[0])
                shown += 1
    print("reshape unique shapes:")
    for k, v in unique_shapes.most_common():
        print(" ", v, k)

    hits = 0
    for name, i in init.items():
        if i.data_type not in (6, 7):
            continue
        try:
            arr = onnx.numpy_helper.to_array(i).reshape(-1)
        except Exception:
            continue
        if arr.size <= 16 and any(int(x) in (64, 72, 80, 16, 18, 1152, 324) for x in arr.tolist()):
            print("init", name, arr.tolist())
            hits += 1
            if hits > 40:
                break

    # custom ops with 64/72
    custom = [n for n in g.node if n.domain and n.domain != "ai.onnx"]
    print("custom nodes", len(custom), "types", Counter(n.op_type for n in custom).most_common(20))


base = Path("/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant")
summarize(base / "visual/hmquant_gr00t_with_act.onnx")
print("\n################ PREFILL ################\n")
summarize(base / "prefill/hmquant_gr00t_with_act.onnx")
