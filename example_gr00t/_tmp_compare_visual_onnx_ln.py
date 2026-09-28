import numpy as np
import onnx
from pathlib import Path


def summarize(path: str, tag: str):
    p = Path(path)
    print("=" * 80)
    print(tag, p, "exists", p.exists(), "size", p.stat().st_size if p.exists() else None)
    m = onnx.load(str(p), load_external_data=True)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    print("n init", len(init), "n nodes", len(g.node), "opset", [o.version for o in m.opset_import[:6]])
    print("inputs", [i.name for i in g.input])
    print("outputs", [o.name for o in g.output])

    ops = {}
    for n in g.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("ops", dict(sorted(ops.items(), key=lambda x: -x[1])[:20]))

    ln_nodes = [n for n in g.node if n.op_type in ("LayerNorm", "RMSNorm")]
    print("LN/RMS nodes", len(ln_nodes))
    identity_w = 0
    identity_b = 0
    w_stats = []
    b_stats = []
    for n in ln_nodes:
        w_name = n.input[1] if len(n.input) > 1 else None
        b_name = n.input[2] if len(n.input) > 2 else None
        if w_name and w_name in init:
            w = onnx.numpy_helper.to_array(init[w_name]).astype(np.float32)
            mae1 = float(np.mean(np.abs(w - 1.0)))
            w_stats.append((n.name, w_name, w.shape, float(w.mean()), float(w.min()), float(w.max()), mae1))
            if mae1 < 1e-6:
                identity_w += 1
        if b_name and b_name in init:
            b = onnx.numpy_helper.to_array(init[b_name]).astype(np.float32)
            mae0 = float(np.mean(np.abs(b)))
            b_stats.append((n.name, b_name, b.shape, float(b.mean()), float(b.min()), float(b.max()), mae0))
            if mae0 < 1e-6:
                identity_b += 1
    print("LN weight identity(ones)", identity_w, "/", len(w_stats))
    print("LN bias identity(zeros)", identity_b, "/", len(b_stats))
    if w_stats:
        print("first weight", w_stats[0])
        print("last weight", w_stats[-1])
        print("max mae_to_1", max(x[-1] for x in w_stats), "mean mae_to_1", float(np.mean([x[-1] for x in w_stats])))
    if b_stats:
        print("first bias", b_stats[0])
        print("last bias", b_stats[-1])
        print("max mae_to_0", max(x[-1] for x in b_stats), "mean mae_to_0", float(np.mean([x[-1] for x in b_stats])))

    # sample first LN node
    if ln_nodes:
        n0 = ln_nodes[0]
        print("first LN", n0.op_type, n0.name, "inputs", list(n0.input), "domain", n0.domain)
        nlast = ln_nodes[-1]
        print("last LN", nlast.op_type, nlast.name, "inputs", list(nlast.input), "domain", nlast.domain)
    return ops, ln_nodes, init


good = "/data01/home/wei.xie/v1.4.0_test/workspace/houmo-examples-xh2/gr00t/output/xh2/hmquant/visual/hmquant_gr00t_with_act.onnx"
bad = "/data01/home/wei.xie/v1.4.0_test/workspace/917_embodied_ai/embodied_ai/gr00t/output/xh2/hmquant/visual/hmquant_gr00t_with_act.onnx"
summarize(good, "GOOD")
summarize(bad, "BAD")
