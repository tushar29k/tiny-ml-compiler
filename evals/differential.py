"""Differential slice: seeded random valid graphs, interpreter vs
optimised vs generated code, compared bit-for-bit.

This is the comparability slice, and the recognised compiler-testing
format it follows is differential testing in the CSmith tradition:
generate many valid programs, run them through independent execution
paths, and treat any disagreement as a bug. The generator here is
hand-rolled for this IR (CSmith targets C; no off-the-shelf
generator speaks this graph format) — that is stated plainly rather
than dressed up. Graphs are valid by construction: operands are only
paired when their shapes broadcast (or are both (4,4) for matmul),
wiring only references earlier nodes, and all values are finite, so
the expected divergence count is exactly zero and any nonzero count
is a miscompilation.

Coverage is reported alongside: how many generated graphs actually
exercised constant folding (node count changed under optimize) and
operator fusion (a fused_mul_add node appeared). A 0-divergence
result over graphs that never trigger the passes would be vacuous;
the coverage numbers are what make it mean something.

Feeds are passed to every path as plain JSON values (lists), which
is inside the documented contract — the README's own usage example
calls the compiled function with a list. The interpreter coerces
feeds with np.asarray; whether the other paths cope is part of what
is being measured. Divergences are therefore reported per path
(passes via opt_eval, codegen raw, codegen on the optimised graph),
so a codegen failure cannot masquerade as a pass failure or hide
one.
"""

import random

import numpy as np

import harness

SEED = 20261103
N_GRAPHS = 240
SHAPES = [(), (4,), (4, 4)]


def _values(rng, shape):
    if not shape:
        return rng.gauss(0.0, 1.0)
    if len(shape) == 1:
        return [rng.gauss(0.0, 1.0) for _ in range(shape[0])]
    return [_values(rng, shape[1:]) for _ in range(shape[0])]


def gen_case(rng, idx):
    """One random valid graph as a harness case dict."""
    nodes = []
    pool = []  # (ref, shape) of values available to consume
    feed = {}
    counter = [0]

    def fresh(prefix):
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    for _ in range(rng.randint(1, 3)):
        ref = fresh("in")
        shape = rng.choice(SHAPES)
        name = f"in_{ref}"
        nodes.append({"ref": ref, "op": "input", "name": name,
                      "shape": list(shape)})
        feed[name] = _values(rng, shape)
        pool.append((ref, shape))

    computed = []
    for _ in range(rng.randint(5, 12)):
        kind = rng.choice(["add", "mul", "relu", "matmul", "const"])
        if kind == "const":
            ref = fresh("c")
            shape = rng.choice(SHAPES)
            nodes.append({"ref": ref, "op": "const",
                          "value": _values(rng, shape)})
            pool.append((ref, shape))
            continue
        if kind == "relu":
            ref = fresh("relu")
            src, shape = rng.choice(pool)
            nodes.append({"ref": ref, "op": "relu", "inputs": [src]})
            pool.append((ref, shape))
            computed.append(ref)
            continue
        if kind == "matmul":
            sq = [p for p in pool if p[1] == (4, 4)]
            if len(sq) < 1:
                continue
            a = rng.choice(sq)
            b = rng.choice(sq)
            ref = fresh("mm")
            nodes.append({"ref": ref, "op": "matmul",
                          "inputs": [a[0], b[0]]})
            pool.append((ref, (4, 4)))
            computed.append(ref)
            continue
        # add / mul: pair two pool entries whose shapes broadcast
        pair = None
        for _try in range(8):
            a, b = rng.choice(pool), rng.choice(pool)
            try:
                shape = tuple(np.broadcast_shapes(a[1], b[1]))
                pair = (a, b, shape)
                break
            except ValueError:
                continue
        if pair is None:
            continue
        a, b, shape = pair
        ref = fresh(kind)
        nodes.append({"ref": ref, "op": kind, "inputs": [a[0], b[0]]})
        pool.append((ref, shape))
        computed.append(ref)
        if kind == "mul" and rng.random() < 0.5:
            # bias toward the fusion pattern: an add consuming this
            # mul (plus any broadcastable partner) — without it the
            # mul -> add shape is too rare to exercise the pass
            partner = None
            for _try in range(8):
                cand = rng.choice(pool[:-1])
                try:
                    pshape = tuple(np.broadcast_shapes(shape, cand[1]))
                    partner = (cand, pshape)
                    break
                except ValueError:
                    continue
            if partner is not None:
                ref = fresh("add")
                nodes.append({"ref": ref, "op": "add",
                              "inputs": [pool[-1][0], partner[0][0]]})
                pool.append((ref, partner[1]))
                computed.append(ref)

    outputs = rng.sample(computed, k=min(len(computed),
                                         rng.randint(1, 3)))
    spec = {"nodes": nodes, "outputs": outputs}
    return {"id": f"diff-{idx:03d}", "family": "differential",
            "goal": "divergence", "graph": spec, "feed": feed}


def run_slice(n=N_GRAPHS, seed=SEED):
    """Run the slice; return (rows, summary). Rows carry the per-graph
    facts and the divergence verdict."""
    rng = random.Random(seed)
    rows = []
    summary = {"n": 0, "diverged": 0, "with_folding": 0,
               "with_fusion": 0, "errors": 0,
               "diverged_opt_eval": 0, "diverged_codegen_raw": 0,
               "diverged_codegen_opt": 0,
               "stage_errors": {}}
    for i in range(n):
        case = gen_case(rng, i)
        facts = harness.run_case(case)
        stages = [facts.get("build")] + \
            list(facts.get("stages", {}).values())
        errored = any(s is not None and s.get("status") == "error"
                      for s in stages)
        summary["n"] += 1
        summary["diverged"] += int(facts["diverged"])
        summary["errors"] += int(errored)
        for path, div in facts.get("path_diverged", {}).items():
            summary[f"diverged_{path}"] += int(div)
        for name, stage in facts.get("stages", {}).items():
            if stage and stage.get("status") == "error":
                summary["stage_errors"][name] = \
                    summary["stage_errors"].get(name, 0) + 1
        details = facts.get("details", {})
        counts = details.get("node_counts", {})
        if counts and counts["optimized"] != counts["input"]:
            summary["with_folding"] += 1
        if details.get("optimized_has_fused"):
            summary["with_fusion"] += 1
        rows.append((case, facts))
    return rows, summary
