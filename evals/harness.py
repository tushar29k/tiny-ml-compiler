"""Harness for the adversarial suites: builds graphs from JSON specs
and runs every pipeline stage against the interpreter.

Cases are data, not code, so the suites are regenerable and the
numbers reproducible. A spec lists nodes in order; the builder first
creates every node through the real builder API (`Graph.add_input` /
`add_const` / `add_op`), then wires inputs in a second pass — which
is what makes the hostile topologies expressible at all: an input may
name a ref that does not exist (dangling), a node created later
(non-topological order), another node mutually (a cycle), or itself
(a self-loop). An undefined ref is passed through as a literal id.

Stages run per case: build, interpret (Graph.evaluate — the ground
truth), optimize (passes.optimize), opt_eval, codegen_raw
(codegen.compile on the graph as written), codegen_opt (compile on
the optimised graph), and infer_shapes on a copy. Each stage records
status, exception type/message and warning categories; nothing is
patched and no stage's failure stops the others from being tried.
"""

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import scoring  # noqa: E402
from tiny_ml_compiler.codegen import compile as codegen_compile
from tiny_ml_compiler.ir import Graph
from tiny_ml_compiler.passes import optimize

_SENTINELS = {"__inf__": float("inf"), "-__inf__": float("-inf"),
              "__nan__": float("nan")}


def convert(value):
    """Recursively replace sentinel strings with IEEE specials."""
    if isinstance(value, str) and value in _SENTINELS:
        return _SENTINELS[value]
    if isinstance(value, list):
        return [convert(v) for v in value]
    if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
    return value


def load_cases(path):
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if case.get("meta"):
                continue
            cases.append(case)
    return cases


def build_graph(spec):
    """Spec -> (Graph, refmap). Raises whatever the builder raises —
    the caller records it as the build stage's result."""
    g = Graph()
    refmap = {}
    wired = []
    for node in spec["nodes"]:
        ref = node["ref"]
        if "chain" in node:
            chain = node["chain"]
            prev = refmap[chain["from"]]
            for _ in range(chain["times"]):
                prev = g.add_op(chain["op"], [prev])
            refmap[ref] = prev
        elif node["op"] == "input":
            shape = node.get("shape")
            refmap[ref] = g.add_input(
                node["name"], shape=tuple(shape) if shape else None)
        elif node["op"] == "const":
            value = node["value"]
            if isinstance(value, dict) and "fill" in value:
                # compact form for large constant tensors:
                # {"shape": [1, 96], "fill": 0.1} — passed as an
                # array, NOT via tolist(): nested lists cannot
                # express zero-size dimensions ((0, 3) -> [])
                refmap[ref] = g.add_const(np.full(
                    value["shape"], convert(value["fill"])))
                continue
            refmap[ref] = g.add_const(convert(value))
        else:
            refmap[ref] = g.add_op(node["op"], [])
            wired.append(node)
    for node in wired:
        nid = refmap[node["ref"]]
        g.nodes[nid].inputs = [refmap.get(r, r)
                               for r in node.get("inputs", [])]
    g.set_outputs([refmap.get(r, r) for r in spec["outputs"]])
    for tamper in spec.get("tamper", []):
        if "rename_node" in tamper:
            t = tamper["rename_node"]
            old = refmap[t["ref"]]
            node = g.nodes.pop(old)
            node.id = t["to"]
            g.nodes[t["to"]] = node
            refmap[t["ref"]] = t["to"]
    return g, refmap


def _stage(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = fn()
            return {"status": "ok", "error_type": None, "error": None,
                    "warnings": sorted({w.category.__name__
                                        for w in caught})}, out
        except Exception as e:  # the stage's failure IS the datum
            return {"status": "error", "error_type": type(e).__name__,
                    "error": str(e)[:160],
                    "warnings": sorted({w.category.__name__
                                        for w in caught})}, None


def _by_ref(results, refmap):
    inv = {v: k for k, v in refmap.items()}
    out = {}
    for k, v in results.items():
        try:
            out[inv.get(k, k)] = np.asarray(v, dtype=np.float64)
        except (ValueError, TypeError):
            # a path returned something that is not a numeric array
            # at all (e.g. a concatenated list) — keep it raw; the
            # equality check will (correctly) call it a divergence
            out[inv.get(k, k)] = v
    return out


def _results_equal(a, b):
    if a is None or b is None or set(a) != set(b):
        return False
    return all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def run_case(case):
    """One case -> the facts dict scoring.py classifies. Raw arrays
    stay out of the facts (they are not serialisable); equality
    verdicts and checker details are computed here, while the graph
    objects are live."""
    spec = case["graph"]
    feed = {k: convert(v) for k, v in case.get("feed", {}).items()}
    param = case.get("goal_param", {})
    facts = {"id": case["id"], "family": case["family"],
             "goal": case["goal"], "goal_fired": False,
             "diverged": False, "details": {}}

    env_key = param.get("env")
    if env_key:
        os.environ.pop(env_key, None)

    facts["build"], built = _stage(lambda: build_graph(spec))
    if built is None:
        return facts
    graph, refmap = built
    stages = {}
    results = {}

    stages["interpret"], raw = _stage(lambda: graph.evaluate(feed))
    if raw is not None:
        results["interpret"] = _by_ref(raw, refmap)

    ograph_holder = {}

    def _opt():
        ograph_holder["g"] = optimize(graph)
        return ograph_holder["g"]
    stages["optimize"], _ = _stage(_opt)
    ograph = ograph_holder.get("g")

    if ograph is not None:
        facts["details"]["node_counts"] = {
            "input": len(graph.nodes), "optimized": len(ograph.nodes)}
        facts["details"]["optimized_has_fused"] = any(
            n.op == "fused_mul_add" for n in ograph.nodes.values())
        stages["opt_eval"], raw = _stage(lambda: ograph.evaluate(feed))
        if raw is not None:
            results["opt_eval"] = _by_ref(raw, refmap)
        facts["details"]["dangling_outputs_after_optimize"] = [
            o for o in ograph.outputs if o not in ograph.nodes]
        if "probe_ref_in_optimized" in param:
            facts["details"]["probe_ref_in_optimized"] = \
                refmap[param["probe_ref_in_optimized"]] in ograph.nodes
        if "fold_probe_ref" in param and \
                refmap[param["fold_probe_ref"]] in ograph.nodes:
            node = ograph.nodes[refmap[param["fold_probe_ref"]]]
            if node.op == "const":
                facts["details"]["optimized_fold_numel"] = \
                    int(node.value.size)
        stages["codegen_opt"], raw = _stage(
            lambda: codegen_compile(ograph)(**feed))
        if raw is not None:
            results["codegen_opt"] = _by_ref(raw, refmap)

    stages["codegen_raw"], raw = _stage(
        lambda: codegen_compile(graph)(**feed))
    if raw is not None:
        results["codegen_raw"] = _by_ref(raw, refmap)

    def _shapes():
        g2 = graph.copy()
        g2.infer_shapes()
        return {ref: (list(g2.nodes[nid].inferred_shape)
                      if getattr(g2.nodes[nid], "inferred_shape", None)
                      is not None else None)
                for ref, nid in refmap.items() if nid in g2.nodes}
    stages["infer_shapes"], inferred = _stage(_shapes)
    if inferred is not None:
        facts["inferred_shapes"] = inferred

    facts["stages"] = stages
    base = results.get("interpret")
    facts["path_diverged"] = {
        k: (not _results_equal(base, results[k]))
        for k in ("opt_eval", "codegen_raw", "codegen_opt")
        if k in results} if base is not None else {}
    facts["diverged"] = any(facts["path_diverged"].values())

    goal = case["goal"]
    all_stages = [facts["build"]] + list(stages.values())
    if goal == "crash":
        facts["goal_fired"] = any(
            scoring.is_crash_error(s) for s in all_stages)
    elif goal == "silent_acceptance":
        facts["goal_fired"] = all(
            s["status"] == "ok" for s in all_stages)
    elif goal == "silent_alias":
        expect = {k: np.asarray(convert(v), dtype=np.float64)
                  for k, v in case["expect_result"].items()}
        facts["goal_fired"] = all(
            s["status"] == "ok" for s in all_stages) and \
            _results_equal(base, expect)
    elif goal == "precision_loss":
        node = graph.nodes[refmap[param["ref"]]]
        stored = np.asarray(node.value)
        facts["details"]["stored_value"] = repr(float(stored.item()))
        facts["goal_fired"] = int(stored.item()) != param["exact_int"]
    elif goal == "injection":
        facts["goal_fired"] = \
            os.environ.get(env_key) == param.get("value", "1")
        if env_key:
            os.environ.pop(env_key, None)
    elif goal == "shape_lie":
        if param.get("mode") == "negative_dim":
            shapes = (facts.get("inferred_shapes") or {}).values()
            facts["goal_fired"] = any(
                s is not None and any(d < 0 for d in s) for s in shapes)
        else:  # concrete shape certified for a graph that cannot run
            target = (facts.get("inferred_shapes") or {}).get(
                param["ref"])
            facts["goal_fired"] = target is not None and \
                stages["interpret"]["status"] == "error"
    elif goal == "divergence":
        facts["goal_fired"] = facts["diverged"]
    return facts
