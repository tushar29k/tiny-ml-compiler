"""Compiler passes: constant folding, dead-code elimination, fusion.

Each pass takes a Graph and returns a new, transformed Graph. None of
them changes the graph's outputs (verified by the tests).
"""

import numpy as np

from .ir import Node


def _eval_const_op(node, values):
    if node.op == "matmul":
        return values[node.inputs[0]] @ values[node.inputs[1]]
    if node.op == "add":
        return values[node.inputs[0]] + values[node.inputs[1]]
    if node.op == "mul":
        return values[node.inputs[0]] * values[node.inputs[1]]
    if node.op == "relu":
        return np.maximum(values[node.inputs[0]], 0.0)
    if node.op == "fused_mul_add":
        a, b, c = (values[i] for i in node.inputs)
        return a * b + c
    return None


def constant_fold(graph):
    """Replace any computable op whose inputs are all constants with a
    single const node holding the computed value. Runs in topological
    order, so a fold can enable further folds downstream in one pass.
    """
    g = graph.copy()
    const_values = {}
    for node in g.nodes.values():
        if node.op == "const":
            const_values[node.id] = node.value
        elif node.op == "input":
            continue
        elif all(i in const_values for i in node.inputs):
            value = _eval_const_op(node, const_values)
            node.op = "const"
            node.inputs = []
            node.value = value
            node.inferred_shape = tuple(value.shape)
            const_values[node.id] = value
    return g


def dead_code_eliminate(graph):
    """Drop every node that does not contribute to a graph output."""
    g = graph.copy()
    reachable = set()
    stack = list(g.outputs)
    while stack:
        node_id = stack.pop()
        if node_id in reachable or node_id not in g.nodes:
            continue
        reachable.add(node_id)
        stack.extend(g.nodes[node_id].inputs)
    g.nodes = {i: n for i, n in g.nodes.items() if i in reachable}
    return g


def fuse_mul_add(graph):
    """Fuse mul followed by add into one fused_mul_add node.

    Pattern: y = mul(a, b); z = add(y, c)  (or add(c, y))
    becomes:  z = fused_mul_add(a, b, c).

    Only applied when the mul node has exactly one consumer (the add),
    so no other node silently loses its input. The orphaned mul node is
    left behind for dead_code_eliminate to remove.
    """
    g = graph.copy()
    consumers = {}
    for node in g.nodes.values():
        for inp in node.inputs:
            consumers.setdefault(inp, []).append(node.id)
    for node in list(g.nodes.values()):
        if node.op != "add" or len(node.inputs) != 2:
            continue
        mul_id = None
        other_id = None
        for inp in node.inputs:
            if g.nodes[inp].op == "mul" and len(consumers.get(inp, [])) == 1:
                mul_id = inp
            else:
                other_id = inp
        if mul_id is None or other_id is None:
            continue
        mul = g.nodes[mul_id]
        node.op = "fused_mul_add"
        node.inputs = [mul.inputs[0], mul.inputs[1], other_id]
    return g


def optimize(graph):
    """Full pipeline: fold constants, fuse mul+add, drop dead code."""
    return dead_code_eliminate(fuse_mul_add(constant_fold(graph)))
