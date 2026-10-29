"""Codegen: lower a graph to executable NumPy source code.

The generated function takes the graph's inputs as keyword arguments
and returns a dict of outputs. Fused nodes become a single expression
(a * b + c); unfused graphs emit one expression per node. The output
is plain Python + NumPy, so it can be inspected, not just trusted.
"""

import numpy as np


def _expr(node, var):
    ins = [var[i] for i in node.inputs]
    if node.op == "matmul":
        return f"({ins[0]} @ {ins[1]})"
    if node.op == "add":
        return f"({ins[0]} + {ins[1]})"
    if node.op == "mul":
        return f"({ins[0]} * {ins[1]})"
    if node.op == "relu":
        return f"np.maximum({ins[0]}, 0.0)"
    if node.op == "fused_mul_add":
        return f"({ins[0]} * {ins[1]} + {ins[2]})"
    raise ValueError(f"cannot generate code for op {node.op}")


def generate(graph):
    """Return the source code of a function `run(**inputs)`."""
    var = {}
    lines = ["def run(**inputs):"]
    for node in graph.nodes.values():
        if node.op == "input":
            var[node.id] = f"inputs[{node.name!r}]"
        elif node.op == "const":
            var[node.id] = f"_const_{node.id}"
        else:
            var[node.id] = f"v_{node.id}"
            lines.append(f"    v_{node.id} = {_expr(node, var)}")
    out = ", ".join(f"{o!r}: {var[o]}" for o in graph.outputs)
    lines.append(f"    return {{{out}}}")
    return "\n".join(lines) + "\n"


def compile(graph):
    """Return a callable `run(**inputs)` built from generated code."""
    namespace = {"np": np}
    for node in graph.nodes.values():
        if node.op == "const":
            namespace[f"_const_{node.id}"] = node.value
    exec(generate(graph), namespace)  # noqa: S102 - generated from own IR
    return namespace["run"]
