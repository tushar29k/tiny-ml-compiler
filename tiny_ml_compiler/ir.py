"""Tiny tensor-graph IR: nodes, graphs, and a NumPy interpreter.

A graph is a list of nodes in topological order. Each node has an op,
a list of input node ids, and optional data (a constant value or an
input name). Supported ops: input, const, matmul, add, mul, relu,
fused_mul_add (produced by the fusion pass, never written by hand).
"""

import numpy as np


OPS = ("input", "const", "matmul", "add", "mul", "relu", "fused_mul_add")


class Node:
    def __init__(self, node_id, op, inputs=(), value=None, name=None):
        if op not in OPS:
            raise ValueError(f"unknown op: {op}")
        self.id = node_id
        self.op = op
        self.inputs = list(inputs)
        self.value = value  # numpy array for const nodes
        self.name = name    # input name for input nodes

    @property
    def shape(self):
        if self.op == "const":
            return tuple(self.value.shape)
        return None  # shapes for computed nodes are inferred by Graph

    def __repr__(self):
        return f"Node({self.id!r}, {self.op}, inputs={self.inputs})"


class Graph:
    """Nodes in topological order; outputs are node ids."""

    def __init__(self):
        self.nodes = {}   # id -> Node, insertion ordered
        self.outputs = []
        self._counter = 0

    def _new_id(self, hint):
        self._counter += 1
        return f"{hint}_{self._counter}"

    def add_input(self, name, shape=None):
        node_id = self._new_id("input")
        node = Node(node_id, "input", name=name)
        node.inferred_shape = tuple(shape) if shape else None
        self.nodes[node_id] = node
        return node_id

    def add_const(self, value):
        node_id = self._new_id("const")
        node = Node(node_id, "const", value=np.asarray(value, dtype=np.float64))
        node.inferred_shape = tuple(node.value.shape)
        self.nodes[node_id] = node
        return node_id

    def add_op(self, op, inputs):
        node_id = self._new_id(op)
        self.nodes[node_id] = Node(node_id, op, inputs)
        return node_id

    def set_outputs(self, outputs):
        self.outputs = list(outputs)

    def copy(self):
        g = Graph()
        for node in self.nodes.values():
            n = Node(node.id, node.op, node.inputs,
                     value=None if node.value is None else node.value.copy(),
                     name=node.name)
            n.inferred_shape = getattr(node, "inferred_shape", None)
            g.nodes[node.id] = n
        g.outputs = list(self.outputs)
        g._counter = self._counter
        return g

    def infer_shapes(self):
        """Fill inferred_shape for every node; None where it can't be derived."""
        for node in self.nodes.values():
            if node.op in ("input", "const"):
                continue
            ins = [self.nodes[i] for i in node.inputs]
            shapes = [getattr(n, "inferred_shape", None) for n in ins]
            if any(s is None for s in shapes):
                node.inferred_shape = None
                continue
            try:
                if node.op == "matmul":
                    node.inferred_shape = (shapes[0][0], shapes[1][1])
                elif node.op in ("add", "mul"):
                    node.inferred_shape = tuple(np.broadcast_shapes(*shapes))
                elif node.op == "relu":
                    node.inferred_shape = shapes[0]
                elif node.op == "fused_mul_add":
                    node.inferred_shape = tuple(
                        np.broadcast_shapes(*shapes))
            except ValueError:
                node.inferred_shape = None
        return self

    def evaluate(self, feed):
        """Run the graph with the NumPy interpreter.

        feed: {input_name: array}. Returns {output_id: array}.
        """
        values = {}
        for node in self.nodes.values():
            if node.op == "input":
                values[node.id] = np.asarray(feed[node.name], dtype=np.float64)
            elif node.op == "const":
                values[node.id] = node.value
            elif node.op == "matmul":
                values[node.id] = values[node.inputs[0]] @ values[node.inputs[1]]
            elif node.op == "add":
                values[node.id] = values[node.inputs[0]] + values[node.inputs[1]]
            elif node.op == "mul":
                values[node.id] = values[node.inputs[0]] * values[node.inputs[1]]
            elif node.op == "relu":
                values[node.id] = np.maximum(values[node.inputs[0]], 0.0)
            elif node.op == "fused_mul_add":
                # Deliberately plain: a * b + c. A single-machine-kernel
                # fusion is what a production compiler would emit; NumPy
                # cannot express that here, so this toy does not pretend
                # to. See README "What the benchmark actually shows".
                a, b, c = (values[i] for i in node.inputs)
                values[node.id] = a * b + c
            else:  # pragma: no cover - constructor guards ops
                raise ValueError(f"cannot evaluate op {node.op}")
        return {o: values[o] for o in self.outputs}
