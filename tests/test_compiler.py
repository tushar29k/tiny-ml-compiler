"""Tests for tiny-ml-compiler: folding, DCE, fusion, end-to-end equivalence."""

import numpy as np
import pytest

from tiny_ml_compiler.codegen import compile, generate
from tiny_ml_compiler.ir import Graph
from tiny_ml_compiler.passes import (
    constant_fold,
    dead_code_eliminate,
    fuse_mul_add,
    optimize,
)


def _const_graph():
    g = Graph()
    a = g.add_const([[1.0, 2.0], [3.0, 4.0]])
    b = g.add_const([[5.0, 6.0], [7.0, 8.0]])
    c = g.add_op("add", [a, b])
    g.set_outputs([c])
    return g


def test_constant_folding_add():
    g = _const_graph()
    folded = constant_fold(g)
    node = folded.nodes[g.outputs[0]]
    assert node.op == "const"
    np.testing.assert_array_equal(
        node.value, [[6.0, 8.0], [10.0, 12.0]])
    # folding alone rewrites the op in place; DCE drops the now-dead inputs
    assert len(optimize(g).nodes) == 1


def test_constant_folding_chain_and_matmul():
    g = Graph()
    a = g.add_const([[1.0, 0.0], [0.0, 1.0]])
    b = g.add_const([[2.0, 3.0], [4.0, 5.0]])
    m = g.add_op("matmul", [a, b])
    r = g.add_op("relu", [m])
    g.set_outputs([r])
    folded = constant_fold(g)
    assert all(n.op == "const" for n in folded.nodes.values())
    out = folded.evaluate({})[r]
    np.testing.assert_array_equal(out, [[2.0, 3.0], [4.0, 5.0]])


def test_constant_folding_keeps_variable_nodes():
    g = Graph()
    x = g.add_input("x")
    c = g.add_const(2.0)
    m = g.add_op("mul", [x, c])
    g.set_outputs([m])
    folded = constant_fold(g)
    assert folded.nodes[m].op == "mul"  # one input is not constant


def test_dead_code_elimination():
    g = Graph()
    x = g.add_input("x")
    live = g.add_op("relu", [x])
    dead = g.add_op("mul", [x, x])  # never reaches an output
    g.set_outputs([live])
    pruned = dead_code_eliminate(g)
    assert dead not in pruned.nodes
    assert live in pruned.nodes


def test_fusion_mul_add():
    g = Graph()
    x = g.add_input("x")
    w = g.add_const(2.0)
    b = g.add_const(1.0)
    m = g.add_op("mul", [x, w])
    a = g.add_op("add", [m, b])
    g.set_outputs([a])
    fused = fuse_mul_add(g)
    assert fused.nodes[a].op == "fused_mul_add"
    assert fused.nodes[a].inputs == [x, w, b]
    optimized = optimize(g)
    assert m not in optimized.nodes  # orphaned mul removed by DCE
    assert optimized.nodes[a].op == "fused_mul_add"


def test_fusion_skips_shared_mul():
    # mul feeding two consumers must not be fused away from the other one.
    g = Graph()
    x = g.add_input("x")
    w = g.add_const(2.0)
    m = g.add_op("mul", [x, w])
    a = g.add_op("add", [m, w])
    r = g.add_op("relu", [m])
    g.set_outputs([a, r])
    fused = fuse_mul_add(g)
    assert fused.nodes[a].op == "add"
    assert fused.nodes[m].op == "mul"


def test_end_to_end_numerical_equivalence():
    rng = np.random.default_rng(42)
    g = Graph()
    x = g.add_input("x", shape=(4, 4))
    w = g.add_const(rng.normal(size=(4, 4)))
    b = g.add_const(rng.normal(size=(4,)))
    m = g.add_op("matmul", [x, w])
    a = g.add_op("add", [m, b])
    r = g.add_op("relu", [a])
    m2 = g.add_op("mul", [r, w])
    out = g.add_op("add", [m2, b])
    g.set_outputs([out])
    feed = {"x": rng.normal(size=(4, 4))}
    expected = g.evaluate(feed)[out]
    got = optimize(g).evaluate(feed)[out]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_codegen_matches_interpreter():
    g = Graph()
    x = g.add_input("x", shape=(3,))
    w = g.add_const([2.0, 3.0, 4.0])
    b = g.add_const([1.0, 1.0, 1.0])
    m = g.add_op("mul", [x, w])
    a = g.add_op("add", [m, b])
    g.set_outputs([a])
    optimized = optimize(g)
    src = generate(optimized)
    assert "fused_mul_add" not in src  # codegen emits numpy, not IR ops
    assert " * " in src and " + " in src
    feed = {"x": np.array([1.0, 2.0, 3.0])}
    expected = g.evaluate(feed)[a]
    got = compile(optimized)(**feed)[a]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_codegen_unoptimized_matches_too():
    g = _const_graph()
    out = g.outputs[0]
    assert np.allclose(
        compile(g)()[out], g.evaluate({})[out])
