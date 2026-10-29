"""Benchmark: fused vs unfused execution on a chain of mul-add pairs.

Builds y = (((x * w0) + b0) * w1 + b1) ... for N pairs, then times
the interpreter on the original graph (2N compute nodes) against the
optimized graph (N fused nodes). Prints real measured numbers.

Read the numbers honestly: fusion halves the compute-node count, but
in this toy interpreter the wall-clock result is roughly flat, because
NumPy kernel time dominates and this toy emits no single fused
machine kernel (see README). The benchmark exists to measure that,
not to claim a speedup that isn't there.
"""

import time

import numpy as np

from tiny_ml_compiler.ir import Graph
from tiny_ml_compiler.passes import optimize

N_PAIRS = 2000
SIZE = 4096
REPEATS = 20


def build_chain(n_pairs=N_PAIRS, size=SIZE):
    rng = np.random.default_rng(0)
    g = Graph()
    x = g.add_input("x", shape=(size,))
    cur = x
    for _ in range(n_pairs):
        w = g.add_const(rng.normal(size=size))
        b = g.add_const(rng.normal(size=size))
        cur = g.add_op("mul", [cur, w])
        cur = g.add_op("add", [cur, b])
    g.set_outputs([cur])
    return g


def time_it(fn, feed, repeats=REPEATS):
    fn(feed)  # warmup
    start = time.perf_counter()
    for _ in range(repeats):
        fn(feed)
    return (time.perf_counter() - start) / repeats * 1000.0


def main():
    rng = np.random.default_rng(1)
    feed = {"x": rng.normal(size=SIZE)}
    original = build_chain()
    optimized = optimize(original)
    unfused_ms = time_it(original.evaluate, feed)
    fused_ms = time_it(optimized.evaluate, feed)
    # correctness: same outputs within tolerance
    a = original.evaluate(feed)
    b = optimized.evaluate(feed)
    max_diff = max(float(np.max(np.abs(a[k] - b[k]))) for k in a)
    print(f"graph: {N_PAIRS} mul-add pairs, vector size {SIZE}")
    print(f"nodes: unfused={len(original.nodes)} fused={len(optimized.nodes)}")
    print(f"unfused: {unfused_ms:.2f} ms/run")
    print(f"fused:   {fused_ms:.2f} ms/run")
    print(f"speedup: {unfused_ms / fused_ms:.2f}x")
    print(f"max output difference: {max_diff:.3e}")
    assert max_diff < 1e-9, "fused output diverged from unfused output"


if __name__ == "__main__":
    main()
