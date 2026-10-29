# tiny-ml-compiler

A toy tensor-graph compiler in pure Python + NumPy. It exists to learn
how a compiler pipeline fits together: a small IR, three real passes,
and a code generator. It is a demo and learning project, **not a
production compiler**, and it makes no claim to production compiler
experience. It does not emit machine code, target GPUs, or compete
with XLA, TVM, MLIR, or any real compiler stack.

## The IR

A `Graph` is a list of nodes in topological order. Each node has an
op, input node ids, and optional data:

- `input` — a named graph input, fed at run time
- `const` — a constant NumPy array
- `matmul`, `add`, `mul`, `relu` — elementwise / matrix ops
- `fused_mul_add` — produced only by the fusion pass, never written
  by hand; it means `a * b + c` as a single graph node

Graphs are evaluated by a small NumPy interpreter
(`Graph.evaluate`), which is the ground truth every other part of the
project is checked against.

## The passes

`optimize(graph)` runs, in order:

1. **Constant folding** — any op whose inputs are all constants is
   replaced by a single `const` node holding the computed value. It
   runs in topological order, so one fold can enable the next.
2. **Operator fusion** — `mul` followed by `add` becomes one
   `fused_mul_add` node, but only when the `mul` has exactly one
   consumer (the `add`). A `mul` feeding two consumers is left alone,
   so no other node loses its input.
3. **Dead-code elimination** — any node that does not contribute to a
   graph output is dropped. This is what removes a `mul` orphaned by
   fusion, and the constant inputs orphaned by folding.

Each pass returns a new graph; the original is untouched.

## Codegen

`tiny_ml_compiler.codegen.generate(graph)` lowers a graph to readable
NumPy source code — one expression per node, with fused nodes emitted
as a single `a * b + c` expression. `compile(graph)` returns a
callable built from that source. Generated code is checked against the
interpreter in the tests, so codegen is verified, not trusted.

## Usage

```python
from tiny_ml_compiler.ir import Graph
from tiny_ml_compiler.passes import optimize
from tiny_ml_compiler.codegen import compile

g = Graph()
x = g.add_input("x")
w = g.add_const([2.0, 3.0])
b = g.add_const([1.0, 1.0])
m = g.add_op("mul", [x, w])
a = g.add_op("add", [m, b])
g.set_outputs([a])

run = compile(optimize(g))
print(run(x=[1.0, 2.0]))  # {'a_5': array([3., 7.])}
```

## Tests

```sh
python3 -m pytest tests/ -q
```

9 tests, all passing at the time of writing: constant folding (plain,
chained/matmul, and the variable-input case that must *not* fold),
dead-code elimination, fusion (including the shared-`mul` case that
must *not* fuse), end-to-end numerical equivalence between the
original and optimized graphs, and codegen matching the interpreter.

## What the benchmark actually shows

```sh
python3 benchmark.py
```

The benchmark times a chain of 2,000 mul-add pairs over a 4,096-float
vector, unfused vs fused. Three runs on the author's machine gave:

| run | unfused | fused | speedup | max output difference |
|-----|---------|-------|---------|----------------------|
| 1 | 16.61 ms | 20.90 ms | 0.80x | 0.0 |
| 2 | 16.43 ms | 16.28 ms | 1.01x | 0.0 |
| 3 | 15.99 ms | 16.44 ms | 0.97x | 0.0 |

Fusion cuts the graph from 8,001 nodes to 6,001 nodes, and the fused
output is bit-identical to the unfused output (difference 0.0). But
the wall-clock result is roughly flat, hovering around 1x. That is
the honest finding, and it is the interesting one: in this toy,
fusion only restructures the graph — NumPy still runs the same two
kernels underneath, so kernel time dominates and there is no real
speedup to report. A production compiler's fusion win comes from
emitting a single machine kernel for the fused op, which this toy
does not do. Re-run `benchmark.py` to reproduce the numbers on your
own machine; they will vary run to run.

## What this is not

- Not a production compiler, not a kernel generator, not GPU code.
- No autotuning, no layout or memory planning, no quantization.
- The "fusion" here is a graph rewrite plus one NumPy expression, and
  the benchmark above is the evidence for exactly how far that goes.

## Layout

```
tiny_ml_compiler/
  ir.py        # Node, Graph, NumPy interpreter
  passes.py    # constant folding, DCE, mul+add fusion
  codegen.py   # graph -> executable NumPy source
benchmark.py   # fused vs unfused timing (real, unflattering numbers)
tests/         # pytest suite
```
