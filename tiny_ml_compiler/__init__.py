"""tiny-ml-compiler: a toy tensor-graph compiler in pure Python + NumPy.

Public API: Graph / Node (ir), constant_fold / dead_code_eliminate /
fuse_mul_add / optimize (passes), generate / compile (codegen).
"""

from .ir import Graph, Node
from .passes import (
    constant_fold,
    dead_code_eliminate,
    fuse_mul_add,
    optimize,
)
from .codegen import compile, generate

__all__ = [
    "Graph",
    "Node",
    "constant_fold",
    "dead_code_eliminate",
    "fuse_mul_add",
    "optimize",
    "compile",
    "generate",
]
