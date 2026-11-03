"""Structural mutations over graph specs (variant generation).

Each mutation takes a deep-copied base spec and returns a full case
dict (graph, goal, description, and expect_result where the goal
needs one), or None when the mutation does not apply to that base.
Mutations are deterministic and operate on the spec, not on built
graphs, so the generated variants regenerate byte-identically.
"""

import copy


def _op_nodes(spec):
    return [n for n in spec["nodes"]
            if n["op"] not in ("input", "const") and "chain" not in n]


def _binary_op_nodes(spec):
    return [n for n in _op_nodes(spec) if len(n.get("inputs", [])) >= 2]


def drop_last_operand(spec):
    targets = _binary_op_nodes(spec)
    if not targets:
        return None
    node = targets[-1]
    node["inputs"] = node["inputs"][:-1]
    return {"goal": "crash",
            "description": f"{node['op']} loses its last operand "
                           f"({node['ref']} keeps {len(node['inputs'])})"}


def append_duplicate_operand(spec):
    targets = [n for n in _binary_op_nodes(spec)
               if len(n.get("inputs", [])) == 2]
    if not targets:
        return None
    node = targets[0]
    node["inputs"] = node["inputs"] + [node["inputs"][0]]
    return {"goal": "silent_acceptance",
            "description": f"{node['op']} gains a duplicate third "
                           f"operand, which every stage ignores"}


def replace_operand_with_ghost(spec):
    targets = _binary_op_nodes(spec)
    if not targets:
        return None
    node = targets[0]
    node["inputs"] = [node["inputs"][0], "ghost_missing"]
    return {"goal": "crash",
            "description": f"{node['op']}'s second operand is a "
                           f"node id that does not exist"}


def rewire_first_input_to_self(spec):
    targets = _op_nodes(spec)
    if not targets:
        return None
    node = targets[0]
    node["inputs"] = [node["ref"]] + node["inputs"][1:]
    return {"goal": "crash",
            "description": f"{node['op']} is rewired to consume itself"}


def move_first_op_to_end(spec):
    nodes = spec["nodes"]
    for i, node in enumerate(nodes):
        if node["op"] not in ("input", "const") and "chain" not in node:
            nodes.append(nodes.pop(i))
            return {"goal": "crash",
                    "description": f"{node['op']} is moved after its "
                                   f"consumer (non-topological order)"}
    return None


def rename_second_input_to_first(spec):
    inputs = [n for n in spec["nodes"] if n["op"] == "input"]
    if len(inputs) < 2:
        return None
    inputs[1]["name"] = inputs[0]["name"]
    # only meaningful for base-two-input-fusion, whose feed and
    # weights make the aliased result exact: x*w + x for x = [1,1,1]
    return {"goal": "silent_alias",
            "description": "the second input node is renamed to the "
                           "first input's name, so both read one feed "
                           "entry",
            "expect_result": {"a": [3.0, 4.0, 5.0]}}


def _poison_first_leaf(value):
    if isinstance(value, list):
        if not value:
            return value
        value[0] = _poison_first_leaf(value[0])
        return value
    return "__nan__"


def poison_const_nan(spec):
    for node in spec["nodes"]:
        if node["op"] == "const" and \
                not (isinstance(node["value"], dict) and "fill" in node["value"]):
            node["value"] = _poison_first_leaf(
                copy.deepcopy(node["value"]))
            return {"goal": "divergence",
                    "description": "one constant element becomes NaN; "
                                   "every path should propagate it "
                                   "identically"}
    return None


def append_ghost_output(spec):
    spec["outputs"] = spec["outputs"] + ["ghost_missing"]
    return {"goal": "crash",
            "description": "the outputs list gains a node id that "
                           "does not exist"}


MUTATIONS = {
    "drop_last_operand": drop_last_operand,
    "append_duplicate_operand": append_duplicate_operand,
    "replace_operand_with_ghost": replace_operand_with_ghost,
    "rewire_first_input_to_self": rewire_first_input_to_self,
    "move_first_op_to_end": move_first_op_to_end,
    "rename_second_input_to_first": rename_second_input_to_first,
    "poison_const_nan": poison_const_nan,
    "append_ghost_output": append_ghost_output,
}
