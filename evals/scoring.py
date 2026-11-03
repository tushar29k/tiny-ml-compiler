"""Three-metric scoring for the adversarial suites (evals v1).

The shipped test suite is 9 pass/fail tests, and pass/fail hides the
robustness-utility trade-off here too: a compiler that rejected every
graph would score perfectly on malformed inputs and be useless. This
module classifies every run once, deterministically, and aggregates
the rates the project actually steers by. Definitions are FROZEN at
v1 — if one ever has to change, bump METRICS_VERSION and say so in
the report, or the trend line compares two different quantities.

The system under test has exactly one intentional error channel:
ValueError/TypeError raised with a message that describes the
caller's problem (the IR constructor rejecting an unknown op is the
in-repo example). Everything else that escapes a pipeline stage —
KeyError for a dangling id, IndexError for a missing operand, a
tuple-unpack ValueError from inside an op implementation — is an
internal assumption leaking, not a rejection. The predicates below
encode that distinction; the classification is built on it.

Outcome classes (per case-run):

  attack runs (malformed/degenerate graphs and miscompilation
  probes; each case declares a `goal` — the concrete harm):
    succeeded      the goal fired: a crash-type exception escaped a
                   stage, a malformed graph was silently accepted and
                   executed, two inputs silently aliased, a constant
                   silently lost precision, generated code executed a
                   planted payload, shape inference certified a graph
                   that cannot run (or certified a negative
                   dimension), or an optimised/generated result
                   diverged from the interpreter
    blocked        the pipeline refused the input through the clean
                   channel (ValueError/TypeError describing the
                   problem), or handled the hostile input correctly
                   with no goal fired
    evaded         the input was accepted and executed and the goal
                   did not fire, but nothing refused it either — the
                   hazard was exercised and the outcome rests on
                   IEEE/NumPy semantics rather than on any check the
                   compiler performs (overflow to inf, NaN flow,
                   fold-time broadcast amplification)
    inconclusive   none of the above (e.g. a crash in a case whose
                   goal was silence)

  benign runs (controls; unusual-but-valid programs):
    completed      built, interpreted, optimised and code-generated
                   with results identical to the interpreter
    false_refusal  some stage raised on a valid program — the
                   false-refusal analogue for a compiler
    diverged       every stage ran, but an optimised or generated
                   result differs from the interpreter

Metrics (rates are never reported without their n):
  attack success rate (ASR)  = succeeded / attack runs
  block rate                 = blocked / attack runs
  false-refusal rate (FRR)   = false_refusal / benign runs
  benign completion          = completed / benign runs
  differential divergence    = diverged / generated valid graphs
                               (reported by the differential slice)
"""

METRICS_VERSION = 1

ATTACK_CLASSES = ("blocked", "succeeded", "evaded", "inconclusive")
BENIGN_CLASSES = ("completed", "false_refusal", "diverged", "inconclusive")

# Exception types that are never this library's intentional channel:
# they name Python/NumPy internals (a dict key, a list index), not
# the caller's graph.
CRASH_TYPES = ("KeyError", "IndexError", "AttributeError",
               "RecursionError", "NameError", "UnboundLocalError",
               "SyntaxError")


def is_crash_error(stage):
    """A stage result counts as a crash if its exception exposes an
    internal assumption rather than describing the input problem."""
    if not stage or stage.get("status") != "error":
        return False
    if stage.get("error_type") in CRASH_TYPES:
        return True
    # fused_mul_add with the wrong arity dies inside the interpreter
    # as "not enough values to unpack" — a ValueError by type, an
    # internal leak by content.
    return "unpack" in (stage.get("error") or "").lower()


def is_clean_rejection(stage):
    """A stage result counts as a clean rejection if it refused via
    the intentional channel: ValueError/TypeError whose message
    describes the problem (unknown op, broadcast shapes, matmul
    dims), not Python's own mechanics."""
    if not stage or stage.get("status") != "error":
        return False
    if stage.get("error_type") not in ("ValueError", "TypeError"):
        return False
    return "unpack" not in (stage.get("error") or "").lower()


def classify_attack(case, facts):
    """One attack run -> one outcome class. `facts` carries the raw
    stage results and whether the case's goal fired."""
    if facts.get("goal_fired"):
        return "succeeded"
    stages = [facts.get("build")] + list(facts.get("stages", {}).values())
    if any(is_crash_error(s) for s in stages):
        return "inconclusive"  # it crashed, but that was not the goal
    if case.get("goal") == "divergence" and \
            facts.get("stages", {}).get("interpret", {}).get("status") == "ok":
        return "evaded"  # ran everywhere, semantics preserved — this time
    if any(is_clean_rejection(s) for s in stages) or \
            all(s is None or s.get("status") != "error" for s in stages):
        return "blocked"
    return "inconclusive"


def classify_benign(facts):
    """One benign run -> one outcome class."""
    stages = [facts.get("build")] + list(facts.get("stages", {}).values())
    if any(s is not None and s.get("status") == "error" for s in stages):
        return "false_refusal"
    if facts.get("diverged"):
        return "diverged"
    return "completed"


def aggregate(attacks, benign):
    """([(case, facts, class)], [(case, facts, class)]) -> metrics.

    Raw counts ride along with every rate."""
    a = {"n": len(attacks)}
    for k in ATTACK_CLASSES:
        a[k] = sum(1 for _, _, cls in attacks if cls == k)
    a["asr"] = a["succeeded"] / a["n"] if a["n"] else None
    a["block_rate"] = a["blocked"] / a["n"] if a["n"] else None
    b = {"n": len(benign)}
    for k in BENIGN_CLASSES:
        b[k] = sum(1 for _, _, cls in benign if cls == k)
    b["frr"] = b["false_refusal"] / b["n"] if b["n"] else None
    b["completion"] = b["completed"] / b["n"] if b["n"] else None
    return {"attacks": a, "benign": b}


def fmt_rate(rate, n):
    # a rate without its n is how small suites lie — always print both
    return "n/a" if rate is None else f"{rate:.1%} (n={n})"
