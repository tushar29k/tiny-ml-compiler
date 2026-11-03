"""Adversarial evals: malformed graphs, silent miscompilation
probes, benign controls, generated variants, differential slice.

Report-only (exit 0 unless the harness itself breaks): the CI gate
stays tests/test_compiler.py. This suite runs every case through the
real pipeline (interpreter, passes, codegen) and scores it with the
frozen three metrics in evals/scoring.py — attack success rate,
false-refusal rate, benign completion — plus the differential
slice's divergence count. Every run appends one summary row to
evals/eval-history.jsonl.

Run from the repo root: python evals/run_adversarial.py
"""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import differential  # noqa: E402
import harness  # noqa: E402
import scoring  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "eval-history.jsonl")
SEEDS = os.path.join(HERE, "adversarial_seeds.jsonl")
CONTROLS = os.path.join(HERE, "benign_controls.jsonl")
VARIANTS = os.path.join(HERE, "adversarial_variants.jsonl")


def run_attacks(cases):
    rows = []
    for case in cases:
        facts = harness.run_case(case)
        rows.append((case, facts, scoring.classify_attack(case, facts)))
    return rows


def run_benign(cases):
    rows = []
    for case in cases:
        facts = harness.run_case(case)
        rows.append((case, facts, scoring.classify_benign(facts)))
    return rows


def metrics_for(attacks, benign):
    return scoring.aggregate(
        [(c, f, cls) for c, f, cls in attacks],
        [(c, f, cls) for c, f, cls in benign])


def collect():
    """Run everything once; return rows + metrics for the report."""
    seed_attacks = run_attacks(harness.load_cases(SEEDS))
    control_benign = run_benign(harness.load_cases(CONTROLS))
    variant_attacks = run_attacks(harness.load_cases(VARIANTS))
    _rows, diff = differential.run_slice()
    return {
        "seed_attacks": seed_attacks,
        "control_benign": control_benign,
        "variant_attacks": variant_attacks,
        "differential": diff,
        "seed_metrics": metrics_for(seed_attacks, control_benign),
        "variant_metrics": metrics_for(variant_attacks, []),
    }


def main():
    data = collect()
    sm, vm = data["seed_metrics"], data["variant_metrics"]
    diff = data["differential"]
    row = {"suite": "adversarial",
           "metrics_version": scoring.METRICS_VERSION,
           "ts": datetime.datetime.now(
               datetime.timezone.utc).isoformat(),
           "seeds": sm, "variants": vm, "differential": diff}
    with open(HISTORY, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    a, b = sm["attacks"], sm["benign"]
    print(f"seeds:    ASR {scoring.fmt_rate(a['asr'], a['n'])} "
          f"(blocked {a['blocked']}, evaded {a['evaded']}, "
          f"inconclusive {a['inconclusive']})")
    print(f"controls: FRR {scoring.fmt_rate(b['frr'], b['n'])}, "
          f"completion {scoring.fmt_rate(b['completion'], b['n'])}")
    va = vm["attacks"]
    print(f"variants: ASR {scoring.fmt_rate(va['asr'], va['n'])} "
          f"(blocked {va['blocked']}, evaded {va['evaded']}, "
          f"inconclusive {va['inconclusive']})")
    print(f"differential: {diff['diverged']} divergences / "
          f"{diff['n']} graphs ({diff['errors']} errored, "
          f"{diff['with_folding']} exercised folding, "
          f"{diff['with_fusion']} exercised fusion; per path: "
          f"opt_eval {diff['diverged_opt_eval']}, codegen_raw "
          f"{diff['diverged_codegen_raw']}, codegen_opt "
          f"{diff['diverged_codegen_opt']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
