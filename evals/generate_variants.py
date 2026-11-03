"""Generate evals/adversarial_variants.jsonl from variant_bases.jsonl.

Every mutation in mutations.MUTATIONS is applied to every base it
fits. The output's first line is a meta record whose sha256 covers
the canonical JSON of the bases and of this generator's mutation
source, so the file regenerates byte-identically and any drift is
visible. Run from the repo root: python evals/generate_variants.py
"""

import copy
import hashlib
import json
import os

import mutations

HERE = os.path.dirname(os.path.abspath(__file__))
BASES = os.path.join(HERE, "variant_bases.jsonl")
OUT = os.path.join(HERE, "adversarial_variants.jsonl")


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def build_cases():
    with open(BASES) as f:
        bases = [json.loads(line) for line in f if line.strip()]
    cases = []
    for base in bases:
        for name, fn in mutations.MUTATIONS.items():
            spec = copy.deepcopy(base["graph"])
            applied = fn(spec)
            if applied is None:
                continue
            case = {"id": f"{base['id']}__{name}",
                    "family": f"mutation:{name}",
                    "goal": applied["goal"],
                    "description": applied["description"],
                    "graph": spec,
                    "feed": base["feed"]}
            if "expect_result" in applied:
                case["expect_result"] = applied["expect_result"]
            cases.append(case)
    return bases, cases


def render():
    bases, cases = build_cases()
    with open(os.path.join(HERE, "mutations.py"), "rb") as f:
        mutations_sha = hashlib.sha256(f.read()).hexdigest()
    stamp = hashlib.sha256(
        (_canon(bases) + mutations_sha).encode()).hexdigest()
    meta = {"meta": True, "generator": "generate_variants.py",
            "sha256": stamp, "n_variants": len(cases)}
    lines = [json.dumps(meta, sort_keys=True)]
    lines += [json.dumps(c, sort_keys=True) for c in cases]
    return "\n".join(lines) + "\n"


def main():
    text = render()
    with open(OUT, "w") as f:
        f.write(text)
    n = text.count("\n") - 1  # minus the meta line
    print(f"wrote {OUT}: {n} variants")


if __name__ == "__main__":
    main()
