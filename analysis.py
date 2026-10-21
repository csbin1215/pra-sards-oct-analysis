"""Dog-level statistical analysis of SD-OCT retinal layer thickness in PRA vs SARDS dogs.

Analyses (as described in Section 2.7 of the manuscript):
  1. Age comparison between groups (Mann-Whitney U test).
  2. Exact permutation test of the difference in group means (SARDS - PRA),
     permuting group labels at the dog level over all C(16,5) = 4,368 splits.
  3. Benjamini-Hochberg false discovery rate (FDR) adjustment across the five parameters.
  4. Dog-level (cluster) bootstrap 95% confidence intervals (10,000 resamples, seed 12345).

Usage:
    python analysis.py

Input:  data/retinal_thickness_eye_level.csv, data/age_by_dog.csv
Output: printed to screen and saved to results/results.csv
"""

import csv
import itertools
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
THICKNESS_FILE = HERE / "data" / "retinal_thickness_eye_level.csv"
AGE_FILE = HERE / "data" / "age_by_dog.csv"
RESULTS_FILE = HERE / "results" / "results.csv"

VAR_NAMES = ["TRT", "IRT", "ORT", "ONL", "PR_ISOS"]
N_BOOT = 10000
SEED = 12345


def load_dogs(path):
    """Return a list of (group, dog_id, eyes), where eyes is a list of 5-value lists."""
    dogs = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row["dog_id"]
            if key not in dogs:
                dogs[key] = (row["group"], key, [])
            dogs[key][2].append([float(row[v]) for v in VAR_NAMES])
    return list(dogs.values())


def group_mean(dogs, indices, v):
    """Mean of all eyes belonging to the given dogs, for parameter v."""
    vals = [eye[v] for i in indices for eye in dogs[i][2]]
    return float(np.mean(vals))


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    fdr = [0.0] * m
    prev_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        adj = min(pvals[idx] * m / rank, prev_min)
        fdr[idx] = adj
        prev_min = adj
    return fdr


def age_comparison(path):
    ages = {"PRA": [], "SARDS": []}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ages[row["group"]].append(float(row["age_years"]))
    res = stats.mannwhitneyu(ages["PRA"], ages["SARDS"], alternative="two-sided")
    print("=== Age comparison (Mann-Whitney U test) ===")
    for g in ("PRA", "SARDS"):
        a = np.array(ages[g])
        print(f"{g}: n={a.size} median={np.median(a):.1f} range={a.min():.0f}-{a.max():.0f} "
              f"mean={a.mean():.1f} SD={a.std(ddof=1):.1f}")
    print(f"U={res.statistic:.1f}, p={res.pvalue:.3f}\n")


def main():
    age_comparison(AGE_FILE)

    dogs = load_dogs(THICKNESS_FILE)
    n = len(dogs)
    pra_idx = [i for i, d in enumerate(dogs) if d[0] == "PRA"]
    sards_idx = [i for i, d in enumerate(dogs) if d[0] == "SARDS"]
    all_idx = list(range(n))

    print("=== Observed group means (SARDS - PRA) ===")
    observed = []
    for v in range(len(VAR_NAMES)):
        m_pra = group_mean(dogs, pra_idx, v)
        m_sards = group_mean(dogs, sards_idx, v)
        observed.append(m_sards - m_pra)
        print(f"{VAR_NAMES[v]}: SARDS={m_sards:.3f} PRA={m_pra:.3f} diff={observed[v]:.3f}")

    print(f"\n=== Exact permutation test: C({n},{len(pra_idx)}) combinations ===")
    combos = list(itertools.combinations(all_idx, len(pra_idx)))
    total = len(combos)
    extreme_count = [0] * len(VAR_NAMES)
    for combo in combos:
        combo_b = [i for i in all_idx if i not in set(combo)]
        for v in range(len(VAR_NAMES)):
            diff = group_mean(dogs, combo_b, v) - group_mean(dogs, combo, v)
            if abs(diff) >= abs(observed[v]) - 1e-9:
                extreme_count[v] += 1
    pvals = [c / total for c in extreme_count]
    for v in range(len(VAR_NAMES)):
        print(f"{VAR_NAMES[v]}: exact p = {pvals[v]:.4f} (extreme={extreme_count[v]}/{total})")

    fdr = bh_fdr(pvals)
    print("\n=== BH-FDR adjusted p-values ===")
    for v in range(len(VAR_NAMES)):
        print(f"{VAR_NAMES[v]}: FDR-p = {fdr[v]:.4f}")

    print(f"\n=== Bootstrap 95% CI ({N_BOOT} resamples, dog-level, seed {SEED}) ===")
    rng = np.random.default_rng(SEED)
    rows = []
    for v in range(len(VAR_NAMES)):
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pra_pick = rng.choice(pra_idx, size=len(pra_idx), replace=True)
            sards_pick = rng.choice(sards_idx, size=len(sards_idx), replace=True)
            boot[b] = group_mean(dogs, sards_pick, v) - group_mean(dogs, pra_pick, v)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{VAR_NAMES[v]}: diff={observed[v]:.1f} 95% CI=({lo:.1f},{hi:.1f}) "
              f"exact-p={pvals[v]:.4f} FDR-p={fdr[v]:.4f}")
        rows.append({"parameter": VAR_NAMES[v], "diff_SARDS_minus_PRA": round(observed[v], 3),
                     "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
                     "exact_p": round(pvals[v], 4), "fdr_p": round(fdr[v], 4)})

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {RESULTS_FILE.relative_to(HERE)}")


if __name__ == "__main__":
    main()
