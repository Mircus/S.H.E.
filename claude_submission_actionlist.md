# Claude ActionList: prepare article + repo for arXiv submission

## Mission
Take the current **coupled geometry–field paper** and the current **`she-geofield` repo** and turn them into a **coherent, reproducible arXiv package**.

This is **not** a journal-polish pass.  
This is a **submission-hardening pass**.

The target state is:

- the paper source compiles cleanly from scratch,
- every numerical/table/figure claim in the paper is produced by the repo,
- the repo implements the **same model the paper describes**,
- the paper and repo no longer drift apart,
- the novelty is stated sharply but modestly,
- the whole package looks disciplined and reproducible.

---

# Part I — Core principle

## 1. One truth only
Right now the biggest risk is divergence between:
- the **paper’s claimed model**, and
- the **repo’s implemented model**.

You must enforce a single source of truth.

### Non-negotiable rule
The **paper must describe exactly the model currently implemented in the repo**, or the **repo must be upgraded until it matches the paper**.

No mixed state.  
No “paper ahead of code.”  
No “code lags behind theory.”  
One truth only.

### Decision
Unless impossible within time, choose this path:

> **Bring the repo up to the paper**, not the other way around.

Reason:
- the paper is now stronger,
- the bidirectional coupling is the main novelty gain,
- downgrading the paper back to one-way coupling would weaken the submission too much.

---

# Part II — Article submission checklist

## 2. Fix LaTeX source compileability
This is immediate.

### Tasks
- Remove all Unicode tree characters from the source:
  - `├`
  - `│`
  - `└`
- Replace them with plain ASCII in the package layout block.
- Compile from the source zip, not from a local already-working PDF.
- Run LaTeX enough times to clear refs/citations.

### Acceptance test
`pdflatex main.tex` runs without fatal errors.

---

## 3. Audit theorem statements line by line
The theorem section must be submission-clean.

### Proposition 8.1
Tasks:
- keep the full argument list in the positivity proof:
  \[
  F(\sigma;w^n,\pi^n,x^n)
  \]
  not a truncated version missing \(x^n\).
- make sure the proposition is stated for the **actual coupled update**, not the old one-way variant.

### Proposition 8.3
This is the dangerous one.

Tasks:
- re-read every line of the proof,
- make the “local” nature absolutely unmistakable,
- ensure the contraction region is correctly stated,
- avoid any impression of a stronger global statement than what is proved.

If the proof is even slightly shaky, weaken the wording rather than bluff.

### Proposition 8.5
Tasks:
- make sure it is presented as a **structural proposition**, not as a deep theorem;
- ensure the statement about operator equality vs trajectory equality is exact.

### Acceptance test
A mathematically picky reader should not be able to say:
- “this proposition is stronger than the proof,”
- or “this proof silently changed models.”

---

## 4. Tighten terminology around curvature
This is essential for honesty.

### Tasks
Search the whole paper for anything that could imply:
- you defined canonical simplicial Forman-Ricci flow,
- or that the triangle curvature is standard Ricci curvature.

Keep the following distinction explicit:
- **edge curvature**: recognizable weighted Forman-style diagnostic expression,
- **triangle curvature**: custom stability-oriented, field-coupled, Forman-inspired functional.

### Required wording
Some version of:
> The triangle curvature used in the flow is not canonical Forman-Ricci curvature. It is a custom bounded functional designed for stability and field feedback.

Keep that in the paper.

---

## 5. Make the coupling description exact
Now that the paper claims bidirectional coupling, it must explain exactly where the feedback enters.

### Tasks
In Section 7:
- explicitly say the coupling is mediated by the field-energy term in triangle curvature;
- explicitly state that the field influences geometry through \(E(t;x)\);
- explicitly distinguish:
  - one-way version: \(\mu=0\)
  - bidirectional version: \(\mu>0\)

### Add one sentence
Something like:
> The present paper studies a minimal bidirectional coupling: the field affects geometry through the curvature functional, but the geometry update is still local and discrete rather than variational or fully dynamical in the continuum sense.

That keeps the claim honest.

---

## 6. Expand the introduction slightly
Yes, the intro needs one more pass.

### Add paragraph A
Why fixed geometry is not enough:
- existing models propagate on a given simplicial substrate,
- here the substrate changes while propagation occurs,
- so the true dynamical object is a sequence of operators induced by evolving geometry.

### Add paragraph B
The social punchline:
- the relevant propagation carrier need not be an individual,
- it may be a bonded micro-group,
- and the group’s local geometry/cohesion can change under the flow.

### Add paragraph C
Scope statement:
- this is a first mathematical-computational note,
- not a complete theory of simplicial geometric flows,
- not broad empirical validation,
- but a minimal coupled mechanism with proofs + computation.

---

## 7. Make the experiment section explicitly mechanism-driven
The experiment is good now, but it should be framed correctly.

### Tasks
At the start of Section 10, add one sentence:
> The purpose of the experiment is to demonstrate mechanism, not to claim broad empirical validation.

### Then ensure each subsection has one clear role:
- setup,
- structural verification,
- signal redistribution,
- spectral shift,
- edge curvature redistribution,
- parameter sweeps,
- seed comparison.

---

## 8. Tighten prose around result claims
Every major result sentence should map to an actual table/figure.

### Tasks
Check every prose claim like:
- “field-dependent weight reversal”
- “spectral stiffening”
- “curvature sign flip”
- “seed-dependent geometry”

For each:
- identify the exact table/figure supporting it,
- cite that table/figure explicitly in the prose.

No floating claims.

---

## 9. Add one sentence on limitations
This helps a lot on arXiv.

### Add to discussion
Something like:
> The present model is intentionally minimal: only triangle weights evolve, the geometric flow is discrete, and the empirical study is synthetic. These restrictions are deliberate and isolate the core mechanism before broader generalization.

That is the right tone.

---

# Part III — Repo submission checklist

## 10. Upgrade repo model to match the paper
This is the biggest repo task.

### Required upgrades
Implement the actual field-coupled triangle curvature:
\[
F(t;w,\pi,x)=\frac{w(t)}{\bar w_e(t)}-1+\lambda\,\mathrm{cohesion}(t)-\mu E(t;x)
\]

### Tasks
- add field argument `x` to triangle curvature,
- add `mu` parameter,
- compute normalized field-energy term \(E(t;x)\),
- pass field state into `geometry_step`,
- pass field state into `coupled_step`,
- ensure `iterate_coupled` uses the real field-coupled update.

### Important
Do not keep the current one-way toy implementation and merely document the stronger paper model. The code must catch up.

---

## 11. Implement the exact experiments the paper reports
Right now the paper reports more than the repo currently generates.

### Required outputs
Repo must generate:
- fixed vs coupled trajectory table,
- triangle weight trajectories,
- spectrum shift table,
- edge curvature table,
- \(\eta\)-sensitivity table,
- \(\lambda\)-sensitivity table,
- \(\mu\)-sensitivity table,
- seed-comparison table.

### Acceptance test
Every table in the paper must be regenerated by one command.

---

## 12. Clean output artifacts
Right now stale files are deadly.

### Tasks
- delete old/stale outputs before regeneration,
- normalize filenames,
- ensure output directory contains only artifacts used by the current paper,
- keep one stable output convention.

### Recommended structure
```text
out/
  fig1_fixed_vs_coupled.png
  fig2_triangle_weights.png
  fig3_hodge_spectrum.png
  fig4_edge_curvature.png
  table1_trajectory.csv
  table2_spectrum.csv
  table3_edge_curvature.csv
  table4_eta_sensitivity.csv
  table5_lambda_sensitivity.csv
  table6_mu_sensitivity.csv
  table7_seed_comparison.csv
  verification.txt
```

---

## 13. Fix CSV hygiene
Current CSVs must be machine-clean.

### Tasks
- avoid tuple names as raw comma-containing headers,
- rename columns like:
  - `w_abc`
  - `w_cde`
  - `w_cef`
- ensure CSVs parse cleanly in pandas with no spurious split columns.

### Acceptance test
`pd.read_csv(...)` works without broken headers.

---

## 14. Make README submission-grade
README must become a reproducibility document.

### Required sections
1. what the package does
2. relation to public SHE
3. installation
4. exact reproduction command
5. expected outputs
6. test command

### Required commands
Something like:
```bash
pip install -e .
pytest -q
python -m she_geofield.experiment
```

### Add one sentence
The repo is a **focused companion extension**, not a replacement for SHE.

---

## 15. Add actual SHE boundary note
You do not need full integration yet, but you must be honest.

### Tasks
- keep `adapters.py` if needed,
- explicitly document current adapter status,
- do not imply live integration if it is still a placeholder.

### Suggested wording
> The current release is a companion extension aligned with the public SHE data model; a direct adapter layer is planned but not yet part of this submission package.

---

## 16. Strengthen tests slightly
Current tests are okay, but not enough for submission confidence.

### Add tests for:
- field-coupled curvature returns finite values for nontrivial field states,
- `mu=0` reproduces one-way behavior,
- positivity under field-coupled update,
- seed-comparison pipeline runs,
- output generation creates all expected files.

These are not huge, but they matter.

---

# Part IV — Paper/repo synchronization

## 17. Create a submission lockstep process
Claude must ensure the paper is generated from repo outputs, not manual transcription.

### Tasks
- regenerate outputs from code,
- update tables/figures in the paper from those outputs,
- then freeze both together.

### Hard rule
No hand-edited numbers in LaTeX after the final experiment run.

---

## 18. Add a paper–repo consistency checklist
Claude should create a short internal checklist file, e.g.:
`SUBMISSION_CHECKLIST.md`

It should verify:
- paper compile passes,
- tests pass,
- experiment runs,
- all table numbers match outputs,
- all figure files exist,
- repo version/date matches paper version/date.

---

# Part V — Final paper polish for arXiv

## 19. Title / abstract check
Current title is fine.

Abstract tasks:
- ensure it does not overclaim geometric sophistication,
- make sure “bidirectional coupling” is justified by implemented code,
- keep the three exhibited phenomena only if repo outputs support them.

---

## 20. Reference sanity pass
Check:
- all refs used are in bibliography,
- no stale citations remain,
- formatting consistent,
- no obvious missing foundational citation for discrete curvature/flow claims.

---

## 21. Final tone pass
Claude must remove any remaining sentence that sounds more ambitious than the paper is.

### The paper should sound like:
- a sharp first note,
- with a real mechanism,
- small theorems,
- real computation,
- honest limits.

### It should not sound like:
- a grand geometric theory,
- a completed social-science validation,
- or a universal higher-order propagation framework.

---

# Part VI — Deliverables Claude should produce

## 22. Final deliverables
Claude should return:

1. **revised LaTeX source** that compiles cleanly,
2. **updated repo** matching the paper’s bidirectional model,
3. **clean output directory** with exactly the paper artifacts,
4. **submission-grade README**,
5. **submission checklist**,
6. **short changelog** summarizing what was synchronized.

---

# Part VII — Acceptance criteria

The submission-prep pass is successful only if:

- the paper source compiles from scratch,
- the repo implements the same model the paper describes,
- all claimed experiments are regenerated by code,
- the theorem section is honest and tight,
- the social/group-level punchline is visible but not exaggerated,
- the paper and repo feel like one object.

If any of those fail, it is not ready.

---

# Final directive to Claude

Be ruthless and literal.

Do **not** preserve mismatches because they are “close enough.”  
Do **not** keep claims that the current repo does not support.  
Do **not** let the paper outrun the code again.  
Do **not** let the code remain a weaker ancestor of the paper.  
Do **not** make this prettier before making it coherent.

The goal is simple:

> **one clean arXiv paper + one clean companion repo + zero ambiguity about what is being claimed and reproduced.**
