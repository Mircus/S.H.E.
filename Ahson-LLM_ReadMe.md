# SHE × LLMs — Higher‑Order Concept Geometry (Ahson Vertical)

A practical README to study LLM representation geometry using **simplicial hyperstructures** (SHE). Focus: measurable wins over pairwise (k‑NN) graphs via Hodge operators, diffusion, and persistent homology.

---

## Table of Contents
- [Overview](#overview)
- [Scenarios](#scenarios)
  - [A) Polysemy & Sense Binding](#a-polysemy--sense-binding-word-in-context)
  - [B) Compositionality Curvature](#b-compositionality-curvature-phrases-vs-parts)
  - [C) RAG Evidence Coverage & Hallucination Risk](#c-rag-evidence-coverage--hallucination-risk)
  - [D) Model Family Comparison via Topological Signatures](#d-model-family-comparison-via-topological-signatures)
  - [E) Circuit‑Level Motifs (Heads × Patterns × Features)](#e-circuit-level-motifs-heads--patterns--features)
  - [F) Knowledge Editing & Local Geometry](#f-knowledge-editing--local-geometry)
  - [G) Prompt‑Injection Propagation (Tool Use)](#g-prompt-injection-propagation-tool-use)
- [Implementation Pattern](#implementation-pattern)
- [Fast Pilot (6–8 weeks)](#fast-pilot-68-weeks)
- [Notes on IP / Publication](#notes-on-ip--publication)

---

## Overview
**Why SHE for LLMs?** Many representational phenomena are *higher‑order*: they involve 3+ entities (token‑in‑context, phrase, operation, behavior). SHE provides operators (Hodge Laplacians \(L_k\)) and topology (persistent homology) that reveal cycles, cavities, and motif flows—signals **invisible** to edge‑only graphs.

**Core ingredients**
- Build **simplicial complexes** (Vietoris–Rips / witness / alpha) on embeddings with **semantic gating** (entailment, MI) to avoid noisy simplices.
- Compute Hodge \(L_k\) for \(k=1,2\), diffusion/heat kernels, and decompose energies (gradient / harmonic / solenoidal).
- Run persistent homology on filtrations (scale, score, time) to capture robust components and cycles.

Deliverables per scenario: dataset → complex → SHE features → baseline comparison → ablations → interpretability plots.

---

## Scenarios

### A) Polysemy & Sense Binding (Word‑in‑Context)
**Goal:** detect and quantify how well an LLM separates senses of a word across contexts.

**Data:** WiC, SCWS, SemCor, or a curated sentence bank.

**Complex:** nodes = contextual instances; k‑simplices if pairwise dist < \(\varepsilon\) **and** contexts share topic cues (VR/witness + semantic gate). Optional anchor nodes (topics) to form \{instance, anchor_j, anchor_k\}.

**SHE signals:** \(\beta_0\) vs #senses; **persistent 1‑cycles** = bridge contexts; **Hodge currents on \(L_1\)** to locate contexts pulling between senses.

**Tasks/metrics:** sense classification (F1); **ambiguity index** = total persistence of \(H_1\) cycles; correlate with SCWS human ratings.

**Ablations:** k‑NN graph vs simplicial PH; show cycles missed by clustering.

---

### B) Compositionality Curvature (Phrases vs Parts)
**Goal:** quantify non‑linear phrase meaning relative to parts; predict failure cases.

**Data:** synthetic templates (e.g., negation/scope), COGS‑style splits, prompt library.

**Complex:** triangles \{u, v, p\} for word1, word2, phrase; add paraphrase variants \{u', v', p'\}. Include a composition map \(f(u,v)\) to create \{p, f(u,v), anchor\} triangles.

**SHE signals:** **compositional curvature** \(\kappa = \|e(p) - f(e(u),e(v))\|\); **Hodge 1‑form energy** flags non‑associative/commutative regimes; persistent 2‑cycles indicate systematic composition gaps.

**Tasks/metrics:** predict correctness on compositional probes (AUROC); rank constructions by non‑compositionality.

---

### C) RAG Evidence Coverage & Hallucination Risk
**Goal:** detect when an answer lacks topological support in retrieved evidence.

**Data:** BEIR/NQ + RAG traces; TruthfulQA/FActScore labels (or manual).

**Complex:** nodes = query, retrieved chunks, answer sentences, entities, citations. Triangles \{a_j, d_i, e_k\} if chunk entails the claim; \{q, d_i, a_j\} link query→evidence→answer. Filtration by retrieval/entailment score.

**SHE signals:** **coverage** (fraction of answer simplices supported); **unsupported‑claim cavities** (holes around \(a_j\)); **diffuser centrality** = most supportive chunks.

**Tasks/metrics:** hallucination detection (AUROC/PR@low FPR); calibrate abstention with topo risk.

---

### D) Model Family Comparison via Topological Signatures
**Goal:** quantify representational deltas across models/versions for a fixed concept bank.

**Data:** same prompts across model families (e.g., Llama‑3, GPT‑X, Mistral).

**Complex:** per model, build complexes for concept sets (animals/tools/professions); compute persistence diagrams \(D^{(m)}\) for \(H_0,H_1,H_2\).

**SHE signals:** **Topological Signature Distance (TSD)** = bottleneck/Wasserstein distance between diagrams; identify which cycles/components shift and link to benchmark gaps.

**Tasks/metrics:** correlate TSD with behavioral deltas (accuracy, calibration, bias indices).

---

### E) Circuit‑Level Motifs (Heads × Patterns × Features)
**Goal:** identify higher‑order motifs spanning attention heads, token patterns, and SAE features.

**Data:** attention activations, SAE features, logit‑lens traces on curated prompts.

**Complex:** nodes = (head, layer), pattern prototypes, SAE features; 3‑simplices \{head_i, pattern_p, feature_f, behavior_b\} if jointly predictive of behavior (e.g., induction, copying). Filtration by joint MI or coefficient.

**SHE signals:** **motif diffuser centrality** to rank tetrahedra; harmonic components on \(L_2\) = robust subcircuits; exact parts = spurious couplings.

**Tasks/metrics:** predict behavior toggling under ablation/steering; rank minimal motifs.

---

### F) Knowledge Editing & Local Geometry
**Goal:** test whether edits (ROME/MEMIT) are local vs global in concept space.

**Data:** pre/post‑edit embeddings for entities/relations and neighborhoods.

**Complex:** build pre/post complexes for \(\mathcal{N}(e)\); compute PH + Hodge energies.

**SHE signals:** **ΔPH** localized near edited relations ⇒ surgical; large global ΔPH ⇒ collateral drift.

**Tasks/metrics:** use ΔPH to gate/accept edits; report locality scores.

---

### G) Prompt‑Injection Propagation (Tool Use)
**Goal:** map injection flow from user text to tool arguments and identify chokepoints.

**Data:** tool‑augmented traces with attribution (attention/IG).

**Complex:** nodes = policy spans, user spans, tool schemas, arg spans; 3‑simplices when user span controls arg span contrary to policy.

**SHE signals:** **flow on \(L_1\)** from user→args; remove top‑k risky motifs and test collapse.

**Tasks/metrics:** attack success vs defense presence; motif‑targeted mitigations.

---

## Implementation Pattern
1) **Embeddings:** choose layer(s), normalize; optional whitening.
2) **Complex:** VR/witness/alpha with **semantic gating** (NLI/MI thresholds) for simplex acceptance.
3) **Operators:** compute \(L_k\) for k=1,2; cache heat kernels; decompose energies.
4) **Topology:** persistent homology over filtrations (scale/score/time); compute diagrams & distances.
5) **Features:** cycle counts/persistence, harmonic/gradient energies, diffuser centralities, coverage.
6) **Models:** simple classifiers/regressors on SHE features; compare to graph‑only baselines.
7) **Ablations:** remove k>1 simplices; shuffle anchors; vary filtrations; sensitivity curves.

---

## Fast Pilot (6–8 weeks)
- **W1:** pick A + C; lock datasets/labels.  
- **W2:** build complexes; compute \(L_k\), PH; derive features.  
- **W3:** train detectors/predictors; compare vs k‑NN/graph baselines.  
- **W4:** ablations + error analysis; produce motif/diagram visualizations.  
- **W5–6:** extend to D (model comparison); compute **TSD** and correlate with benchmark deltas.  
- **W7:** draft paper + demo.  
- **W8:** finalize figures; arXiv or workshop (NeurIPS D&B / ICLR Safety / ICML Topology).

---

## Notes on IP / Publication
- **Publishable:** tasks, complexes, metrics, ablations, and results.  
- **Proprietary:** optimized engine implementations and performance tricks.  
- **Credit:** Ahson as vertical lead/co‑author for chosen scenarios.
