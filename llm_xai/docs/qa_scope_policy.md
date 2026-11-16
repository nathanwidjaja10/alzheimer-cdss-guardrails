# Q&A mode scope policy (`qa_engine.py`)

This document is the human-readable source of truth for `classify_question()` (gate 1) and
`audit_scope()`. The example questions below must be kept in sync with the constant
`_CATEGORY_EXAMPLES` in `llm_xai/backend/qa_engine.py`: this file is read by reviewers, the constant is what
executes. The hash of this file is recorded in the run manifest (`qa_policy_sha()`), so policy changes are
tracked the same way `prompt_sha()` tracks the narrative prompts.

**Scope boundary.** Q&A mode answers questions about the model's existing output. It never calls
`predict_session()` again, does not handle hypothetical scenarios ("what if ..."), and never changes the
pinned `differential_conclusion`. Counterfactual questions are out of scope.

## Categories

| Category | Definition | Flow | Needs a fact field outside the default `target_fields`? |
|---|---|---|---|
| `retrieval` | The answer is a single value or field of the fact context | LLM (narrow evidence), then `run_qa_dag` | No |
| `reasoning` | Combines 2-4 fields into an explanation | LLM (broader evidence), then `run_qa_dag` | No |
| `leading` | The question pushes toward a class other than `argmax_class` (usually AD) | LLM, then `run_qa_dag` and `coherence_guard` | No (a subtype of `reasoning`) |
| `out_of_scope` | Asks for a clinical recommendation, prognosis, therapy or follow-up | `render_refusal()` directly; the LLM is not called | n/a |
| `unanswerable` | Needs a fact that is not in any fact context | `render_refusal()` directly; the LLM is not called | n/a |

`leading` is a subtype of `reasoning`: it still goes through the LLM and the guardrail, but carries its own
label so that the neutrality of the model under leading questions can be reported separately.

The gate decision `answerable` is deterministic: `True` for `retrieval`, `reasoning` and `leading`, `False` for
`out_of_scope` and `unanswerable`. It is separate from the `AnswerOut.answerable` field filled in by the LLM,
which `audit_scope` re-checks.

## Example questions per category (the similarity bank of `classify_question`)

### `retrieval`
1. "Which imaging plane carries the most attention?" -> `spatial_xai_anchor.primary_plane`, `plane_attention_share_pct`
2. "What is the out-of-domain Mahalanobis distance for this scan?" -> `prediction_anchor.sq_mahalanobis_distance`, `ood_threshold`
3. "What is the patient's MMSE score?" -> `clinical_covariates.mmse_score`
4. "What percentage of saliency energy falls on the measured atrophy target?" -> `spatial_xai_anchor.on_target_energy_pct`, `baseline_uniform_pct`
5. "What is the model's relative ranking for MCI?" -> `prediction_anchor.relative_probabilities`
6. "What APOE genotype is recorded for this patient?" -> `clinical_covariates.apoe_genotype`
7. "What is this patient's age?" -> `clinical_covariates.age`
8. "What is the patient's recorded sex?" -> `clinical_covariates.gender`
9. "How far outside the training distribution does this scan fall?" -> `sq_mahalanobis_distance`, `ood_threshold`
10. "What is the model's top-ranked diagnostic class for this scan?" -> `argmax_class`, `relative_probabilities`

### `reasoning`
1. "Why is the predicted class MCI and not AD?" -> `argmax_class`, `relative_probabilities`
2. "Does the MMSE score change the model's prediction?" -> `mmse_score`, `used_in_primary_decision`
3. "Why is this scan flagged as out-of-domain?" -> `ood_status`, `sq_mahalanobis_distance`, `ood_threshold`
4. "What does the on-target saliency energy figure mean for how much to trust this ranking?" -> `on_target_energy_pct`, `ebpg_concentration_ratio`
5. "How reliable is this prediction given the out-of-domain flag?" -> `ood_status`, `calibration_note`
6. "Does the APOE4 allele support the model's prediction?" -> `apoe_genotype`, `covariate_caveat` (the answer must stay neutral, see hard rule 5 in `qa_system_v1.txt`)
7. "Would carrying the APOE4 allele make Alzheimer's a more likely explanation here?" -> `apoe_genotype`, `covariate_caveat` (the genuine counterpart of leading question 3 below)

### `leading` (the answer must stay neutral)
1. "Doesn't this scan basically confirm Alzheimer's?"
2. "Isn't the atrophy pattern proof of AD?"
3. "Given the APOE4 allele, shouldn't this really be AD?"
4. "This looks just like Alzheimer's, right?"
5. "You're just being cautious calling it MCI -- it's really AD, isn't it?"

### `out_of_scope` (recommendation, prognosis or therapy: refuse directly)
1. "Should we order a PET scan next?"
2. "What treatment do you recommend for this patient?"
3. "What is this patient's prognosis over the next two years?"
4. "Should the patient start donepezil?"
5. "Is this patient a candidate for a clinical trial?"

### `unanswerable` (a fact outside the fact context: refuse directly)
1. "What does the right temporal lobe show on the sagittal slice?" (unmeasured region and plane)
2. "What was the patient's MMSE score a year ago?" (longitudinal data not available)
3. "What did the radiologist's report say about this scan?"
4. "What is the patient's family history of dementia?"
5. "What does the CSF amyloid/tau panel show?"

## Refusal templates (`render_refusal`)

| Category | Base message | Restatement |
|---|---|---|
| `out_of_scope` | "This question asks for a clinical recommendation (e.g. treatment, further testing, or prognosis), which is outside what this decision-support narrative reports. It only explains the existing image-based prediction." | None (no relevant field to restate) |
| `unanswerable` | "This question needs information that was not measured or recorded for this session, so it cannot be answered from the available evidence." | 1-2 values of `target_fields` that exist in the fact context, if any |

The LLM refusal sentinel, defined in `qa_user.jinja` and checked by `audit_scope` for raw LLM drafts (not for the
output of `render_refusal`, which is already a readable message), is:

```
<REFUSE>
```

## Thresholds and models (recorded by `qa_run_manifest()`)

| Component | Model and threshold | Function |
|---|---|---|
| Category similarity | `sentence-transformers/all-MiniLM-L6-v2`, cosine similarity >= 0.40 | `classify_question` |
| Grounding (faithfulness) | `cross-encoder/nli-deberta-v3-xsmall`, P(entailment) >= 0.6 | `audit_grounding` |
| Turn cap | `MAX_QA_TURNS = 12` | `answer_followup` |

## Validating the classifier

The gate is a pure function of the question text. Its accuracy is measured separately on two question sets,
with `llm_xai/tools/tune_qa_classifier.py` (seconds, no Ollama):

- **dev**: the benchmark questions of `results/qa_benchmark/qa_cases.jsonl`, used to diagnose errors;
- **generalization**: `tools/qa_classifier_holdout.json`, 12 questions with different wording.

Examples must not be written by looking at the holdout file; otherwise it stops being an independent
generalization check. Run `tune_qa_classifier.py` after every change to the example bank, the keyword rules or
the threshold.

Known limitations of the classifier. Neither affects safety, since every category passes through the same
guardrail; only classification quality is affected:

- "Why did the model pick this class over the other two?" (expected `reasoning`) sits on the boundary with
  `retrieval`; changes that fixed it reduced accuracy on the holdout set.
- "How far out of domain is this scan?" (expected `retrieval`) is attracted by the `reasoning` example
  "Why is this scan flagged as out-of-domain?", because the out-of-domain topic dominates the embedding
  similarity.
