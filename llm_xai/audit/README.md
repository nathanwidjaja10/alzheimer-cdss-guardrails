# Blind human audit of llama31_8b variant C narratives

The automatic guardrail measures hallucination rate (HR) and numeric consistency rate (NCR) against
facts it also verifies, so those metrics are circular. This tool adds an independent validation layer:
a human rates each narrative without seeing the guardrail verdict until after submission.

It runs locally with the Python standard library only and has no effect on the backend; the `audit/`
folder can be deleted without side effects.

## Coverage

All 232 test sessions of `llama31_8b` variant C (a census, not a sample). Narratives are read from
`results/eval_raw/*__llama31_8b__c.json` and facts from `results/gold_facts/`; nothing is duplicated.
Cases are ordered round-robin over strata (`argmax_class` x `ood_status`) with a fixed seed, so any
prefix of the order stays representative across strata.

The repository ships the 232 `eval_raw/*__llama31_8b__c.json` files read by the UI, `results/gold_facts/`
and the prebuilt `cases_manifest.json`, so the server can be started directly and the `build_manifest.py`
step below is optional. Other raw evaluation outputs are not shipped.

## Usage

```bash
# Once at the start, or when eval_raw changes
python3 audit/build_manifest.py

# Start the server and keep it running during the audit session
python3 audit/server.py
# open http://localhost:8420/
```

Enter an auditor name or initials and click **Mulai / Lanjutkan**; the UI continues with the first case
that auditor has not rated. Progress is recomputed from `audit/results/audit_log.jsonl` on every request,
so the audit can be stopped and resumed at any time and no session state can be lost.

Progress without a browser:

```bash
python3 audit/summarize_audit.py --progress-only
```

Full summary (human HR with Wilson interval, category breakdown, inter-auditor kappa when auditors
overlap) and the table `results/T7_human_audit_summary.csv`:

```bash
python3 audit/summarize_audit.py
```

## Design notes

- **Blinding.** `guardrail.passed`, `ncr` and `violation_types` are sent to the browser but hidden in the
  UI until the verdict is submitted, so the rating is not influenced by the automatic verdict.
- **Highlighting.** Numbers and anatomy terms in the narrative are highlighted only as a visual aid. It is
  not a verdict and is incomplete (terms outside the whitelist are not highlighted); the auditor must
  read the full text.
- **Revisions.** Resubmitting a session overwrites the previous verdict. Earlier JSONL rows are kept, and
  `summarize_audit.py` always uses the last row per `(auditor_id, session_id)`.
- **Second auditor (optional).** Build the manifest once (it is shared) and let the second auditor enter a
  different name. The overlapping sessions are used to compute kappa.
- **Language.** The audit UI text is in Indonesian, the language of the auditors.
