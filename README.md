# Deterministic guardrails for LLM narratives in a FHIR-based Alzheimer's neuroimaging CDSS

Code for a research prototype of a clinical decision support system (CDSS) around a 2.5D MobileViT classifier of
structural brain MRI (CN / MCI / AD). A local large language model (LLM) turns the frozen explainable-AI (XAI)
output into a short narrative and answers follow-up questions, a **deterministic guardrail** (no second generative
model as verifier) checks every sentence against the underlying facts, and the audited narrative is translated
into HL7 FHIR R4 and shown in a web interface.

> **This is a research prototype, not a medical device.** Nothing here is validated for clinical use.
> The evidence supports audit-level factuality and schema-level interoperability, not clinical validity.

## What is in this repository, and what is not

| Included | Not included |
|---|---|
| Backend (FastAPI): inference, Grad-CAM / HiResCAM / Integrated Gradients, out-of-domain check, FHIR R4 generator and validator | OASIS-3 and ADNI images, clinical covariates, and their derived slice caches |
| `llm_xai/`: guardrails, scoped Q&A, prompts, evaluation and audit tools, summary result tables | Per-narrative raw outputs of the evaluation sweeps, gold facts of the OASIS-3 test sessions |
| React CDSS frontend | Training notebooks |
| Trained checkpoint of the model served by the backend (`checkpoints/`) | The large language models (run locally through Ollama) |
| **12 synthetic demonstration sessions** (`dummy_data/`) so that anyone can run the system end to end | FastSurfer (cloned separately, only needed to rebuild the demo data) |
| A screen recording of the system running on the synthetic sessions (`demo_video/`) | |

OASIS-3 and ADNI are governed by data use agreements and are not redistributed. The synthetic sessions are
simulated normal brains from BrainWeb with made-up covariates. **They are not patient data, carry no clinical
meaning, and are not part of the experiments in the paper.** See [`dummy_data/README.md`](dummy_data/README.md)
and [`NOTICE`](NOTICE).

## Quick start

Python 3.10 is what the code was developed on.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-runtime.txt        # backend and LLM layer
# requirements-lock.txt is the full pinned research environment; requirements.txt is its unpinned form
```

The first start downloads the initial MobileViT weights (`apple/mobilevit-small`) from Hugging Face, and the first
use of the guardrails or the Q&A engine downloads two sentence-transformers models (an embedding model and an NLI
cross-encoder), so it needs internet access once.

### Backend

```bash
.venv/bin/uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/analyze/BWP01_MR_d0000       # prediction, probabilities, heatmaps, FHIR bundle
```

`/api/health` reports a missing checkpoint or a dimension mismatch. `BASE_DIR` overrides the repository root.

### Frontend

```bash
cd alzheimer-xai-cdss
npm install
npx vite                 # http://localhost:5173, API at http://localhost:8000
```

The sessions appear under the cohort filter **Synthetic**. Use `npx vite`, not `npm run dev`: the `predev` hook of
`npm run dev` rebuilds `src/data.json` from whatever data folders exist on your machine. **Run AI Analysis** also
requests the LLM narrative, so start the LLM layer below first (the analysis itself needs only the backend).

### LLM layer (optional, local only)

```bash
OLLAMA_NUM_PARALLEL=1 ollama serve
ollama pull llama3.1:8b-instruct-q4_K_M       # default model (variant C, guarded generation)
```

Nothing is sent to a cloud API; entries of kind `openai` / `google` in `llm_xai/configs/llm_models.yaml` are
rejected unless `--allow-cloud` is passed and `llm_xai/configs/ethics_clearance.json` exists. Comparison models
(`biomistral`, `mistral7b`, `qwen25_14b`) are listed in the same file.

### Docker

```bash
docker compose up --build                      # backend :8000 and frontend :5173
docker compose --profile fhir up -d hapi       # local HAPI FHIR R4 server on :8080
```

## Layout

```text
backend/                 app.py, inference_engine.py, fhir_generator.py, test_fhir_cohort_labels.py
alzheimer-xai-cdss/      React + Vite frontend (src/App.jsx is the main application)
llm_xai/
  backend/               pipeline, guardrails, Q&A engine, prompts, tests
  configs/               model roster, Ollama Modelfile
  tools/                 evaluation, statistics, benchmark building
  audit/                 blind human-audit web tool (its case manifest is built from the omitted sweep outputs)
  results/               summary tables (T1-T7, McNemar), edge latency, AGA structure summary
checkpoints/             best_model_v2_unified.pt, training_metadata_v2_unified.json, rep_train_640d.npy
dummy_data/              12 synthetic sessions (4 dummy patients x 3 sessions) and how they were built
tools/hapi_fhir/         FHIR tests against a local HAPI server
tools/make_brainweb_demo.py   builds dummy_data/ from BrainWeb
demo_video/              screen recording of the CDSS on the synthetic sessions
```

## Tests

```bash
python -I backend/test_fhir_cohort_labels.py                                   # FHIR labels per cohort
python -m llm_xai.backend.test_guardrails                                      # guardrail rules
python -m llm_xai.backend.test_qa_engine                                       # Q&A rules (no pytest needed)
```

With a HAPI server running (`docker compose --profile fhir up -d hapi`):

```bash
python tools/hapi_fhir/hapi_roundtrip_test.py --base http://127.0.0.1:8080/fhir
```

`hapi_fullset_test.py` and `hapi_narrative_test.py` reproduce the validation of all 232 test sessions of the paper
and need OASIS-3-derived inputs that are not distributed; their reports from that run are kept next to them.

## FHIR R4 export

`backend/fhir_generator.py` builds a transaction bundle of six resources (`Patient`, `Device`, `ImagingStudy`,
`Observation`, `DocumentReference`, `DiagnosticReport`), validated with `fhir.resources`. Identifiers of the
synthetic sessions use `urn:alzai:synthetic:*` systems; the model provenance in `Device` points to this repository.

## Known limits

- The model is validated on OASIS-3 only. On the external ADNI cohort it degrades substantially, and the CDSS
  flags such scans as out of domain. All 12 synthetic sessions are staged CN, as expected for simulated normal
  brains; one of them is flagged out of domain.
- Probabilities are relative rankings, not calibrated confidences.
- The guardrail is deterministic and rule-based: it catches the violation classes it encodes (numbers, anatomy
  outside the whitelist, self-diagnosis, scope) and nothing beyond them. Local LLM decoding is seeded but not
  guaranteed to be bit-for-bit deterministic.
- The API has no authentication, CORS is open, and Q&A sessions are kept in memory without a limit. It is meant to run
  on localhost for research, not to be exposed on a network.
- The statements above about the paper's experiments refer to the OASIS-3 data, which this repository cannot
  reproduce without the data.

## License, notices, and citation

- Code and model weights: [PolyForm Noncommercial License 1.0.0](LICENSE). Noncommercial use is permitted;
  for any other use, including commercial use and redistribution of modified versions beyond what the license
  allows, ask first: nathanwidjaja10.nw@gmail.com. Keep the copyright notice in `LICENSE` and `NOTICE`.
- Third-party material (OASIS-3 acknowledgment, BrainWeb images and citations, FastSurfer, MobileViT initial
  weights): [`NOTICE`](NOTICE). The BrainWeb-derived images are not the authors' work and are not covered by
  the license above.
- Citation metadata: [`CITATION.cff`](CITATION.cff).
