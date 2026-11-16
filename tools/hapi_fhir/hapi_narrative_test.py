"""HAPI R4 test of bundles that carry LLM narratives, rebuilt from the evaluation cache.

For each llm_xai/results/eval_raw file with a narrative, the bundle is rebuilt as in production
(`build_bundle_with_narrative`: narrative in DiagnosticReport.conclusion plus an audit tag), then
validated in process, posted to /Bundle/$validate, posted as a transaction and read back. No heatmap
is attached; bundles with heatmaps are covered by hapi_fullset_test.py.

    python tools/hapi_fhir/hapi_narrative_test.py --base http://127.0.0.1:8090/fhir
"""
import argparse, json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hapi_roundtrip_test import http, ROOT  # noqa: E402
sys.path.insert(0, str(ROOT))
from backend.fhir_generator import validate_fhir_bundle  # noqa: E402
from llm_xai.backend.fhir_wrap import build_bundle_with_narrative  # noqa: E402

E_RAW = ROOT / "llm_xai/results/eval_raw"; GOLD = ROOT / "llm_xai/results/gold_facts"


def build(f):
    """Rebuild the bundle of one eval_raw file from its narrative and the session's gold facts."""
    d = json.loads(f.read_text()); sid, m, v = f.stem.split("__")
    gf = json.loads((GOLD / f"{sid}.json").read_text()); cc = gf["clinical_covariates"]; pa = gf["prediction_anchor"]; dx = pa["argmax_class"]
    b = build_bundle_with_narrative(
        narrative=d["narrative"],
        provenance={"fact_digest": gf["fact_digest"], "guardrail_status": "PASS" if d.get("guardrail", {}).get("passed") else "FAIL",
                    "narrative_source": d.get("source", "n/a"), "model": m},
        ai_result={"diagnosis": dx, "confidence": pa["relative_probabilities"][dx], "probabilities": pa["relative_probabilities"],
                   "model_info": {"name": m}, "latency_ms": (d.get("wall_s") or 0) * 1000},
        session_id=sid,
        clinical={"age": cc.get("age"), "gender": cc.get("gender"), "mmse": cc.get("mmse_score") if cc.get("mmse_available") else None,
                  "cdr": cc.get("cdr_score"), "apoe": cc.get("apoe4_allele") if cc.get("apoe_available") else None})
    return b, m, v


def main():
    """Check every narrative-bearing result against the server and write the report."""
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="http://127.0.0.1:8090/fhir"); a = ap.parse_args()
    base = a.base.rstrip("/")
    st, meta = http("GET", f"{base}/metadata"); assert st == 200
    # Per (model, variant) pass counts
    tot = collections.Counter(); ok = collections.Counter(); fails = []; skipped = 0; n_res = 0
    for f in sorted(E_RAW.glob("*.json")):
        if json.loads(f.read_text()).get("narrative") is None:
            skipped += 1; continue
        b, m, v = build(f); k = f"{m}/{v}"; tot[k] += 1
        inproc = validate_fhir_bundle(b)["valid"]
        s1, out = http("POST", f"{base}/Bundle/$validate", b)
        errs = [i for i in out.get("issue", []) if i.get("severity") in ("error", "fatal")]
        s2, resp = http("POST", base, b); ents = resp.get("entry", [])
        all201 = len(ents) == 6 and all(str(e.get("response", {}).get("status", "")).startswith("201") for e in ents)
        rb = [http("GET", f"{base}/{e['response']['location'].split('/_history')[0]}")[0] for e in ents if e.get("response", {}).get("location")]
        good = inproc and s1 == 200 and not errs and s2 == 200 and all201 and len(rb) == 6 and all(x == 200 for x in rb)
        if good: ok[k] += 1; n_res += 6
        else: fails.append({"file": f.name, "inproc": inproc, "validate": s1, "errs": [e.get("diagnostics") for e in errs][:2], "tx": s2})
    rep = {"server": meta.get("software"), "fhirVersion": meta.get("fhirVersion"), "skipped_no_narrative": skipped,
           "per_combo": {k: f"{ok[k]}/{tot[k]}" for k in sorted(tot)}, "n": sum(tot.values()), "n_pass": sum(ok.values()),
           "resources_created": n_res, "failures": fails[:20]}
    (Path(__file__).parent / "hapi_narrative_report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
