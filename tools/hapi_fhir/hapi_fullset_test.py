"""FHIR R4 test of every internal test session against a local HAPI server.

For each session of the internal test fold (checkpoints/xai_results_hirescam.csv, n=232):
real predict_session() (HiResCAM) -> create_fhir_bundle() -> in-process validation ->
POST /Bundle/$validate -> POST transaction -> read back all six resources. Metadata is resolved as in
backend/app.py::analyze_session.

Start the server first, then run:
    HAPI_PORT=8090 docker compose --profile fhir up -d hapi
    python tools/hapi_fhir/hapi_fullset_test.py --base http://127.0.0.1:8090/fhir
Per-session results are appended to hapi_fullset_results.jsonl, so an interrupted run can resume.
"""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hapi_roundtrip_test import http, ROOT  # noqa: E402
from fhir_generator import create_fhir_bundle, validate_fhir_bundle  # noqa: E402
from inference_engine import predict_session, MODEL_READY  # noqa: E402

OUT = Path(__file__).parent
clinical = json.load(open(ROOT / "oasis_data" / "clinical_metadata.json"))


def meta_for(sid, canon):
    """Clinical metadata record of a session (canonical id first)."""
    return (clinical.get(canon) or clinical.get(sid) or clinical.get(sid.replace("_MR_", "_")) or {})


def one(base, sid):
    """Run the full FHIR check for one session and return the result record."""
    t0 = time.time()
    ai = predict_session(session_id=sid, include_mmse=True, include_apoe=True, xai_method="hirescam")
    c = meta_for(sid, ai.get("canonical_session_id", sid))
    subj = c.get("subject_id") or sid.split("_")[0]
    bundle = create_fhir_bundle(
        patient_id=subj, age=c.get("age_at_scan", "Unknown"), gender=c.get("gender", "Unknown"),
        ai_diagnosis=ai["diagnosis"], ai_confidence=ai["confidence"], probabilities=ai["probabilities"],
        session_id=sid, slice_count=ai["slice_count"], gradcam_base64=ai["gradcam_base64"],
        model_name=ai["model_info"]["name"], checkpoint_version="2.0.0", execution_latency_ms=ai["latency_ms"],
        mmse=c.get("mmse"), cdr=c.get("cdr"), apoe=c.get("apoe"))
    # In-process validation and server-side $validate
    r = {"session": sid, "diagnosis": ai["diagnosis"], "inprocess_valid": validate_fhir_bundle(bundle)["valid"]}
    st, out = http("POST", f"{base}/Bundle/$validate", bundle)
    iss = out.get("issue", [])
    r["validate_status"] = st
    r["validate_errors"] = [i.get("diagnostics") or i.get("details", {}).get("text") for i in iss if i.get("severity") in ("error", "fatal")]
    r["validate_warnings"] = sum(1 for i in iss if i.get("severity") == "warning")
    # Transaction POST and readback of the created resources
    st, resp = http("POST", base, bundle)
    ents = resp.get("entry", [])
    locs = [e.get("response", {}).get("location") for e in ents]
    r["tx_status"] = st
    r["tx_n"] = len(ents)
    r["tx_all_201"] = bool(ents) and all(str(e.get("response", {}).get("status", "")).startswith("201") for e in ents)
    rb = [http("GET", f"{base}/{l.split('/_history')[0]}")[0] for l in locs if l]
    r["readback_ok"] = len(rb) == 6 and all(x == 200 for x in rb)
    r["pass"] = bool(r["inprocess_valid"] and st == 200 and not r["validate_errors"] and r["tx_n"] == 6
                     and r["tx_all_201"] and r["readback_ok"])
    r["seconds"] = round(time.time() - t0, 1)
    return r


def main():
    """Process the sessions not yet in the results file and write the summary report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090/fhir")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    assert MODEL_READY, "model not ready"
    st, meta = http("GET", f"{base}/metadata")
    assert st == 200, f"HAPI not reachable ({st})"
    sessions = [r["session_id"] for r in csv.DictReader(open(ROOT / "checkpoints" / "xai_results_hirescam.csv"))]
    if a.limit:
        sessions = sessions[:a.limit]
    # Resume from the results file
    res_path = OUT / "hapi_fullset_results.jsonl"
    done = {json.loads(l)["session"] for l in open(res_path)} if res_path.exists() else set()
    with open(res_path, "a") as f:
        for i, sid in enumerate(sessions, 1):
            if sid in done:
                continue
            try:
                r = one(base, sid)
            except Exception as e:  # noqa: BLE001
                r = {"session": sid, "pass": False, "exception": repr(e)}
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[{i}/{len(sessions)}] {sid} pass={r['pass']} {r.get('seconds','')}s", flush=True)
    # Summary over the requested sessions
    rows = [json.loads(l) for l in open(res_path)]
    rows = [r for r in rows if r["session"] in set(sessions)]
    summ = {"server": meta.get("software"), "fhirVersion": meta.get("fhirVersion"), "n": len(rows),
            "n_pass": sum(r["pass"] for r in rows), "n_inprocess_valid": sum(r.get("inprocess_valid", False) for r in rows),
            "n_server_validate_0err": sum(1 for r in rows if r.get("validate_status") == 200 and not r.get("validate_errors")),
            "n_tx_201x6": sum(1 for r in rows if r.get("tx_all_201") and r.get("tx_n") == 6),
            "n_readback_ok": sum(r.get("readback_ok", False) for r in rows),
            "failures": [r for r in rows if not r["pass"]][:20]}
    (OUT / "hapi_fullset_report.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
