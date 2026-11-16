"""Round-trip test of bundles from backend/fhir_generator.py against a local HAPI R4 server.

Uses synthetic values only (as backend/test_pipeline.py), so no patient data is involved. Start the
server first (`docker compose --profile fhir up -d hapi` with the repository docker-compose.yml), then:

    python tools/hapi_fhir/hapi_roundtrip_test.py [--base http://127.0.0.1:8080/fhir]

Checks per class (CN, MCI, AD):
  1. in-process validate_fhir_bundle() passes
  2. POST /Bundle/$validate returns no error-severity issues
  3. POST of the transaction returns 200 with six 201 Created entries
  4. every created resource can be read back and DiagnosticReport.result references resolve
Negative controls:
  5. a malformed bundle (Observation without status) is rejected with a 4xx status
  6. a transaction with one valid and one invalid entry leaves no new resource behind
A JSON report is written next to this script.
"""
import argparse, copy, json, sys, uuid
from pathlib import Path
from urllib import request, error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from fhir_generator import create_fhir_bundle, validate_fhir_bundle  # noqa: E402

CASES = {
    "CN":  dict(ai_diagnosis="CN",  ai_confidence=0.85, probabilities={"CN": 0.85, "MCI": 0.10, "AD": 0.05}),
    "MCI": dict(ai_diagnosis="MCI", ai_confidence=0.65, probabilities={"CN": 0.20, "MCI": 0.65, "AD": 0.15}),
    "AD":  dict(ai_diagnosis="AD",  ai_confidence=0.894, probabilities={"CN": 0.041, "MCI": 0.065, "AD": 0.894}),
}


def http(method, url, body=None):
    """Send a FHIR JSON request and return (status, parsed body)."""
    data = json.dumps(body).encode() if body is not None else None
    req = request.Request(url, data=data, method=method,
                          headers={"Content-Type": "application/fhir+json", "Accept": "application/fhir+json"})
    try:
        with request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw.decode(errors="replace")}


def make_bundle(label, spec):
    """Build a bundle with synthetic demographics for one prediction case."""
    return create_fhir_bundle(
        patient_id=f"SYNTH-{label}-{uuid.uuid4().hex[:6]}", age=72.0, gender="female",
        session_id=f"SYNTH_MR_{label}", slice_count=10,
        gradcam_base64="data:image/jpeg;base64,dGVzdA==", execution_latency_ms=1240.5,
        mmse=27.0, cdr=0.5, apoe="33", **spec)


def count(base, rtype):
    """Number of resources of a type on the server."""
    _, b = http("GET", f"{base}/{rtype}?_summary=count")
    return b.get("total", -1)


def main():
    """Run the per-class checks and the negative controls, then write the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080/fhir")
    ap.add_argument("--tag", default="", help="suffix for the report file name")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    report = {"base": base, "cases": {}, "negative": {}}
    ok = True

    st, meta = http("GET", f"{base}/metadata")
    report["server"] = {"status": st, "software": meta.get("software"), "fhirVersion": meta.get("fhirVersion")}
    if st != 200:
        sys.exit(f"HAPI not reachable at {base} (status {st}). Start it first.")

    # Per-class checks
    for label, spec in CASES.items():
        bundle = make_bundle(label, spec)
        r = {"inprocess_valid": validate_fhir_bundle(bundle)["valid"]}
        st, out = http("POST", f"{base}/Bundle/$validate", bundle)
        issues = out.get("issue", [])
        r["server_validate"] = {"status": st,
                                "errors": [i.get("diagnostics") or i.get("details", {}).get("text") for i in issues if i.get("severity") in ("error", "fatal")],
                                "warnings": sum(1 for i in issues if i.get("severity") == "warning")}
        st, resp = http("POST", base, bundle)
        entries = resp.get("entry", [])
        locs = [e.get("response", {}).get("location") for e in entries]
        r["transaction"] = {"status": st, "n_entries": len(entries),
                            "all_201": all(str(e.get("response", {}).get("status", "")).startswith("201") for e in entries)}
        readback = {}
        for loc in locs:
            gst, res = http("GET", f"{base}/{loc.split('/_history')[0]}") if loc else (0, {})
            readback[loc.split("/_history")[0] if loc else "?"] = gst
        r["readback"] = readback
        rtype_ids = {l.split("/")[0]: l.split("/")[1] for l in locs if l}
        if "DiagnosticReport" in rtype_ids:
            _, dr = http("GET", f"{base}/DiagnosticReport/{rtype_ids['DiagnosticReport']}")
            refs = [x.get("reference") for x in dr.get("result", [])]
            r["diagnosticreport_result_refs"] = refs
            r["refs_resolved"] = bool(refs) and all(x and x.startswith("Observation/") for x in refs)
        case_ok = (r["inprocess_valid"] and not r["server_validate"]["errors"] and st == 200
                   and r["transaction"]["n_entries"] == 6 and r["transaction"]["all_201"]
                   and all(v == 200 for v in readback.values()) and r.get("refs_resolved"))
        r["pass"] = bool(case_ok)
        ok &= bool(case_ok)
        report["cases"][label] = r

    # Negative control: malformed bundle is rejected
    bad = make_bundle("BAD", CASES["AD"])
    for e in bad["entry"]:
        if e["resource"]["resourceType"] == "Observation":
            e["resource"].pop("status", None)
            break
    st, out = http("POST", base, bad)
    report["negative"]["malformed_rejected"] = {"status": st, "pass": 400 <= st < 500}

    # Negative control: atomic rollback (valid Patient and invalid Observation in one transaction)
    before = count(base, "Patient")
    mixed = make_bundle("ATOMIC", CASES["AD"])
    for e in mixed["entry"]:
        if e["resource"]["resourceType"] == "Observation":
            e["resource"].pop("status", None)
    st, _ = http("POST", base, mixed)
    after = count(base, "Patient")
    report["negative"]["atomic_rollback"] = {"status": st, "patients_before": before, "patients_after": after,
                                             "pass": st >= 400 and before == after}
    ok &= all(v["pass"] for v in report["negative"].values())

    report["overall_pass"] = bool(ok)
    (Path(__file__).parent / f"hapi_roundtrip_report{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
