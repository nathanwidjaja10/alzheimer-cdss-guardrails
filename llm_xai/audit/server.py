"""Local server for the blind human audit of llama31_8b variant C narratives.

Standard library only. Progress is derived from the append-only results/audit_log.jsonl on every
request, so the server keeps no state and the audit can be stopped and resumed at any time.

Run:
    python3 audit/server.py [--port 8420]
and open http://localhost:8420/
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

AUDIT_DIR = Path(__file__).resolve().parent
ROOT = AUDIT_DIR.parent
UI_DIR = AUDIT_DIR / "ui"
RESULTS_DIR = AUDIT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH = RESULTS_DIR / "audit_log.jsonl"
MANIFEST_PATH = AUDIT_DIR / "cases_manifest.json"

EVAL_RAW = ROOT / "results" / "eval_raw"
GOLD_FACTS = ROOT / "results" / "gold_facts"

# URL path -> (file in ui/, content type)
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/anatomy_terms.json": ("anatomy_terms.json", "application/json; charset=utf-8"),
}

# Hallucination categories offered to the auditor
CATEGORIES = [
    "ungrounded_number",
    "fabricated_structure",
    "diagnosis_mismatch",
    "ood_contradiction",
    "covariate_caveat_breach",
    "label_or_comparison_error",
    "other",
]


def _load_manifest() -> dict:
    """Read the case manifest; exit with a hint when it has not been built."""
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"{MANIFEST_PATH} does not exist. Run first: python3 audit/build_manifest.py"
        )
    return json.loads(MANIFEST_PATH.read_text())


def _read_log() -> list[dict]:
    """Read all records of the audit log."""
    if not LOG_PATH.exists():
        return []
    records = []
    with LOG_PATH.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _last_records_by_auditor(records: list[dict], auditor: str) -> dict[str, dict]:
    """Map session_id to the auditor's latest record (a resubmission replaces the earlier one)."""
    out: dict[str, dict] = {}
    for r in records:
        if r.get("auditor_id") == auditor:
            out[r["session_id"]] = r
    return out


def _progress_for(auditor: str, manifest: dict, records: list[dict]) -> dict:
    """Count the auditor's completed and flagged cases."""
    done = _last_records_by_auditor(records, auditor)
    total = manifest["total"]
    n_flagged = sum(1 for r in done.values() if r.get("has_hallucination"))
    return {
        "auditor": auditor,
        "done": len(done),
        "total": total,
        "flagged": n_flagged,
    }


def _case_payload(case: dict, manifest: dict, previous: dict | None) -> dict:
    """Assemble the narrative, facts and (hidden) guardrail verdict for one case."""
    session_id = case["session_id"]
    eval_raw_path = EVAL_RAW / f"{session_id}__{manifest['model']}__{manifest['variant']}.json"
    gold_path = GOLD_FACTS / f"{session_id}.json"
    eval_raw = json.loads(eval_raw_path.read_text())
    facts = json.loads(gold_path.read_text())
    return {
        "order_index": case["order_index"],
        "total": manifest["total"],
        "session_id": session_id,
        "narrative": eval_raw["narrative"],
        "facts": {
            "prediction_anchor": facts["prediction_anchor"],
            "spatial_xai_anchor": facts["spatial_xai_anchor"],
            "clinical_covariates": facts["clinical_covariates"],
            "model_provenance": facts["model_provenance"],
        },
        "fact_digest": facts.get("fact_digest"),
        # The guardrail verdict is shown only after submission, which keeps the audit blind
        "reveal": {
            "guardrail_passed": eval_raw["guardrail"]["passed"],
            "guardrail_ncr": eval_raw["guardrail"]["ncr"],
            "guardrail_violation_types": eval_raw["guardrail"]["violation_types"],
            "guardrail_source": eval_raw["guardrail"]["source"],
        },
        "categories": CATEGORIES,
        "previous_verdict": previous,
    }


class Handler(BaseHTTPRequestHandler):
    """HTTP handler: static UI files and the JSON API (/api/next, /api/case, /api/progress, /api/submit)."""

    def log_message(self, fmt, *args):  # noqa: A003 -- quieter default logging
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, filename: str, content_type: str):
        path = UI_DIR / filename
        if not path.exists():
            self._send_json({"error": f"missing {filename}"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        # Static files, then API routes
        parsed = urlparse(self.path)
        if parsed.path in STATIC_FILES:
            filename, ctype = STATIC_FILES[parsed.path]
            self._send_static(filename, ctype)
            return

        qs = parse_qs(parsed.query)
        auditor = (qs.get("auditor") or [""])[0].strip()

        # Next case the auditor has not completed
        if parsed.path == "/api/next":
            if not auditor:
                self._send_json({"error": "auditor is empty"}, 400)
                return
            manifest = _load_manifest()
            records = _read_log()
            done_ids = set(_last_records_by_auditor(records, auditor).keys())
            remaining = [c for c in manifest["cases"] if c["session_id"] not in done_ids]
            progress = _progress_for(auditor, manifest, records)
            if not remaining:
                self._send_json({"finished": True, "progress": progress})
                return
            payload = _case_payload(remaining[0], manifest, previous=None)
            payload["progress"] = progress
            self._send_json(payload)
            return

        # A specific case, with the auditor's previous verdict
        if parsed.path == "/api/case":
            session_id = (qs.get("session_id") or [""])[0]
            if not auditor or not session_id:
                self._send_json({"error": "auditor/session_id is empty"}, 400)
                return
            manifest = _load_manifest()
            case = next((c for c in manifest["cases"] if c["session_id"] == session_id), None)
            if case is None:
                self._send_json({"error": "session_id not found in manifest"}, 404)
                return
            records = _read_log()
            previous = _last_records_by_auditor(records, auditor).get(session_id)
            payload = _case_payload(case, manifest, previous=previous)
            payload["progress"] = _progress_for(auditor, manifest, records)
            self._send_json(payload)
            return

        # Progress counters
        if parsed.path == "/api/progress":
            if not auditor:
                self._send_json({"error": "auditor is empty"}, 400)
                return
            manifest = _load_manifest()
            records = _read_log()
            self._send_json(_progress_for(auditor, manifest, records))
            return

        self._send_json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        # Validate a verdict and append it to the log
        parsed = urlparse(self.path)
        if parsed.path != "/api/submit":
            self._send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, 400)
            return

        required = ["auditor_id", "session_id", "has_hallucination"]
        missing = [k for k in required if k not in body or body[k] in (None, "")]
        if missing:
            self._send_json({"error": f"required field(s) empty: {missing}"}, 400)
            return

        manifest = _load_manifest()
        case = next((c for c in manifest["cases"] if c["session_id"] == body["session_id"]), None)
        if case is None:
            self._send_json({"error": "session_id not found in manifest"}, 404)
            return

        record = {
            "auditor_id": body["auditor_id"].strip(),
            "session_id": body["session_id"],
            "order_index": case["order_index"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_hallucination": bool(body["has_hallucination"]),
            "categories": body.get("categories", []),
            "other_category_note": body.get("other_category_note", ""),
            "field_grounding": body.get("field_grounding"),
            "confidence": body.get("confidence"),
            "notes": body.get("notes", ""),
            "duration_ms": body.get("duration_ms"),
            "manifest_seed": manifest["seed"],
            "fact_digest": case["fact_digest"],
        }
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")

        records = _read_log()
        self._send_json({"ok": True, "progress": _progress_for(record["auditor_id"], manifest, records)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8420)
    args = parser.parse_args()
    _load_manifest()  # exit early when the manifest has not been built
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Audit UI: http://localhost:{args.port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
