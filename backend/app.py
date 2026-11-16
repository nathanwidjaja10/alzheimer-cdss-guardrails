import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Make sibling backend modules importable regardless of the working directory
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from inference_engine import (
    predict_session, precompute_xai, xai_cache_exists, BASE_DIR, MODEL_READY, MODEL_LOAD_ERROR,
)

from fhir_generator import create_fhir_bundle, validate_fhir_bundle

app = FastAPI(
    title="OASIS-3 MobileViT Alzheimer CDSS & FHIR R4 API",
    version="2.1.0",
    description="Clinical Decision Support System integrating MobileViT Deep Learning classification, "
                "HiResCAM/Integrated-Gradients explainability (10-slice, attention-weighted), and "
                "HL7 FHIR R4 interoperability."
)

# Allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: mount the llm_xai narrative router. The core API runs without it; an import or
# mount failure is only logged.
try:
    import sys as _llm_sys
    from pathlib import Path as _LlmPath
    _llm_sys.path.insert(0, str(_LlmPath(__file__).resolve().parent.parent))  # root repo

    from llm_xai.backend.api_router import router as _llm_router
    app.include_router(_llm_router)
except Exception as _llm_e:  # noqa: BLE001
    print(f"[llm_xai] narrative router not mounted: {_llm_e}")

# Clinical metadata of both cohorts (OASIS-3 and ADNI), keyed by session id
CLINICAL_METADATA_PATHS = [
    BASE_DIR / 'oasis_data' / 'clinical_metadata.json',
    BASE_DIR / 'adni_data' / 'clinical_metadata.json',
    BASE_DIR / 'dummy_data' / 'clinical_metadata.json',
    BASE_DIR / 'clinical_metadata.json',
    BASE_DIR.parent / 'clinical_metadata.json'
]

clinical_data = {}
for meta_path in CLINICAL_METADATA_PATHS:
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                loaded = json.load(f)
                clinical_data.update(loaded)
            print(f"✅ Loaded metadata from {meta_path.name} (Combined records: {len(clinical_data)})")
        except Exception as e:
            print(f"⚠️ Error reading {meta_path}: {e}")


@app.get("/")
def read_root():
    return {
        "system": "OASIS-3 MobileViT Alzheimer CDSS",
        "status": "online",
        "version": "2.0.0",
        "fhir_version": "HL7 FHIR Release 4 / Release 5",
        "endpoints": [
            "/api/health",
            "/api/analyze/{session_id}",
            "/api/xai/precompute/{session_id}",
            "/api/fhir/validate"
        ]
    }


@app.get("/api/health")
def health_check():
    """Report service health, including whether the model checkpoint loaded successfully."""
    return {
        "status": "healthy" if MODEL_READY else "degraded",
        "model_ready": MODEL_READY,
        "model_load_error": MODEL_LOAD_ERROR,
        "clinical_records_count": len(clinical_data),
        "base_directory": str(BASE_DIR)
    }


@app.get("/api/analyze/{session_id}")
@app.post("/api/analyze/{session_id}")
def analyze_session(
    session_id: str,
    include_mmse: bool = True,
    include_apoe: bool = True,
    custom_age: Optional[float] = None,
    custom_gender: Optional[str] = None,
    xai_method: str = "hirescam",
    target_class: Optional[str] = None,
):
    """Classify a session, explain it, and package the result as a validated FHIR R4 bundle.

    xai_method: 'hirescam' (default), 'gradcam', or 'ig' (slower, cached).
    target_class: class explained by the heatmaps; None uses the predicted class.
    """
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail=f"Model not ready: {MODEL_LOAD_ERROR}")
    if xai_method not in ("hirescam", "gradcam", "ig"):
        raise HTTPException(status_code=400, detail=f"Invalid xai_method: {xai_method!r}")
    if target_class not in (None, "CN", "MCI", "AD"):
        raise HTTPException(status_code=400, detail=f"Invalid target_class: {target_class!r}")
    try:
        # Inference and explanation
        ai_result = predict_session(
            session_id=session_id,
            include_mmse=include_mmse,
            include_apoe=include_apoe,
            custom_age=custom_age,
            custom_gender=custom_gender,
            xai_method=xai_method,
            target_class=target_class,
        )
        
        # Clinical and demographic metadata of the session
        canonical_id = ai_result.get('canonical_session_id', session_id)
        c_data = (
            clinical_data.get(canonical_id) or
            clinical_data.get(session_id) or
            clinical_data.get(session_id.replace('_MR_', '_')) or
            clinical_data.get(f"ADNI_{session_id.replace('_MR_', '_')}") or
            {}
        )
        subject_id = c_data.get('subject_id') or (
            canonical_id.split('_')[0] if canonical_id.startswith('OAS')
            else (session_id.split('_')[0] if not session_id.startswith('ADNI_') else '_'.join(session_id.split('_')[1:4]))
        )
        age = custom_age if custom_age is not None else c_data.get('age_at_scan', 'Unknown')
        gender = custom_gender or c_data.get('gender', 'Unknown')
        mmse = c_data.get('mmse', None)
        cdr = c_data.get('cdr', None)
        apoe = c_data.get('apoe', None)
        
        # FHIR R4 bundle
        fhir_bundle = create_fhir_bundle(
            patient_id=subject_id,
            age=age,
            gender=gender,
            ai_diagnosis=ai_result['diagnosis'],
            ai_confidence=ai_result['confidence'],
            probabilities=ai_result['probabilities'],
            session_id=session_id,
            slice_count=ai_result['slice_count'],
            gradcam_base64=ai_result['gradcam_base64'],
            model_name=ai_result['model_info']['name'],
            checkpoint_version="2.0.0",
            execution_latency_ms=ai_result['latency_ms'],
            mmse=mmse,
            cdr=cdr,
            apoe=apoe
        )

        # Bundle validation
        validation_report = validate_fhir_bundle(fhir_bundle)
        
        return {
            "session_id": session_id,
            "patient_id": subject_id,
            "demographics": {
                "age": age,
                "gender": gender
            },
            "diagnosis": ai_result['diagnosis'],
            "diagnosis_calibrated": ai_result['diagnosis_calibrated'],
            "diagnosis_argmax": ai_result['diagnosis_argmax'],
            "confidence": ai_result['confidence'],
            "probabilities": ai_result['probabilities'],
            "thresholds_used": ai_result.get('thresholds_used', {}),
            "decision_rule_active": ai_result.get('decision_rule_active'),
            "attention_weights": ai_result.get('attention_weights', []),
            "top_slice": ai_result.get('top_slice', {}),
            "multimodal_inputs_used": ai_result.get('multimodal_inputs_used', {}),
            "slice_count": ai_result['slice_count'],
            "xai_method": ai_result.get('xai_method'),
            "cam_target_class": ai_result.get('cam_target_class'),
            "ig_completeness_residual": ai_result.get('ig_completeness_residual'),
            # Whether the Integrated Gradients view of the same class is already cached (opens instantly)
            "ig_cache_ready": xai_cache_exists(canonical_id, 'ig', ai_result.get('cam_target_class')),
            "slice_heatmaps": ai_result.get('slice_heatmaps', []),
            "out_of_domain": ai_result.get('out_of_domain', {}),
            "latency_ms": ai_result['latency_ms'],
            "device_used": ai_result['device_used'],
            "model_info": ai_result['model_info'],
            "gradcam_base64": ai_result['gradcam_base64'],
            "raw_slice_base64": ai_result.get('raw_slice_base64'),
            "fhir_bundle": fhir_bundle,
            "fhir_validation": validation_report
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/xai/precompute/{session_id}")
def precompute_xai_endpoint(session_id: str, xai_method: str = "ig", target_class: Optional[str] = None):
    """Compute and cache the XAI result for (session, method, class) so a later request is served from cache."""
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail=f"Model not ready: {MODEL_LOAD_ERROR}")
    if xai_method not in ("hirescam", "gradcam", "ig"):
        raise HTTPException(status_code=400, detail=f"Invalid xai_method: {xai_method!r}")
    try:
        return precompute_xai(session_id, xai_method=xai_method, target_class=target_class)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Precompute error: {str(e)}")


@app.post("/api/fhir/validate")
def validate_bundle_endpoint(bundle: Dict[str, Any] = Body(...)):
    """Validate an arbitrary FHIR R4 bundle."""
    try:
        report = validate_fhir_bundle(bundle)
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
