import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

try:
    from fhir.resources.R4B.bundle import Bundle as FHIRBundle  # 8.x top-level modules are R5; R4B is the R4-family schema
    FHIR_RESOURCES_AVAILABLE = True
except ImportError:
    FHIR_RESOURCES_AVAILABLE = False


# SNOMED CT codes used for the predicted diagnosis
SNOMED_DIAGNOSES = {
    'CN': {
        'code': '449888003',
        'display': 'Cognitively normal (finding)',
        'sct_display': 'Normal cognition',
        'definition': 'Normal cognitive function with no signs of mild cognitive impairment or dementia'
    },
    'MCI': {
        'code': '386805003',
        'display': 'Mild cognitive impairment (disorder)',
        'sct_display': 'Mild neurocognitive disorder',
        'definition': 'Mild cognitive impairment characterized by noticeable cognitive decline without severe daily impairment'
    },
    'AD': {
        'code': '26929004',
        'display': 'Alzheimer\'s disease (disorder)',
        'sct_display': 'Alzheimer\'s disease',
        'definition': 'Progressive neurodegenerative disease causing dementia, memory loss, and cognitive dysfunction'
    }
}

# Labels that depend on the cohort of the session. Sessions of the synthetic BrainWeb demo cohort
# (ids starting with "BWP") must not present themselves as OASIS-3 scans. Every other session keeps
# the OASIS-3 labels exactly as before (ADNI sessions included; they are not relabelled here).
_OASIS_LABELS: Dict[str, str] = {
    "subject_system": "https://www.oasis-brains.org/identifiers/subjects",
    "experiment_system": "https://www.oasis-brains.org/identifiers/experiments",
    "study_description": "OASIS-3 Structural Brain MRI (T1w / FreeSurfer Normalized Slices)",
    "scan_display": "OASIS-3 Scan",
    "series_uid_root": "2.16.840.1.113883.3.oasis3.series",
}
_SYNTHETIC_LABELS: Dict[str, str] = {
    "subject_system": "urn:alzai:synthetic:subjects",
    "experiment_system": "urn:alzai:synthetic:experiments",
    "study_description": "Synthetic BrainWeb phantom MRI (simulated T1w / FastSurfer normalized slices)",
    "scan_display": "Synthetic BrainWeb phantom",
    "series_uid_root": "2.16.840.1.113883.3.synthetic.series",
}


def cohort_labels(session_clean: str) -> Dict[str, str]:
    """Return the identifier systems and display strings for the cohort a session id belongs to."""
    return _SYNTHETIC_LABELS if session_clean.upper().startswith("BWP") else _OASIS_LABELS


def create_fhir_bundle(
    patient_id: str,
    age: Any,
    gender: str,
    ai_diagnosis: str,
    ai_confidence: float,
    probabilities: Optional[Dict[str, float]] = None,
    session_id: Optional[str] = None,
    slice_count: Optional[int] = None,
    gradcam_base64: Optional[str] = None,
    model_name: str = "apple/mobilevit-small",
    checkpoint_version: str = "1.0.0",
    execution_latency_ms: Optional[float] = None,
    mmse: Optional[Any] = None,
    cdr: Optional[Any] = None,
    apoe: Optional[str] = None
) -> Dict[str, Any]:
    """Build an HL7 FHIR R4 transaction bundle for one AI-assisted MRI assessment.

    The transaction type lets a FHIR server process the bundle atomically. It contains six
    linked resources:
    1. Patient: demographics and subject identifier.
    2. Device: AI model provenance.
    3. ImagingStudy: MRI session metadata.
    4. Observation: predicted class, class probabilities and optional clinical components.
    5. DocumentReference: explanation heatmap (Base64 JPEG).
    6. DiagnosticReport: consolidated report linking all findings.
    """
    # Derive class probabilities from the confidence when not supplied
    if probabilities is None:
        probabilities = {
            'CN': ai_confidence if ai_diagnosis == 'CN' else (1.0 - ai_confidence) / 2,
            'MCI': ai_confidence if ai_diagnosis == 'MCI' else (1.0 - ai_confidence) / 2,
            'AD': ai_confidence if ai_diagnosis == 'AD' else (1.0 - ai_confidence) / 2
        }

    snomed_info = SNOMED_DIAGNOSES.get(ai_diagnosis, {
        'code': '26929004',
        'display': 'Alzheimer\'s disease (disorder)',
        'sct_display': 'Alzheimer\'s disease',
        'definition': 'Alzheimer disease prediction'
    })

    # Resource identifiers
    patient_uuid = str(uuid.uuid4())
    device_uuid = str(uuid.uuid4())
    imaging_uuid = str(uuid.uuid4())
    observation_uuid = str(uuid.uuid4())
    docref_uuid = str(uuid.uuid4())
    report_uuid = str(uuid.uuid4())
    bundle_uuid = str(uuid.uuid4())

    now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    session_clean = session_id or f"{patient_id}_MR_session"
    labels = cohort_labels(session_clean)
    clean_series_uid = f"{labels['series_uid_root']}.{session_clean.replace('_', '.')}"

    # Patient
    patient_resource: Dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_uuid,
        "identifier": [
            {
                "use": "usual",
                "system": labels["subject_system"],
                "value": patient_id
            }
        ],
        "active": True,
        "gender": gender.lower() if gender and gender.lower() in ["male", "female", "other", "unknown"] else "unknown"
    }

    # Age extension when a numeric age is available
    if age and str(age).lower() != 'unknown':
        try:
            age_float = float(age)
            patient_resource["extension"] = [
                {
                    "url": "http://hl7.org/fhir/StructureDefinition/patient-age",
                    "valueQuantity": {
                        "value": age_float,
                        "unit": "years",
                        "system": "http://unitsofmeasure.org",
                        "code": "a"
                    }
                }
            ]
        except ValueError:
            pass

    # Device (model provenance)
    device_resource: Dict[str, Any] = {
        "resourceType": "Device",
        "id": device_uuid,
        "identifier": [
            {
                "system": "https://github.com/nathanwidjaja10/alzheimer-cdss-guardrails",
                "value": f"{model_name}-v{checkpoint_version}"
            }
        ],
        "status": "active",
        "manufacturer": "AlzheimerAI MobileViT Engine",
        "deviceName": [
            {
                "name": f"MobileViT-Small Alzheimer Classifier ({model_name})",
                "type": "model-name"
            }
        ],
        "type": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "706689003",
                    "display": "Application program software"
                }
            ],
            "text": "Deep Learning Vision Transformer Classifier for Structural MRI"
        },
        "version": [
            {
                "value": checkpoint_version
            }
        ],
        "note": [
            {
                "text": "Architecture: 2.5D MobileViT-small with gated-attention multiple-instance learning over multi-slice input, fused with clinical covariates; primary decision is vision-only argmax. Resolution: 224x224 RGB."
            }
        ]
    }

    # ImagingStudy (MRI session reference)
    imaging_resource: Dict[str, Any] = {
        "resourceType": "ImagingStudy",
        "id": imaging_uuid,
        "identifier": [
            {
                "use": "official",
                "system": labels["experiment_system"],
                "value": session_clean
            }
        ],
        "status": "available",
        "modality": [
            {
                "system": "http://dicom.nema.org/resources/ontology/DCM",
                "code": "MR",
                "display": "Magnetic Resonance"
            }
        ],
        "subject": {
            "reference": f"urn:uuid:{patient_uuid}",
            "display": f"Patient {patient_id}"
        },
        "started": now_iso,
        "numberOfSeries": 1,
        "numberOfInstances": slice_count if slice_count else 50,
        "description": labels["study_description"],
        "series": [
            {
                "uid": clean_series_uid,
                "number": 1,
                "modality": {
                    "system": "http://dicom.nema.org/resources/ontology/DCM",
                    "code": "MR",
                    "display": "Magnetic Resonance"
                },
                "description": "Coronal T1w Slices for Alzheimer Staging",
                "numberOfInstances": slice_count if slice_count else 50,
                "bodySite": {
                    "system": "http://snomed.info/sct",
                    "code": "12738006",
                    "display": "Brain structure"
                }
            }
        ]
    }

    # Observation (AI prediction with components)
    observation_components = []
    for cls_key in ['CN', 'MCI', 'AD']:
        prob_val = probabilities.get(cls_key, 0.0)
        cls_snomed = SNOMED_DIAGNOSES[cls_key]
        observation_components.append({
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": cls_snomed['code'],
                        "display": cls_snomed['sct_display']
                    }
                ],
                "text": f"Probability of {cls_key} ({cls_snomed['display']})"
            },
            "valueQuantity": {
                "value": round(float(prob_val) * 100, 2),
                "unit": "%",
                "system": "http://unitsofmeasure.org",
                "code": "%"
            }
        })

    # MMSE component (LOINC 72106-8)
    if mmse is not None and str(mmse).lower() not in ['unknown', 'none', 'nan', '']:
        try:
            mmse_num = float(mmse)
            observation_components.append({
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "72106-8",
                            "display": "Total score [MMSE]"
                        }
                    ],
                    "text": "Mini-Mental State Examination (MMSE) Score"
                },
                "valueQuantity": {
                    "value": mmse_num,
                    "unit": "{score}",
                    "system": "http://unitsofmeasure.org",
                    "code": "{score}"
                }
            })
        except ValueError:
            pass

    # CDR component (LOINC 72088-8, the scale-level CDR code; LOINC has no global-score code)
    if cdr is not None and str(cdr).lower() not in ['unknown', 'none', 'nan', '']:
        try:
            cdr_num = float(cdr)
            observation_components.append({
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "72088-8",
                            "display": "Clinical Dementia Rating scale [CDR]"
                        }
                    ],
                    "text": "Clinical Dementia Rating (CDR) Global Score"
                },
                "valueQuantity": {
                    "value": cdr_num,
                    "unit": "{score}",
                    "system": "http://unitsofmeasure.org",
                    "code": "{score}"
                }
            })
        except ValueError:
            pass

    # APOE genotype component (LOINC 42315-2)
    if apoe and str(apoe).lower() not in ['unknown', 'none', 'nan', '']:
        apoe_str = str(apoe).strip()
        apoe_clean = f"E{apoe_str[0]}/E{apoe_str[1]}" if (len(apoe_str) == 2 and apoe_str.isdigit()) else apoe_str
        observation_components.append({
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "42315-2",
                        "display": "APOE gene alleles e2 and e3 and e4 [Identifier] in Blood or Tissue by Molecular genetics method Nominal"
                    }
                ],
                "text": "APOE Genotype"
            },
            "valueString": apoe_clean
        })

    observation_resource: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": observation_uuid,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging"
                    },
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "exam",
                        "display": "Exam"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "24590-2",
                    "display": "MR Brain"
                }
            ],
            "text": "MobileViT Structural MRI AI Cognitive Status Assessment"
        },
        "subject": {
            "reference": f"urn:uuid:{patient_uuid}",
            "display": f"Patient {patient_id}"
        },
        "device": {
            "reference": f"urn:uuid:{device_uuid}",
            "display": f"MobileViT-Small ({model_name})"
        },
        "derivedFrom": [
            {
                "reference": f"urn:uuid:{imaging_uuid}",
                "display": f"{labels['scan_display']} {session_clean}"
            }
        ],
        "effectiveDateTime": now_iso,
        "valueCodeableConcept": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": snomed_info['code'],
                    "display": snomed_info['sct_display']
                }
            ],
            "text": f"Predicted Stage: {ai_diagnosis} ({snomed_info['display']})"
        },
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": "A" if ai_diagnosis in ['MCI', 'AD'] else "N",
                        "display": "Abnormal" if ai_diagnosis in ['MCI', 'AD'] else "Normal"
                    }
                ],
                "text": f"Highest confidence prediction: {ai_diagnosis} ({ai_confidence:.1%})"
            }
        ],
        "component": observation_components
    }

    # DocumentReference (explanation heatmap)
    clean_gradcam_b64 = ""
    if gradcam_base64:
        clean_gradcam_b64 = gradcam_base64.split(",")[-1] if "," in gradcam_base64 else gradcam_base64

    docref_resource: Dict[str, Any] = {
        "resourceType": "DocumentReference",
        "id": docref_uuid,
        "status": "current",
        "docStatus": "final",
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "55113-5",
                    "display": "Key images Document Radiology"
                }
            ],
            "text": "Explainable AI Attribution Heatmap"
        },
        "subject": {
            "reference": f"urn:uuid:{patient_uuid}",
            "display": f"Patient {patient_id}"
        },
        "date": now_iso,
        "author": [
            {
                "reference": f"urn:uuid:{device_uuid}",
                "display": "MobileViT AI Engine"
            }
        ],
        "description": f"Attribution heatmap (HiResCAM by default) highlighting salient regions for {ai_diagnosis} prediction.",
        "content": [
            {
                "attachment": {
                    "contentType": "image/jpeg",
                    "language": "en",
                    "data": clean_gradcam_b64 if clean_gradcam_b64 else "dGVzdA==",
                    "title": f"Attribution Heatmap - Session {session_clean}"
                }
            }
        ]
    }

    # DiagnosticReport (consolidated report)
    conclusion_text = (
        f"Automated AI neuroimaging evaluation of MRI session {session_clean} predicts '{ai_diagnosis}' "
        f"({snomed_info['display']}) with a confidence of {ai_confidence:.1%}. "
        f"Probability breakdown: CN: {probabilities.get('CN', 0.0):.1%}, "
        f"MCI: {probabilities.get('MCI', 0.0):.1%}, AD: {probabilities.get('AD', 0.0):.1%}."
    )

    report_resource: Dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": report_uuid,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "18748-4",
                    "display": "Diagnostic imaging study"
                },
                {
                    "system": "http://loinc.org",
                    "code": "24590-2",
                    "display": "MR Brain"
                }
            ],
            "text": "Brain MRI AI-Assisted Cognitive Screening Report"
        },
        "subject": {
            "reference": f"urn:uuid:{patient_uuid}",
            "display": f"Patient {patient_id}"
        },
        "effectiveDateTime": now_iso,
        "issued": now_iso,
        "imagingStudy": [
            {
                "reference": f"urn:uuid:{imaging_uuid}",
                "display": f"{labels['scan_display']} {session_clean}"
            }
        ],
        "result": [
            {
                "reference": f"urn:uuid:{observation_uuid}",
                "display": f"MobileViT Inference: {ai_diagnosis} ({ai_confidence:.1%})"
            }
        ],
        "presentedForm": [
            {
                "contentType": "image/jpeg",
                "language": "en",
                "data": clean_gradcam_b64 if clean_gradcam_b64 else "dGVzdA==",
                "title": f"Attribution Heatmap - {session_clean}"
            }
        ],
        "conclusion": conclusion_text,
        "conclusionCode": [
            {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": snomed_info['code'],
                        "display": snomed_info['sct_display']
                    }
                ],
                "text": snomed_info['display']
            }
        ]
    }

    # Assemble the transaction bundle
    bundle: Dict[str, Any] = {
        "resourceType": "Bundle",
        "id": bundle_uuid,
        "type": "transaction",
        "timestamp": now_iso,
        "entry": [
            {
                "fullUrl": f"urn:uuid:{patient_uuid}",
                "resource": patient_resource,
                "request": {"method": "POST", "url": "Patient"}
            },
            {
                "fullUrl": f"urn:uuid:{device_uuid}",
                "resource": device_resource,
                "request": {"method": "POST", "url": "Device"}
            },
            {
                "fullUrl": f"urn:uuid:{imaging_uuid}",
                "resource": imaging_resource,
                "request": {"method": "POST", "url": "ImagingStudy"}
            },
            {
                "fullUrl": f"urn:uuid:{observation_uuid}",
                "resource": observation_resource,
                "request": {"method": "POST", "url": "Observation"}
            },
            {
                "fullUrl": f"urn:uuid:{docref_uuid}",
                "resource": docref_resource,
                "request": {"method": "POST", "url": "DocumentReference"}
            },
            {
                "fullUrl": f"urn:uuid:{report_uuid}",
                "resource": report_resource,
                "request": {"method": "POST", "url": "DiagnosticReport"}
            }
        ]
    }

    return bundle


def validate_fhir_bundle(bundle_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a bundle dictionary against the FHIR Bundle schema using `fhir.resources`.

    Falls back to a structural check when the library is unavailable.
    Returns {"valid", "validator", "errors", "resource_types", "timestamp"}.
    """
    resource_types = [entry.get("resource", {}).get("resourceType") for entry in bundle_dict.get("entry", [])]

    if not FHIR_RESOURCES_AVAILABLE:
        is_bundle = bundle_dict.get("resourceType") == "Bundle"
        has_entries = isinstance(bundle_dict.get("entry"), list) and len(bundle_dict["entry"]) > 0
        return {
            "valid": is_bundle and has_entries,
            "validator": "structural_fallback",
            "errors": [] if (is_bundle and has_entries) else ["Invalid resourceType or missing entries"],
            "resource_types": resource_types,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    try:
        FHIRBundle.model_validate(bundle_dict)
        return {
            "valid": True,
            "validator": "fhir.resources",
            "errors": [],
            "resource_types": resource_types,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "valid": False,
            "validator": "fhir.resources",
            "errors": [str(e)],
            "resource_types": resource_types,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
