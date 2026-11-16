import React, { useState, useEffect, useMemo, useRef } from 'react';
import NiiVueViewer from './NiiVueViewer';
import RoiDemoViewer from './RoiDemoViewer';
import { ROI_LEGEND } from './roiDemoData';
import { 
  User, Activity, Brain, Clock, ChevronRight, ChevronLeft, Stethoscope, 
  AlertTriangle, AlertCircle, Search, Filter, PieChart, Layers, Image as ImageIcon, 
  Info, Download, Copy, Check, ShieldCheck, ShieldAlert, FileText, CheckCircle2,
  Cpu, Sparkles, RotateCcw, Eye, SplitSquareVertical, RefreshCw, Database, Wrench,
  MessageCircle
} from 'lucide-react';
import generatedData from './data.json';
import './index.css';

// Class-level recall with 95% bootstrap intervals (2000 resamples of the 232-session internal test fold,
// vision-only argmax, the primary decision path), taken from the training notebook. Shown as a
// population-level reliability band next to a prediction, not as a per-patient confidence interval.
const CLASS_TESTFOLD_CI = {
  CN:  { metric: 'recall', point: 84.7, lo: 79.3, hi: 89.6 },
  MCI: { metric: 'recall', point: 64.9, lo: 48.6, hi: 80.0 },
  AD:  { metric: 'recall', point: 91.7, lo: 71.4, hi: 100.0 },
};
const MODEL_MACRO_F1_CI = { point: 75.2, lo: 66.5, hi: 82.2 };

// Map the diagnosis text and CDR to a broad category (Normal, MCI, AD, Other)
const getDiagnosisCategory = (diagnosisText, cdr) => {
  // The CDR score takes precedence
  try {
    const cdrVal = parseFloat(cdr);
    if (!isNaN(cdrVal)) {
      if (cdrVal === 0) return 'Normal';
      if (cdrVal === 0.5) return 'MCI';
      if (cdrVal >= 1.0) return 'AD';
    }
  } catch (e) {
    // Fall through to the text diagnosis
  }

  // Text diagnosis when the CDR is invalid or absent
  if (!diagnosisText) return 'Other';
  const text = diagnosisText.toLowerCase();
  if (text.includes('normal')) return 'Normal';
  if (text.includes('mci') || text.includes('impairment') || text.includes('mild cognitive')) return 'MCI';
  if (text.includes('ad') || text.includes('dementia') || text.includes('alzheimer')) return 'AD';
  return 'Other';
};

// 2D slice viewer with orientation markers, window presets and keyboard scrolling
const SliceViewer2D = ({ sliceUrls, cohortName, onSwitchTo3D }) => {
  const [sliceIndex, setSliceIndex] = useState(0);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [invert, setInvert] = useState(false);
  const [activePreset, setActivePreset] = useState('standard');

  // Return to the first slice when the slice set changes
  useEffect(() => {
    setSliceIndex(0);
  }, [sliceUrls]);

  // Arrow keys scroll through the slices
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        setSliceIndex(prev => Math.min(sliceUrls.length - 1, prev + 1));
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        setSliceIndex(prev => Math.max(0, prev - 1));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [sliceUrls]);

  const applyPreset = (preset) => {
    setActivePreset(preset);
    if (preset === 'standard') {
      setBrightness(100);
      setContrast(100);
      setInvert(false);
    } else if (preset === 'high_contrast') {
      setBrightness(115);
      setContrast(145);
      setInvert(false);
    } else if (preset === 'invert') {
      setBrightness(100);
      setContrast(110);
      setInvert(true);
    }
  };

  const resetWindowing = () => {
    setBrightness(100);
    setContrast(100);
    setInvert(false);
    setActivePreset('standard');
  };

  if (!sliceUrls || sliceUrls.length === 0) {
    return (
      <div className="empty-state" style={{ height: '400px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#000', borderRadius: '8px', padding: '20px' }}>
        <Brain size={48} color="#38bdf8" style={{ opacity: 0.85, marginBottom: '12px' }} />
        <p style={{ marginTop: '5px', color: '#f8fafc', fontWeight: 600, fontSize: '15px' }}>2D Pre-extracted PNG Slices Not Available</p>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '6px', maxWidth: '400px', textAlign: 'center', lineHeight: 1.5 }}>
          This subject is from the <strong>{cohortName || 'ADNI'}</strong> cohort. Explore the native 3D NIfTI volume in <strong>3D NiiVue</strong> mode with multi-planar crosshairs.
        </p>
        {onSwitchTo3D && (
          <button 
            className="toggle-btn active" 
            onClick={onSwitchTo3D}
            style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 20px', border: 'none', background: 'var(--primary-color)', color: '#fff', borderRadius: '20px', cursor: 'pointer', fontWeight: 600, fontSize: '13px' }}
          >
            <Brain size={16} /> Switch to 3D NiiVue View
          </button>
        )}
      </div>
    );
  }

  const imageFilterStyle = `brightness(${brightness}%) contrast(${contrast}%) ${invert ? 'invert(100%)' : ''}`;

  return (
    <div className="dicom-viewport-container" role="region" aria-label="DICOM 2D Slice Viewport">
      <div className="dicom-canvas-area">
        {/* Anatomical orientation markers */}
        <div className="dicom-marker dicom-marker-top" title="Superior / Anterior (Anatomical Top)">S</div>
        <div className="dicom-marker dicom-marker-bottom" title="Inferior / Posterior (Anatomical Bottom)">I</div>
        <div className="dicom-marker dicom-marker-left" title="Patient's Right (Radiological Convention: Left of screen is Subject's Right)">R</div>
        <div className="dicom-marker dicom-marker-right" title="Patient's Left (Radiological Convention: Right of screen is Subject's Left)">L</div>

        {/* Slice image with brightness and contrast filter */}
        <img 
          src={sliceUrls[sliceIndex]} 
          alt={`MRI Slice ${sliceIndex + 1}`} 
          className="dicom-slice-img"
          style={{ filter: imageFilterStyle }}
        />

        {/* 50 mm scale reference */}
        <div className="dicom-scale-bar" title="Calibrated 50 mm Spatial Scale Reference">
          <div className="dicom-scale-line"></div>
          <span className="dicom-scale-text">50 mm</span>
        </div>

        {/* Slice index badge */}
        <div style={{ position: 'absolute', top: 10, right: 10, background: 'rgba(15,23,42,0.85)', padding: '4px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.15)', color: '#38bdf8', fontSize: '11px', fontFamily: 'monospace', fontWeight: 600 }}>
          Slice: {sliceIndex + 1} / {sliceUrls.length}
        </div>
      </div>

      {/* Window controls */}
      <div className="dicom-controls-toolbar">
        {/* Slice scrubber */}
        <div className="dicom-scrubber-row">
          <button 
            className="dicom-btn" 
            onClick={() => setSliceIndex(prev => Math.max(0, prev - 1))}
            disabled={sliceIndex === 0}
            title="Previous Slice (Arrow Left)"
          >
            <ChevronLeft size={14} /> Prev
          </button>

          <input 
            type="range" 
            min="0" 
            max={sliceUrls.length - 1} 
            value={sliceIndex} 
            onChange={(e) => setSliceIndex(parseInt(e.target.value))}
            style={{ flex: 1, accentColor: '#38bdf8', cursor: 'ew-resize' }}
            aria-label="Slice scrub slider"
          />

          <button 
            className="dicom-btn" 
            onClick={() => setSliceIndex(prev => Math.min(sliceUrls.length - 1, prev + 1))}
            disabled={sliceIndex === sliceUrls.length - 1}
            title="Next Slice (Arrow Right)"
          >
            Next <ChevronRight size={14} />
          </button>
        </div>

        {/* Contrast presets */}
        <div className="dicom-windowing-row">
          <div className="dicom-preset-group">
            <span style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Windowing (WL/WW):</span>
            <button 
              className={`dicom-btn ${activePreset === 'standard' ? 'active' : ''}`}
              onClick={() => applyPreset('standard')}
            >
              Standard
            </button>
            <button 
              className={`dicom-btn ${activePreset === 'high_contrast' ? 'active' : ''}`}
              onClick={() => applyPreset('high_contrast')}
            >
              High Contrast
            </button>
            <button 
              className={`dicom-btn ${activePreset === 'invert' ? 'active' : ''}`}
              onClick={() => applyPreset('invert')}
            >
              Inverted
            </button>
            <button 
              className="dicom-btn" 
              onClick={resetWindowing}
              title="Reset Windowing Filters"
            >
              <RotateCcw size={12} />
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="keyboard-shortcut-tag">← / → Arrow Keys to Scrub</span>
          </div>
        </div>
      </div>
    </div>
  );
};

function App() {
  const [activeSubject, setActiveSubject] = useState(generatedData.length > 0 ? generatedData[0] : null);
  const [activeSession, setActiveSession] = useState(generatedData.length > 0 && generatedData[0].sessions.length > 0 ? generatedData[0].sessions[0] : null);

  // Subject filters
  const [searchTerm, setSearchTerm] = useState('');
  const [cohortFilter, setCohortFilter] = useState('All'); // 'All' | 'OASIS-3' | 'ADNI'
  const [genderFilter, setGenderFilter] = useState('All');
  const [diagnosisFilter, setDiagnosisFilter] = useState('All');
  const [cdrFilter, setCdrFilter] = useState('All');
  const [mmseFilter, setMmseFilter] = useState('All');

  // Viewer mode
  const [viewMode, setViewMode] = useState('3D'); // '3D' or '2D'
  // Anatomical ROI reference pop-up: a static CN (CDR 0) and AD (CDR 2) pair with the AD target structures
  const [showRoiModal, setShowRoiModal] = useState(false);

  // Inference and FHIR state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  const [showAIModal, setShowAIModal] = useState(false);
  const [cdssTab, setCdssTab] = useState('diagnosis'); // 'diagnosis' | 'summary' | 'json' | 'rules' | 'conformance'
  const [copied, setCopied] = useState(false);

  // Narrative panel state. Generation starts in the background once the prediction arrives, so the
  // local-LLM latency does not block the prediction and FHIR panels; "Generate" and "Retry" call the
  // same function on demand.
  const [narrativeResult, setNarrativeResult] = useState(null);
  const [narrativeLoading, setNarrativeLoading] = useState(false);
  const [narrativeError, setNarrativeError] = useState(null);

  // Follow-up Q&A state. A session pins the facts and base narrative on the server behind a token; every
  // turn is answered from that pinned state. The panel is available only once a narrative exists. The
  // token lives in server memory, so a backend restart or the turn cap ends the session.
  const [qaToken, setQaToken] = useState(null);
  const [qaMessages, setQaMessages] = useState([]); // [{question, answer, cited_fields, answerable, category, guardrail, source, llm_calls, wall_s}]
  const [qaSessionLoading, setQaSessionLoading] = useState(false);
  const [qaSessionError, setQaSessionError] = useState(null);
  const [qaInput, setQaInput] = useState('');
  const [qaAsking, setQaAsking] = useState(false);
  const [qaAskError, setQaAskError] = useState(null);
  const [qaEnded, setQaEnded] = useState(false); // turn cap reached or token expired: a new session is needed

  // Clinical covariates (MMSE, APOE, CDR, age, sex) are read-only context: the deployed model is
  // image-only and the FHIR bundle includes them regardless, so there is no include/exclude toggle.
  // The decision rule is fixed to image-only argmax; calibrated thresholds are shown only for comparison.

  // Explainability view controls
  const [xaiMode, setXaiMode] = useState('overlay'); // 'overlay' | 'raw' | 'split'
  const [xaiOpacity, setXaiOpacity] = useState(0.65); // 0.0 - 1.0
  // HiResCAM is the default method; Integrated Gradients is an optional, slower detail mode
  const [xaiMethod, setXaiMethod] = useState('hirescam'); // 'hirescam' | 'ig'
  // Integrated Gradients cache of the current analysis. IG is precomputed in the background only
  // after the narrative finishes, so the two never compete for the GPU.
  // 'unknown' | 'queued' (waiting for the narrative) | 'computing' | 'ready' | 'failed'
  const [igStatus, setIgStatus] = useState('unknown');
  // The IG view was requested while IG was still queued or computing: open it once ready
  const [igPendingOpen, setIgPendingOpen] = useState(false);
  const igPendingOpenRef = useRef(false);
  // Incremented by every new analysis. Responses of an older analysis (another session) are dropped,
  // so a slow narrative, Q&A answer or IG precompute never lands in the panel of a different session.
  const analysisRunRef = useRef(0);
  const narrativeAbortRef = useRef(null);
  // Slice heatmap shown enlarged (the highest-attention slice once a result arrives)
  const [selectedSliceIdx, setSelectedSliceIdx] = useState(0);
  // Differential CN/MCI/AD explanation comparison
  const [showDiffView, setShowDiffView] = useState(false);
  const [diffResults, setDiffResults] = useState(null);
  const [diffLoading, setDiffLoading] = useState(false);

  const downloadFHIRBundle = () => {
    if (!aiResult?.fhir_bundle) return;
    const jsonStr = JSON.stringify(aiResult.fhir_bundle, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `FHIR_Bundle_${aiResult.session_id || 'session'}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const copyToClipboard = () => {
    if (!aiResult?.fhir_bundle) return;
    navigator.clipboard.writeText(JSON.stringify(aiResult.fhir_bundle, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Latest diagnosis per subject, used by the filters
  const processedSubjects = useMemo(() => {
    return generatedData.map(subject => {
      // The session with the highest day count defines the subject's status
      const latestSession = [...subject.sessions].sort((a, b) => b.days - a.days)[0];
      const latestCategory = latestSession ? getDiagnosisCategory(latestSession.diagnosis, latestSession.cdr) : 'Other';
      const latestCdr = latestSession ? latestSession.cdr : 'N/A';
      const latestMmse = latestSession ? latestSession.mmse : 'N/A';
      return {
        ...subject,
        latestCategory,
        latestCdr,
        latestMmse
      };
    });
  }, []);

  const filteredSubjects = useMemo(() => {
    return processedSubjects.filter(subject => {
      const matchSearch = subject.id.toLowerCase().includes(searchTerm.toLowerCase());
      const matchCohort = cohortFilter === 'All' || subject.cohort === cohortFilter;
      const matchGender = genderFilter === 'All' || subject.gender === genderFilter;
      const matchDiagnosis = diagnosisFilter === 'All' || subject.latestCategory === diagnosisFilter;
      
      let matchCdr = true;
      if (cdrFilter !== 'All') {
        if (cdrFilter === 'N/A') matchCdr = subject.latestCdr === 'N/A';
        else matchCdr = parseFloat(subject.latestCdr) === parseFloat(cdrFilter);
      }

      let matchMmse = true;
      if (mmseFilter !== 'All') {
        const mmse = parseInt(subject.latestMmse);
        if (mmseFilter === 'N/A') matchMmse = isNaN(mmse);
        else if (mmseFilter === 'Normal (24-30)') matchMmse = mmse >= 24 && mmse <= 30;
        else if (mmseFilter === 'Mild (10-23)') matchMmse = mmse >= 10 && mmse <= 23;
        else if (mmseFilter === 'Severe (0-9)') matchMmse = mmse >= 0 && mmse <= 9;
      }

      return matchSearch && matchCohort && matchGender && matchDiagnosis && matchCdr && matchMmse;
    });
  }, [processedSubjects, searchTerm, cohortFilter, genderFilter, diagnosisFilter, cdrFilter, mmseFilter]);

  const resolveFullSessionId = () => activeSession.sessionId || (
    (activeSubject.cohort === 'ADNI' || activeSubject.id.includes('_S_'))
      ? `ADNI_${activeSubject.id}_${activeSession.id}`
      : `${activeSubject.id}_MR_${activeSession.id}`
  );

  // customXaiMethod is set when only the XAI method of the current analysis is switched; a call
  // without it is a new analysis.
  const runAIAnalysis = async (customXaiMethod) => {
    if (!activeSession) return;
    const methodActive = customXaiMethod || xaiMethod;
    const isNewAnalysis = !customXaiMethod;
    if (isNewAnalysis) {
      analysisRunRef.current += 1;
      // Drop the previous analysis's narrative request (its response is ignored from here on)
      narrativeAbortRef.current?.abort();
      // A new prediction invalidates the narrative and Q&A session of the previous one. Switching
      // only the XAI method keeps them: the narrative is always built from the HiResCAM facts.
      setNarrativeResult(null);
      setNarrativeError(null);
      setNarrativeLoading(false);
      resetQaState();
      setIgStatus('unknown');
      setIgPending(false);
    }
    const runId = analysisRunRef.current;
    setIsAnalyzing(true);
    setShowAIModal(true);
    setAiResult(null);
    setSelectedSliceIdx(0);
    setDiffResults(null);
    setShowDiffView(false);
    try {
      const fullSessionId = resolveFullSessionId();
      // include_mmse and include_apoe only make the response carry the covariate values for display
      const response = await fetch(`http://localhost:8000/api/analyze/${fullSessionId}?include_mmse=true&include_apoe=true&xai_method=${methodActive}`);
      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.detail || `API Error: ${response.statusText}`);
      }
      const data = await response.json();
      if (runId !== analysisRunRef.current) return;
      setAiResult(data);
      if (data.top_slice && typeof data.top_slice.index === 'number') {
        setSelectedSliceIdx(data.top_slice.index);
      }
      if (methodActive === 'ig' || data.ig_cache_ready) {
        setIgStatus('ready');
      }
      if (isNewAnalysis) {
        // Narrative first, then the background Integrated Gradients precompute (only when the IG
        // cache is missing), so the two never run at the same time
        const igNeeded = methodActive !== 'ig' && !data.ig_cache_ready;
        if (igNeeded) setIgStatus('queued');
        runNarrativeGeneration().finally(() => {
          if (igNeeded) precomputeIg(fullSessionId, runId);
        });
      }
    } catch (err) {
      if (runId === analysisRunRef.current) setAiResult({ error: err.message });
    } finally {
      if (runId === analysisRunRef.current) setIsAnalyzing(false);
    }
  };

  // Warm the Integrated Gradients cache so the detail view opens instantly
  const precomputeIg = async (fullSessionId, runId) => {
    if (runId !== analysisRunRef.current) return;
    setIgStatus('computing');
    try {
      const response = await fetch(`http://localhost:8000/api/xai/precompute/${fullSessionId}?xai_method=ig`, { method: 'POST' });
      if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
      if (runId !== analysisRunRef.current) return;
      setIgStatus('ready');
      // Open the IG view that was requested while it was still queued or computing
      if (igPendingOpenRef.current) switchXaiMethod('ig');
    } catch {
      if (runId !== analysisRunRef.current) return;
      setIgStatus('failed');
      setIgPending(false);
    }
  };

  // Closing the CDSS modal cancels a pending IG request, so it never reopens the modal later
  const closeAIModal = () => {
    setIgPending(false);
    setShowAIModal(false);
  };

  const setIgPending = (pending) => {
    igPendingOpenRef.current = pending;
    setIgPendingOpen(pending);
  };

  const switchXaiMethod = (method) => {
    setIgPending(false);
    setXaiMethod(method);
    runAIAnalysis(method);
  };

  // IG button: while the background precompute is queued or running, wait for it instead of
  // starting a second computation (and instead of competing with the narrative)
  const requestIgView = () => {
    if (igStatus === 'queued' || igStatus === 'computing') {
      setIgPending(true);
      return;
    }
    switchXaiMethod('ig');
  };

  // Generate the guardrail-audited narrative (fixed to llama31_8b, variant c, the evaluated configuration).
  // Fails gracefully when the llm_xai router is not mounted.
  const runNarrativeGeneration = async () => {
    if (!activeSession) return;
    const runId = analysisRunRef.current;
    narrativeAbortRef.current?.abort();
    const controller = new AbortController();
    narrativeAbortRef.current = controller;
    setNarrativeLoading(true);
    setNarrativeError(null);
    try {
      const fullSessionId = resolveFullSessionId();
      // The endpoint accepts POST only
      const response = await fetch(`http://localhost:8000/api/xai/narrative/${fullSessionId}?model=llama31_8b&variant=c`, {
        method: 'POST',
        signal: controller.signal,
      });
      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.detail || `API Error: ${response.statusText}`);
      }
      const data = await response.json();
      if (controller.signal.aborted || runId !== analysisRunRef.current) return;
      setNarrativeResult(data);
      // A new narrative invalidates the pinned Q&A session; a failed attempt leaves it untouched
      resetQaState();
    } catch (err) {
      if (controller.signal.aborted || runId !== analysisRunRef.current) return;
      setNarrativeError(err.message);
    } finally {
      if (narrativeAbortRef.current === controller) {
        narrativeAbortRef.current = null;
        if (runId === analysisRunRef.current) setNarrativeLoading(false);
      }
    }
  };

  const resetQaState = () => {
    setQaToken(null);
    setQaMessages([]);
    setQaSessionError(null);
    setQaAskError(null);
    setQaInput('');
    setQaEnded(false);
  };

  // Start a Q&A session: pin the facts and base narrative (served from the narrative cache)
  const startQaSession = async () => {
    if (!activeSession) return;
    resetQaState();
    const runId = analysisRunRef.current;
    setQaSessionLoading(true);
    try {
      const fullSessionId = resolveFullSessionId();
      const response = await fetch(`http://localhost:8000/api/xai/narrative/${fullSessionId}/session?model=llama31_8b&variant=c`, {
        method: 'POST',
      });
      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.detail || `API Error: ${response.statusText}`);
      }
      const data = await response.json();
      if (runId !== analysisRunRef.current) return;  // the session changed meanwhile
      setQaToken(data.qa_token);
    } catch (err) {
      if (runId !== analysisRunRef.current) return;
      setQaSessionError(err.message);
    } finally {
      setQaSessionLoading(false);
    }
  };

  // Ask one question; answered from the pinned facts only
  const askQaQuestion = async (e) => {
    e.preventDefault();
    const question = qaInput.trim();
    if (!qaToken || !question || qaAsking) return;
    const runId = analysisRunRef.current;
    setQaAsking(true);
    setQaAskError(null);
    try {
      const response = await fetch(`http://localhost:8000/api/xai/qa/${qaToken}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      if (runId !== analysisRunRef.current) return;  // the session changed meanwhile
      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        // The turn cap (400) and an expired token (404) both end the session
        if (response.status === 400 || response.status === 404) {
          setQaEnded(true);
        }
        throw new Error(errorJson.detail || `API Error: ${response.statusText}`);
      }
      const data = await response.json();
      setQaMessages(prev => [...prev, { question, ...data }]);
      setQaInput('');
    } catch (err) {
      if (runId !== analysisRunRef.current) return;
      setQaAskError(err.message);
    } finally {
      setQaAsking(false);
    }
  };

  // Compute the explanation of each class (CN, MCI, AD) for the selected slice
  const runDifferentialView = async () => {
    if (!aiResult || aiResult.error) return;
    setDiffLoading(true);
    setShowDiffView(true);
    try {
      const fullSessionId = resolveFullSessionId();
      const classes = ['CN', 'MCI', 'AD'];
      const results = await Promise.all(classes.map(async (cls) => {
        const resp = await fetch(`http://localhost:8000/api/analyze/${fullSessionId}?include_mmse=true&include_apoe=true&xai_method=${xaiMethod}&target_class=${cls}`);
        const data = await resp.json();
        return [cls, data];
      }));
      setDiffResults(Object.fromEntries(results));
    } catch (err) {
      setDiffResults({ error: err.message });
    } finally {
      setDiffLoading(false);
    }
  };

  // Keep the active subject within the filtered list
  useEffect(() => {
    if (filteredSubjects.length > 0 && !filteredSubjects.find(s => s.id === activeSubject?.id)) {
      handleSubjectClick(filteredSubjects[0]);
    }
  }, [filteredSubjects]);

  const handleSubjectClick = (subject) => {
    if (!subject) return;
    setActiveSubject(subject);
    setActiveSession(subject.sessions[0]); // Reset to first session
  };

  // Cohort statistics of the filtered subjects
  const stats = useMemo(() => {
    const counts = { 
      Total: filteredSubjects.length, 
      Normal: 0, 
      MCI: 0, 
      AD: 0, 
      Other: 0,
      Male: 0,
      Female: 0,
      ApoeE4: 0,   // e4 carriers
      ApoeNet: 0,  // no e4 allele
      ApoeUnk: 0,  // unknown genotype
      AvgAge: 0,
      OasisCount: 0,
      AdniCount: 0,
      SyntheticCount: 0
    };
    
    let totalAge = 0;
    let ageCount = 0;
    
    filteredSubjects.forEach(s => {
      // Cohort
      if (s.cohort === 'ADNI') counts.AdniCount++;
      else if (s.cohort === 'Synthetic') counts.SyntheticCount++;
      else counts.OasisCount++;

      // Diagnostic category
      const cat = s.latestCategory;
      if (counts[cat] !== undefined) {
        counts[cat]++;
      } else {
        counts.Other++;
      }
      
      // Sex
      if (s.gender === 'Male') counts.Male++;
      else if (s.gender === 'Female') counts.Female++;
      
      // APOE e4 carrier: the genotype contains a 4
      if (s.apoe && s.apoe !== 'Unknown' && s.apoe !== '') {
        if (s.apoe.includes('4')) {
          counts.ApoeE4++;
        } else {
          counts.ApoeNet++;
        }
      } else {
        counts.ApoeUnk++;
      }
      
      // Age
      const ageVal = parseFloat(s.age);
      if (!isNaN(ageVal)) {
        totalAge += ageVal;
        ageCount++;
      }
    });
    
    counts.AvgAge = ageCount > 0 ? (totalAge / ageCount).toFixed(1) : 'N/A';
    return counts;
  }, [filteredSubjects]);

  if (generatedData.length === 0) {
    return (
      <div className="app-container" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <div className="glass-panel" style={{ padding: '40px', textAlign: 'center' }}>
          <Brain size={64} style={{ opacity: 0.5, marginBottom: '20px' }} />
          <h2>No Data Found</h2>
          <p style={{ color: 'var(--text-muted)', marginTop: '10px' }}>Waiting for Freesurfer data downloads to complete...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Sidebar: subject list and filters */}
      <aside className="sidebar glass-panel" role="complementary" aria-label="Subject Selector & Cohort Filters">
        <div className="sidebar-header">
          <h1>ALZHEIMER XAI CDSS</h1>
          <div className="sidebar-subtitle">
            <Database size={12} color="#38bdf8" />
            <span>Multi-Cohort • {stats.Total} Active Subjects</span>
          </div>
        </div>
        
        {/* Filters */}
        <div className="filter-section">
          <div className="search-box">
            <Search size={16} color="var(--text-muted)" />
            <input 
              type="text" 
              placeholder="Search Subject ID..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              aria-label="Search Subject ID"
            />
          </div>

          {/* Cohort filter */}
          <div className="filter-row" style={{ marginTop: '8px' }}>
            <div className="filter-group" style={{ width: '100%' }}>
              <label><Database size={12}/> Cohort Source</label>
              <select value={cohortFilter} onChange={(e) => setCohortFilter(e.target.value)} aria-label="Filter by Cohort Dataset">
                <option value="All">All Cohorts (OASIS-3 & ADNI)</option>
                <option value="OASIS-3">OASIS-3 (1,316 Subjects)</option>
                <option value="ADNI">ADNI 1.5T (333 Subjects)</option>
                <option value="Synthetic">Synthetic demo (BrainWeb phantoms, 4 dummy patients)</option>
              </select>
            </div>
          </div>
          
          <div className="filter-row" style={{ marginTop: '8px' }}>
            <div className="filter-group">
              <label><User size={12}/> Gender</label>
              <select value={genderFilter} onChange={(e) => setGenderFilter(e.target.value)} aria-label="Filter by Gender">
                <option value="All">All Genders</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
            
            <div className="filter-group">
              <label><Stethoscope size={12}/> Diagnosis</label>
              <select value={diagnosisFilter} onChange={(e) => setDiagnosisFilter(e.target.value)} aria-label="Filter by Diagnosis">
                <option value="All">All Status</option>
                <option value="Normal">Normal (CN)</option>
                <option value="MCI">MCI</option>
                <option value="AD">AD / Dementia</option>
              </select>
            </div>
          </div>
          <div className="filter-row" style={{ marginTop: '8px' }}>
            <div className="filter-group">
              <label><Activity size={12}/> CDR</label>
              <select value={cdrFilter} onChange={(e) => setCdrFilter(e.target.value)} aria-label="Filter by CDR Score">
                <option value="All">All CDR</option>
                <option value="0">0 (Normal)</option>
                <option value="0.5">0.5 (Questionable)</option>
                <option value="1">1 (Mild)</option>
                <option value="2">2 (Moderate)</option>
                <option value="3">3 (Severe)</option>
                <option value="N/A">N/A</option>
              </select>
            </div>
            
            <div className="filter-group">
              <label><Brain size={12}/> MMSE</label>
              <select value={mmseFilter} onChange={(e) => setMmseFilter(e.target.value)} aria-label="Filter by MMSE Range">
                <option value="All">All MMSE</option>
                <option value="Normal (24-30)">24-30 (Normal)</option>
                <option value="Mild (10-23)">10-23 (Mild/Moderate)</option>
                <option value="Severe (0-9)">0-9 (Severe)</option>
                <option value="N/A">N/A</option>
              </select>
            </div>
          </div>
        </div>

        <div className="subject-list" role="listbox" aria-label="Subjects List">
          {filteredSubjects.length === 0 ? (
            <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
              No subjects match the filter.
            </div>
          ) : (
            filteredSubjects.map((subject) => (
            <div 
              key={subject.id}
              role="option"
              aria-selected={activeSubject.id === subject.id}
              className={`subject-item ${activeSubject.id === subject.id ? 'active' : ''}`}
              onClick={() => handleSubjectClick(subject)}
            >
              <div className="subject-icon">
                <User size={18} />
              </div>
              <div className="subject-info">
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <h3>{subject.id}</h3>
                  <span className={`cohort-badge ${subject.cohort === 'ADNI' ? 'adni-badge' : subject.cohort === 'Synthetic' ? 'synthetic-badge' : 'oasis-badge'}`}>
                    {subject.cohort}
                  </span>
                </div>
                <p>Age {subject.age} • {subject.gender}</p>
              </div>
              <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '6px' }}>
                {subject.latestCategory === 'Normal' && <CheckCircle2 size={14} color="#4ade80" title="Cognitively Normal" />}
                {subject.latestCategory === 'MCI' && <AlertCircle size={14} color="#fbbf24" title="Mild Cognitive Impairment" />}
                {subject.latestCategory === 'AD' && <AlertTriangle size={14} color="#f87171" title="Alzheimer's Disease" />}
                <ChevronRight size={16} color="var(--text-muted)" />
              </div>
            </div>
          ))
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content" role="main">
        
        {/* Research-use and safety disclaimer */}
        <div className="medical-disclaimer-banner" role="alert" aria-label="Regulatory Notice">
          <div className="disclaimer-left">
            <span className="disclaimer-badge">CDSS / SAMD RESEARCH</span>
            <ShieldAlert size={14} color="#38bdf8" />
            <span><strong>Clinical Decision Support Protocol:</strong> Investigational research use only. Not for autonomous diagnostic determination. Requires qualified medical review.</span>
          </div>
        </div>

        {/* Cohort dashboard */}
        <div className="dashboard-container glass-panel" role="region" aria-label="Demographic Summary Dashboard">
          <div className="dashboard-title">
            <PieChart size={18} color="var(--primary-color)" />
            <h2>Demographic Cohort Summary</h2>
          </div>
          <div className="dashboard-stats" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '12px' }}>
            <div className="stat-card">
              <span className="stat-value">{stats.Total}</span>
              <span className="stat-label">Total Subjects</span>
            </div>
            <div className="stat-card stat-normal">
              <span className="stat-value" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '5px' }}>
                <CheckCircle2 size={16} /> {stats.Normal}
              </span>
              <span className="stat-label">Normal (CN)</span>
            </div>
            <div className="stat-card stat-mci">
              <span className="stat-value" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '5px' }}>
                <AlertCircle size={16} /> {stats.MCI}
              </span>
              <span className="stat-label">MCI (Transition)</span>
            </div>
            <div className="stat-card stat-ad">
              <span className="stat-value" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '5px' }}>
                <AlertTriangle size={16} /> {stats.AD}
              </span>
              <span className="stat-label">AD / Dementia</span>
            </div>
            <div className="stat-card" style={{ borderLeft: '3px solid #38bdf8', background: 'rgba(56, 189, 248, 0.05)' }}>
              <span className="stat-value" style={{ color: '#38bdf8', fontSize: '15px' }}>
                OASIS: {stats.OasisCount} | ADNI: {stats.AdniCount}{stats.SyntheticCount > 0 ? ` | Synthetic: ${stats.SyntheticCount}` : ''}
              </span>
              <span className="stat-label">Cohort Source</span>
            </div>
            <div className="stat-card" style={{ borderLeft: '3px solid #60a5fa', background: 'rgba(96, 165, 250, 0.05)' }}>
              <span className="stat-value" style={{ color: '#60a5fa' }}>{stats.AvgAge}</span>
              <span className="stat-label">Average Age</span>
            </div>
            <div className="stat-card" style={{ borderLeft: '3px solid #22d3ee', background: 'rgba(34, 211, 238, 0.05)' }}>
              <span className="stat-value" style={{ color: '#22d3ee', fontSize: '15px' }}>
                M: {stats.Male} | F: {stats.Female}
              </span>
              <span className="stat-label">Gender Ratio</span>
            </div>
            <div className="stat-card" style={{ borderLeft: '3px solid #7dd3fc', background: 'rgba(125, 211, 252, 0.05)' }}>
              <span className="stat-value" style={{ color: '#7dd3fc', fontSize: '15px' }}>
                Carrier: {stats.ApoeE4}
              </span>
              <span className="stat-label">APOE ε4 Carrier</span>
            </div>
          </div>
        </div>

        {/* Imaging viewer */}
        <div className="viewer-container glass-panel" role="region" aria-label="Medical Imaging Viewport">
          
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
            <div className="mode-toggle" style={{ display: 'flex', background: 'rgba(0,0,0,0.3)', borderRadius: '20px', padding: '4px' }}>
              <button
                className={`toggle-btn ${viewMode === '3D' ? 'active' : ''}`}
                onClick={() => setViewMode('3D')}
                style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 16px', border: 'none', background: viewMode === '3D' ? 'var(--primary-color)' : 'transparent', color: viewMode === '3D' ? '#fff' : 'var(--text-muted)', borderRadius: '16px', cursor: 'pointer', transition: '0.2s', fontSize: '13px', fontWeight: '500' }}
              >
                <Brain size={16} /> 3D NiiVue
              </button>
              <button
                className={`toggle-btn ${viewMode === '2D' ? 'active' : ''}`}
                onClick={() => setViewMode('2D')}
                style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 16px', border: 'none', background: viewMode === '2D' ? 'var(--primary-color)' : 'transparent', color: viewMode === '2D' ? '#fff' : 'var(--text-muted)', borderRadius: '16px', cursor: 'pointer', transition: '0.2s', fontSize: '13px', fontWeight: '500' }}
              >
                <Layers size={16} /> 2D DICOM Slices
              </button>
            </div>
            <button
              className="roi-ref-btn"
              onClick={() => setShowRoiModal(true)}
              title="Anatomical ROI reference for Alzheimer's disease (CN vs AD, FreeSurfer segmentation)"
              aria-label="Open anatomical ROI reference"
            >
              <Info size={14} /> Anatomical ROI
            </button>
          </div>

          {/* Viewer content */}
          {activeSession ? (
            viewMode === '3D' ? (
              <NiiVueViewer key={activeSession.url} volumeUrl={activeSession.url} maskUrl={activeSession.maskUrl} />
            ) : (
              <SliceViewer2D sliceUrls={activeSession.sliceUrls} cohortName={activeSubject?.cohort} onSwitchTo3D={() => setViewMode('3D')} />
            )
          ) : (
            <div className="empty-state">
              <Brain size={48} />
              <p>Select a session to view MRI</p>
            </div>
          )}
          
          <div className="viewer-overlay">
            <h2>{activeSubject.id} ({activeSubject.cohort || 'OASIS-3'}) - {activeSession.type}</h2>
            <p>Session: {activeSession.date} • Mode: {viewMode}</p>
          </div>
        </div>

        {/* Session timeline and diagnosis */}
        <div className="timeline-container glass-panel" style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: '20px', height: 'auto' }}>
          
          <div style={{ flex: 1 }}>
            <h3 className="timeline-title">
              <Clock size={14} style={{ display: 'inline', marginRight: '6px', verticalAlign: '-2px' }} />
              Longitudinal Timeline (Days from Entry)
            </h3>
            <div className="timeline-track">
              {activeSubject.sessions.map((session) => (
                <div 
                  key={session.id}
                  className={`timeline-node ${activeSession.id === session.id ? 'active' : ''}`}
                  onClick={() => setActiveSession(session)}
                >
                  <div className="node-date">Day {session.days}</div>
                  <div className="node-label">{session.id}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Diagnosis side panel */}
          {activeSession && activeSession.diagnosis && (
            <div className="clinical-info-side" style={{ width: '260px', background: 'rgba(0, 0, 0, 0.2)', padding: '15px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
              <h3 className="timeline-title" style={{ marginBottom: '12px' }}>
                <Stethoscope size={14} style={{ display: 'inline', marginRight: '6px', verticalAlign: '-2px' }} />
                Session Diagnostics
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Database size={16} color="var(--primary-color)" />
                  <span><strong>Cohort:</strong> <span className={`cohort-badge ${activeSubject.cohort === 'ADNI' ? 'adni-badge' : activeSubject.cohort === 'Synthetic' ? 'synthetic-badge' : 'oasis-badge'}`}>{activeSubject.cohort || 'OASIS-3'}</span></span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <User size={16} color="var(--primary-color)" />
                  <span><strong>Gender:</strong> {activeSubject.gender}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Clock size={16} color="var(--primary-color)" />
                  <span><strong>Age at Scan:</strong> {activeSubject.age}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {getDiagnosisCategory(activeSession.diagnosis, activeSession.cdr) === 'Normal' ? (
                    <CheckCircle2 size={16} color="#4ade80" />
                  ) : getDiagnosisCategory(activeSession.diagnosis, activeSession.cdr) === 'MCI' ? (
                    <AlertCircle size={16} color="#fbbf24" />
                  ) : (
                    <AlertTriangle size={16} color="var(--danger-color)" />
                  )}
                  <span><strong>Diagnosis:</strong> {activeSession.diagnosis}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Activity size={16} color="var(--accent-color)" />
                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <strong>CDR:</strong> {activeSession.cdr}
                    <span className="tooltip-container">
                      <Info size={13} className="info-icon" />
                      <span className="tooltip-text">
                        <strong>Clinical Dementia Rating (CDR)</strong><br />
                        Cognitive impairment severity scale:<br />
                        • <strong>0</strong>: Cognitively Normal (CN)<br />
                        • <strong>0.5</strong>: Very Mild / MCI<br />
                        • <strong>1.0</strong>: Mild Dementia (AD)<br />
                        • <strong>2.0</strong>: Moderate Dementia (AD)<br />
                        • <strong>3.0</strong>: Severe Dementia (AD)
                      </span>
                    </span>
                    <span style={{ margin: '0 4px', opacity: 0.3 }}>|</span>
                    <strong>MMSE:</strong> {activeSession.mmse}
                    <span className="tooltip-container">
                      <Info size={13} className="info-icon" />
                      <span className="tooltip-text">
                        <strong>Mini-Mental State Exam (MMSE)</strong><br />
                        Cognitive function score (0–30 scale):<br />
                        • <strong>24–30</strong>: Normal Cognition<br />
                        • <strong>20–23</strong>: Mild Cognitive Impairment (MCI)<br />
                        • <strong>10–19</strong>: Moderate Impairment<br />
                        • <strong>&lt;10</strong>: Severe Cognitive Decline
                      </span>
                    </span>
                  </span>
                </div>
              </div>
              
              <button 
                className="ai-analyze-btn"
                onClick={() => runAIAnalysis()}
                disabled={isAnalyzing}
                aria-label="Run MobileViT AI Analysis"
              >
                {isAnalyzing ? <span className="spinner"></span> : <Brain size={16} />}
                {isAnalyzing ? "Analyzing..." : "Run AI Analysis"}
              </button>
            </div>
          )}

        </div>

        {/* CDSS result modal */}
        {showAIModal && (
          <div className="modal-backdrop" onClick={closeAIModal} role="dialog" aria-modal="true" aria-label="Clinical Decision Support System">
            <div className="modal-content glass-panel" onClick={e => e.stopPropagation()}>
              <div className="modal-header">
                <div>
                  <h2>
                    <Brain size={22} color="var(--primary-color)" /> 
                    MobileViT Clinical Decision Support System (CDSS)
                  </h2>
                  {aiResult && !aiResult.error && (
                    <div className="modal-header-meta">
                      <span><strong>Session:</strong> {aiResult.session_id}</span>
                      <span>•</span>
                      <span><strong>Subject:</strong> {aiResult.patient_id} ({aiResult.demographics?.gender || activeSubject?.gender}, {aiResult.demographics?.age || activeSubject?.age} yrs)</span>
                      <span>•</span>
                      <span className="cdss-badge model"><Cpu size={12} /> {aiResult.device_used?.toUpperCase() || 'CPU'} ({aiResult.latency_ms || 0}ms)</span>
                      <span className="cdss-badge valid tooltip-container">
                        <ShieldCheck size={12} /> HL7 FHIR R4 Valid
                        <span className="tooltip-text">
                          This transaction bundle is fully compliant with standard HL7 FHIR R4 servers (HAPI FHIR,
                          Google Cloud Healthcare, hospital EHRs) and ready for immediate export.
                        </span>
                      </span>
                      <span className="cdss-badge investigational tooltip-container">
                        <ShieldAlert size={12} /> SaMD — IMDRF N12:2014, Category II.i
                        <span className="tooltip-text">
                          <strong>Clinical Guidance &amp; Safety Advisory:</strong> This output serves as a Clinical
                          Decision Support System (CDSS) for investigational research. AI predictions and
                          HiResCAM/Integrated-Gradients saliency maps must be corroborated with complete medical
                          history, comprehensive neuropsychological assessment (CDR, MMSE, MoCA), and certified
                          radiological/neurological review.
                        </span>
                      </span>
                      {(() => {
                        const ood = aiResult.out_of_domain;
                        const oodKnown = !!ood?.available;
                        const inDomain = ood?.in_validated_domain;
                        return (
                          <button
                            type="button"
                            className={`cdss-badge rules-tab-trigger ${cdssTab === 'rules' ? 'active' : ''} ${oodKnown ? (inDomain ? 'ood-ok' : 'ood-warn') : ''}`}
                            onClick={() => setCdssTab('rules')}
                            title={oodKnown
                              ? `Out-of-domain gate: Mahalanobis distance ${ood.distance} vs. threshold ${ood.threshold} -- click for the decision rule, reliability band & calibration caveat too`
                              : 'Decision rule, reliability band, calibration caveat & domain validation'}
                          >
                            {oodKnown
                              ? (inDomain ? <ShieldCheck size={12} /> : <ShieldAlert size={12} />)
                              : <Info size={12} />}
                            {oodKnown
                              ? `OOD gate: ${inDomain ? 'within range' : 'OUTSIDE range'}`
                              : 'Rules & Validation'}
                          </button>
                        );
                      })()}
                    </div>
                  )}
                </div>
                <button className="close-btn" onClick={closeAIModal} aria-label="Close CDSS modal">×</button>
              </div>

              <div className="modal-body">
                {isAnalyzing ? (
                  <div className="loading-state">
                    <div className="ai-spinner"></div>
                    <p style={{ fontSize: '15px', color: '#f1f5f9', fontWeight: 500 }}>Running MobileViT Multi-Slice Inference...</p>
                    <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Analyzing 10 multi-plane MRI slices for {activeSubject?.id} ({activeSession?.id})</p>
                    {xaiMethod === 'ig' && igStatus !== 'ready' && (
                      <p style={{ fontSize: '12px', color: '#fbbf24' }}>Computing Integrated Gradients (slower than HiResCAM; cached afterwards)</p>
                    )}
                  </div>
                ) : aiResult?.error ? (
                  <div className="error-state" style={{ padding: '30px 20px', textAlign: 'center' }}>
                    <AlertTriangle size={44} color="var(--danger-color)" style={{ marginBottom: '12px' }} />
                    <p style={{ fontSize: '16px', fontWeight: 600, color: '#f87171' }}>{aiResult.error}</p>
                    <div style={{ maxWidth: '520px', margin: '15px auto', padding: '14px', background: 'rgba(15, 23, 42, 0.7)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)', textAlign: 'left', fontSize: '12px', color: '#cbd5e1', lineHeight: 1.6 }}>
                      <strong style={{ color: '#e2e8f0', display: 'inline-flex', alignItems: 'center', gap: '6px' }}><Wrench size={13} /> Clinical Troubleshooting Guide:</strong>
                      <ol style={{ paddingLeft: '18px', marginTop: '6px' }}>
                        <li>Ensure the FastAPI backend service is running (<code>python backend/app.py</code> on port 8000).</li>
                        <li>Verify that multi-plane 2D slices have been extracted into <code>slice_cache/</code> for this session.</li>
                        <li>Check local network connectivity between Vite frontend (port 5173) and the inference backend.</li>
                      </ol>
                    </div>
                    <button 
                      className="action-btn-primary" 
                      onClick={() => runAIAnalysis()}
                      style={{ margin: '10px auto 0', display: 'inline-flex' }}
                    >
                      <RefreshCw size={13} /> Retry Analysis
                    </button>
                  </div>
                ) : aiResult ? (
                  <div className="cdss-grid">

                    {/* Main tabbed panel: Diagnosis & XAI, Clinical Summary, FHIR R4 JSON, Rules & Validation, Conformance & Provenance */}
                    <div className="cdss-right-panel">

                      {/* Tab header */}
                      <div className="cdss-tabs-header">
                        <div className="tab-buttons">
                          <button 
                            className={`cdss-tab-btn ${cdssTab === 'diagnosis' ? 'active' : ''}`}
                            onClick={() => setCdssTab('diagnosis')}
                          >
                            <Brain size={14} /> Diagnosis & XAI
                          </button>
                          <button 
                            className={`cdss-tab-btn ${cdssTab === 'summary' ? 'active' : ''}`}
                            onClick={() => setCdssTab('summary')}
                          >
                            <FileText size={14} /> Clinical Summary
                          </button>
                          <button 
                            className={`cdss-tab-btn ${cdssTab === 'json' ? 'active' : ''}`}
                            onClick={() => setCdssTab('json')}
                          >
                            <Sparkles size={14} /> FHIR R4 JSON
                          </button>
                          <button 
                            className={`cdss-tab-btn ${cdssTab === 'rules' ? 'active' : ''}`}
                            onClick={() => setCdssTab('rules')}
                          >
                            <Info size={14} /> Rules & Validation
                          </button>
                          <button 
                            className={`cdss-tab-btn ${cdssTab === 'conformance' ? 'active' : ''}`}
                            onClick={() => setCdssTab('conformance')}
                          >
                            <ShieldCheck size={14} /> Conformance & Provenance
                          </button>
                        </div>

                        <div className="tab-actions">
                          <button className="action-btn-secondary" onClick={copyToClipboard}>
                            {copied ? <Check size={13} color="#4ade80" /> : <Copy size={13} />}
                            {copied ? 'Copied!' : 'Copy JSON'}
                          </button>
                          <button className="action-btn-primary" onClick={downloadFHIRBundle}>
                            <Download size={13} /> Download FHIR (.json)
                          </button>
                        </div>
                      </div>

                      {/* Tab content */}
                      <div className="tab-content-body">

                        {/* Tab: Diagnosis & XAI */}
                        {cdssTab === 'diagnosis' && (
                          <div className="cdss-left-panel">
                      
                      {/* Clinical covariates: read-only context, not model inputs */}
                      <div className="multimodal-controls-panel">
                        <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                          Clinical Covariates (context only)
                        </span>

                        <div className="multimodal-covariate-readout">
                          {(() => {
                            const mm = aiResult.multimodal_inputs_used || {};
                            const rows = [
                              ['MMSE', mm.mmse_available ? mm.mmse_score : '—'],
                              ['APOE ε4', mm.apoe_available ? mm.apoe_genotype : '—'],
                              ['CDR', activeSession?.cdr ?? '—'],
                              ['Age / Sex', `${mm.age ?? activeSubject?.age ?? '—'} / ${mm.gender ?? activeSubject?.gender ?? '—'}`],
                            ];
                            return rows.map(([k, v]) => (
                              <span key={k} className="mm-cov-chip"><span className="mm-cov-k">{k}</span> {v}</span>
                            ));
                          })()}
                          <span className="mm-cov-note">
                            On file · recorded in the FHIR bundle for the reviewing clinician · <strong>not fed to the image-only model</strong>.
                          </span>
                        </div>

                        {/* Decision rule: fixed to image-only argmax; the calibrated thresholds are shown for comparison only */}
                        <div style={{ marginTop: '4px' }}>
                          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>Decision Rule (fixed):</div>
                          <div className="decision-rule-compare">
                            <div className="drc-row drc-primary">
                              <span className="drc-tag">Argmax · primary</span>
                              <strong>{aiResult.diagnosis_argmax || aiResult.diagnosis}</strong>
                            </div>
                            <div className="drc-row">
                              <span className="drc-tag">Calibrated T_AD={(aiResult.thresholds_used?.AD ?? 0.25).toFixed(2)} / T_MCI={(aiResult.thresholds_used?.MCI ?? 0.33).toFixed(2)} · comparison only</span>
                              <strong>{aiResult.diagnosis_calibrated || '—'}</strong>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Diagnosis and attention bar-chart cards, side by side */}
                      <div className="cdss-diag-attention-row">
                      <div className="cdss-card">
                        <div className="cdss-card-title">
                          <span>Primary Diagnosis (Argmax, image-only)</span>
                          <span>Confidence</span>
                        </div>

                        {(() => {
                          const activeDiag = aiResult.diagnosis_argmax || aiResult.diagnosis;
                          const confVal = (aiResult.probabilities?.[activeDiag] || aiResult.confidence) * 100;
                          const ci = CLASS_TESTFOLD_CI[activeDiag];
                          return (
                            <div className={`diagnosis-badge-large ${activeDiag.toLowerCase()}`}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                {activeDiag === 'CN' && <CheckCircle2 size={28} color="#4ade80" />}
                                {activeDiag === 'MCI' && <AlertCircle size={28} color="#fbbf24" />}
                                {activeDiag === 'AD' && <AlertTriangle size={28} color="#f87171" />}
                                <div>
                                  <div className="diag-title">{activeDiag}</div>
                                  <div className="diag-sub">
                                    {activeDiag === 'CN' ? 'Cognitively Normal' : activeDiag === 'MCI' ? 'Mild Cognitive Impairment' : 'Alzheimer\'s Disease'}
                                  </div>
                                </div>
                              </div>
                              <div className="diag-conf">
                                <div>{confVal.toFixed(1)}%</div>
                                <div style={{ fontSize: '9.5px', fontWeight: 400, opacity: 0.85, color: '#cbd5e1', maxWidth: '190px', textAlign: 'right', lineHeight: 1.35 }}>
                                  {ci
                                    ? `${activeDiag} ${ci.metric} ${ci.point}% · internal test-fold 95% CI ${ci.lo}–${ci.hi}% (bootstrap 2000×)`
                                    : `macro-F1 ${MODEL_MACRO_F1_CI.point}% · 95% CI ${MODEL_MACRO_F1_CI.lo}–${MODEL_MACRO_F1_CI.hi}%`}
                                </div>
                              </div>
                            </div>
                          );
                        })()}

                        <div className="cdss-card-title" style={{ marginTop: '16px', marginBottom: '8px' }}>
                          <span>Probability Distribution (3-Class)</span>
                        </div>

                        <div className="prob-list">
                          {/* CN */}
                          <div className="prob-row">
                            <div className="prob-label-row">
                              <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                <CheckCircle2 size={12} color="#4ade80" /> Cognitively Normal (CN)
                              </span>
                              <strong>{((aiResult.probabilities?.CN || 0) * 100).toFixed(1)}%</strong>
                            </div>
                            <div className="prob-track">
                              <div className="prob-fill cn" style={{ width: `${(aiResult.probabilities?.CN || 0) * 100}%` }}></div>
                            </div>
                          </div>

                          {/* MCI */}
                          <div className="prob-row">
                            <div className="prob-label-row">
                              <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                <AlertCircle size={12} color="#fbbf24" /> Mild Cognitive Impairment (MCI)
                              </span>
                              <strong>{((aiResult.probabilities?.MCI || 0) * 100).toFixed(1)}%</strong>
                            </div>
                            <div className="prob-track">
                              <div className="prob-fill mci" style={{ width: `${(aiResult.probabilities?.MCI || 0) * 100}%` }}></div>
                            </div>
                          </div>

                          {/* AD */}
                          <div className="prob-row">
                            <div className="prob-label-row">
                              <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                <AlertTriangle size={12} color="#f87171" /> Alzheimer's Disease (AD)
                              </span>
                              <strong>{((aiResult.probabilities?.AD || 0) * 100).toFixed(1)}%</strong>
                            </div>
                            <div className="prob-track">
                              <div className="prob-fill ad" style={{ width: `${(aiResult.probabilities?.AD || 0) * 100}%` }}></div>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Attention breakdown over the 10 slices */}
                      {aiResult.attention_weights && aiResult.attention_weights.length > 0 && (
                        <div className="cdss-card">
                          <div className="cdss-card-title">
                            <span>10-Slice Multi-Plane Attention</span>
                            <span style={{ fontSize: '11px', textTransform: 'none', color: '#34d399' }}>
                              Top: {aiResult.top_slice?.name} ({aiResult.top_slice?.attention_pct}%)
                            </span>
                          </div>
                          <div className="attention-grid">
                            {aiResult.attention_weights.map((item, idx) => {
                              const planeType = item.plane.toLowerCase().includes('axial') 
                                ? 'axial' 
                                : item.plane.toLowerCase().includes('coronal') 
                                ? 'coronal' 
                                : 'sagittal';
                              return (
                                <div key={idx} className="attention-row">
                                  <span className="attention-label">{item.plane}</span>
                                  <div className="attention-track">
                                    <div 
                                      className={`attention-fill ${planeType}`} 
                                      style={{ width: `${Math.min(100, item.percentage * 3)}%` }}
                                    ></div>
                                  </div>
                                  <span className="attention-pct">{item.percentage}%</span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      </div>

                      {/* Explainability card: 10-slice panel and HiResCAM / Integrated Gradients toggle */}
                      <div className="cdss-card">
                        <div className="cdss-card-title">
                          <span>Explainable AI ({xaiMethod === 'ig' ? 'Integrated Gradients' : 'HiResCAM'})</span>
                          <span style={{ fontSize: '11px', textTransform: 'none', color: '#38bdf8' }}>
                            Slice {selectedSliceIdx + 1}/10 ({aiResult.slice_heatmaps?.[selectedSliceIdx]?.plane || aiResult.top_slice?.name})
                          </span>
                        </div>

                        {/* Method toggle: HiResCAM (default) or Integrated Gradients (detail) */}
                        <div className="xai-mode-selector" style={{ marginBottom: '8px' }}>
                          <button
                            className={`xai-mode-btn ${xaiMethod === 'hirescam' ? 'active' : ''}`}
                            onClick={() => switchXaiMethod('hirescam')}
                            disabled={isAnalyzing}
                          >
                            <Sparkles size={13} /> HiResCAM (Fast)
                          </button>
                          <button
                            className={`xai-mode-btn ${xaiMethod === 'ig' ? 'active' : ''}`}
                            onClick={requestIgView}
                            disabled={isAnalyzing}
                            title={{
                              ready: 'Cached: opens instantly',
                              queued: 'Computed in the background once the narrative finishes',
                              computing: 'Being computed in the background',
                            }[igStatus] || 'Computed when opened (slower than HiResCAM)'}
                          >
                            <Cpu size={13} /> Integrated Gradients (Detail{{
                              ready: ', ready',
                              queued: ', queued',
                              computing: ', computing…',
                            }[igStatus] || ''})
                          </button>
                        </div>

                        {/* Status of an IG view requested before its background computation finished */}
                        {igPendingOpen && xaiMethod !== 'ig' && (
                          <div style={{ fontSize: '11px', marginBottom: '8px', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span className="spinner" style={{ width: '10px', height: '10px' }}></span>
                            {igStatus === 'queued'
                              ? 'Integrated Gradients is queued: it starts after the narrative finishes and opens automatically.'
                              : 'Computing Integrated Gradients in the background; the view opens automatically when ready.'}
                          </div>
                        )}
                        {igStatus === 'failed' && xaiMethod !== 'ig' && (
                          <div style={{ fontSize: '11px', marginBottom: '8px', color: '#f87171' }}>
                            Background Integrated Gradients computation failed; clicking the button computes it directly.
                          </div>
                        )}

                        {/* Integrated Gradients completeness-residual reliability badge */}
                        {xaiMethod === 'ig' && aiResult.ig_completeness_residual != null && (() => {
                          const res = aiResult.ig_completeness_residual;
                          const level = res < 0.10 ? 'good' : res < 0.20 ? 'medium' : 'low';
                          const label = level === 'good' ? 'Map reliability: good' : level === 'medium' ? 'Map reliability: medium' : 'Map reliability: low, interpret with caution';
                          const color = level === 'good' ? '#4ade80' : level === 'medium' ? '#fbbf24' : '#f87171';
                          return (
                            <div style={{ fontSize: '11px', marginBottom: '8px', color, display: 'flex', alignItems: 'center', gap: '4px' }}>
                              <span>● {label} (completeness residual {res.toFixed(3)})</span>
                              <span className="tooltip-container">
                                <Info size={12} className="info-icon" />
                                <span className="tooltip-text">
                                  <strong>Completeness residual</strong> checks Integrated Gradients' completeness axiom:
                                  the attributions should sum to exactly the model's output change from baseline to this
                                  scan. The residual is how far off that sum is (0 = perfect). A high value (like this one)
                                  means the map's attributions don't fully account for the prediction -- treat which
                                  regions it highlights as less trustworthy for this particular scan, and lean more on
                                  HiResCAM or the on-target % instead.
                                </span>
                              </span>
                            </div>
                          );
                        })()}

                        {/* Slice thumbnails grouped by plane, with on-target badges */}
                        {aiResult.slice_heatmaps && aiResult.slice_heatmaps.length > 0 && (
                          <div className="xai-slice-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '6px', marginBottom: '10px' }}>
                            {aiResult.slice_heatmaps.map((sh, idx) => (
                              <button
                                key={idx}
                                onClick={() => setSelectedSliceIdx(idx)}
                                title={`${sh.plane} — attention ${sh.attention_pct}%${sh.on_target_pct != null ? `, on-target ${sh.on_target_pct}%` : ''}`}
                                style={{
                                  position: 'relative', padding: 0, border: idx === selectedSliceIdx ? '2px solid #38bdf8' : '1px solid rgba(255,255,255,0.12)',
                                  borderRadius: '6px', overflow: 'hidden', cursor: 'pointer', background: '#000',
                                }}
                              >
                                <img src={sh.overlay_base64} alt={sh.plane} style={{ width: '100%', display: 'block', aspectRatio: '1/1', objectFit: 'cover' }} />
                                <span style={{ position: 'absolute', bottom: 0, left: 0, right: 0, fontSize: '8.5px', background: 'rgba(0,0,0,0.7)', color: '#e2e8f0', padding: '1px 2px', textAlign: 'center' }}>
                                  {sh.plane} · {sh.attention_pct}%
                                </span>
                                {sh.on_target_pct != null && (
                                  <span style={{ position: 'absolute', top: 2, right: 2, fontSize: '8px', background: 'rgba(15,23,42,0.85)', color: '#4ade80', padding: '1px 4px', borderRadius: '4px' }}>
                                    {sh.on_target_pct}%
                                  </span>
                                )}
                              </button>
                            ))}
                          </div>
                        )}

                        <div className="xai-interactive-panel">
                          {/* Display mode selector */}
                          <div className="xai-mode-selector">
                            <button
                              className={`xai-mode-btn ${xaiMode === 'overlay' ? 'active' : ''}`}
                              onClick={() => setXaiMode('overlay')}
                            >
                              <Layers size={13} /> Overlay Heatmap
                            </button>
                            <button
                              className={`xai-mode-btn ${xaiMode === 'raw' ? 'active' : ''}`}
                              onClick={() => setXaiMode('raw')}
                            >
                              <Eye size={13} /> Raw MRI Slice
                            </button>
                            <button
                              className={`xai-mode-btn ${xaiMode === 'split' ? 'active' : ''}`}
                              onClick={() => setXaiMode('split')}
                            >
                              <SplitSquareVertical size={13} /> Side-by-Side
                            </button>
                          </div>

                          {/* Opacity slider (overlay mode) */}
                          {xaiMode === 'overlay' && (
                            <div className="xai-opacity-bar">
                              <span>Heatmap Opacity:</span>
                              <input
                                type="range"
                                min="0"
                                max="100"
                                value={Math.round(xaiOpacity * 100)}
                                onChange={(e) => setXaiOpacity(parseFloat(e.target.value) / 100)}
                                className="xai-opacity-slider"
                                aria-label="Heatmap Opacity"
                              />
                              <strong style={{ minWidth: '36px', textAlign: 'right', color: '#38bdf8' }}>
                                {Math.round(xaiOpacity * 100)}%
                              </strong>
                            </div>
                          )}

                          {(() => {
                            const selected = aiResult.slice_heatmaps?.[selectedSliceIdx];
                            const overlaySrc = selected?.overlay_base64 || aiResult.gradcam_base64;
                            const rawSrc = aiResult.raw_slice_base64 || overlaySrc;
                            if (xaiMode === 'overlay') {
                              return (
                                <div className="xai-canvas-stack">
                                  <img src={rawSrc} alt="Base Anatomical MRI" className="base-mri" />
                                  <img src={overlaySrc} alt="XAI Heatmap Overlay" className="cam-layer" style={{ opacity: xaiOpacity }} />
                                </div>
                              );
                            }
                            if (xaiMode === 'raw') {
                              return (
                                <div className="xai-canvas-stack">
                                  <img src={rawSrc} alt="Raw Anatomical MRI" className="base-mri" />
                                </div>
                              );
                            }
                            return (
                              <div className="xai-split-grid">
                                <div className="xai-split-box">
                                  <img src={rawSrc} alt="Raw MRI Slice" />
                                  <span className="xai-split-label">Raw Structural MRI (T1w)</span>
                                </div>
                                <div className="xai-split-box">
                                  <img src={overlaySrc} alt="XAI Activation Map" />
                                  <span className="xai-split-label">{xaiMethod === 'ig' ? 'Integrated Gradients' : 'HiResCAM'} Activation Map</span>
                                </div>
                              </div>
                            );
                          })()}

                          {aiResult.slice_heatmaps?.[selectedSliceIdx]?.on_target_pct == null && (
                            <p style={{ fontSize: '10.5px', color: 'var(--text-muted)', marginTop: '4px' }}>
                              No FreeSurfer/FastSurfer segmentation available for this session — on-target badge hidden.
                            </p>
                          )}

                          {/* Differential explanation comparison */}
                          <div style={{ marginTop: '10px' }}>
                            <button className="action-btn-secondary" onClick={runDifferentialView} disabled={diffLoading}>
                              <SplitSquareVertical size={13} /> {diffLoading ? 'Comparing classes…' : 'Compare CN vs MCI vs AD explanation'}
                            </button>
                          </div>
                          {showDiffView && diffResults && !diffResults.error && (
                            <div className="xai-split-grid" style={{ marginTop: '8px', gridTemplateColumns: 'repeat(3, 1fr)' }}>
                              {['CN', 'MCI', 'AD'].map((cls) => (
                                <div className="xai-split-box" key={cls}>
                                  <img src={diffResults[cls]?.slice_heatmaps?.[selectedSliceIdx]?.overlay_base64} alt={`${cls} explanation`} />
                                  <span className="xai-split-label">Target: {cls}{cls === aiResult.diagnosis ? ' (predicted)' : ''}</span>
                                </div>
                              ))}
                            </div>
                          )}

                          {(() => {
                            const sel = aiResult.slice_heatmaps?.[selectedSliceIdx];
                            const method = xaiMethod === 'ig' ? 'Integrated Gradients' : 'HiResCAM';
                            const pred = aiResult.diagnosis_argmax || aiResult.diagnosis;
                            return (
                              <p className="xai-caption">
                                <strong>Visual Interpretation:</strong> {method} attribution for the predicted class <strong>{pred}</strong>,
                                pooled across all 10 slices by the gated-attention weights.
                                {sel && <> Shown: <strong>{sel.plane} (slice {selectedSliceIdx + 1}/10)</strong>, which carries{' '}
                                <strong>{sel.attention_pct}%</strong> of the attention mass
                                {sel.on_target_pct != null
                                  ? <> and lands <strong>{sel.on_target_pct}%</strong> of its {method} energy inside the combined
                                     AD-pathology reference from the FreeSurfer/FastSurfer <em>aseg</em>: hippocampus (labels 17/53),
                                     amygdala (18/54) and the ventricular system — lateral, temporal-horn and 3rd ventricle
                                     (4/43/5/44/14). The same reference mask is applied to every slice; structures a given slice
                                     does not intersect simply contribute nothing.</>
                                  : <>; no FreeSurfer/FastSurfer segmentation for this session, so no on-target figure.</>}
                                </>}
                                {' '}All numbers are per-case measurements for this session — not fixed anatomical claims.
                              </p>
                            );
                          })()}
                        </div>
                      </div>

                          </div>
                        )}

                        {/* Tab: Clinical Summary */}
                        {cdssTab === 'summary' && (
                          <div>
                            <table className="clinical-summary-table">
                              <tbody>
                                <tr>
                                  <td>Patient Identifier</td>
                                  <td><strong>{aiResult.patient_id}</strong> ({activeSubject?.cohort === 'Synthetic' ? 'Synthetic demo patient, not real data' : activeSubject?.cohort === 'ADNI' ? 'ADNI Subject' : 'OASIS-3 Subject'})</td>
                                </tr>
                                <tr>
                                  <td>MRI Scan Session</td>
                                  <td><code>{aiResult.session_id}</code> ({aiResult.slice_count} Multi-Plane Slices)</td>
                                </tr>
                                <tr>
                                  <td>Anatomic Region & Modality</td>
                                  <td>
                                    <span className="code-pill">SNOMED 12738006</span> Brain Structure &bull; 
                                    <span className="code-pill" style={{ marginLeft: '6px' }}>DCM MR</span> Magnetic Resonance
                                  </td>
                                </tr>
                                <tr>
                                  <td>LOINC Observation Code</td>
                                  <td>
                                    <span className="code-pill">LOINC 24590-2</span> MR Brain
                                  </td>
                                </tr>
                                <tr>
                                  <td>SNOMED CT Diagnosis</td>
                                  <td>
                                    {aiResult.diagnosis === 'CN' && <><span className="code-pill">449888003</span> Normal cognition</>}
                                    {aiResult.diagnosis === 'MCI' && <><span className="code-pill">386805003</span> Mild neurocognitive disorder</>}
                                    {aiResult.diagnosis === 'AD' && <><span className="code-pill">26929004</span> Alzheimer's disease</>}
                                  </td>
                                </tr>
                                <tr>
                                  <td>AI Classification Model</td>
                                  <td>
                                    <span className="code-pill">SNOMED 706689003</span> MobileViT 2.5D Multi-Plane Vision Transformer (v2.0.0)
                                  </td>
                                </tr>
                                <tr>
                                  <td>Diagnostic Conclusion</td>
                                  <td style={{ lineHeight: 1.5, color: '#e2e8f0' }}>
                                    {aiResult.fhir_bundle?.entry?.find(e => e.resource.resourceType === 'DiagnosticReport')?.resource?.conclusion || 
                                     `AI predicts ${aiResult.diagnosis} with ${(aiResult.confidence * 100).toFixed(1)}% confidence.`}
                                  </td>
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        )}

                        {/* Tab: FHIR R4 Raw JSON Viewer */}
                        {cdssTab === 'json' && (
                          <div className="json-viewer-container">
                            <pre>{JSON.stringify(aiResult.fhir_bundle, null, 2)}</pre>
                          </div>
                        )}

                        {/* Tab: Rules & Validation (decision rule, reliability band, calibration caveat, out-of-domain scan) */}
                        {cdssTab === 'rules' && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div className="cdss-card" style={{ padding: '10px 14px', fontSize: '11.5px', lineHeight: 1.5, color: '#cbd5e1', background: 'rgba(56,189,248,0.06)', border: '1px solid rgba(56,189,248,0.18)' }}>
                        <div>
                          <strong style={{ color: '#38bdf8' }}>Active decision rule:</strong> {aiResult.decision_rule_active === 'argmax' ? 'Argmax (primary)' : (aiResult.decision_rule_active || 'argmax (primary)')} —
                          the reported <strong>Primary Diagnosis</strong> (and the explanation heatmap) always uses this rule, image-only
                          (the clinical covariates are informational, not fed to the model — see <em>Clinical Covariates</em>).
                          The calibrated-threshold class (AD ≥ {(aiResult.thresholds_used?.AD ?? 0.25).toFixed(2)}, MCI ≥ {(aiResult.thresholds_used?.MCI ?? 0.33).toFixed(2)}) is shown in the <em>Decision Rule</em> box for comparison only and never changes the reported diagnosis or the heatmap.
                        </div>
                        <div style={{ marginTop: '6px', opacity: 0.85 }}>
                          <strong>Reliability band:</strong> the range next to the confidence value is the predicted class's
                          recall on the 232-session internal test fold (bootstrap 2000×) — a population-level operating
                          characteristic, <em>not</em> a per-patient confidence interval.
                        </div>
                        <div style={{ marginTop: '6px', opacity: 0.85 }}>
                          <strong>Calibration caveat:</strong> probabilities are not temperature-scaled (scaling was found to
                          worsen calibration on this test set) — read them as a relative ranking across CN/MCI/AD, not as an
                          absolute confidence percentage.
                        </div>
                      </div>
                      {aiResult.out_of_domain?.available && (
                        <div
                          className="cdss-card"
                          style={{
                            padding: '10px 14px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '10px',
                            background: aiResult.out_of_domain.in_validated_domain ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)',
                            border: `1px solid ${aiResult.out_of_domain.in_validated_domain ? 'rgba(34,197,94,0.25)' : 'rgba(239,68,68,0.3)'}`,
                          }}
                        >
                          {aiResult.out_of_domain.in_validated_domain
                            ? <ShieldCheck size={18} color="#4ade80" style={{ flexShrink: 0 }} />
                            : <ShieldAlert size={18} color="#f87171" style={{ flexShrink: 0 }} />}
                          <div>
                            <strong style={{ color: aiResult.out_of_domain.in_validated_domain ? '#4ade80' : '#f87171' }}>
                              Domain scan: {aiResult.out_of_domain.in_validated_domain ? 'within validated range' : 'OUTSIDE validated range'}
                            </strong>
                            <div style={{ color: 'var(--text-muted)', marginTop: '2px' }}>
                              Mahalanobis distance {aiResult.out_of_domain.distance} vs. threshold {aiResult.out_of_domain.threshold} (95th pct. of training scans).
                              {' '}{aiResult.out_of_domain.validated_domain_note}
                            </div>
                          </div>
                        </div>
                      )}
                          </div>
                        )}

                        {/* Tab: Conformance & Provenance */}
                        {cdssTab === 'conformance' && (
                          <div className="conformance-grid">
                            <div className="conformance-card">
                              <h4><ShieldCheck size={15} color="#4ade80" /> HL7 FHIR Conformance Status</h4>
                              <p style={{ margin: '4px 0', fontSize: '12px', color: '#cbd5e1' }}>
                                Status: <strong style={{ color: '#4ade80' }}>100% VALID</strong> &bull; Validator: <code>{aiResult.fhir_validation?.validator || 'fhir.resources (HL7 FHIR R4)'}</code>
                              </p>
                              <div className="resource-chip-list">
                                {(aiResult.fhir_validation?.resource_types || ['Patient', 'Device', 'ImagingStudy', 'Observation', 'DocumentReference', 'DiagnosticReport']).map((rt, idx) => (
                                  <span key={idx} className="resource-chip">{rt}</span>
                                ))}
                              </div>
                            </div>

                            <div className="conformance-card">
                              <h4><ShieldCheck size={15} color="#38bdf8" /> Design &amp; Regulatory Alignment</h4>
                              <p style={{ margin: '4px 0 8px', fontSize: '11.5px', color: '#cbd5e1', lineHeight: 1.5 }}>
                                Interface designed with reference to the standards below. These are development references,
                                not certifications; no summative usability study or regulatory clearance has been performed.
                              </p>
                              <div className="resource-chip-list">
                                <span className="resource-chip">FDA CDS Software Guidance (FDA-2017-D-6569)</span>
                                <span className="resource-chip">FDA AI-Enabled Device SW Functions (draft, 2025)</span>
                                <span className="resource-chip">IEC 62366-1:2015</span>
                                <span className="resource-chip">IMDRF N12:2014 — SaMD Category II.i</span>
                                <span className="resource-chip">DICOM PS3.3 / PS3.14 / PS3.17</span>
                                <span className="resource-chip">WCAG 2.1 AA</span>
                              </div>
                            </div>

                            <div className="conformance-card">
                              <h4><Cpu size={15} color="#38bdf8" /> Computational Execution Provenance</h4>
                              <table className="clinical-summary-table" style={{ marginTop: '6px' }}>
                                <tbody>
                                  <tr>
                                    <td>Architecture</td>
                                    <td>MobileViT-Small 2.5D Multi-Plane (10 Slices) + Selective Tabular Fusion</td>
                                  </tr>
                                  <tr>
                                    <td>Weights Checkpoint</td>
                                    <td><code>{aiResult.model_info?.checkpoint_file ?? 'best_model_v2_unified.pt'}</code> ({aiResult.model_info?.checkpoint_loaded ? 'Trained Checkpoint' : 'Pretrained'})</td>
                                  </tr>
                                  <tr>
                                    <td>Input Resolution</td>
                                    <td>224 × 224 RGB (10 Slices: 5 Coronal, 3 Axial, 2 Sagittal)</td>
                                  </tr>
                                  <tr>
                                    <td>Inference Latency</td>
                                    <td>{aiResult.latency_ms} ms on {aiResult.device_used}</td>
                                  </tr>
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                      </div>

                    </div>
                    {/* /cdss-right-panel (main tabbed panel) */}

                    {/* Second column: LLM-generated clinical narrative */}
                    <div className="cdss-narrative-panel cdss-card">
                      <div className="narrative-panel-header">
                        <span className="narrative-panel-title">
                          <Sparkles size={15} color="#a78bfa" /> AI-Generated Clinical Narrative
                        </span>
                        {narrativeResult && (
                          <span
                            className="cdss-badge"
                            style={{ background: 'rgba(167,139,250,0.15)', color: '#c4b5fd', border: '1px solid rgba(167,139,250,0.3)' }}
                          >
                            {narrativeResult.model || 'llama31_8b'} · variant {(narrativeResult.variant || 'c').toUpperCase()}
                          </span>
                        )}
                      </div>

                      {!narrativeResult && !narrativeLoading && !narrativeError && (
                        <div className="narrative-empty-state">
                          <p>
                            Generate a guardrail-audited clinical narrative summarizing this session's imaging
                            findings, covariates, and uncertainty, written by a local LLM from the structured
                            prediction above. Optional -- not required to review the diagnosis, heatmap, or FHIR bundle.
                          </p>
                          <button className="action-btn-primary" onClick={runNarrativeGeneration} style={{ alignSelf: 'flex-start' }}>
                            <Sparkles size={13} /> Generate Narrative
                          </button>
                        </div>
                      )}

                      {narrativeLoading && (
                        <div className="loading-state" style={{ padding: '28px 8px' }}>
                          <div className="ai-spinner" style={{ width: '32px', height: '32px' }}></div>
                          <p style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                            Running local LLM (~20-30s) + deterministic guardrail audit…
                          </p>
                        </div>
                      )}

                      {narrativeError && (
                        <div className="error-state" style={{ padding: '18px 8px', gap: '10px' }}>
                          <AlertTriangle size={26} color="var(--danger-color)" />
                          <p style={{ fontSize: '12px', color: '#f87171', margin: 0 }}>{narrativeError}</p>
                          <p style={{ fontSize: '10.5px', color: 'var(--text-muted)', margin: 0 }}>
                            If this persists, verify the llm_xai router is mounted in backend/app.py and
                            <code> ollama serve</code> is running (see llm_xai/README.md).
                          </p>
                          <button className="action-btn-secondary" onClick={runNarrativeGeneration}>
                            <RefreshCw size={13} /> Retry
                          </button>
                        </div>
                      )}

                      {narrativeResult && narrativeResult.narrative && (
                        <>
                          <div className="narrative-source-row">
                            <span
                              className="cdss-badge"
                              style={
                                narrativeResult.guardrail?.passed && narrativeResult.guardrail?.source !== 'deterministic_fallback'
                                  ? { background: 'rgba(34,197,94,0.15)', color: '#4ade80', border: '1px solid rgba(34,197,94,0.3)' }
                                  : { background: 'rgba(251,191,36,0.15)', color: '#fbbf24', border: '1px solid rgba(251,191,36,0.3)' }
                              }
                            >
                              {narrativeResult.guardrail?.source === 'deterministic_fallback'
                                ? <><ShieldAlert size={12} /> Deterministic fallback (LLM draft rejected)</>
                                : <><ShieldCheck size={12} /> LLM-generated, guardrail-verified</>}
                            </span>
                            <span style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>
                              NCR {(narrativeResult.guardrail?.ncr ?? 1).toFixed(2)}
                              {narrativeResult.llm_meta?.total_duration_ns
                                ? ` · ${(narrativeResult.llm_meta.total_duration_ns / 1e9).toFixed(1)}s`
                                : ''}
                            </span>
                          </div>
                          {xaiMethod === 'ig' && (
                            <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                              The narrative is based on the HiResCAM explanation, not the Integrated Gradients view shown.
                            </div>
                          )}

                          {[
                            ['Visual Findings', narrativeResult.narrative.visual_findings],
                            ['Multimodal Correlation', narrativeResult.narrative.multimodal_correlation],
                            ['Differential Justification', narrativeResult.narrative.differential_justification],
                            ['Differential Conclusion', narrativeResult.narrative.differential_conclusion],
                            ['Uncertainty Note', narrativeResult.narrative.uncertainty_note],
                          ].map(([label, text]) => (
                            text ? (
                              <div key={label} className="narrative-field">
                                <div className="narrative-field-label">{label}</div>
                                <p>{text}</p>
                              </div>
                            ) : null
                          ))}

                          <button className="action-btn-secondary" onClick={runNarrativeGeneration} style={{ marginTop: '2px', alignSelf: 'flex-start' }}>
                            <RefreshCw size={12} /> Regenerate
                          </button>

                          {/* Follow-up Q&A, available once a narrative is shown */}
                          <div className="qa-section">
                            <div className="qa-section-header">
                              <MessageCircle size={13} color="#a78bfa" />
                              <span>Ask a follow-up question</span>
                            </div>

                            {!qaToken && !qaSessionLoading && !qaEnded && (
                              <div className="narrative-empty-state" style={{ padding: '2px 0 0' }}>
                                <p>
                                  Ask about this session's evidence (e.g. MMSE, out-of-domain distance, which
                                  structures the model attended to). Every answer is grounded to the same frozen
                                  facts and independently guardrail-checked before it is shown; questions outside
                                  what this narrative reports (treatment, prognosis, next steps) are declined.
                                </p>
                                <button className="action-btn-secondary" onClick={startQaSession} style={{ alignSelf: 'flex-start' }}>
                                  <MessageCircle size={13} /> Start Q&A Session
                                </button>
                                {qaSessionError && (
                                  <p style={{ fontSize: '11px', color: '#f87171', margin: 0 }}>{qaSessionError}</p>
                                )}
                              </div>
                            )}

                            {qaSessionLoading && (
                              <div className="loading-state" style={{ padding: '14px 8px' }}>
                                <div className="ai-spinner" style={{ width: '22px', height: '22px' }}></div>
                                <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Freezing session facts…</p>
                              </div>
                            )}

                            {qaToken && (
                              <>
                                <div className="qa-messages">
                                  {qaMessages.length === 0 && (
                                    <p style={{ fontSize: '11px', color: 'var(--text-muted)', margin: '0 0 4px' }}>
                                      Session ready -- ask your first question below.
                                    </p>
                                  )}
                                  {qaMessages.map((turn, i) => (
                                    <div key={i} className="qa-turn">
                                      <div className="qa-question">
                                        <Stethoscope size={12} style={{ flexShrink: 0, marginTop: '2px' }} /> {turn.question}
                                      </div>
                                      <div className="qa-answer">
                                        <div className="qa-answer-badges">
                                          <span
                                            className="cdss-badge"
                                            style={
                                              turn.source === 'deterministic_refusal'
                                                ? { background: 'rgba(148,163,184,0.15)', color: '#cbd5e1', border: '1px solid rgba(148,163,184,0.3)' }
                                                : { background: 'rgba(34,197,94,0.15)', color: '#4ade80', border: '1px solid rgba(34,197,94,0.3)' }
                                            }
                                          >
                                            {turn.source === 'deterministic_refusal'
                                              ? <><ShieldAlert size={11} /> Declined (outside grounded evidence)</>
                                              : <><ShieldCheck size={11} /> {turn.source === 'llm_regenerated' ? 'Guardrail-corrected' : 'Guardrail-verified'}</>}
                                          </span>
                                          <span className="qa-category-tag">{turn.category}</span>
                                        </div>
                                        <p>{turn.answer}</p>
                                        {turn.cited_fields && turn.cited_fields.length > 0 && (
                                          <p className="qa-cited-fields">Cited: {turn.cited_fields.join(', ')}</p>
                                        )}
                                      </div>
                                    </div>
                                  ))}
                                </div>

                                {qaAsking && (
                                  <div className="loading-state" style={{ padding: '8px', flexDirection: 'row', gap: '8px' }}>
                                    <div className="ai-spinner" style={{ width: '16px', height: '16px' }}></div>
                                    <p style={{ fontSize: '10.5px', color: 'var(--text-muted)', margin: 0 }}>
                                      Answering + running guardrail audit (up to ~30-40s, may retry once)…
                                    </p>
                                  </div>
                                )}

                                {qaAskError && !qaEnded && (
                                  <p style={{ fontSize: '11px', color: '#f87171', margin: '4px 0' }}>{qaAskError}</p>
                                )}

                                {!qaEnded ? (
                                  <form className="qa-input-row" onSubmit={askQaQuestion}>
                                    <input
                                      type="text"
                                      value={qaInput}
                                      onChange={(e) => setQaInput(e.target.value)}
                                      placeholder="e.g. What is this patient's MMSE score?"
                                      disabled={qaAsking}
                                      className="qa-input"
                                    />
                                    <button type="submit" className="action-btn-primary" disabled={qaAsking || !qaInput.trim()}>
                                      Ask
                                    </button>
                                  </form>
                                ) : (
                                  <div className="narrative-empty-state" style={{ padding: '6px 0 0' }}>
                                    <p style={{ margin: 0 }}>
                                      This session has ended (question limit reached, or the session expired).
                                      Start a new one to keep asking.
                                    </p>
                                    <button className="action-btn-secondary" onClick={startQaSession} style={{ alignSelf: 'flex-start' }}>
                                      <RefreshCw size={13} /> Start New Q&A Session
                                    </button>
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                          {/* /qa-section */}
                        </>
                      )}

                      <div className="cdss-clinical-advisory">
                        <ShieldAlert size={16} color="#a78bfa" style={{ flexShrink: 0, marginTop: '2px' }} />
                        <div>
                          Investigational summary generated by a local LLM ({narrativeResult?.model || 'llama31_8b'}) from
                          the structured prediction above, audited by a deterministic guardrail (numeric range,
                          anatomical whitelist, and contradiction checks). Not a substitute for clinical judgment --
                          always corroborate against the source imaging and FHIR record. Follow-up answers above are
                          grounded to the same frozen facts and separately guardrail-checked before being shown.
                        </div>
                      </div>
                    </div>
                    {/* /cdss-narrative-panel */}

                  </div>
                ) : null}
              </div>
            </div>
          </div>
        )}

        {/* Anatomical ROI reference pop-up */}
        {showRoiModal && (
          <div className="modal-backdrop" onClick={() => setShowRoiModal(false)} role="dialog" aria-modal="true" aria-label="Anatomical ROI Reference">
            <div className="modal-content glass-panel roi-modal" onClick={e => e.stopPropagation()}>
              <div className="modal-header">
                <div>
                  <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Brain size={18} /> Anatomical ROI Reference for Alzheimer's Disease
                  </h2>
                  <p style={{ margin: '4px 0 0', fontSize: '12px', color: 'var(--text-muted)' }}>
                    Synthetic BrainWeb phantoms, FastSurfer <code>aseg.mgz</code> segmentation reduced to the AD-pathology target
                    structures (identical to <em>ad_target</em> in the quantitative XAI localization test).
                  </p>
                </div>
                <button className="close-btn" onClick={() => setShowRoiModal(false)} aria-label="Close ROI reference">×</button>
              </div>

              <div className="modal-body">
                <p className="roi-rationale">
                  The earliest and most characteristic atrophy in typical amnestic AD dementia occurs in the <strong>medial temporal
                  lobe</strong> — the <strong>hippocampus</strong> and <strong>amygdala</strong> — and reads most clearly in the
                  <strong> coronal plane</strong>. That tissue loss is followed by passive (<em>ex vacuo</em>) enlargement of the
                  <strong> lateral ventricle</strong>, especially its <strong>temporal horn</strong>. The <em>entorhinal</em> cortex is
                  the earliest site of pathology but has no separate label in <code>aseg.mgz</code>, so it isn't quantified here.
                  Posterior structures (precuneus, parietal cortex) are also involved in AD, but the model's explanation energy
                  is concentrated in the coronal medial temporal lobe.
                </p>

                <div className="roi-legend">
                  {ROI_LEGEND.map((r) => (
                    <div className="roi-legend-item" key={r.name}>
                      <span className="roi-swatch" style={{ background: r.color }} />
                      <span><strong>{r.name}</strong> — {r.note}</span>
                    </div>
                  ))}
                </div>

                <div className="roi-demo-stack">
                  <div className="roi-demo-panel">
                    <div className="roi-demo-caption">
                      <span className="roi-tag roi-tag-cn">CN · CDR 0</span>
                      BWP01_MR_d0000 (dummy) · age 68 · MMSE 30 &nbsp;—&nbsp;
                      volumes from an OASIS-3 CN sample, not measured on this image: hippocampus <strong>7.6 mL</strong> · temporal horn <strong>1.2 mL</strong> · lateral ventricle <strong>26 mL</strong>
                    </div>
                    <RoiDemoViewer brainUrl="/demo/brain_cn.mgz" roiUrl="/demo/roi_cn.nii.gz" />
                  </div>
                  <div className="roi-demo-panel">
                    <div className="roi-demo-caption">
                      <span className="roi-tag roi-tag-ad">AD · CDR 2</span>
                      BWP04_MR_d0728 (dummy) · age 81 · MMSE 14 &nbsp;—&nbsp;
                      volumes from an OASIS-3 AD sample, not measured on this image: hippocampus <strong>2.7 mL</strong> · temporal horn <strong>6.8 mL</strong> · lateral ventricle <strong>74 mL</strong>
                    </div>
                    <RoiDemoViewer brainUrl="/demo/brain_ad.mgz" roiUrl="/demo/roi_ad.nii.gz" />
                  </div>
                </div>
                <p className="roi-footnote">
                  <em>Multiplanar</em> view + NiiVue 3D render. Drag the panel to rotate/pan the slices. The images are simulated
                  BrainWeb phantoms with normal anatomy and made-up covariates, so neither panel shows atrophy. The structure
                  volumes are typical <code>aseg.mgz</code> voxel-count values (1&nbsp;mm³/voxel) of two OASIS-3 samples, shown
                  only to illustrate the CN versus AD contrast.
                </p>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default App;
