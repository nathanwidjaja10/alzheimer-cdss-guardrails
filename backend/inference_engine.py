import gc
import os
import io
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from transformers import MobileViTForImageClassification
from sklearn.covariance import LedoitWolf

# Paths (BASE_DIR may be overridden through the environment)
BASE_DIR = Path(os.environ.get('BASE_DIR', Path(__file__).resolve().parent.parent))
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
SLICE_CACHE_DIR = BASE_DIR / 'oasis_data' / 'slice_cache'
if not SLICE_CACHE_DIR.exists():
    SLICE_CACHE_DIR = BASE_DIR / 'slice_cache'
ADNI_SLICE_CACHE_DIR = BASE_DIR / 'adni_data' / 'slice_cache'
FREESURFER_DIR = BASE_DIR / 'oasis_data' / 'freesurfer_data'
ADNI_FREESURFER_DIR = BASE_DIR / 'adni_data' / 'freesurfer_data'
# Synthetic demo sessions (BrainWeb phantoms); see dummy_data/README.md
DUMMY_DATA_DIR = BASE_DIR / 'dummy_data'
DUMMY_SLICE_CACHE_DIR = DUMMY_DATA_DIR / 'slice_cache'
DUMMY_FREESURFER_DIR = DUMMY_DATA_DIR / 'freesurfer_data'
XAI_CACHE_DIR = CHECKPOINT_DIR / 'xai_cache'
XAI_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def resolve_slice_cache_dir(session_id: str) -> Path:
    """Return the slice-cache root of the cohort a session belongs to.

    OASIS-3 session ids start with 'OAS3'; ADNI session ids start with 'ADNI_' or contain '_S_'.
    """
    if session_id.startswith('ADNI_') or '_S_' in session_id:
        return ADNI_SLICE_CACHE_DIR
    return SLICE_CACHE_DIR


def _freesurfer_dir_for(session_id: str) -> Path:
    if (DUMMY_FREESURFER_DIR / session_id).is_dir():
        return DUMMY_FREESURFER_DIR
    if session_id.startswith('ADNI_') or '_S_' in session_id:
        return ADNI_FREESURFER_DIR
    return FREESURFER_DIR


def locate_session_cache(session_id: str) -> tuple[Path, str]:
    """Locate the slice-cache directory and canonical session id for a session query.

    Accepts OASIS-3 ids (e.g. OAS30001_MR_d0129) and ADNI ids in any of their spellings
    (ADNI_002_S_0729_sc_I40708, 002_S_0729_MR_sc_I40708, 002_S_0729_sc_I40708).
    Returns (session_dir_path, canonical_session_id).
    """
    clean_id = session_id.strip()

    # Spelling variants of the query, de-duplicated in order
    candidates = [
        clean_id,
        f"ADNI_{clean_id}",
        clean_id.replace('_MR_', '_'),
        f"ADNI_{clean_id.replace('_MR_', '_')}",
        clean_id.replace('ADNI_', ''),
        clean_id.replace('ADNI_', '').replace('_MR_', '_'),
    ]
    seen = set()
    uniq_candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    search_roots = [ADNI_SLICE_CACHE_DIR, SLICE_CACHE_DIR, BASE_DIR / 'slice_cache', DUMMY_SLICE_CACHE_DIR]

    # Exact directory match
    for cand in uniq_candidates:
        for root in search_roots:
            p = root / cand
            if p.is_dir() and any(p.glob('*.png')):
                return p, cand

    # Fuzzy match: substring containment or all id tokens present in the directory name
    for root in search_roots:
        if not root.exists():
            continue
        parts = [part for part in clean_id.split('_') if part not in ('ADNI', 'MR', 'sc')]
        for d in root.iterdir():
            if d.is_dir() and any(d.glob('*.png')):
                d_name = d.name
                if clean_id in d_name or d_name in clean_id:
                    return d, d_name
                if len(parts) >= 2 and all(part in d_name for part in parts):
                    return d, d_name

    # Fall back to the expected location (the caller reports it as missing)
    default_dir = resolve_slice_cache_dir(clean_id) / clean_id
    return default_dir, clean_id


# Clinical metadata of both cohorts, keyed by session id
CLINICAL_PATHS = [
    BASE_DIR / 'oasis_data' / 'clinical_metadata.json',
    BASE_DIR / 'adni_data' / 'clinical_metadata.json',
    DUMMY_DATA_DIR / 'clinical_metadata.json',
]
clinical_meta: Dict[str, Any] = {}
for p in CLINICAL_PATHS:
    if p.exists():
        try:
            with open(p, 'r') as f:
                clinical_meta.update(json.load(f))
        except Exception as e:
            print(f"⚠️ Warning loading clinical metadata from {p}: {e}")

MODEL_NAME = 'apple/mobilevit-small'
CLASS_NAMES = ['CN', 'MCI', 'AD']
LABEL_MAP = {'CN': 0, 'MCI': 1, 'AD': 2}
NUM_CLASSES = 3

N_AXIAL = 5
N_CORONAL = 3
N_SAGITTAL = 2
SLICES_PER_VOL = N_AXIAL + N_CORONAL + N_SAGITTAL  # 10
IMG_SIZE = 224

# Covariate vector layout: (gender, has_gender, age_norm, has_age, mmse_norm, has_mmse, apoe4, has_apoe)
TABULAR_DIM = 8
TABULAR_HIDDEN_DIM = 64

# Decision-layer defaults, used only when the training metadata cannot be read
CALIB_THRESHOLD_AD = 0.25
CALIB_THRESHOLD_MCI = 0.33
OOD_DISTANCE_THRESHOLD: Optional[float] = None
VALIDATED_DOMAIN_NOTE = 'OASIS-3 structural MRI processed with FreeSurfer.'

METADATA_JSON = CHECKPOINT_DIR / 'training_metadata_v2_unified.json'
meta_v2: Dict[str, Any] = {}
if METADATA_JSON.exists():
    try:
        with open(METADATA_JSON, 'r') as f:
            meta_v2 = json.load(f)
        # Calibrated thresholds and out-of-domain settings from the training metadata
        _decision_rules = meta_v2.get('decision_rules', {})
        CALIB_THRESHOLD_AD = float(_decision_rules.get('calibrated_ad_threshold', CALIB_THRESHOLD_AD))
        CALIB_THRESHOLD_MCI = float(_decision_rules.get('calibrated_mci_threshold', CALIB_THRESHOLD_MCI))
        _pdp = meta_v2.get('primary_decision_path', {})
        if _pdp.get('out_of_domain_sq_mahalanobis_threshold') is not None:
            OOD_DISTANCE_THRESHOLD = float(_pdp['out_of_domain_sq_mahalanobis_threshold'])
        if _pdp.get('validated_domain'):
            VALIDATED_DOMAIN_NOTE = _pdp['validated_domain']
    except Exception as e:
        print(f"⚠️ Warning loading training metadata: {e}")

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

# Limit CPU threads to half the cores so the co-hosted LLM service is not starved
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

# Image preprocessing (ImageNet statistics)
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =====================================================================
# Model architecture (identical to the training and XAI notebooks)
# =====================================================================
class GatedAttentionFusion(nn.Module):
    """Gated attention MIL pooling (Ilse et al., ICML 2018) with a residual mean path."""
    def __init__(self, in_features: int = 640, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.fusion_norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        # h shape: (Batch, Slices=10, Features=640)
        A_V = self.attention_V(h)                   # (B, S, hidden_dim)
        A_U = self.attention_U(h)                   # (B, S, hidden_dim)
        A_gate = self.attention_weights(A_V * A_U)  # (B, S, 1)
        attn_scores = torch.softmax(A_gate, dim=1)  # (B, S, 1)

        # Attention-weighted sum plus residual mean over slices
        h_att = torch.sum(attn_scores * h, dim=1)   # (B, Features)
        h_mean = h.mean(dim=1)                      # (B, Features)

        volume_rep = self.fusion_norm(h_att + h_mean)
        return self.dropout(volume_rep), attn_scores.squeeze(-1)


class MobileViT25DMultimodalV2(nn.Module):
    """2.5D multimodal MobileViT classifier.

    - 10 multi-plane slices (5 axial-band, 3 coronal-band, 2 sagittal) through a shared backbone
    - learnable positional embedding per slice position
    - gated attention MIL fusion across slices
    - tabular branch: 8-D covariate vector (values and availability flags) -> 64-D embedding
    - MLP classifier head over the concatenated visual and tabular representations
    """
    def __init__(self, model_name: str = 'apple/mobilevit-small', num_classes: int = 3,
                 num_slices: int = SLICES_PER_VOL, embed_dim: int = 640,
                 hidden_dim: int = 128, tabular_input_dim: int = TABULAR_DIM,
                 tabular_hidden_dim: int = TABULAR_HIDDEN_DIM):
        super().__init__()
        self.num_slices = num_slices
        self.embed_dim  = embed_dim
        self.tabular_input_dim = tabular_input_dim

        # Shared MobileViT-small backbone
        hf_model = MobileViTForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.backbone = hf_model.mobilevit

        # Learnable positional encoding per slice
        self.slice_pos_embed = nn.Parameter(torch.randn(1, num_slices, embed_dim) * 0.02)

        # Gated attention MIL fusion
        self.fusion = GatedAttentionFusion(in_features=embed_dim, hidden_dim=hidden_dim, dropout=0.2)

        # Tabular covariate branch
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_input_dim, tabular_hidden_dim),
            nn.SiLU(),
            nn.Linear(tabular_hidden_dim, tabular_hidden_dim),
            nn.LayerNorm(tabular_hidden_dim)
        )

        # Classifier head over 640 visual + 64 tabular = 704 features
        fused_dim = embed_dim + tabular_hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def encode_volume(self, x):
        """Encode a (B, S, 3, H, W) slice stack into a pooled volume representation and slice attention weights."""
        B, S, C, H, W = x.shape
        features = self.backbone(pixel_values=x.view(B * S, C, H, W)).last_hidden_state
        slice_emb = features.mean(dim=[-2, -1]).view(B, S, self.embed_dim)
        slice_emb = slice_emb + self.slice_pos_embed[:, :S, :]
        return self.fusion(slice_emb)

    def forward(self, x, tabular_feat=None, return_attention: bool = False):
        # x shape: (Batch, Slices=10, Channels=3, H=224, W=224)
        volume_rep, attn_weights = self.encode_volume(x)

        if tabular_feat is None:
            tabular_feat = torch.zeros((x.size(0), self.tabular_input_dim), device=x.device, dtype=torch.float32)
        else:
            tabular_feat = tabular_feat.to(x.device).float()

        t_emb = self.tabular_proj(tabular_feat)  # (B, 64)
        fused_rep = torch.cat([volume_rep, t_emb], dim=-1)  # (B, 704)
        logits = self.classifier(fused_rep)

        if return_attention:
            return logits, attn_weights
        return logits


# =====================================================================
# Model and checkpoint loading. MODEL_READY gates every prediction entry point so that an
# untrained or mismatched model is never served.
# =====================================================================
model = MobileViT25DMultimodalV2(num_classes=NUM_CLASSES)
v2_ckpt = CHECKPOINT_DIR / 'best_model_v2_unified.pt'
checkpoint_loaded = False
MODEL_READY = False
MODEL_LOAD_ERROR: Optional[str] = None

if not v2_ckpt.exists():
    MODEL_LOAD_ERROR = f'Checkpoint not found at {v2_ckpt}'
    print(f"⚠️ {MODEL_LOAD_ERROR}")
else:
    try:
        ckpt_data = torch.load(v2_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt_data['model_state_dict'] if 'model_state_dict' in ckpt_data else ckpt_data
        model.load_state_dict(state_dict)  # strict=True (default): raises on any shape mismatch
        checkpoint_loaded = True
        MODEL_READY = True
        print(f"✅ Successfully loaded MobileViT v2 Unified champion model from {v2_ckpt}")
    except Exception as e:
        MODEL_LOAD_ERROR = f'Failed to load {v2_ckpt}: {e}'
        print(f"🛑 {MODEL_LOAD_ERROR}")
        print("🛑 MODEL_READY=False -- predict_session() will refuse to run rather than serve "
              "predictions from an untrained model. Fix the architecture/checkpoint mismatch "
              "and restart.")

model = model.to(device)
model.eval()

# One model instance serves every request, and FastAPI runs the sync endpoints in a threadpool.
# Without serialization, overlapping requests stack their GPU memory peaks (several IG runs at once)
# and cam_attr's forward hook can capture another request's activation. predict_session holds this
# lock for its whole model section, so model work runs one request at a time.
_MODEL_LOCK = threading.Lock()


def _release_accelerator_cache() -> None:
    """Return cached accelerator blocks to the system after a large attribution run.

    The MPS caching allocator otherwise keeps the peak of an IG run reserved for the life of the process.
    """
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()


# =====================================================================
# Attribution kernels (as in xai_gradcam_anatomical_benchmark.ipynb). They run the full
# 10-slice forward pass (attention, positional embeddings, covariates) and differentiate the
# target-class logit with respect to one backbone layer for all slices in one backward pass.
# =====================================================================
def _get_backbone_module(m: nn.Module, dotted: str):
    """Resolve a dotted layer path (e.g. 'encoder.layer.3') inside the backbone."""
    mod = m.backbone
    for part in dotted.split('.'):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def cam_attr(m: nn.Module, vol_tensor: torch.Tensor, tab_tensor: torch.Tensor, target_id: int,
             layer: str = 'encoder.layer.3', mode: str = 'hirescam') -> np.ndarray:
    """Class-activation maps for all SLICES_PER_VOL slices from a single backward pass.

    mode='hirescam': element-wise gradient * activation, summed over channels, then ReLU.
    mode='gradcam': channel-mean gradient weights applied to the activations.
    layer selects the hooked backbone layer ('encoder.layer.3' is the 14x14 stage).
    Returns an array (slices, IMG_SIZE, IMG_SIZE), each map scaled to [0, 1].
    """
    m.eval()
    # Capture the hooked layer's activation during the forward pass
    store: Dict[str, torch.Tensor] = {}
    h = _get_backbone_module(m, layer).register_forward_hook(
        lambda _mod, _i, _o: store.__setitem__('a', _o[0] if isinstance(_o, (tuple, list)) else _o))
    try:
        with torch.enable_grad():
            v = vol_tensor.unsqueeze(0).to(device)
            t = tab_tensor.unsqueeze(0).to(device)
            logits = m(v, tabular_feat=t)
            g = torch.autograd.grad(logits[0, int(target_id)], store['a'])[0]
        a = store['a'].detach()
    finally:
        h.remove()
    if mode == 'gradcam':
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(dim=1, keepdim=True))
    elif mode == 'hirescam':
        cam = torch.relu((g * a).sum(dim=1, keepdim=True))
    else:
        raise ValueError(f'unknown mode {mode!r}')
    # Upsample to the input resolution and scale each slice to [0, 1]
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)[:, 0]
    cam = cam.detach().cpu().numpy()
    for s in range(cam.shape[0]):
        mx = cam[s].max()
        if mx > 0:
            cam[s] = cam[s] / mx
    return cam


def _ig_raw(m: nn.Module, vol_tensor: torch.Tensor, tab_tensor: torch.Tensor, target_id: int,
            steps: int = 16, chunk: int = 4, baseline: str = 'blur'):
    """Integrated gradients along a straight path from the baseline to the input.

    Returns (attribution tensor, F(x) - F(baseline), sum of attributions).
    """
    m.eval()
    v0 = vol_tensor.to(device)
    px = v0.cpu().numpy() * IMAGENET_STD[None, :, None, None] + IMAGENET_MEAN[None, :, None, None]
    # Baseline: heavily blurred slices ('blur') or a black image
    if baseline == 'blur':
        bpx = np.stack([gaussian_filter(px[s], sigma=(0, 8, 8)) for s in range(px.shape[0])])
        base = torch.tensor((bpx - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None],
                            dtype=torch.float32, device=device)
    else:
        base = torch.tensor((0.0 - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None],
                            dtype=torch.float32, device=device).expand_as(v0).contiguous()
    # Accumulate gradients over interpolation steps, processed in chunks
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)
    diff = v0 - base
    grad_acc = torch.zeros_like(v0)
    t = tab_tensor.unsqueeze(0).to(device)
    for i in range(0, steps, chunk):
        al = alphas[i:i + chunk]
        xb = (base.unsqueeze(0) + al[:, None, None, None, None] * diff.unsqueeze(0)).requires_grad_(True)
        gb = torch.autograd.grad(m(xb, tabular_feat=t.expand(xb.size(0), -1))[:, int(target_id)].sum(), xb)[0]
        grad_acc += gb.sum(0).detach()
    ig = diff * grad_acc / steps
    with torch.no_grad():
        fx = float(m(v0.unsqueeze(0), tabular_feat=t)[0, int(target_id)])
        fb = float(m(base.unsqueeze(0), tabular_feat=t)[0, int(target_id)])
    return ig, fx - fb, float(ig.sum())


def ig_attr(m: nn.Module, vol_tensor: torch.Tensor, tab_tensor: torch.Tensor, target_id: int,
            steps: int = 16, chunk: int = 4, baseline: str = 'blur') -> np.ndarray:
    """Integrated-gradients heatmaps with the same signature and [0, 1] output convention as cam_attr.

    Roughly 20-30x slower than cam_attr, so results are cached on disk (XAI_CACHE_DIR).
    """
    ig, _, _ = _ig_raw(m, vol_tensor, tab_tensor, target_id, steps, chunk, baseline)
    return _ig_to_map(ig)


def _ig_to_map(ig: torch.Tensor) -> np.ndarray:
    """Per-slice IG heatmaps: channel sum, ReLU, each slice scaled to [0, 1]."""
    att = torch.relu(ig.sum(dim=1)).cpu().numpy()
    for s in range(att.shape[0]):
        mx = att[s].max()
        if mx > 0:
            att[s] = att[s] / mx
    return att


def _completeness_residual(delta: float, s_ig: float) -> float:
    return abs(s_ig - delta) / (abs(delta) + 1e-6)


def ig_completeness(m: nn.Module, vol_tensor: torch.Tensor, tab_tensor: torch.Tensor, target_id: int,
                     steps: int = 16) -> float:
    """Relative completeness residual |sum(IG) - (F(x) - F(baseline))| / |F(x) - F(baseline)|.

    A small value indicates the attribution is faithful to the model output for this scan.
    """
    _, delta, s_ig = _ig_raw(m, vol_tensor, tab_tensor, target_id, steps=steps, chunk=4)
    return _completeness_residual(delta, s_ig)


def ig_attr_with_completeness(m: nn.Module, vol_tensor: torch.Tensor, tab_tensor: torch.Tensor,
                              target_id: int, steps: int = 16, chunk: int = 4,
                              baseline: str = 'blur') -> tuple:
    """ig_attr heatmaps and the ig_completeness residual from one IG run.

    Same values as calling ig_attr and ig_completeness with these arguments, which each ran the
    identical _ig_raw; the residual now describes exactly the attribution that is displayed.
    """
    ig, delta, s_ig = _ig_raw(m, vol_tensor, tab_tensor, target_id, steps, chunk, baseline)
    return _ig_to_map(ig), _completeness_residual(delta, s_ig)


# =====================================================================
# Anatomical grounding: optional, available only when a FreeSurfer aseg.mgz exists for the session
# =====================================================================
ROI_LABELS = {
    'hippocampus': [17, 53],
    'ventricles':  [4, 43, 5, 44, 14],
    'amygdala':    [18, 54],
    'cortex':      [3, 42],
    'ad_target':   [17, 53, 4, 43, 5, 44, 14, 18, 54],
}
AXIAL_SLICE_PCT, CORONAL_SLICE_PCT, SAGITTAL_SLICE_PCT = (0.35, 0.65), (0.35, 0.65), (0.40, 0.60)
PX_TO_MM = 256.0 / IMG_SIZE


def _get_slice_coordinate(plane: str, local_idx: int, shape) -> int:
    """Voxel index of the local_idx-th extracted slice of a plane within the volume."""
    if plane == 'axial':
        lo, hi, n, axis_len = AXIAL_SLICE_PCT[0], AXIAL_SLICE_PCT[1], N_AXIAL, shape[2]
    elif plane == 'coronal':
        lo, hi, n, axis_len = CORONAL_SLICE_PCT[0], CORONAL_SLICE_PCT[1], N_CORONAL, shape[1]
    else:
        lo, hi, n, axis_len = SAGITTAL_SLICE_PCT[0], SAGITTAL_SLICE_PCT[1], N_SAGITTAL, shape[0]
    return int(np.linspace(int(axis_len * lo), int(axis_len * hi) - 1, n, dtype=int)[local_idx])


def _load_aseg_for_session(session_id: str, canonical_id: str) -> Optional[np.ndarray]:
    """Load the session's aseg segmentation volume, or None when unavailable."""
    fs_dir = _freesurfer_dir_for(session_id)
    for cand in (canonical_id, session_id):
        p = fs_dir / cand / 'aseg.mgz'
        if not p.exists():
            p = fs_dir / cand / 'mri' / 'aseg.mgz'
        if p.exists():
            try:
                return nib.load(str(p)).get_fdata()
            except Exception:
                return None
    return None


def _slice_from_aseg(data: np.ndarray, plane: str, local_idx: int) -> np.ndarray:
    """Extract the segmentation slice matching an extracted image slice (same rotation and size)."""
    coord = _get_slice_coordinate(plane, local_idx, data.shape)
    s = data[:, :, coord] if plane == 'axial' else (data[:, coord, :] if plane == 'coronal' else data[coord, :, :])
    s_rot = np.rot90(s, k=1)
    return np.array(Image.fromarray(s_rot.astype(np.uint16)).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))


def _on_target_pct(aseg_vol: Optional[np.ndarray], plane_key: str, local_idx: int,
                   cam_slice: np.ndarray, brain_mask: np.ndarray) -> Optional[float]:
    """Percentage of brain-masked attribution energy that falls on AD-related structures (slice-level EBPG).

    Returns None when the session has no segmentation.
    """
    if aseg_vol is None:
        return None
    aseg_slice = _slice_from_aseg(aseg_vol, plane_key, local_idx)
    target = np.isin(aseg_slice, ROI_LABELS['ad_target']).astype(np.float32)
    masked = cam_slice * brain_mask
    denom = float(np.sum(masked)) + 1e-8
    return round(float(np.sum(masked * target) / denom) * 100.0, 2)


# For this LIA-oriented data, files 'slice_*.png' are anatomically coronal and 'coronal_*.png'
# are anatomically axial. PLANE_NAMES holds the anatomical display names; _PLANE_KEYS holds the
# extraction-band key used to locate the matching segmentation slice.
PLANE_NAMES = [f"Coronal {i+1}" for i in range(N_AXIAL)] + \
              [f"Axial {i+1}" for i in range(N_CORONAL)] + \
              [f"Sagittal {i+1}" for i in range(N_SAGITTAL)]
_PLANE_KEYS = ['axial'] * N_AXIAL + ['coronal'] * N_CORONAL + ['sagittal'] * N_SAGITTAL  # extraction-band key


# =====================================================================
# Out-of-domain check: squared Mahalanobis distance of the 640-D volume representation to the
# training feature distribution (Ledoit-Wolf covariance), compared with OOD_DISTANCE_THRESHOLD
# (95th percentile of the training distances).
# =====================================================================
_REP_TRAIN_CACHE = CHECKPOINT_DIR / 'rep_train_640d.npy'
_ood_estimator: Optional[LedoitWolf] = None
if _REP_TRAIN_CACHE.exists():
    try:
        _rep_train = np.load(_REP_TRAIN_CACHE)
        _ood_estimator = LedoitWolf().fit(_rep_train)
        print(f"✅ Out-of-domain estimator fitted on {_rep_train.shape} cached training features "
              f"(threshold={OOD_DISTANCE_THRESHOLD})")
    except Exception as e:
        print(f"⚠️ Could not fit out-of-domain estimator: {e}")
else:
    print(f"⚠️ {_REP_TRAIN_CACHE} not found -- out-of-domain badge will be unavailable.")


def _ood_check(volume_rep: torch.Tensor) -> Dict[str, Any]:
    """Report whether the volume representation lies within the validated domain."""
    if _ood_estimator is None or OOD_DISTANCE_THRESHOLD is None:
        return {'available': False}
    dist = float(_ood_estimator.mahalanobis(volume_rep.detach().cpu().numpy()))
    return {
        'available': True,
        'distance': round(dist, 1),
        'threshold': round(OOD_DISTANCE_THRESHOLD, 1),
        'in_validated_domain': dist <= OOD_DISTANCE_THRESHOLD,
        'validated_domain_note': VALIDATED_DOMAIN_NOTE,
    }


# =====================================================================
# Disk cache of XAI results, keyed by (session, method, target class). The checkpoint is fixed
# at runtime, so entries never expire.
# =====================================================================
def _cache_path(canonical_id: str, xai_method: str, target_class: str) -> Path:
    key = f'{canonical_id}__{xai_method}__{target_class}'
    return XAI_CACHE_DIR / (hashlib.sha256(key.encode()).hexdigest()[:24] + '.json')


def _cache_read(canonical_id: str, xai_method: str, target_class: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(canonical_id, xai_method, target_class)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _cache_write(canonical_id: str, xai_method: str, target_class: str, payload: Dict[str, Any]) -> None:
    p = _cache_path(canonical_id, xai_method, target_class)
    try:
        p.write_text(json.dumps(payload))
    except Exception as e:
        print(f"⚠️ Could not write XAI cache {p}: {e}")


def base64_encode(b: bytes) -> str:
    """Base64-encode bytes as a UTF-8 string."""
    import base64
    return base64.b64encode(b).decode('utf-8')


def _encode_jpeg_b64(rgb_uint8: np.ndarray, quality: int = 80) -> str:
    pil = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    return base64_encode(buf.getvalue())


def _overlay_cam(orig_rgb01: np.ndarray, cam01: np.ndarray) -> np.ndarray:
    """Blend a jet-colormap heatmap over the raw slice and return a uint8 RGB image."""
    import matplotlib.cm as cm
    heat = (cm.get_cmap('jet')(cam01)[..., :3] * 255.0).astype(np.float32)
    base = (orig_rgb01 * 255.0).astype(np.float32)
    blended = np.clip(0.55 * heat + 0.45 * base, 0, 255).astype(np.uint8)
    return blended


def _read_tabular_from_meta(session_id: str, canonical_id: str, include_mmse: bool, include_apoe: bool,
                            custom_age: Optional[float], custom_gender: Optional[str]) -> tuple:
    """Build the 8-D covariate vector and a dict of the raw values from the session metadata."""
    sess_meta = (
        clinical_meta.get(canonical_id) or
        clinical_meta.get(session_id) or
        clinical_meta.get(session_id.replace('_MR_', '_')) or
        clinical_meta.get(f"ADNI_{session_id.replace('_MR_', '_')}") or
        {}
    )
    gender_str = custom_gender or sess_meta.get('gender', 'Unknown')
    gender_lower = str(gender_str).lower()
    is_male = 1.0 if ('male' in gender_lower and 'female' not in gender_lower) or str(gender_str) == '1' else 0.0

    age_raw = custom_age if custom_age is not None else sess_meta.get('age_at_scan', sess_meta.get('age', 70.0))
    try:
        age_val = float(age_raw)
        age_norm = float(np.clip((age_val - 47.0) / 50.0, 0.0, 1.0))
    except (ValueError, TypeError):
        age_val, age_norm = 70.0, 0.5

    mmse_raw = sess_meta.get('mmse', None)
    try:
        if include_mmse and mmse_raw is not None and str(mmse_raw).strip() not in ('', '#N/A', 'nan', 'None', 'N/A', 'Unknown'):
            mmse_val = float(mmse_raw)
            mmse_norm = float(np.clip((mmse_val - 7.0) / 23.0, 0.0, 1.0))
            has_mmse = 1.0
        else:
            mmse_val, mmse_norm, has_mmse = None, 0.0, 0.0
    except (ValueError, TypeError):
        mmse_val, mmse_norm, has_mmse = None, 0.0, 0.0

    apoe_raw = str(sess_meta.get('apoe', '')).strip()
    if include_apoe and apoe_raw not in ('', '#N/A', '0', 'nan', 'None', 'Unknown', 'N/A'):
        has_apoe, is_apoe4, apoe_str = 1.0, (1.0 if '4' in apoe_raw else 0.0), apoe_raw
    else:
        has_apoe, is_apoe4, apoe_str = 0.0, 0.0, None

    # Vector order matches the training notebooks; gender and age are always marked available
    tab = [is_male, 1.0, age_norm, 1.0, mmse_norm, has_mmse, is_apoe4, has_apoe]
    extras = {
        'gender_str': gender_str, 'age_val': age_val, 'has_mmse': has_mmse, 'mmse_val': mmse_val,
        'has_apoe': has_apoe, 'apoe_str': apoe_str,
    }
    return tab, extras


def _load_session_tensors(session_id: str) -> tuple:
    """Load the 10 cached slices of a session as normalised tensors and PIL images."""
    session_dir, canonical_id = locate_session_cache(session_id)
    if not session_dir.exists():
        raise FileNotFoundError(f"Slice cache for {session_id} not found at {session_dir}")

    axial_paths = [session_dir / f"slice_{i:03d}.png" for i in range(N_AXIAL)]
    coronal_paths = [session_dir / f"coronal_{i:03d}.png" for i in range(N_CORONAL)]
    sagittal_paths = [session_dir / f"sagittal_{i:03d}.png" for i in range(N_SAGITTAL)]
    all_paths = axial_paths + coronal_paths + sagittal_paths

    tensors, pil_images = [], []
    for p in all_paths:
        if not p.exists():
            existing = sorted(session_dir.glob('*.png'))
            p = existing[0] if existing else None
            if not p:
                raise ValueError(f"No slices found in {session_dir}")
        img = Image.open(p).convert('RGB')
        tensors.append(val_transform(img))
        pil_images.append(img.resize((IMG_SIZE, IMG_SIZE)))
    vol_tensor = torch.stack(tensors, dim=0)
    return vol_tensor, tensors, pil_images, session_dir, canonical_id


def predict_session(
    session_id: str,
    include_mmse: bool = True,
    include_apoe: bool = True,
    custom_age: Optional[float] = None,
    custom_gender: Optional[str] = None,
    xai_method: Literal['hirescam', 'gradcam', 'ig'] = 'hirescam',
    target_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify one session from its 10 multi-plane slices and explain the prediction.

    Returns the diagnosis, class probabilities, slice attention weights, a heatmap overlay for
    every slice, an out-of-domain assessment and model metadata.

    xai_method: 'hirescam' (default, one backward pass), 'gradcam', or 'ig' (integrated
    gradients, slower, cached to disk, reported with a completeness residual).
    target_class: class explained by the heatmaps; None uses the predicted class.

    Calls are serialized by _MODEL_LOCK. A request that waited behind a computation of the same
    (session, method, class) is then served from the disk cache that computation wrote.
    """
    with _MODEL_LOCK:
        return _predict_session_unlocked(
            session_id, include_mmse, include_apoe, custom_age, custom_gender, xai_method, target_class)


def xai_cache_exists(canonical_id: str, xai_method: str, target_class: str) -> bool:
    """Whether the disk cache already holds the heatmaps of (session, method, class)."""
    return _cache_path(canonical_id, xai_method, target_class).exists()


def _predict_session_unlocked(
    session_id: str,
    include_mmse: bool,
    include_apoe: bool,
    custom_age: Optional[float],
    custom_gender: Optional[str],
    xai_method: str,
    target_class: Optional[str],
) -> Dict[str, Any]:
    """Body of predict_session; callers must hold _MODEL_LOCK."""
    if not MODEL_READY:
        raise RuntimeError(
            f"Model is not ready to serve predictions (MODEL_LOAD_ERROR={MODEL_LOAD_ERROR!r}). "
            f"Refusing to run inference on an untrained/mismatched model."
        )

    start_time = time.perf_counter()
    vol_tensor_stack, tensors, pil_images, session_dir, canonical_id = _load_session_tensors(session_id)
    vol_tensor_batch = vol_tensor_stack.unsqueeze(0).to(device)

    tab_list, tab_extras = _read_tabular_from_meta(
        session_id, canonical_id, include_mmse, include_apoe, custom_age, custom_gender)
    tab_tensor_real = torch.tensor([tab_list], dtype=torch.float32).to(device)
    # The primary decision path is image-only (primary_decision_path.mode == 'image_only_argmax'
    # in the training metadata): covariates are parsed and reported but not fed to the model.
    tab_tensor = torch.zeros_like(tab_tensor_real)

    # Forward pass: pooled volume representation, slice attention, class probabilities
    with torch.no_grad():
        volume_rep, attn_weights_raw = model.encode_volume(vol_tensor_batch)
        t_emb = model.tabular_proj(tab_tensor)
        logits = model.classifier(torch.cat([volume_rep, t_emb], dim=-1))
        probs = torch.softmax(logits, dim=1)[0]
        attns = attn_weights_raw[0].cpu().numpy()

    p_cn, p_mci, p_ad = float(probs[0].item()), float(probs[1].item()), float(probs[2].item())
    argmax_idx = int(torch.argmax(probs).item())
    pred_label_argmax = CLASS_NAMES[argmax_idx]

    # Threshold-calibrated label, reported alongside the argmax label
    if p_ad >= CALIB_THRESHOLD_AD:
        pred_label_calib = 'AD'
    elif p_mci >= CALIB_THRESHOLD_MCI:
        pred_label_calib = 'MCI'
    else:
        pred_label_calib = 'CN'

    # The primary diagnosis and the default explanation target are the argmax class;
    # target_class overrides the explanation target only
    primary_diagnosis = pred_label_argmax
    pred_id = argmax_idx
    confidence = float(probs[pred_id].item())
    cam_target_id = LABEL_MAP[target_class] if target_class is not None else pred_id
    cam_target_label = target_class if target_class is not None else pred_label_argmax

    ood_info = _ood_check(volume_rep)

    probabilities = {'CN': round(p_cn, 4), 'MCI': round(p_mci, 4), 'AD': round(p_ad, 4)}
    attention_breakdown = [
        {"index": i, "plane": PLANE_NAMES[i], "weight": round(float(attns[i]), 4),
         "percentage": round(float(attns[i] * 100), 2)}
        for i in range(SLICES_PER_VOL)
    ]
    top_slice_idx = int(np.argmax(attns))

    # Heatmaps: read from the disk cache or compute
    cache_key_class = cam_target_label
    cached = _cache_read(canonical_id, xai_method, cache_key_class)
    ig_completeness_residual = None
    if cached is not None:
        slice_heatmaps = cached['slice_heatmaps']
        ig_completeness_residual = cached.get('ig_completeness_residual')
    else:
        vol_for_cam = vol_tensor_stack
        tab_for_cam = tab_tensor[0].cpu()
        if xai_method == 'ig':
            cams, residual = ig_attr_with_completeness(
                model, vol_for_cam, tab_for_cam, cam_target_id, steps=16, chunk=4, baseline='blur')
            ig_completeness_residual = round(residual, 4)
            _release_accelerator_cache()
        else:
            mode = 'gradcam' if xai_method == 'gradcam' else 'hirescam'
            layer = 'conv_1x1_exp' if xai_method == 'gradcam' else 'encoder.layer.3'
            cams = cam_attr(model, vol_for_cam, tab_for_cam, cam_target_id, layer=layer, mode=mode)

        # Per-slice overlays and on-target percentages
        aseg_vol = _load_aseg_for_session(session_id, canonical_id)
        slice_heatmaps = []
        for i in range(SLICES_PER_VOL):
            img_np = np.array(pil_images[i]).astype(np.float32) / 255.0
            brain_mask = (img_np.mean(axis=2) > 0.04).astype(np.float32)
            cam_i = cams[i]
            if xai_method == 'ig':
                # Smooth the sparse IG attribution for display; on_target_pct uses the unsmoothed map
                cam_i = gaussian_filter(cam_i, sigma=3.0)
            masked_cam = cam_i * brain_mask
            if masked_cam.max() > 0:
                masked_cam = masked_cam / masked_cam.max()
            overlay = _overlay_cam(img_np, masked_cam)
            if xai_method == 'ig':
                # Upscale the IG overlay for display
                overlay = np.array(Image.fromarray(overlay).resize((IMG_SIZE * 2, IMG_SIZE * 2), Image.LANCZOS))
            plane_key = _PLANE_KEYS[i]
            local_idx = i if i < N_AXIAL else (i - N_AXIAL if i < N_AXIAL + N_CORONAL else i - N_AXIAL - N_CORONAL)
            on_target = _on_target_pct(aseg_vol, plane_key, local_idx, cams[i], brain_mask)
            slice_heatmaps.append({
                'index': i, 'plane': PLANE_NAMES[i],
                'attention_pct': round(float(attns[i] * 100), 2),
                'overlay_base64': f"data:image/jpeg;base64,{_encode_jpeg_b64(overlay)}",
                'on_target_pct': on_target,
            })
        _cache_write(canonical_id, xai_method, cache_key_class, {
            'slice_heatmaps': slice_heatmaps, 'ig_completeness_residual': ig_completeness_residual,
        })

    top = slice_heatmaps[top_slice_idx]
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "session_id": session_id,
        "canonical_session_id": canonical_id,
        "diagnosis": primary_diagnosis,
        "diagnosis_calibrated": pred_label_calib,
        "diagnosis_argmax": pred_label_argmax,
        "confidence": confidence,
        "probabilities": probabilities,
        "thresholds_used": {"AD": CALIB_THRESHOLD_AD, "MCI": CALIB_THRESHOLD_MCI},
        "decision_rule_active": "argmax",
        "slice_count": SLICES_PER_VOL,
        "attention_weights": attention_breakdown,
        "top_slice": {
            "index": top_slice_idx, "name": PLANE_NAMES[top_slice_idx],
            "attention_pct": round(float(attns[top_slice_idx] * 100), 2),
        },
        "multimodal_inputs_used": {
            # Clinical values are shown for transparency only; the primary diagnosis is image-only
            "used_in_primary_decision": False,
            "gender": tab_extras['gender_str'], "age": tab_extras['age_val'],
            "mmse_available": include_mmse and tab_extras['has_mmse'] > 0.5,
            "mmse_score": tab_extras['mmse_val'] if tab_extras['has_mmse'] > 0.5 else None,
            "apoe_available": include_apoe and tab_extras['has_apoe'] > 0.5,
            "apoe_genotype": tab_extras['apoe_str'] if tab_extras['has_apoe'] > 0.5 else None,
        },
        "xai_method": xai_method,
        "cam_target_class": cam_target_label,
        "ig_completeness_residual": ig_completeness_residual,
        "slice_heatmaps": slice_heatmaps,
        # Top-attention slice fields, used by the FHIR export and the UI
        "gradcam_base64": top['overlay_base64'],
        "raw_slice_base64": f"data:image/jpeg;base64,{_encode_jpeg_b64((np.array(pil_images[top_slice_idx])).astype(np.uint8))}",
        "out_of_domain": ood_info,
        "model_info": {
            "name": "MobileViT25DMultimodalV2-Unified",
            "architecture": "MobileViT-Small 2.5D Multi-Plane (10 Slices) + Selective Tabular Gated Fusion",
            "checkpoint_loaded": checkpoint_loaded,
            "checkpoint_file": v2_ckpt.name,
            "input_resolution": [224, 224],
            "num_classes": NUM_CLASSES,
        },
        "device_used": str(device),
        "latency_ms": latency_ms,
    }


def precompute_xai(session_id: str, xai_method: str = 'ig', target_class: Optional[str] = None) -> Dict[str, Any]:
    """Warm the disk cache for a (session, method, class) combination and return a short summary."""
    result = predict_session(session_id, xai_method=xai_method, target_class=target_class)
    return {
        'session_id': session_id, 'xai_method': xai_method, 'target_class': target_class,
        'cached': True, 'ig_completeness_residual': result.get('ig_completeness_residual'),
    }
