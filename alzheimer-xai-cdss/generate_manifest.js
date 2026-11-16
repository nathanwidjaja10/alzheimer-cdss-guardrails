import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

// Generates src/data.json, the subject/session catalogue of both cohorts, from the local data folders.

// OASIS-3 locations (oasis_data/)
const oasisDataDir = path.join(projectRoot, 'oasis_data', 'freesurfer_data');
const sliceCacheDir = path.join(projectRoot, 'oasis_data', 'slice_cache');
const oasisClinicalMetadataPath = path.join(projectRoot, 'oasis_data', 'clinical_metadata.json');

// ADNI locations (adni_data/)
const adniDataDir = path.join(projectRoot, 'adni_data', 'freesurfer_data');
const adniRawDataDir = path.join(projectRoot, 'adni_data', 'raw_scans_data');
const adniSliceCacheDir = path.join(projectRoot, 'adni_data', 'slice_cache');
const adniClinicalMetadataPath = path.join(projectRoot, 'adni_data', 'clinical_metadata.json');

// Synthetic demo sessions (dummy_data/): BrainWeb phantoms with made-up covariates, see dummy_data/README.md
const dummyDataDir = path.join(projectRoot, 'dummy_data', 'freesurfer_data');
const dummySliceCacheDir = path.join(projectRoot, 'dummy_data', 'slice_cache');
const dummyClinicalMetadataPath = path.join(projectRoot, 'dummy_data', 'clinical_metadata.json');

const outputFile = path.join(__dirname, 'src', 'data.json');

// Approximate day offset of an ADNI visit code ('sc'/'bl' = 0, 'mNN' = NN months)
function parseAdniDays(visitCode) {
  if (!visitCode) return 0;
  const v = visitCode.toLowerCase();
  if (v === 'sc' || v === 'bl' || v === 'baseline') return 0;
  if (v.startsWith('m') && !isNaN(parseInt(v.slice(1), 10))) {
    const months = parseInt(v.slice(1), 10);
    return Math.round(months * 30.4375);
  }
  return 0;
}

// Build the manifest: group sessions by subject, attach clinical metadata and slice URLs, sort and write
function generateManifest() {
  const subjectsMap = new Map();

  // OASIS-3 sessions
  if (fs.existsSync(oasisDataDir)) {
    let oasisMetadata = {};
    if (fs.existsSync(oasisClinicalMetadataPath)) {
      try {
        oasisMetadata = JSON.parse(fs.readFileSync(oasisClinicalMetadataPath, 'utf8'));
      } catch(e) {
        console.warn("[WARN] Could not parse OASIS clinical_metadata.json");
      }
    }

    const dirs = fs.readdirSync(oasisDataDir, { withFileTypes: true });
    for (const dir of dirs) {
      if (!dir.isDirectory()) continue;
      const sessionName = dir.name; // e.g. OAS30001_MR_d0129
      if (sessionName.startsWith('.')) continue;

      const parts = sessionName.split('_');
      if (parts.length < 3) continue;

      const subjectId = parts[0];
      const daysStr = parts[2].replace('d', '');
      const days = parseInt(daysStr, 10) || 0;

      const basePath = `/@fs${path.join(oasisDataDir, sessionName)}`;
      const brainUrl = `${basePath}/brain.mgz`;
      const asegUrl = `${basePath}/aseg.mgz`;

      const physicalBrain = path.join(oasisDataDir, sessionName, 'brain.mgz');
      const physicalAseg = path.join(oasisDataDir, sessionName, 'aseg.mgz');

      if (!fs.existsSync(physicalBrain)) continue;

      const clinicalInfo = oasisMetadata[sessionName] || {};

      // Cached 2D slices
      const sessionSliceDir = path.join(sliceCacheDir, sessionName);
      let sliceUrls = [];
      if (fs.existsSync(sessionSliceDir)) {
        const sliceFiles = fs.readdirSync(sessionSliceDir)
          .filter(f => f.endsWith('.png'))
          .sort((a, b) => {
            const numA = parseInt(a.split('_').pop().replace('.png', ''), 10) || 0;
            const numB = parseInt(b.split('_').pop().replace('.png', ''), 10) || 0;
            return numA - numB;
          });
        sliceUrls = sliceFiles.map(f => `/@fs${path.join(sessionSliceDir, f)}`);
      }

      if (!subjectsMap.has(subjectId)) {
        subjectsMap.set(subjectId, {
          id: subjectId,
          cohort: 'OASIS-3',
          age: clinicalInfo.age_at_scan || 'Unknown',
          gender: clinicalInfo.gender || 'Unknown',
          apoe: clinicalInfo.apoe || 'Unknown',
          education: clinicalInfo.education || 'N/A',
          sessions: []
        });
      }

      const subject = subjectsMap.get(subjectId);
      subject.sessions.push({
        id: parts[2],
        sessionId: sessionName,
        days: days,
        type: 'OASIS-3 Freesurfer Volume (.mgz)',
        date: `Day ${days}`,
        url: brainUrl,
        maskUrl: fs.existsSync(physicalAseg) ? asegUrl : null,
        sliceUrls: sliceUrls,
        diagnosis: clinicalInfo.diagnosis || 'Unknown',
        cdr: clinicalInfo.cdr || 'N/A',
        mmse: clinicalInfo.mmse || 'N/A'
      });
    }
  }

  // ADNI sessions
  if (fs.existsSync(adniDataDir)) {
    let adniMetadata = {};
    if (fs.existsSync(adniClinicalMetadataPath)) {
      try {
        adniMetadata = JSON.parse(fs.readFileSync(adniClinicalMetadataPath, 'utf8'));
      } catch(e) {
        console.warn("[WARN] Could not parse ADNI clinical_metadata.json");
      }
    }

    const dirs = fs.readdirSync(adniDataDir, { withFileTypes: true });
    for (const dir of dirs) {
      if (!dir.isDirectory() || dir.name.startsWith('.')) continue;

      const sessionKey = dir.name;
      const meta = adniMetadata[sessionKey] || {};
      const subjectId = meta.subject_id || sessionKey.split('_').slice(1, 4).join('_') || 'ADNI_Unknown';
      const visit = meta.visit || 'sc';
      const days = parseAdniDays(visit);

      const adniFsDir = path.join(adniDataDir, sessionKey);
      const fsBrain = path.join(adniFsDir, 'brain.mgz');
      const fsAseg = path.join(adniFsDir, 'aseg.mgz');
      const rawFile = path.join(adniRawDataDir, `${sessionKey}.nii.gz`);

      let volumeUrl = null;
      let maskUrl = null;
      let sessionType = 'ADNI FreeSurfer 7.x Volume (.mgz)';

      if (fs.existsSync(fsBrain)) {
        volumeUrl = `/@fs${fsBrain}`;
        if (fs.existsSync(fsAseg)) {
          maskUrl = `/@fs${fsAseg}`;
        }
      } else if (fs.existsSync(rawFile)) {
        volumeUrl = `/@fs${rawFile}`;
        sessionType = 'ADNI 1.5T 3D MRI Volume (.nii.gz)';
      }

      if (!volumeUrl) continue;

      // Cached 2D slices (may be missing for sessions whose extraction is not finished)
      const adniSessionSliceDir = path.join(adniSliceCacheDir, sessionKey);
      let adniSliceUrls = [];
      if (fs.existsSync(adniSessionSliceDir)) {
        const sliceFiles = fs.readdirSync(adniSessionSliceDir)
          .filter(f => f.endsWith('.png'))
          .sort((a, b) => {
            const numA = parseInt(a.split('_').pop().replace('.png', ''), 10) || 0;
            const numB = parseInt(b.split('_').pop().replace('.png', ''), 10) || 0;
            return numA - numB;
          });
        adniSliceUrls = sliceFiles.map(f => `/@fs${path.join(adniSessionSliceDir, f)}`);
      }

      if (!subjectsMap.has(subjectId)) {
        subjectsMap.set(subjectId, {
          id: subjectId,
          cohort: 'ADNI',
          age: meta.age_at_scan || 'Unknown',
          gender: meta.gender || 'Unknown',
          apoe: meta.apoe || 'Unknown',
          education: meta.education || 'N/A',
          sessions: []
        });
      }

      const subject = subjectsMap.get(subjectId);
      subject.sessions.push({
        id: `${visit}_${meta.image_uid || sessionKey.split('_').pop()}`,
        sessionId: sessionKey,
        days: days,
        type: sessionType,
        date: `Visit ${visit.toUpperCase()}`,
        url: volumeUrl,
        maskUrl: maskUrl,
        sliceUrls: adniSliceUrls,
        diagnosis: meta.diagnosis || 'Unknown',
        cdr: meta.cdr || 'N/A',
        mmse: meta.mmse || 'N/A'
      });
    }
  }

  // Synthetic demo sessions (same layout as OASIS-3: <patient>_MR_d<days>)
  if (fs.existsSync(dummyDataDir)) {
    let dummyMetadata = {};
    if (fs.existsSync(dummyClinicalMetadataPath)) {
      try {
        dummyMetadata = JSON.parse(fs.readFileSync(dummyClinicalMetadataPath, 'utf8'));
      } catch(e) {
        console.warn("[WARN] Could not parse dummy_data/clinical_metadata.json");
      }
    }

    for (const dir of fs.readdirSync(dummyDataDir, { withFileTypes: true })) {
      if (!dir.isDirectory() || dir.name.startsWith('.')) continue;
      const sessionName = dir.name; // e.g. BWP01_MR_d0000
      const parts = sessionName.split('_');
      if (parts.length < 3) continue;

      const physicalBrain = path.join(dummyDataDir, sessionName, 'brain.mgz');
      const physicalAseg = path.join(dummyDataDir, sessionName, 'aseg.mgz');
      if (!fs.existsSync(physicalBrain)) continue;

      const clinicalInfo = dummyMetadata[sessionName] || {};
      const subjectId = clinicalInfo.subject_id || parts[0];
      const days = parseInt(parts[2].replace('d', ''), 10) || 0;

      const sessionSliceDir = path.join(dummySliceCacheDir, sessionName);
      let sliceUrls = [];
      if (fs.existsSync(sessionSliceDir)) {
        sliceUrls = fs.readdirSync(sessionSliceDir)
          .filter(f => f.endsWith('.png'))
          .sort((a, b) => {
            const numA = parseInt(a.split('_').pop().replace('.png', ''), 10) || 0;
            const numB = parseInt(b.split('_').pop().replace('.png', ''), 10) || 0;
            return numA - numB;
          })
          .map(f => `/@fs${path.join(sessionSliceDir, f)}`);
      }

      if (!subjectsMap.has(subjectId)) {
        subjectsMap.set(subjectId, {
          id: subjectId,
          cohort: 'Synthetic',
          age: clinicalInfo.age_at_scan || 'Unknown',
          gender: clinicalInfo.gender || 'Unknown',
          apoe: clinicalInfo.apoe || 'Unknown',
          education: clinicalInfo.education || 'N/A',
          sessions: []
        });
      }

      subjectsMap.get(subjectId).sessions.push({
        id: parts[2],
        sessionId: sessionName,
        days: days,
        type: 'Synthetic BrainWeb phantom, FastSurfer volume (.mgz)',
        date: `Day ${days}`,
        url: `/@fs${path.join(dummyDataDir, sessionName, 'brain.mgz')}`,
        maskUrl: fs.existsSync(physicalAseg) ? `/@fs${physicalAseg}` : null,
        sliceUrls: sliceUrls,
        diagnosis: clinicalInfo.diagnosis || 'Unknown',
        cdr: clinicalInfo.cdr || 'N/A',
        mmse: clinicalInfo.mmse || 'N/A'
      });
    }
  }

  // Sort sessions by day and subjects by id, then write the manifest
  const subjects = Array.from(subjectsMap.values()).map(sub => {
    sub.sessions.sort((a, b) => a.days - b.days);
    return sub;
  });

  subjects.sort((a, b) => a.id.localeCompare(b.id));

  fs.writeFileSync(outputFile, JSON.stringify(subjects, null, 2));
  console.log(`[OK] Generated manifest with ${subjects.length} total subjects (OASIS-3 + ADNI + synthetic demo).`);
}

generateManifest();
