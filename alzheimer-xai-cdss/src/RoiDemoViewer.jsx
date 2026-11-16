import { useEffect, useRef } from 'react';
import { Niivue, SLICE_TYPE, SHOW_RENDER } from '@niivue/niivue';
import { ROI_LUT } from './roiDemoData';

// Two-volume NiiVue panel of the Anatomical ROI Reference pop-up: brain.mgz with the AD target structures
// (labels 1..5) as an overlay. A new <canvas> is created for each effect run and removed on cleanup, so the
// double invocation of effects in React StrictMode cannot leave two viewers on one WebGL context.
export default function RoiDemoViewer({ brainUrl, roiUrl }) {
  const hostRef = useRef(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !brainUrl || !roiUrl) return;
    let cancelled = false;

    // Fresh canvas for this effect run
    const canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    host.appendChild(canvas);

    const nv = new Niivue({
      backColor: [0.05, 0.06, 0.09, 1],
      show3Dcrosshair: true,
      crosshairColor: [1, 1, 1, 0.4],
      isColorbar: false,
      isOrientCube: true,
      fontMinPx: 13,
    });
    nv.attachToCanvas(canvas);

    // Load both volumes, apply the label colour table and show the multiplanar view with a 3D render
    (async () => {
      try {
        await nv.loadVolumes([{ url: brainUrl }, { url: roiUrl, opacity: 1.0 }]);
        if (cancelled || !nv.volumes[1]) return;
        nv.volumes[1].setColormapLabel(ROI_LUT);
        nv.volumes[1].opacity = 1.0;
        nv.opts.multiplanarShowRender = SHOW_RENDER.ALWAYS;
        nv.setSliceType(SLICE_TYPE.MULTIPLANAR);
        nv.setRenderAzimuthElevation(135, 15);
        nv.updateGLVolume();
        nv.drawScene();
      } catch (err) {
        console.error('RoiDemoViewer load error:', err);
      }
    })();

    // Release the viewer and the WebGL context
    return () => {
      cancelled = true;
      try {
        nv.cleanup?.();
        nv.gl?.getExtension('WEBGL_lose_context')?.loseContext();
      } catch { /* ignore teardown errors */ }
      canvas.remove();
    };
  }, [brainUrl, roiUrl]);

  return <div ref={hostRef} className="roi-demo-canvas" />;
}
