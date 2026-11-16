import React, { useEffect, useRef } from 'react';
import { Niivue } from '@niivue/niivue';

// 3D/multiplanar NiiVue viewer for a brain volume with an optional segmentation overlay
export default function NiiVueViewer({ volumeUrl, maskUrl }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !volumeUrl) return;

    // Create the viewer
    const nv = new Niivue({
      dragAndDropEnabled: true,
      backColor: [0, 0, 0, 1],
      show3Dcrosshair: true,
      textHeight: 0.04
    });
    
    nv.attachToCanvas(canvasRef.current);
    
    // Load the volume and, when given, the segmentation overlay
    const loadVolume = async () => {
      try {
        const volumes = [{ url: volumeUrl }];
        if (maskUrl) {
          volumes.push({ 
            url: maskUrl, 
            colormap: 'freesurfer', 
            opacity: 0.5 
          });
        }
        await nv.loadVolumes(volumes);
      } catch (err) {
        console.error("Error loading NIfTI volume:", err);
      }
    };
    
    loadVolume();

    return () => {
      // Nothing to release explicitly
    };
  }, [volumeUrl]);

  return <canvas ref={canvasRef} id="gl"></canvas>;
}
