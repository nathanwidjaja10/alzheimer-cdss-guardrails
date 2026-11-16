// Constants of the Anatomical ROI Reference pop-up, kept in a plain module so React Fast Refresh keeps working.

// Colour table for the compact labels 1..5 written by generate_roi_demo.py
// (index 0 is the background, which NiiVue renders transparent)
export const ROI_LUT = {
  R: [0, 232, 245, 140, 38, 150],
  G: [0, 67, 166, 180, 214, 175],
  B: [0, 59, 66, 235, 196, 205],
  I: [0, 1, 2, 3, 4, 5],
  labels: ['', 'Hippocampus', 'Amygdala', 'Lateral ventricle', 'Temporal horn', 'Third ventricle'],
};

// Legend entries shown next to the viewer
export const ROI_LEGEND = [
  { name: 'Hippocampus', color: 'rgb(232,67,59)', note: 'aseg 17/53 — memory center, earliest atrophy' },
  { name: 'Amygdala', color: 'rgb(245,166,66)', note: 'aseg 18/54 — shrinks alongside the hippocampus from early on' },
  { name: 'Lateral ventricle temporal horn', color: 'rgb(38,214,196)', note: 'aseg 5/44 — enlarges following MTL atrophy (most visible in the coronal plane)' },
  { name: 'Lateral ventricle (body/atrium)', color: 'rgb(140,180,235)', note: 'aseg 4/43 — ex vacuo enlargement' },
  { name: 'Third ventricle', color: 'rgb(150,175,205)', note: 'aseg 14 — midline ex vacuo enlargement' },
];
