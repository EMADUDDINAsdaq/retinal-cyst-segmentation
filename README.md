# Retinal Cyst Segmentation in OCT Images
### A Classical (Non-CNN) Hybrid Segmentation Pipeline

Coursework project for Newcastle University's MSc Data Science & AI programme (module **CSC8628**, Autumn 2025). Designs, implements, and evaluates a fully classical image-processing pipeline that detects and segments retinal cysts in Optical Coherence Tomography (OCT) scans — without using any Convolutional Neural Network.

Full written report: [`report/Hybrid_Image_Segmentation_Report.pdf`](report/Hybrid_Image_Segmentation_Report.pdf)
Notebook: [`notebooks/Retinal_Cyst_Segmentation.ipynb`](notebooks/Retinal_Cyst_Segmentation.ipynb)

---

## 1. Background

Retinal cysts are fluid-filled, hypo-reflective (dark) lesions that form inside retinal tissue. They are a key biomarker in several sight-threatening eye diseases, most notably **diabetic macular oedema (DME)** and **age-related macular degeneration (AMD)**. Optical Coherence Tomography (OCT) is the clinical gold-standard imaging modality for visualising retinal microstructure at micrometre-level precision, and it is the primary tool clinicians use to detect, size, and track these cysts over time.

Accurately segmenting cysts from OCT scans matters for three clinical reasons:
- **Diagnosis** — confirming the presence and extent of fluid accumulation.
- **Treatment planning** — informing decisions such as anti-VEGF injection timing.
- **Monitoring** — tracking how a lesion changes across repeat scans to judge treatment response.

Manual delineation by clinicians is accurate but labour-intensive, slow, and subject to inter-grader variability — it does not scale to high-volume screening. This motivates automated segmentation.

## 2. Problem Statement & Brief

This project was set as the CSC8628 assignment (Autumn 2025, Newcastle University). The brief required:

1. **Image segmentation** — design and implement an original segmentation algorithm for retinal cysts in OCT scans, at the pixel level, **without using Convolutional Neural Networks (CNNs)** — classical computer-vision methods only. Literature-inspired design and iterative, staged improvement from a baseline were explicitly permitted and encouraged.
2. **Evaluation** — compare predicted segmentations against ground-truth annotations using a measurable metric, specifically **Mean Intersection over Union (mIoU)**.
3. **Reporting** — present results with graphs, tables, and sample segmented images, and write a structured report (introduction, methodology, implementation, results, discussion, conclusions) referencing at least 2–3 works from the current literature.

The task deliberately scopes out disease classification or higher-level recognition — the focus is segmentation only, and the explicit exclusion of CNNs pushes the solution toward classical, explainable image-processing techniques (thresholding, morphology, texture filtering, region-growing/watershed) rather than learned feature representations. This matters in practice too: classical pipelines are lightweight, require no training data or GPU, and are fully interpretable — relevant where computational resources, labelled data, or explainability requirements rule out deep learning.

## 3. Dataset

The coursework-provided OCT dataset consists of **10 training images** (`TRAINING1.tif`–`TRAINING10.tif`), each with a corresponding pixel-level ground-truth annotation mask marking the cyst regions.

**Access:** this dataset was distributed privately through the university's CSC8628 course platform for this assignment and is not publicly redistributable, so it is *not included in this repository*. To run the notebook, point `img_dir` at a folder of raw `.tif` OCT scans and `ann_dir` at a matching folder of ground-truth masks (same filenames).

## 4. Solution — Why This Approach

Early in development, several simpler baselines were tried and found insufficient on their own:
- **K-means clustering** on pixel intensities — too sensitive to global intensity variation across scans, poor at separating cysts from other dark, low-texture regions.
- **Global (Otsu) thresholding alone** — reliably separates retina tissue from background, but is not discriminative enough to isolate cysts specifically.
- **Raw watershed segmentation** — oversegments noisy OCT texture without prior candidate-region constraints.

These experiments converged on a design principle drawn from the literature on classical OCT cyst segmentation [1]–[3]: no single cue is sufficient, but **combining intensity, texture, and geometric/morphological cues** is robust. Specifically:
- Cysts are typically **dark with smooth (low-texture) interiors** [1].
- Retinal layer structure introduces **strong horizontal gradients** that must be excluded to avoid false positives [2].
- **Region-based segmentation constrained to the plausible retina area** improves reliability over unconstrained approaches [3].

The final design is a **six-phase hybrid classical pipeline**, each phase targeting one of these cues and refining the output of the previous phase.

## 5. Methodology

```
Input OCT Image
      │
      ▼
┌─────────────────────────────┐
│ 1. Pre-processing            │  Median + Gaussian smoothing, intensity normalisation
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ 2. Retina Region Extraction  │  Otsu thresholding, Canny edges, anatomical band prior,
│    (Region of Interest)      │  hole-filling
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ 3. Cyst Enhancement          │  Percentile contrast stretch, light Gaussian smoothing
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ 4. Strict Cyst Extraction    │  CLAHE, darkest-12% pixel selection, local adaptive
│                               │  thresholding, morphological refinement
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ 5. Entropy-Guided Watershed  │  Entropy filtering, distance transform + peak detection,
│    Refinement                │  watershed splitting, shape-based morphological constraints
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ 6. Evaluation & Output       │  Dice coefficient, IoU, visual overlays
└─────────────────────────────┘
```

### 5.1 Pre-processing

Each raw OCT image is converted to `float32` and normalised to the range [0, 255], which is required for the contrast and thresholding operations downstream to behave consistently across scans:

```
I_norm = 255 · (I − min(I)) / (max(I) − min(I) + ε)
```

A **median filter** (size 3) removes salt-and-pepper / speckle noise, which is common in OCT acquisition. A **Gaussian filter** (σ = 1.0) then smooths homogeneous regions while preserving retinal structure, following the pre-processing approach used by Fabritius et al. [2] to stabilise texture descriptors for later steps.

### 5.2 Retina Region Extraction (Region of Interest)

Because cysts only occur within retinal tissue, isolating the retina region first prevents false positives elsewhere in the image and reduces computation. Three signals are combined:

1. **Otsu thresholding** — coarse global separation of tissue from background. Not precise enough to find cysts directly, but reliable for isolating the retina from surrounding noise.
2. **Canny edge detection** (weak thresholds 0.1 / 0.3) — reinforces the transition between retinal layers while preserving internal gradients.
3. **Anatomical band prior** (rows 80–300) — a fixed vertical band derived from dataset analysis, following González et al. [1], who similarly restrict segmentation to the plausible retina region:

```
Mask = OtsuMask ∩ CannyEdges ∩ [80:300 rows]
```

4. **Hole filling** — binary fill-holes produces a single coherent, closed retina region suitable for region-based segmentation.
5. **Mask application** — everything outside the filled retina mask is zeroed out, mirroring the region-limiting pre-processing described by Chiu et al. [3].

### 5.3 Cyst Enhancement

Cysts present as dark, low-contrast, hypo-reflective pockets, so a local enhancement step improves their separability before extraction:

1. **Percentile-based contrast stretching** (1st–99th percentile) within the ROI, which normalises local contrast without amplifying noise:
   ```
   I' = rescale_intensity(I_roi, p2, p98)
   ```
2. **Light Gaussian smoothing** (σ = 0.6) to remove minor irregularities, consistent with smoothing strategies used in region-flooding literature [1].

### 5.4 Strict Cyst Extraction (core of the pipeline)

This phase produces the first candidate cyst mask and is where most of the pipeline's precision comes from:

- **CLAHE** (Contrast-Limited Adaptive Histogram Equalisation, `clip_limit = 0.02`) applied within the ROI, based on its established effectiveness for local contrast enhancement in OCT segmentation:
  ```
  I_clahe = equalize_adapthist(I / 255, clip_limit=0.02) · 255
  ```
- **Darkest-12% pixel selection** — González et al. [1] observed that cyst pixels occupy the lowest intensity percentile of the retinal structure. Following this, the 12th percentile of `I_clahe` within the ROI is computed and pixels darker than this threshold are treated as cyst candidates.
- **Local adaptive thresholding** (block size 35, offset 0.03 — values tuned empirically through repeated visual evaluation) adapts the threshold to local intensity variation rather than using one global cut-off.
- **Morphological refinement**: opening (disk = 2) removes small noise specks, closing (disk = 3) seals small gaps in candidate regions, and small-object removal (< 20 px) eliminates tiny false positives.

### 5.5 Entropy-Guided Watershed Refinement

This phase refines the Phase 4 output by combining texture analysis with region-flooding (watershed) splitting:

- **Entropy filtering** — cyst interiors have low local texture, while the surrounding retinal microstructure has high entropy. Rank entropy is computed with a disk of radius 3, and a threshold of 5.5 (set via visual analysis and experimentation) keeps only the low-entropy regions that overlap with cyst candidates:
  ```
  E(x, y) = entropy(I, disk(3))
  ```
- **Watershed segmentation** — a distance transform plus local-maxima detection generates markers, and `watershed(-D, markers, mask=entropy_mask)` splits merged cyst clusters into separate, more shape-accurate regions.
- **Morphological constraints** (post-watershed) — opening (disk = 2), closing (disk = 3), removal of regions smaller than 15 px, and shape filters (eccentricity < 0.98, extent < 0.95) discard elongated artefacts and residual retinal-layer fragments, following a strategy similar to Fabritius et al. [2].

The output of this phase is the `refined_mask` — the final predicted cyst segmentation.

### 5.6 Evaluation

Predicted masks are compared against ground truth per image using:

```
Dice = 2|P ∩ G| / (|P| + |G|)
IoU  = |P ∩ G| / |P ∪ G|
mIoU = (1/N) · Σ IoU_i
```

## 6. Implementation Details

Implemented entirely in Python on Google Colab.

| Library | Role |
|---|---|
| `scikit-image` | Core classical CV toolkit — filters, thresholding (Otsu, local adaptive), CLAHE, Canny edges, morphology, rank entropy, watershed segmentation |
| `scipy.ndimage` | Median filtering, binary hole-filling, Euclidean distance transform |
| `scikit-learn` | `jaccard_score` (IoU) metric computation |
| `scipy.spatial.distance` | Dice coefficient computation |
| `NumPy` | Array operations and metric aggregation |
| `Matplotlib` | Qualitative visualisation — 5-column overlay plots, Dice/IoU graphs |

`scikit-image` was chosen as the backbone because it provides essentially every classical computer-vision primitive the brief required — adaptive thresholding, morphological operations, texture/entropy filtering, and watershed — directly aligned with the Segmentation Fundamentals and Segmentation Algorithms lecture content the method is built on. This combination deliberately satisfies the assignment's core constraint: classical image-processing algorithms only, no CNN-based or learned methods anywhere in the pipeline.

## 7. Results

### 7.1 Quantitative

Evaluated across all 10 provided OCT training images:

| Image | Dice Coefficient | IoU |
|---|---|---|
| TRAINING1 | 0.3831 | 0.2370 |
| TRAINING2 | 0.5771 | 0.4056 |
| TRAINING3 | 0.0557 | 0.0286 |
| TRAINING4 | 0.5983 | 0.4268 |
| TRAINING5 | 0.5398 | 0.3697 |
| TRAINING6 | 0.5208 | 0.3521 |
| TRAINING7 | 0.1519 | 0.0822 |
| TRAINING8 | 0.5737 | 0.4022 |
| TRAINING9 | 0.2690 | 0.1554 |
| TRAINING10 | 0.4846 | 0.3198 |

**Average Dice: 0.4154 · Mean IoU (mIoU): 0.2779**

### 7.2 Qualitative

For each image, the notebook renders a five-panel comparison: the original (Gaussian-smoothed) scan, the masked retina ROI, the model's predicted final mask, the resized ground truth, and the per-image Dice/IoU scores — reproduced in full in the report (`report/Hybrid_Image_Segmentation_Report.pdf`, Figures 2–3).

### 7.3 What the results mean

Performance is strongest (IoU ≈ 0.35–0.43) on images with clearly delineated, moderately sized cyst clusters — TRAINING2, 4, 5, 6, and 8 — where the darkest-12% and entropy cues align well with the true cyst boundaries. Performance drops sharply on TRAINING3 and TRAINING7 (IoU < 0.09), where cysts are faint, small, or have irregular boundaries that don't cleanly separate from surrounding low-entropy tissue — the fixed entropy (< 5.5) and darkest-percentile (12%) thresholds, tuned on this dataset as a whole, don't adapt per-image and so under- or over-segment these harder cases. Watershed splitting is additionally sensitive to small local texture variation, which compounds the effect on lower-contrast scans.

## 8. Discussion

**Strengths.** Combining intensity, texture, and morphological cues avoids the weaknesses of any single-strategy approach (as seen in the discarded K-means/global-threshold/raw-watershed baselines). Entropy-guided watershed effectively separates merged cyst clusters, while darkest-12% selection gives high precision on clearly hypo-reflective regions. The pipeline is transparent, lightweight (no GPU or training required), and fully reproducible.

**Limitations.** The key thresholds — entropy < 5.5 and the darkest-12% cut-off — were optimised specifically for this dataset and are not expected to generalise unchanged to OCT scans from different machines or acquisition protocols. Watershed segmentation is also sensitive to small texture variation, which explains the lower scores on fainter or more irregular images.

**Comparison with literature.** The overall method aligns closely with González et al. [1], who similarly combine contrast enhancement with texture-based refinement; the entropy-based filtering follows the texture-driven approach of Fabritius et al. [2]; and the watershed-based splitting resembles the boundary-detection strategy of Chiu et al. [3]. The hybrid combination used here helps control over-segmentation and improves cyst isolation compared with using region-flooding alone.

## 9. Conclusion

This project developed a fully classical, six-phase retinal cyst segmentation pipeline for OCT scans — combining CLAHE enhancement, percentile-based dark-region detection, anatomically constrained retina-region extraction, entropy filtering, and watershed refinement — achieving a mean IoU of 0.2779 (mean Dice 0.4154) across the 10 provided training images, without using any CNN. The pipeline satisfies the assignment's core requirement (classical methods only) while remaining explainable, lightweight, and reproducible, and its design choices are consistently grounded in prior OCT cyst-segmentation literature [1]–[3].

**Future directions** suggested in the report: adaptive (rather than fixed) entropy and intensity thresholding, learning-based threshold selection, and extending the pipeline to 3-D volumetric OCT segmentation.

## 10. Repository Structure

```
retinal-cyst-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Retinal_Cyst_Segmentation.ipynb
└── report/
    └── Hybrid_Image_Segmentation_Report.pdf
```

## 11. Setup & Running the Notebook

Developed and run on **Google Colab**. The notebook mounts Google Drive and reads OCT images/annotations directly from a personal Drive folder (`drive.mount(...)`, then `img_dir` / `ann_dir` paths under `/content/drive/My Drive/Assignment/OCT_Dataset/`).

To run it elsewhere:

1. **Remove or skip** the `google.colab` import and `drive.mount(...)` cell.
2. **Obtain the dataset** (see [Dataset](#3-dataset) above) and set your own paths:
   ```python
   img_dir = "path/to/OCT_Dataset/img_dir"   # raw .tif OCT scans
   ann_dir = "path/to/OCT_Dataset/ann_dir"   # ground-truth .tif masks
   res_dir = "path/to/OCT_Results"           # where predicted masks are saved
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run cells top to bottom** — core function definitions (pre-processing, ROI extraction, cyst enhancement, strict extraction, entropy/watershed refinement), then the main loop over `TRAINING1`–`TRAINING10`, then the final results table and Dice/IoU graphs.

## 12. References

[1] A. González et al., "Automatic Cyst Detection in OCT Retinal Images Combining Region Flooding and Texture Analysis," 2017.
[2] T. Fabritius et al., "Segmentation of Cystoid Macular Edema in Optical Coherence Tomography," 2009.
[3] S. J. Chiu et al., "Automatic Segmentation of the Choroid in OCT Images," 2012.

---

*Author: Emaduddin Asdaq Syed Mohammed — School of Computing, Newcastle University.*
