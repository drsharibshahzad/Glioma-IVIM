# Glioma-IVIM: Preoperative Glioma Characterization via Bi-Exponential IVIM-DWI

**Study Title:** *Intravoxel Incoherent Motion Diffusion-Weighted Imaging (IVIM-DWI) for Non-invasive Characterization of Preoperative Gliomas: Correlation with Tumor Grade, Isocitrate Dehydrogenase (IDH) Status, and Ki-67 Index*  
**Institution:** Department of Radiodiagnosis, Jawaharlal Institute of Postgraduate Medical Education and Research (JIPMER), Puducherry, India

---

## 1. Overview & Scientific Rationale

Under the WHO 2021 Classification of CNS Tumors, preoperative glioma characterization relies on molecular and proliferative biomarkers—specifically **Isocitrate Dehydrogenase-1 (IDH1)** mutation status and the **Ki-67 (MIB-1)** proliferation index—which traditionally require invasive neurosurgical biopsy. Conventional mono-exponential Diffusion-Weighted Imaging (ADC) conflates true restricted water diffusion within hypercellular tumor tissue with capillary microcirculation.

**Intravoxel Incoherent Motion (IVIM-DWI)** utilizes multi-$b$-value diffusion acquisitions ($0\text{–}2000\text{ s/mm}^2$) to separate parenchymal diffusivity from microvascular perfusion without exogenous gadolinium contrast:

$$\frac{S(b)}{S_0} = f \cdot \exp(-b \cdot D^*) + (1 - f) \cdot \exp(-b \cdot D)$$

* **$D$ (True Diffusion Coefficient, $\times 10^{-3}\text{ mm}^2/\text{s}$):** Reflects pure intracellular/extracellular water mobility inversely correlated with tumor cellularity.
* **$D^*$ (Pseudo-Diffusion Coefficient, $\times 10^{-3}\text{ mm}^2/\text{s}$):** Reflects incoherent blood flow velocity within the randomly oriented capillary network.
* **$f$ (Perfusion Fraction, $\%$):** Reflects the fractional volume of capillary microcirculation within each voxel.

---

## 2. Key Pipeline Features

1. **Multi-Direction Shell Averaging (Powder Averaging):** Automatically groups and averages multi-directional diffusion volumes (e.g., 298-volume `.bval`/`.bvec` acquisitions) into isotropic $b$-value shells prior to curve fitting to eliminate white matter anisotropy bias.
2. **Automated 3D $b_0$ Brain Extraction (Skull Stripping):** Uses 3D Otsu thresholding (`dipy.segment.mask.median_otsu`), 3D largest connected-component isolation, and morphological erosion on the $b_0$ volume to strip low-SNR background air, scalp fat, skull marrow, and paranasal sinuses before fitting—eliminating extra-cranial outlier artifacts.
3. **Two-Step Segmented Bi-Exponential Fitting:**
   * **Step 1 ($b \ge b_{\text{cutoff}}$, default $200\text{ s/mm}^2$):** Fits mono-exponential decay $\ln(S(b)/S_0) = \ln(1 - f_{\text{init}}) - bD$ at high $b$-values where microvascular pseudo-diffusion is negligible.
   * **Step 2 (Fixed $D$):** Holds $D$ fixed from Step 1 and fits $D^*$ and $f$ across low/all $b$-values within physiological cerebral bounds ($D \in [0, 4.0]\times 10^{-3}\text{ mm}^2/\text{s}$, $D^* \in [D, 50.0]\times 10^{-3}\text{ mm}^2/\text{s}$, $f \in [0, 40.0]\%$).
4. **Interactive Drag-and-Drop GUI (`ivim_gui.py`):** Supports both raw 3T DICOM folders and NIfTI files, prompts for the cutoff $b$-value prior to fitting, and tags all generated NIfTI maps with the selected cutoff (e.g., `_b200_D_map.nii.gz`).
5. **3D Slicer Co-Registration & Single-Row Statistical Export (`slicer_ivim_pipeline.py`):** Automatically co-registers structural scans (`T1+C`, `FLAIR`, `T2W`, `T1W`) to the skull-stripped `b0_ref` volume using 12-DOF Rigid + Affine `BRAINSFit`, generates 5-mm concentric peritumoral shells, and exports `Mean`, `SD`, `Min`, `Max`, and contralateral normalized ratios (`rD`, `rD*`, `rf`) across all ROIs as a single horizontal row per patient for direct import into IBM SPSS.

---

## 3. Repository Structure

```text
Glioma-IVIM/
├── README.md                          # Step-by-step documentation
├── requirements.txt                   # Python package dependencies
├── .gitignore                         # Excludes confidential patient DICOM/NIfTI files
├── ivim_gui.py                        # Standalone PyQt5 Drag-and-Drop GUI
├── batch_1_dicom_to_ivim.py           # Automated 45-patient batch DICOM-to-IVIM script
├── slicer_ivim_pipeline.py            # 3D Slicer Python console automation script
└── templates/
    ├── IVIM_Glioma_Master_SPSS.xlsx   # 81-column SPSS-ready master database & codebook
    └── Data_Collection_Proforma.docx  # Clinical & histopathological proforma
```

---

## 4. Installation & Prerequisites (100% Native Windows)

### A. Python Environment Setup
1. Install **Python 3.9+** (or Anaconda/Miniconda) on Windows.
2. Clone this repository and install the required packages via Command Prompt:
   ```cmd
   git clone [https://github.com/](https://github.com/)<your-username>/Glioma-IVIM.git
   cd Glioma-IVIM
   pip install -r requirements.txt
   ```
   *(Note: `pip install -r requirements.txt` automatically installs `dcm2niix`, `dipy`, `nibabel`, `scipy`, `numpy`, and `PyQt5`).*

### B. 3D Slicer Setup
1. Download and install **3D Slicer** (v5.6 or higher) from [slicer.org](https://www.slicer.org/).
2. *(Optional)* Open 3D Slicer's **Extension Manager** and install `SlicerElastix`, `SlicerMedSAM`, or `MONAIAuto3DSeg` for AI-assisted 3D tumor core segmentation.

---

## 5. Step-by-Step Usage Guide

### Workflow A: Interactive Single-Patient GUI (`ivim_gui.py`)

Use `ivim_gui.py` during the pilot feasibility phase to test different $b$-value cutoffs (e.g., `b150` vs. `b200` vs. `b250`) or process individual patient scans:

1. Launch the GUI from Command Prompt:
   ```cmd
   python ivim_gui.py
   ```
2. **Drag and Drop Input:**
   * Drag a **raw 3T DICOM patient folder** directly onto the window (automatically converts via `dcm2niix` into a `Converted_NIfTI` subfolder and selects the multi-$b$ IVIM series), **OR**
   * Drag a pre-converted **`.nii` / `.nii.gz` file** (automatically pairs with its companion `.bval` and `.bvec`/`.bvac` files).
3. **Confirm Cutoff $b$-Value:**
   * A popup dialog displays the total number of volumes (e.g., `298`) and all unique $b$-value shells detected in the scan.
   * Enter your desired cutoff $b$-value (default: `200` $\text{s/mm}^2$) and click **OK**.
4. **Tagged Output Files:**
   * Creates a folder named `<scan_prefix>_IVIM_Maps_b200` containing:
     * `<scan_prefix>_b200_b0_ref.nii.gz` *(Skull-stripped $b_0$ anatomical reference)*
     * `<scan_prefix>_b200_b0_brain_mask.nii.gz` *(3D binary brain mask)*
     * `<scan_prefix>_b200_D_map.nii.gz` *(True diffusion $D$ in $10^{-3}\text{ mm}^2/\text{s}$)*
     * `<scan_prefix>_b200_D_star_map.nii.gz` *(Pseudo-diffusion $D^*$ in $10^{-3}\text{ mm}^2/\text{s}$)*
     * `<scan_prefix>_b200_f_map.nii.gz` *(Perfusion fraction $f$ in $\%$)*

---

### Workflow B: Automated 45-Patient Batch Processing (`batch_1_dicom_to_ivim.py`)

To process your entire 45-patient prospective cohort automatically:

1. Organize your main cohort directory on your hard drive:
   ```text
   C:\IVIM_Thesis_Cohort\
    ├── Patient_01\
    │    └── Raw_DICOM\   <-- Copy raw 3T DICOM export here
    ├── Patient_02\
    │    └── Raw_DICOM\
    ...
    └── Patient_45\
         └── Raw_DICOM\
   ```
2. Run the batch processor:
   ```cmd
   python batch_1_dicom_to_ivim.py
   ```
3. For every patient folder, the script automatically converts raw DICOMs into `Patient_XX\NIfTI_Ready\`, skull-strips the $b_0$ volume, and generates `b0_ref.nii.gz`, `b0_brain_mask.nii.gz`, `D_map.nii.gz`, `D_star_map.nii.gz`, and `f_map.nii.gz`.

---

### Workflow C: 3D Slicer Co-Registration, ROI Segmentation & One-Row Export (`slicer_ivim_pipeline.py`)

1. Open **3D Slicer** and press **`Ctrl + 3`** to open the **Python Console**.
2. Copy and paste the contents of `slicer_ivim_pipeline.py` into the console once per session.
3. **Load & Co-Register Patient Scans:**
   ```python
   load_and_register("Patient_01")
   ```
   * Loads `b0_ref`, `D_map`, `D_star_map`, and `f_map` with clinical `Rainbow` lookup tables (`0.0–3.0` for $D$, `0.0–50.0` for $D^*$, `0.0–40.0` for $f$).
   * Co-registers all structural sequences (`T1+C`, `FLAIR`, `T2W`) to the skull-stripped `b0_ref` volume using **Rigid (6 DOF) + Affine (12 DOF)** `BRAINSFit` alignment.
   * Initializes a `Segmentation` node named `ROIs` with pre-labeled segments: `Core`, `InnerRim_Edema`, `OuterRim`, `Perilesional`, and `Contra_NAWM`.
4. **Delineate ROIs (Blinded to Histopathology, Axial Plane Only):**
   * **Solid Tumor Core (`Core`):** Outline the solid enhancing tumor (or central hyperintense lesion in non-enhancing gliomas) on the representative axial slice, carefully excluding necrosis, cysts, hemorrhage, calcification, and macro-vessels.
   * **Peritumoral Zones (`InnerRim_Edema`, `OuterRim`, `Perilesional`):** Either outline the FLAIR hyperintense edema manually, or run:
     ```python
     auto_grow_shells(5.0)
     ```
     to automatically expand three concentric 5-mm rings outward from `Core` on the active axial slice(s).
   * **Contralateral Normal White Matter (`Contra_NAWM`):** Place a mirror-image circular ROI in healthy contralateral centrum semiovale / white matter.
5. **Export All Metrics into One Horizontal Table Row:**
   ```python
   export_patient_stats("Patient_01")
   ```
   * Simultaneously extracts `Mean`, `SD`, `Min`, and `Max` across `D_map`, `D_star_map`, and `f_map`, computes normalized contralateral ratios (`rD_Core`, `rDstar_Core`, `rf_Core`, `rD_Edema`, `rDstar_Edema`, `rf_Edema`), and appends a single horizontal row to `C:\IVIM_Thesis_Cohort\All_45_Patients_IVIM_Stats.csv`.

---

## 6. Master Spreadsheet & SPSS Statistical Analysis Plan

Open `All_45_Patients_IVIM_Stats.csv` and paste the extracted rows directly into **Columns Y–BT** of `templates/IVIM_Glioma_Master_SPSS.xlsx` (`Master_Data_SPSS` tab).

### Summary of Statistical Plan (IBM SPSS / R)
* **Normality Testing:** Shapiro–Wilk test on all continuous variables (`Mean ± SD` for parametric; `Median [IQR]` for non-parametric).
* **Binary Group Comparisons (LGG [WHO Grade 1–2] vs. HGG [WHO Grade 3–4], IDH-mutant [$\ge 10\%$] vs. IDH-wildtype [$< 10\%$], High Ki-67 [$\ge 10\%$] vs. Low Ki-67 [$< 10\%$]):** Independent samples *t*-test (parametric) or Mann–Whitney *U* test (non-parametric).
* **Multi-Grade Comparisons (WHO Grades 2, 3, and 4):** One-way ANOVA or Kruskal–Wallis test with post-hoc Bonferroni–Dunn correction.
* **Proliferation Correlation:** Pearson’s ($r$) or Spearman’s rank ($\rho$) correlation with continuous Ki-67 index (%).
* **Diagnostic Performance:** Receiver Operating Characteristic (ROC) curve analysis (AUC with 95% CI, sensitivity, specificity), optimal cutoffs via Youden’s index, binary logistic regression for multi-parameter combinations, and DeLong’s test for AUC comparisons.
* **Within-Patient Spatial Gradients (Core vs. Inner Rim vs. Outer Rim vs. Perilesional vs. Contralateral NAWM):** Repeated Measures ANOVA (parametric) or Friedman test (non-parametric).
* **Confounder & Reliability Analyses:** Stratified subgroup analysis for prior corticosteroid exposure (`0 = Naive`, `1 = Exposed`) and Intraclass Correlation Coefficient (ICC with 95% CI) across two blinded radiologists.

---

## 7. Key References

1. **Togao O, et al.** Differentiation of high-grade and low-grade diffuse gliomas by intravoxel incoherent motion MR imaging. *Neuro-Oncology*. 2016;18(1):132–141. [doi:10.1093/neuonc/nov147](https://doi.org/10.1093/neuonc/nov147)
2. **Bai Y, et al.** Grading of gliomas by using monoexponential, biexponential, and stretched exponential diffusion-weighted MR imaging and diffusion kurtosis MR imaging. *Radiology*. 2016;278(2):496–504. [doi:10.1148/radiol.2015142173](https://doi.org/10.1148/radiol.2015142173)
3. **Gu T, et al.** Evaluation of gliomas peritumoral diffusion and prediction of IDH1 mutation by IVIM-DWI. *Aging (Albany NY)*. 2021;13(7):9948–9959. [doi:10.18632/aging.202751](https://doi.org/10.18632/aging.202751)
4. **Luo H, et al.** The diagnostic value of intravoxel incoherent motion imaging in differentiating high-grade from low-grade gliomas: a systematic review and meta-analysis. *Br J Radiol*. 2021;94(1121):20201321. [doi:10.1259/bjr.20201321](https://doi.org/10.1259/bjr.20201321)
5. **Federau C.** Intravoxel incoherent motion MRI as a means to measure in vivo perfusion: A review of the evidence. *NMR Biomed*. 2017;30(11):e3780. [doi:10.1002/nbm.3780](https://doi.org/10.1002/nbm.3780)
6. **Federau C, et al.** Quantitative measurement of brain perfusion with intravoxel incoherent motion MR imaging. *Radiology*. 2014;265(3):874–881. [doi:10.1148/radiol.12120584](https://doi.org/10.1148/radiol.12120584)
7. **Cao M, et al.** A three-component multi-b-value diffusion-weighted imaging might be a useful biomarker for detecting microstructural features in gliomas with differences in malignancy and IDH-1 mutation status. *Eur Radiol*. 2023;33(4):2871–2880. [doi:10.1007/s00330-022-09212-5](https://doi.org/10.1007/s00330-022-09212-5)
8. **Guo D, Jiang B.** Noninvasively evaluating the grade and IDH mutation status of gliomas by using mono-exponential, bi-exponential diffusion-weighted imaging and three-dimensional pseudo-continuous arterial spin labeling. *Eur J Radiol*. 2023;160:110721. [doi:10.1016/j.ejrad.2023.110721](https://doi.org/10.1016/j.ejrad.2023.110721)
9. **Karschnia P, et al.** Prognostic validation of a new classification system for extent of resection in glioblastoma: A report of the RANO resect group. *Neuro-Oncology*. 2023;25(5):940–954. [doi:10.1093/neuonc/noac193](https://doi.org/10.1093/neuonc/noac193)