import os
import sys
import glob
import shutil
import subprocess
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_erosion, label
from dipy.segment.mask import median_otsu

# ================= CONFIGURATION =================
COHORT_DIR = r"D:\IVIM_Thesis_Cohort"

# Physiological upper bounds (Federau 2017 & Glioma IVIM standards)
MAX_D = 4.0       # x 10^-3 mm^2/s (0.004 mm^2/s)
MAX_DSTAR = 50.0  # x 10^-3 mm^2/s (0.050 mm^2/s)
MAX_F = 40.0      # % (0.40 perfusion fraction)
# =================================================


def find_dcm2niix():
    """Finds dcm2niix executable in system PATH or Python Scripts folder."""
    exe = shutil.which("dcm2niix")
    if exe:
        return exe
    py_scripts = os.path.join(os.path.dirname(sys.executable), "Scripts", "dcm2niix.exe")
    if os.path.exists(py_scripts):
        return py_scripts
    return "dcm2niix"


def extract_brain_mask_b0(S0, affine, header, out_dir, b_tag, prefix=""):
    """Generates a 3D binary brain mask and skull-stripped b0 volume."""
    raw_b0_path = os.path.join(out_dir, f"{prefix}b0_raw_{b_tag}.nii.gz")
    brain_b0_path = os.path.join(out_dir, f"{prefix}b0_ref_{b_tag}.nii.gz")
    mask_path = os.path.join(out_dir, f"{prefix}b0_brain_mask_{b_tag}.nii.gz")

    nib.save(nib.Nifti1Image(S0.astype(np.float32), affine, header), raw_b0_path)

    if shutil.which("mri_synthstrip"):
        print("   -> Running SynthStrip brain extraction on b0...")
        subprocess.run(["mri_synthstrip", "-i", raw_b0_path, "-o", brain_b0_path, "-m", mask_path], check=True)
        brain_mask = nib.load(mask_path).get_fdata() > 0
        S0_brain = S0 * brain_mask
        return S0_brain, brain_mask

    print("   -> Running 3D Otsu + Morphological Brain Extraction (Skull Stripping) on b0...")
    _, mask = median_otsu(S0, median_radius=3, numpass=4)

    # Keep only the largest 3D connected component (removes detached scalp/neck/eyeball islands)
    labeled_array, num_features = label(mask)
    if num_features > 0:
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0
        largest_label = component_sizes.argmax()
        mask = (labeled_array == largest_label)

    # Fill internal ventricular/tumor cavities and erode 1 boundary voxel
    mask = binary_fill_holes(mask)
    for z in range(mask.shape[2]):
        mask[:, :, z] = binary_fill_holes(mask[:, :, z])

    mask = binary_erosion(mask, iterations=1)
    S0_brain = np.where(mask, S0, 0.0).astype(np.float32)

    nib.save(nib.Nifti1Image(S0_brain, affine, header), brain_b0_path)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine, header), mask_path)
    return S0_brain, mask


def fit_ivim_from_nifti(dwi_path, bval_path, out_dir, cutoff_b, prefix=""):
    """
    Direction-averages multi-b volumes, skull-strips b0, and fits the two-step
    segmented bi-exponential IVIM model (with 1500 s/mm^2 upper cap to prevent kurtosis bias).
    """
    img = nib.load(dwi_path)
    data = img.get_fdata(dtype=np.float32)
    b_tag = f"b{int(round(cutoff_b))}"

    with open(bval_path, "r") as f:
        bvals = np.array([float(x) for x in f.read().split()], dtype=np.float32)

    if data.ndim != 4 or data.shape[3] != len(bvals):
        return False, f"Skipped {os.path.basename(dwi_path)} (not a matching 4D series)."

    # Group b-values to nearest 5 s/mm^2 and direction-average (powder average)
    bvals_rounded = np.round(bvals / 5.0) * 5.0
    unique_b = np.unique(bvals_rounded)

    if not (np.any((unique_b > 0) & (unique_b < cutoff_b)) and np.any(unique_b >= cutoff_b)):
        return False, f"Skipped non-IVIM scan (b-values: {unique_b.tolist()})"

    print(f"   -> Direction-averaging {len(bvals)} volumes into {len(unique_b)} b-shells: {unique_b.tolist()}")
    shell_data = np.zeros(data.shape[:3] + (len(unique_b),), dtype=np.float32)
    for i, b in enumerate(unique_b):
        shell_data[..., i] = np.mean(data[..., bvals_rounded == b], axis=-1)

    # 1. Extract b=0 and run 3D Skull Stripping
    b0_mask = unique_b <= 5
    S0_raw = np.mean(shell_data[..., b0_mask], axis=-1) if np.any(b0_mask) else shell_data[..., 0]
    _, brain_mask = extract_brain_mask_b0(S0_raw, img.affine, img.header, out_dir, b_tag, prefix=prefix)

    # 2. Mild 3D spatial smoothing across b-shells to suppress Rician noise spikes
    for i in range(len(unique_b)):
        shell_data[..., i] = gaussian_filter(shell_data[..., i], sigma=(0.6, 0.6, 0.3))

    S0_smooth = np.mean(shell_data[..., b0_mask], axis=-1) if np.any(b0_mask) else shell_data[..., 0]
    vox_signals = shell_data[brain_mask]
    vox_S0 = np.maximum(S0_smooth[brain_mask], 1e-4)
    eps = 1e-6

    # 3. STEP 1: Fit D using high b-values (cutoff_b <= b <= 1500 s/mm^2 to avoid kurtosis bias)
    high_b_mask = (unique_b >= cutoff_b) & (unique_b <= 1500)
    if np.sum(high_b_mask) < 2:
        high_b_mask = unique_b >= cutoff_b

    b_high = unique_b[high_b_mask]
    S_high = np.maximum(vox_signals[:, high_b_mask], eps)
    Y_high = np.log(np.clip(S_high / vox_S0[:, None], 1e-5, 1.0))

    A_high = np.vstack([-b_high, np.ones_like(b_high)]).T
    coeffs_high, _, _, _ = np.linalg.lstsq(A_high, Y_high.T, rcond=None)
    D_vox = np.clip(coeffs_high[0], 0.0, MAX_D * 1e-3)
    f_init = np.clip(1.0 - np.exp(coeffs_high[1]), 0.0, MAX_F / 100.0)

    # 4. STEP 2: Hold D fixed and fit D* and f across low/all b-values
    low_b_mask = (unique_b > 0) & (unique_b <= cutoff_b)
    b_low = unique_b[low_b_mask]
    S_low_norm = np.clip(vox_signals[:, low_b_mask] / vox_S0[:, None], 1e-5, 1.0)
    slow_comp = (1.0 - f_init[:, None]) * np.exp(-D_vox[:, None] * b_low[None, :])
    residual = np.maximum(S_low_norm - slow_comp, eps)

    f_safe = np.maximum(f_init[:, None], 1e-3)
    Y_low = np.log(np.clip(residual / f_safe, 1e-4, 1.0))
    D_star_vox = -np.sum(Y_low * b_low[None, :], axis=1) / np.sum(b_low ** 2)
    D_star_vox = np.clip(D_star_vox, np.maximum(D_vox * 1.5, 0.002), MAX_DSTAR * 1e-3)

    # Refine f across b-values <= 1500 s/mm^2 holding D and D* fixed
    fit_b_mask = (unique_b > 0) & (unique_b <= 1500)
    if np.sum(fit_b_mask) < 2:
        fit_b_mask = unique_b > 0

    all_b = unique_b[fit_b_mask]
    S_all_norm = np.clip(vox_signals[:, fit_b_mask] / vox_S0[:, None], 1e-5, 1.0)
    E_slow = np.exp(-D_vox[:, None] * all_b[None, :])
    E_fast = np.exp(-D_star_vox[:, None] * all_b[None, :])
    diff_E = E_fast - E_slow
    f_refined = np.sum((S_all_norm - E_slow) * diff_E, axis=1) / (np.sum(diff_E ** 2, axis=1) + 1e-8)
    f_vox = np.clip(f_refined, 0.0, MAX_F / 100.0)

    # 5. Reconstruct 3D Skull-Stripped Maps in clinical units
    D_map = np.zeros(S0_raw.shape, dtype=np.float32)
    D_star_map = np.zeros(S0_raw.shape, dtype=np.float32)
    f_map = np.zeros(S0_raw.shape, dtype=np.float32)

    D_map[brain_mask] = np.clip(D_vox * 1000.0, 0.0, MAX_D)
    D_star_map[brain_mask] = np.clip(D_star_vox * 1000.0, 0.0, MAX_DSTAR)
    f_map[brain_mask] = np.clip(f_vox * 100.0, 0.0, MAX_F)

    nib.save(nib.Nifti1Image(D_map, img.affine, img.header), os.path.join(out_dir, f"{prefix}D_map_{b_tag}.nii.gz"))
    nib.save(nib.Nifti1Image(D_star_map, img.affine, img.header), os.path.join(out_dir, f"{prefix}D_star_map_{b_tag}.nii.gz"))
    nib.save(nib.Nifti1Image(f_map, img.affine, img.header), os.path.join(out_dir, f"{prefix}f_map_{b_tag}.nii.gz"))

    return True, (
        f"Success! {b_tag} Skull-stripped maps saved.\n"
        f"   -> Brain Mean ± SD | D: {D_map[brain_mask].mean():.2f}±{D_map[brain_mask].std():.2f}, "
        f"D*: {D_star_map[brain_mask].mean():.2f}±{D_star_map[brain_mask].std():.2f}, "
        f"f: {f_map[brain_mask].mean():.2f}±{f_map[brain_mask].std():.2f}%"
    )


def process_all_patients(cutoff_b):
    dcm2niix_exe = find_dcm2niix()
    patient_dirs = sorted(glob.glob(os.path.join(COHORT_DIR, "Patient_*")))

    if not patient_dirs:
        print(f"\n[Warning] No folders starting with 'Patient_' found in {COHORT_DIR}")
        return

    for p_dir in patient_dirs:
        p_name = os.path.basename(p_dir)
        raw_dicom = os.path.join(p_dir, "Raw_DICOM")
        nifti_dir = os.path.join(p_dir, "NIfTI_Ready")
        os.makedirs(nifti_dir, exist_ok=True)

        # Convert DICOMs if Raw_DICOM exists and NIfTI_Ready has no .bval files yet
        if os.path.exists(raw_dicom) and not glob.glob(os.path.join(nifti_dir, "*.bval*")):
            print(f"\n[{p_name}] Converting DICOMs to NIfTI via dcm2niix...")
            cmd = [dcm2niix_exe, "-z", "y", "-f", "%d_%s", "-o", nifti_dir, raw_dicom]
            subprocess.run(cmd, check=False)

        # Search for .bval files in both NIfTI_Ready and directly inside Patient_XX
        bval_files = glob.glob(os.path.join(nifti_dir, "*.bval*")) + glob.glob(os.path.join(p_dir, "*.bval*"))
        if not bval_files:
            print(f"[{p_name}] No .bval files found in {p_dir}. Skipping.")
            continue

        fitted = False
        for bval_file in bval_files:
            base = bval_file[:-5] if bval_file.lower().endswith(".bval") else os.path.splitext(bval_file)[0]
            nii_candidates = [
                f for f in glob.glob(base + ".nii*")
                if not any(tag in os.path.basename(f).lower() for tag in ["d_map", "d_star_map", "f_map", "b0_ref", "b0_raw", "brain_mask"])
            ]
            for nii_path in nii_candidates:
                ok, msg = fit_ivim_from_nifti(nii_path, bval_file, nifti_dir, cutoff_b, prefix="")
                if ok:
                    print(f"[{p_name}] {msg}")
                    fitted = True
                    break
            if fitted:
                break


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" Glioma IVIM-DWI Batch Processor")
    print("=" * 50)

    user_input = input(
        "\nEnter the cutoff b-value (s/mm²) to separate perfusion from diffusion [Press Enter for default: 200]: "
    ).strip()

    try:
        cutoff_b = float(user_input) if user_input else 200.0
    except ValueError:
        print("\nInvalid input detected. Defaulting to 200.0 s/mm².")
        cutoff_b = 200.0

    print(f"\nStarting batch processing for all patients using cutoff b = {cutoff_b:g} s/mm²...\n")
    process_all_patients(cutoff_b)
    print("\nBatch processing complete.")