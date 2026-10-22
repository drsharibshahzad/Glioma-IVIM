import sys
import os
import glob
import time
import shutil
import subprocess
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_erosion, label
from dipy.segment.mask import median_otsu
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QInputDialog, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Physiological brain IVIM upper limits (Federau 2017 & Glioma standards)
MAX_D = 4.0       # x 10^-3 mm^2/s
MAX_DSTAR = 50.0  # x 10^-3 mm^2/s
MAX_F = 40.0      # %


def strip_nii_ext(filepath):
    """Removes .nii.gz or .nii to get the clean base filename."""
    base = os.path.basename(filepath)
    if base.lower().endswith(".nii.gz"):
        return base[:-7]
    elif base.lower().endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


def find_dcm2niix_executable():
    """Finds dcm2niix in Python Scripts, system PATH, or MRIcroGL folders."""
    exe = shutil.which("dcm2niix")
    if exe:
        return exe

    py_scripts = os.path.join(os.path.dirname(sys.executable), "Scripts", "dcm2niix.exe")
    if os.path.exists(py_scripts):
        return py_scripts

    common_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcm2niix.exe"),
        r"C:\Program Files\MRIcroGL\Resources\dcm2niix.exe",
        r"C:\MRIcroGL\Resources\dcm2niix.exe",
        os.path.expanduser(r"~\Desktop\MRIcroGL\Resources\dcm2niix.exe"),
        os.path.expanduser(r"~\Downloads\MRIcroGL\Resources\dcm2niix.exe"),
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    return None


def read_bvals_flexible(bval_path):
    """Reads space, tab, or newline separated b-values (e.g., 298 values)."""
    with open(bval_path, "r") as f:
        return np.array([float(x) for x in f.read().split()], dtype=np.float32)


class IVIMWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str, float, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, dwi_path, bval_path, out_dir, prefix, cutoff_b):
        super().__init__()
        self.dwi_path = dwi_path
        self.bval_path = bval_path
        self.out_dir = out_dir
        self.prefix = prefix
        self.cutoff_b = float(cutoff_b)
        self.b_tag = f"b{int(round(self.cutoff_b))}"

    def extract_brain_mask_b0(self, S0, affine, header):
        """Skull-strips b0 using 3D Otsu + largest connected component + morphology."""
        b0_brain_path = os.path.join(self.out_dir, f"{self.prefix}_{self.b_tag}_b0_ref.nii.gz")
        mask_path = os.path.join(self.out_dir, f"{self.prefix}_{self.b_tag}_b0_brain_mask.nii.gz")

        _, mask = median_otsu(S0, median_radius=3, numpass=4)

        labeled_array, num_features = label(mask)
        if num_features > 0:
            component_sizes = np.bincount(labeled_array.ravel())
            component_sizes[0] = 0
            mask = (labeled_array == component_sizes.argmax())

        mask = binary_fill_holes(mask)
        for z in range(mask.shape[2]):
            mask[:, :, z] = binary_fill_holes(mask[:, :, z])
        mask = binary_erosion(mask, iterations=1)

        S0_brain = np.where(mask, S0, 0.0).astype(np.float32)
        nib.save(nib.Nifti1Image(S0_brain, affine, header), b0_brain_path)
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine, header), mask_path)
        return S0_brain, mask

    def run(self):
        try:
            t0 = time.time()
            os.makedirs(self.out_dir, exist_ok=True)

            self.progress.emit(10, "Loading 4D IVIM NIfTI and b-values...")
            img = nib.load(self.dwi_path)
            data = img.get_fdata(dtype=np.float32)
            bvals = read_bvals_flexible(self.bval_path)

            if data.ndim != 4 or data.shape[3] != len(bvals):
                raise ValueError(
                    f"4D volume count ({data.shape[3] if data.ndim == 4 else '3D'}) "
                    f"does not match .bval count ({len(bvals)})."
                )

            bvals_rounded = np.round(bvals / 5.0) * 5.0
            unique_b = np.unique(bvals_rounded)

            low_b_mask = (unique_b > 0) & (unique_b < self.cutoff_b)
            # Cap high b-values at 1500 s/mm^2 to avoid non-Gaussian kurtosis bias (Luo et al. / Su et al.)
            high_b_mask = (unique_b >= self.cutoff_b) & (unique_b <= 1500)
            if np.sum(high_b_mask) < 2:
                high_b_mask = unique_b >= self.cutoff_b  # Fallback if scan only has b=0, low, and >1500

            if np.sum(low_b_mask) == 0 or np.sum(high_b_mask) < 2:
                raise ValueError(
                    f"With cutoff b = {self.cutoff_b:g} s/mm², you need at least one low b-value "
                    f"(0 < b < {self.cutoff_b:g}) and two high b-values (b >= {self.cutoff_b:g}).\n"
                    f"Detected b-shells: {unique_b.tolist()}"
                )

            self.progress.emit(25, f"Direction-averaging {len(bvals)} volumes into {len(unique_b)} b-shells...")
            shell_data = np.zeros(data.shape[:3] + (len(unique_b),), dtype=np.float32)
            for i, b in enumerate(unique_b):
                shell_data[..., i] = np.mean(data[..., bvals_rounded == b], axis=-1)

            self.progress.emit(40, "Running 3D Brain Extraction (Skull Stripping) on b0...")
            S0_raw = np.mean(shell_data[..., b0_mask], axis=-1) if np.any(b0_mask) else shell_data[..., 0]
            _, brain_mask = self.extract_brain_mask_b0(S0_raw, img.affine, img.header)

            for i in range(len(unique_b)):
                shell_data[..., i] = gaussian_filter(shell_data[..., i], sigma=(0.6, 0.6, 0.3))

            S0_smooth = np.mean(shell_data[..., b0_mask], axis=-1) if np.any(b0_mask) else shell_data[..., 0]
            vox_signals = shell_data[brain_mask]
            vox_S0 = np.maximum(S0_smooth[brain_mask], 1e-4)
            eps = 1e-6

            self.progress.emit(65, f"Step 1/2: Fitting True Diffusion (D) for b >= {self.cutoff_b:g} s/mm²...")
            b_high = unique_b[high_b_mask]
            S_high = np.maximum(vox_signals[:, high_b_mask], eps)
            Y_high = np.log(np.clip(S_high / vox_S0[:, None], 1e-5, 1.0))

            A_high = np.vstack([-b_high, np.ones_like(b_high)]).T
            coeffs_high, _, _, _ = np.linalg.lstsq(A_high, Y_high.T, rcond=None)
            D_vox = np.clip(coeffs_high[0], 0.0, MAX_D * 1e-3)
            f_init = np.clip(1.0 - np.exp(coeffs_high[1]), 0.0, MAX_F / 100.0)

            self.progress.emit(80, f"Step 2/2: Holding D fixed & fitting D* and f (cutoff b = {self.cutoff_b:g})...")
            b_low = unique_b[low_b_mask]
            S_low_norm = np.clip(vox_signals[:, low_b_mask] / vox_S0[:, None], 1e-5, 1.0)
            slow_comp = (1.0 - f_init[:, None]) * np.exp(-D_vox[:, None] * b_low[None, :])
            residual = np.maximum(S_low_norm - slow_comp, eps)

            f_safe = np.maximum(f_init[:, None], 1e-3)
            Y_low = np.log(np.clip(residual / f_safe, 1e-4, 1.0))
            D_star_vox = -np.sum(Y_low * b_low[None, :], axis=1) / np.sum(b_low ** 2)
            D_star_vox = np.clip(D_star_vox, np.maximum(D_vox * 1.5, 0.002), MAX_DSTAR * 1e-3)

            all_b = unique_b[unique_b > 0]
            S_all_norm = np.clip(vox_signals[:, unique_b > 0] / vox_S0[:, None], 1e-5, 1.0)
            E_slow = np.exp(-D_vox[:, None] * all_b[None, :])
            E_fast = np.exp(-D_star_vox[:, None] * all_b[None, :])
            diff_E = E_fast - E_slow
            f_refined = np.sum((S_all_norm - E_slow) * diff_E, axis=1) / (np.sum(diff_E ** 2, axis=1) + 1e-8)
            f_vox = np.clip(f_refined, 0.0, MAX_F / 100.0)

            self.progress.emit(92, f"Saving {self.b_tag} Maps (D, D*, f, b0)...")
            D_map = np.zeros(S0_raw.shape, dtype=np.float32)
            D_star_map = np.zeros(S0_raw.shape, dtype=np.float32)
            f_map = np.zeros(S0_raw.shape, dtype=np.float32)

            D_map[brain_mask] = np.clip(D_vox * 1000.0, 0.0, MAX_D)
            D_star_map[brain_mask] = np.clip(D_star_vox * 1000.0, 0.0, MAX_DSTAR)
            f_map[brain_mask] = np.clip(f_vox * 100.0, 0.0, MAX_F)

            nib.save(nib.Nifti1Image(D_map, img.affine, img.header),
                     os.path.join(self.out_dir, f"{self.prefix}_{self.b_tag}_D_map.nii.gz"))
            nib.save(nib.Nifti1Image(D_star_map, img.affine, img.header),
                     os.path.join(self.out_dir, f"{self.prefix}_{self.b_tag}_D_star_map.nii.gz"))
            nib.save(nib.Nifti1Image(f_map, img.affine, img.header),
                     os.path.join(self.out_dir, f"{self.prefix}_{self.b_tag}_f_map.nii.gz"))

            summary = (
                f"Brain Mean ± SD ({self.b_tag}) -> "
                f"D: {D_map[brain_mask].mean():.2f}±{D_map[brain_mask].std():.2f} | "
                f"D*: {D_star_map[brain_mask].mean():.2f}±{D_star_map[brain_mask].std():.2f} | "
                f"f: {f_map[brain_mask].mean():.2f}±{f_map[brain_mask].std():.2f}%"
            )
            elapsed = time.time() - t0
            self.progress.emit(100, "Done!")
            self.finished_ok.emit(self.out_dir, elapsed, summary)

        except Exception as e:
            self.error_occurred.emit(str(e))


class IVIMDropWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setWindowTitle("Glioma-IVIM Map Generator (DICOM & NIfTI + Custom b-Cutoff)")
        self.resize(600, 400)

        self.layout = QVBoxLayout()

        top_bar = QHBoxLayout()
        top_label = QLabel("Default Cutoff b-value (s/mm²):")
        top_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.default_b_spin = QDoubleSpinBox()
        self.default_b_spin.setRange(10.0, 1000.0)
        self.default_b_spin.setSingleStep(10.0)
        self.default_b_spin.setDecimals(0)
        self.default_b_spin.setValue(200.0)
        self.default_b_spin.setStyleSheet("font-size: 13px; padding: 4px;")
        top_bar.addWidget(top_label)
        top_bar.addWidget(self.default_b_spin)
        top_bar.addStretch()
        self.layout.addLayout(top_bar)

        self.drop_label = QLabel(
            "Drag & Drop DICOM Folder or NIfTI Files Here\n\n"
            "• Drop a Raw 3T DICOM Patient Folder (auto-converts via dcm2niix)\n"
            "• Or drop a .nii / .nii.gz file (auto-finds .bval & .bvec/.bvac)\n"
            "• Or drop a folder containing NIfTI files\n\n"
            "(You will be prompted to confirm the cutoff b-value before fitting)"
        )
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet(
            "border: 3px dashed #555; font-size: 14px; font-weight: bold; "
            "padding: 25px; background-color: #f4f4f4; border-radius: 8px;"
        )
        self.layout.addWidget(self.drop_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Waiting for DICOM folder or NIfTI files...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 12px; color: #333; margin-top: 8px;")
        self.layout.addWidget(self.status_label)

        self.setLayout(self.layout)
        self.worker = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        dropped_paths = [u.toLocalFile() for u in event.mimeData().urls()]
        self.resolve_and_run(dropped_paths)

    def find_companion_file(self, folder, base_name, exts):
        for ext in exts:
            candidates = [
                os.path.join(folder, f"{base_name}.{ext}"),
                os.path.join(folder, f"{base_name}.{ext}.{ext}")
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c
        for ext in exts:
            all_matches = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(f".{ext}")]
            if len(all_matches) == 1:
                return all_matches[0]
        return None

    def convert_dicom_folder_to_nifti(self, dicom_folder):
        dcm2niix_exe = find_dcm2niix_executable()
        if not dcm2niix_exe:
            raise RuntimeError(
                "Could not find 'dcm2niix' to convert DICOMs!\n"
                "Run 'pip install dcm2niix' in Command Prompt, or place dcm2niix.exe next to ivim_gui.py."
            )

        conv_dir = os.path.join(dicom_folder, "Converted_NIfTI")
        os.makedirs(conv_dir, exist_ok=True)

        self.status_label.setText("Converting raw DICOM folder to NIfTI via dcm2niix... Please wait.")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        QApplication.processEvents()

        cmd = [dcm2niix_exe, "-z", "y", "-f", "%d_%s", "-o", conv_dir, dicom_folder]
        subprocess.run(cmd, check=False)

        bval_files = glob.glob(os.path.join(conv_dir, "*.bval*"))
        if not bval_files:
            raise RuntimeError("DICOM conversion finished, but no diffusion (.bval) series was found in that folder!")

        best_nii, best_bval = None, None
        max_shells = -1
        for bf in bval_files:
            base = bf[:-5] if bf.lower().endswith(".bval") else os.path.splitext(bf)[0]
            nii_matches = [f for f in glob.glob(base + ".nii*") if not f.endswith("_map.nii.gz")]
            if nii_matches:
                bvals = read_bvals_flexible(bf)
                n_shells = len(np.unique(np.round(bvals / 5.0) * 5.0))
                if n_shells > max_shells:
                    max_shells = n_shells
                    best_nii = nii_matches[0]
                    best_bval = bf

        return best_nii, best_bval

    def resolve_and_run(self, dropped_paths):
        dwi_path, bval_path = None, None

        try:
            if len(dropped_paths) == 1 and os.path.isdir(dropped_paths[0]):
                folder = dropped_paths[0]
                files = [os.path.join(folder, f) for f in os.listdir(folder)]
                nii_in_folder = [
                    f for f in files
                    if (f.lower().endswith(".nii.gz") or f.lower().endswith(".nii"))
                    and not f.lower().endswith(("_map.nii.gz", "_ref.nii.gz", "_mask.nii.gz"))
                ]
                if not nii_in_folder:
                    dwi_path, bval_path = self.convert_dicom_folder_to_nifti(folder)
                else:
                    files = nii_in_folder + [f for f in files if f.lower().endswith((".bval", ".bvec", ".bvac"))]
            else:
                files = dropped_paths

            if not dwi_path:
                for f in files:
                    low = f.lower()
                    if (low.endswith(".nii.gz") or low.endswith(".nii")) and not low.endswith(("_map.nii.gz", "_ref.nii.gz", "_mask.nii.gz")):
                        dwi_path = f
                    elif low.endswith(".bval"):
                        bval_path = f

                if not dwi_path and any(f.lower().endswith((".dcm", ".ima")) for f in files):
                    parent_dicom_dir = os.path.dirname(files[0])
                    dwi_path, bval_path = self.convert_dicom_folder_to_nifti(parent_dicom_dir)

            if not dwi_path:
                self.status_label.setText("Error: No NIfTI (.nii/.nii.gz) or DICOM series found in your drop!")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                return

            folder = os.path.dirname(dwi_path)
            base_name = strip_nii_ext(dwi_path)

            if not bval_path:
                bval_path = self.find_companion_file(folder, base_name, ["bval"])

            if not bval_path:
                self.status_label.setText(f"Error: Could not find matching .bval file for '{base_name}'.")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                return

            bvals = read_bvals_flexible(bval_path)
            unique_b = np.unique(np.round(bvals / 5.0) * 5.0).astype(int).tolist()

            cutoff_b, ok = QInputDialog.getDouble(
                self,
                "Select Cutoff b-Value",
                f"Scan: {base_name}\n"
                f"Total volumes: {len(bvals)} | Detected b-shells (s/mm²):\n{unique_b}\n\n"
                f"Enter Cutoff (Threshold) b-value (s/mm²):",
                value=self.default_b_spin.value(),
                min=10.0,
                max=2000.0,
                decimals=0
            )
            if not ok:
                self.status_label.setText("Status: Map generation cancelled by user.")
                self.status_label.setStyleSheet("color: #555;")
                return

            b_tag = f"b{int(round(cutoff_b))}"
            out_dir = os.path.join(folder, f"{base_name}_IVIM_Maps_{b_tag}")

            self.progress_bar.setValue(5)
            self.status_label.setText(f"Processing '{base_name}' with cutoff {b_tag} ({int(cutoff_b)} s/mm²)...")
            self.status_label.setStyleSheet("color: blue; font-weight: bold;")

            self.worker = IVIMWorker(dwi_path, bval_path, out_dir, base_name, cutoff_b)
            self.worker.progress.connect(self.on_progress)
            self.worker.finished_ok.connect(self.on_finished)
            self.worker.error_occurred.connect(self.on_error)
            self.worker.start()

        except Exception as e:
            self.on_error(str(e))

    def on_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.status_label.setText(msg)

    def on_finished(self, out_dir, elapsed, summary):
        self.status_label.setText(
            f"Success in {elapsed:.1f}s! Saved to:\n{out_dir}\n{summary}"
        )
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

    def on_error(self, err_msg):
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Error: {err_msg}")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IVIMDropWindow()
    window.show()
    sys.exit(app.exec_())