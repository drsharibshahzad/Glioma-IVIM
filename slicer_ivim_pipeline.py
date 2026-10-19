import os
import glob
import csv
import numpy as np
import slicer
import vtk
import SegmentStatistics

# ================= CONFIGURATION =================
COHORT_DIR = r"C:\IVIM_Thesis_Cohort"
MASTER_CSV = os.path.join(COHORT_DIR, "All_45_Patients_IVIM_Stats.csv")
ROI_NAMES = ["Core", "InnerRim_Edema", "OuterRim", "Perilesional", "Contra_NAWM"]
# =================================================


def find_map_file(search_dirs, pattern_End):
    """Finds a map file matching pattern_End across candidate directories."""
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        # Match both *b0_ref.nii.gz and *b0_ref_b200.nii.gz styles
        matches = glob.glob(os.path.join(d, f"*{pattern_End}*.nii*"))
        if matches:
            return sorted(matches)[-1]
    return None


def load_and_register(patient_folder_name, b_cutoff=None):
    """
    Auto-detects and loads skull-stripped b0_ref, D, D*, and f maps (with any prefix or b-tag),
    applies clinical Rainbow colormaps, co-registers structural scans (T1C, FLAIR, T2, T1),
    and initializes pre-labeled ROI segments in Segment Editor.
    """
    slicer.mrmlScene.Clear(0)
    p_root = os.path.join(COHORT_DIR, patient_folder_name)
    if not os.path.exists(p_root):
        raise FileNotFoundError(f"Patient folder not found: {p_root}")

    # Search directly inside Patient_XX, NIfTI_Ready, and any *_IVIM_Maps* subfolders
    subfolders = [f for f in glob.glob(os.path.join(p_root, "*")) if os.path.isdir(f)]
    search_dirs = [p_root, os.path.join(p_root, "NIfTI_Ready")] + subfolders

    tag = f"b{int(b_cutoff)}" if b_cutoff else ""

    b0_path = find_map_file(search_dirs, f"{tag}_b0_ref" if tag else "b0_ref")
    d_path = find_map_file(search_dirs, f"{tag}_D_map" if tag else "D_map")
    ds_path = find_map_file(search_dirs, f"{tag}_D_star_map" if tag else "D_star_map")
    f_path = find_map_file(search_dirs, f"{tag}_f_map" if tag else "f_map")

    for name, path in [("b0_ref", b0_path), ("D_map", d_path), ("D_star_map", ds_path), ("f_map", f_path)]:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Could not find '{name}' inside {p_root}. Check that your maps are in this folder!")

    print(f"Loading detected maps from: {os.path.dirname(b0_path)}")
    b0_node = slicer.util.loadVolume(b0_path)
    b0_node.SetName("b0_ref")

    d_node = slicer.util.loadVolume(d_path)
    d_node.SetName("D_map")

    ds_node = slicer.util.loadVolume(ds_path)
    ds_node.SetName("D_star_map")

    f_node = slicer.util.loadVolume(f_path)
    f_node.SetName("f_map")

    # Apply Rainbow color table & clinical ranges
    rainbow_id = slicer.util.getNode("Rainbow").GetID()
    for node, upper in [(d_node, 3.0), (ds_node, 50.0), (f_node, 40.0)]:
        disp = node.GetDisplayNode()
        disp.SetAndObserveColorNodeID(rainbow_id)
        disp.AutoWindowLevelOff()
        disp.SetWindowLevelMinMax(0.0, upper)

    # Find anatomical scans (T1, T1C, T2, FLAIR) in any of the patient folders
    anat_files = []
    for d in search_dirs:
        if os.path.exists(d):
            for f in glob.glob(os.path.join(d, "*.nii*")):
                low = os.path.basename(f).lower()
                if not any(k in low for k in ["b0_", "d_map", "d_star_map", "f_map", "dwi", "ivim", "ep2d"]):
                    if f not in anat_files:
                        anat_files.append(f)

    reg_nodes = []
    for af in anat_files:
        anat_node = slicer.util.loadVolume(af)
        out_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"Reg_{anat_node.GetName()}")
        params = {
            "fixedVolume": b0_node.GetID(),
            "movingVolume": anat_node.GetID(),
            "outputVolume": out_node.GetID(),
            "useRigid": True,
            "useAffine": True,
            "initializeTransformMode": "useMomentsAlign",
            "samplingPercentage": 0.05
        }
        print(f"Co-registering {anat_node.GetName()} to b0_ref (Rigid + Affine)...")
        slicer.cli.runSync(slicer.modules.brainsfit, None, params)
        slicer.mrmlScene.RemoveNode(anat_node)
        reg_nodes.append(out_node)

    # Create pre-labeled Segmentation node ready for drawing
    seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ROIs")
    seg_node.CreateDefaultDisplayNodes()
    master_vol = reg_nodes[0] if reg_nodes else b0_node
    seg_node.SetReferenceImageGeometryParameterFromVolumeNode(master_vol)
    for r_name in ROI_NAMES:
        seg_node.GetSegmentation().AddEmptySegment(r_name, r_name)

    slicer.util.selectModule("SegmentEditor")
    print(f"=== {patient_folder_name} loaded & ready! Draw ROIs, then run: export_patient_stats('{patient_folder_name}') ===")


def auto_grow_shells(shell_thickness_mm=5.0):
    """
    After drawing 'Core', automatically generates 3 concentric 5-mm hollow rings outward
    ('InnerRim_Edema' 0-5mm, 'OuterRim' 5-10mm, 'Perilesional' 10-15mm) on the active axial slice(s).
    """
    seg_node = slicer.util.getNode("ROIs")
    ref_vol = slicer.util.getNode("b0_ref")
    core_arr = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, "Core", ref_vol)

    if np.sum(core_arr) == 0:
        raise ValueError("Please draw the 'Core' segment first before running auto_grow_shells()!")

    from scipy.ndimage import distance_transform_edt
    spacing = ref_vol.GetSpacing()
    inplane_spacing = (spacing[1], spacing[0])

    inner_arr = np.zeros_like(core_arr, dtype=np.uint8)
    outer_arr = np.zeros_like(core_arr, dtype=np.uint8)
    peri_arr = np.zeros_like(core_arr, dtype=np.uint8)

    active_slices = np.where(np.any(core_arr > 0, axis=(1, 2)))[0]
    for z in active_slices:
        slice_core = core_arr[z] > 0
        dist_mm = distance_transform_edt(~slice_core, sampling=inplane_spacing)
        inner_arr[z] = ((dist_mm > 0) & (dist_mm <= shell_thickness_mm)).astype(np.uint8)
        outer_arr[z] = ((dist_mm > shell_thickness_mm) & (dist_mm <= 2 * shell_thickness_mm)).astype(np.uint8)
        peri_arr[z] = ((dist_mm > 2 * shell_thickness_mm) & (dist_mm <= 3 * shell_thickness_mm)).astype(np.uint8)

    slicer.util.updateSegmentBinaryLabelmapFromArray(inner_arr, seg_node, "InnerRim_Edema", ref_vol)
    slicer.util.updateSegmentBinaryLabelmapFromArray(outer_arr, seg_node, "OuterRim", ref_vol)
    slicer.util.updateSegmentBinaryLabelmapFromArray(peri_arr, seg_node, "Perilesional", ref_vol)
    print(f"=== Generated {shell_thickness_mm}mm concentric shells (InnerRim_Edema, OuterRim, Perilesional)! ===")


def export_patient_stats(patient_folder_name):
    """
    Extracts Mean, SD, Min, Max across D, D*, and f maps simultaneously, computes
    normalized contralateral ratios (rD, rD*, rf), and appends 1 row to All_45_Patients_IVIM_Stats.csv.
    """
    seg_node = slicer.util.getNode("ROIs")
    maps = {
        "D": slicer.util.getNode("D_map"),
        "Dstar": slicer.util.getNode("D_star_map"),
        "f": slicer.util.getNode("f_map")
    }

    results = {}
    for m_key, m_node in maps.items():
        logic = SegmentStatistics.SegmentStatisticsLogic()
        p_node = logic.getParameterNode()
        p_node.SetParameter("Segmentation", seg_node.GetID())
        p_node.SetParameter("ScalarVolume", m_node.GetID())
        p_node.SetParameter("ScalarVolumeSegmentStatisticsPlugin.mean.enabled", str(True))
        p_node.SetParameter("ScalarVolumeSegmentStatisticsPlugin.stdev.enabled", str(True))
        p_node.SetParameter("ScalarVolumeSegmentStatisticsPlugin.min.enabled", str(True))
        p_node.SetParameter("ScalarVolumeSegmentStatisticsPlugin.max.enabled", str(True))
        logic.computeStatistics()
        stats = logic.getStatistics()

        for r_name in ROI_NAMES:
            seg_id = seg_node.GetSegmentation().GetSegmentIdBySegmentName(r_name)
            for stat_key, col_suffix in [("mean", "Mean"), ("stdev", "SD"), ("min", "Min"), ("max", "Max")]:
                val = stats.get((seg_id, f"ScalarVolumeSegmentStatisticsPlugin.{stat_key}"), "")
                results[f"{r_name}_{m_key}_{col_suffix}"] = round(float(val), 4) if val != "" else ""

    headers = ["Folder_Name"]
    for r_name in ["Core", "InnerRim_Edema"]:
        for m_key in ["D", "Dstar", "f"]:
            for st in ["Mean", "SD", "Min", "Max"]:
                headers.append(f"{r_name}_{m_key}_{st}")
    for r_name in ["OuterRim", "Perilesional"]:
        for m_key in ["D", "Dstar", "f"]:
            headers.append(f"{r_name}_{m_key}_Mean")
    for m_key in ["D", "Dstar", "f"]:
        for st in ["Mean", "SD", "Min", "Max"]:
            headers.append(f"Contra_NAWM_{m_key}_{st}")

    for r_name, label in [("Core", "Core"), ("InnerRim_Edema", "Edema")]:
        for m_key in ["D", "Dstar", "f"]:
            num = results.get(f"{r_name}_{m_key}_Mean", "")
            den = results.get(f"Contra_NAWM_{m_key}_Mean", "")
            ratio_col = f"r{m_key}_{label}"
            headers.append(ratio_col)
            results[ratio_col] = round(num / den, 4) if (num != "" and den not in ["", 0]) else ""

    row_dict = {"Folder_Name": patient_folder_name}
    for h in headers[1:]:
        row_dict[h] = results.get(h, "")

    write_header = not os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)

    print(f"=== SUCCESS! {patient_folder_name} D, D*, and f stats appended as 1 row to: {MASTER_CSV} ===")