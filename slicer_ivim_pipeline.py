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


def load_and_register(patient_folder_name):
    """
    Loads skull-stripped b0_ref, D, D*, and f maps, applies clinical Rainbow colormaps,
    co-registers structural scans (T1C, FLAIR, T2, T1) using Rigid + Affine BRAINSFit,
    and initializes pre-labeled ROI segments in Segment Editor.
    """
    slicer.mrmlScene.Clear(0)
    p_dir = os.path.join(COHORT_DIR, patient_folder_name, "NIfTI_Ready")

    b0_node = slicer.util.loadVolume(os.path.join(p_dir, "b0_ref.nii.gz"))
    d_node = slicer.util.loadVolume(os.path.join(p_dir, "D_map.nii.gz"))
    ds_node = slicer.util.loadVolume(os.path.join(p_dir, "D_star_map.nii.gz"))
    f_node = slicer.util.loadVolume(os.path.join(p_dir, "f_map.nii.gz"))

    rainbow_id = slicer.util.getNode("Rainbow").GetID()
    for node, upper in [(d_node, 3.0), (ds_node, 50.0), (f_node, 40.0)]:
        disp = node.GetDisplayNode()
        disp.SetAndObserveColorNodeID(rainbow_id)
        disp.AutoWindowLevelOff()
        disp.SetWindowLevelMinMax(0.0, upper)

    anat_files = [
        f for f in glob.glob(os.path.join(p_dir, "*.nii*"))
        if not any(k in os.path.basename(f).lower() for k in ["b0_", "d_map", "d_star_map", "f_map", "dwi", "ivim", "ep2d"])
    ]

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
        print(f"Co-registering {anat_node.GetName()} to skull-stripped b0_ref (Rigid + Affine)...")
        slicer.cli.runSync(slicer.modules.brainsfit, None, params)
        slicer.mrmlScene.RemoveNode(anat_node)
        reg_nodes.append(out_node)

    seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ROIs")
    seg_node.CreateDefaultDisplayNodes()
    master_vol = reg_nodes[0] if reg_nodes else b0_node
    seg_node.SetReferenceImageGeometryParameterFromVolumeNode(master_vol)
    for r_name in ROI_NAMES:
        seg_node.GetSegmentation().AddEmptySegment(r_name, r_name)

    slicer.util.selectModule("SegmentEditor")
    print(f"=== {patient_folder_name} loaded & registered! Draw ROIs, then run: export_patient_stats('{patient_folder_name}') ===")


def auto_grow_shells(shell_thickness_mm=5.0):
    """
    Optional helper: After drawing 'Core', this automatically generates 3 concentric
    5-mm hollow rings outward ('InnerRim_Edema' 0-5mm, 'OuterRim' 5-10mm, 'Perilesional' 10-15mm)
    restricted to the axial plane(s) containing 'Core'.
    """
    seg_node = slicer.util.getNode("ROIs")
    ref_vol = slicer.util.getNode("b0_ref")
    core_arr = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, "Core", ref_vol)

    if np.sum(core_arr) == 0:
        raise ValueError("Please draw the 'Core' segment first before running auto_grow_shells()!")

    from scipy.ndimage import distance_transform_edt
    spacing = ref_vol.GetSpacing()  # (X, Y, Z) in mm
    inplane_spacing = (spacing[1], spacing[0])  # (Y, X) for 2D axial slices

    inner_arr = np.zeros_like(core_arr, dtype=np.uint8)
    outer_arr = np.zeros_like(core_arr, dtype=np.uint8)
    peri_arr = np.zeros_like(core_arr, dtype=np.uint8)

    # Expand strictly on the axial slice(s) where 'Core' was drawn
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