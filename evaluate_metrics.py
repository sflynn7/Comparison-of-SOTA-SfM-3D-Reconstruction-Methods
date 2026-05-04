"""
3D Reconstruction Evaluation Metrics (No ICP - Pre-aligned PLYs)
Computes Chamfer Distance, Hausdorff Distance, Completeness, Accuracy, and F-Score.

Usage:
    python evaluate_metrics.py /path/to/3D_Model_Meshes/

Requirements:
    pip install open3d numpy scipy

Naming convention expected:
    - Reconstructions: VGGT_Ukulele_Day_aligned.ply, Pi3_Fan_Night_aligned.ply, etc.
    - Ground truth:    GT_Ukulele_Day.ply, GT_Fan_Night.ply, etc.

IMPORTANT: All PLY files should already be aligned to ground truth (via CloudCompare ICP).
"""

import os
import sys
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import csv


# ── Configuration ─────────────────────────────────────────────────────────────

# Threshold for Completeness, Accuracy, and F-Score (in same units as your PLY).
# Since your clouds are at ~1000 scale, start with 50.0 and adjust if needed.
# - If F-Score is 0% for everything: increase threshold
# - If F-Score is 100% for everything: decrease threshold
THRESHOLD = 50.0

# Methods to evaluate
METHODS = ["VGGT", "Pi3", "COLMAP", "DA3"]

# Scenes to evaluate (edit to match your actual scene names)
SCENES = [
    "Ukulele_Day",
    "Ukulele_Night",
    "ArtificialPlant_Day",
    "ElectricFan_Day",
    "LabChair",
]

# ──────────────────────────────────────────────────────────────────────────────


def load_point_cloud(path):
    """Load a PLY file and return points as numpy array."""
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError(f"No points loaded from {path}")
    return points


def compute_metrics(reconstruction, ground_truth, threshold=THRESHOLD):
    """
    Compute all metrics between reconstruction and ground truth point clouds.

    Args:
        reconstruction: Nx3 numpy array of reconstruction points
        ground_truth:   Mx3 numpy array of ground truth points
        threshold:      distance threshold for completeness/accuracy/f-score

    Returns:
        dict of metrics
    """
    # Build KD-trees for fast nearest neighbor search
    tree_gt = cKDTree(ground_truth)
    tree_rec = cKDTree(reconstruction)

    # Distances from reconstruction to ground truth
    dist_rec_to_gt, _ = tree_gt.query(reconstruction, k=1)

    # Distances from ground truth to reconstruction
    dist_gt_to_rec, _ = tree_rec.query(ground_truth, k=1)

    # Chamfer Distance (mean of both directions)
    chamfer = (np.mean(dist_rec_to_gt) + np.mean(dist_gt_to_rec)) / 2.0

    # Hausdorff Distance (max of both directions)
    hausdorff = max(np.max(dist_rec_to_gt), np.max(dist_gt_to_rec))

    # Accuracy: % of reconstruction points within threshold of ground truth
    accuracy = np.mean(dist_rec_to_gt < threshold) * 100.0

    # Completeness: % of ground truth points within threshold of reconstruction
    completeness = np.mean(dist_gt_to_rec < threshold) * 100.0

    # F-Score: harmonic mean of accuracy and completeness
    if accuracy + completeness > 0:
        fscore = 2 * (accuracy * completeness) / (accuracy + completeness)
    else:
        fscore = 0.0

    return {
        "Chamfer Distance": round(chamfer, 4),
        "Hausdorff Distance": round(hausdorff, 4),
        "Accuracy (%)": round(accuracy, 2),
        "Completeness (%)": round(completeness, 2),
        "F-Score (%)": round(fscore, 2),
    }


def find_file(folder, method, scene):
    """Try to find the aligned reconstruction file."""
    candidates = [
        f"{method}_{scene}_aligned.ply",
        f"{method}_{scene}.ply",
        f"{method.lower()}_{scene.lower()}_aligned.ply",
        f"{method.lower()}_{scene.lower()}.ply",
    ]
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def find_gt_file(folder, scene):
    """Try to find the ground truth file."""
    candidates = [
        f"GT_{scene}.ply",
        f"gt_{scene}.ply",
        f"GroundTruth_{scene}.ply",
        f"ground_truth_{scene}.ply",
    ]
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def main(folder):
    print(f"\n{'='*75}")
    print(f"  3D Reconstruction Evaluation (Pre-aligned PLYs)")
    print(f"  Folder: {folder}")
    print(f"  Threshold: {THRESHOLD} units")
    print(f"{'='*75}\n")

    results = []
    missing = []

    for scene in SCENES:
        gt_path = find_gt_file(folder, scene)
        if gt_path is None:
            print(f"⚠️  Ground truth not found for scene: {scene} — skipping")
            continue

        print(f"Scene: {scene}")
        print(f"  Ground truth: {os.path.basename(gt_path)}")

        try:
            gt_points = load_point_cloud(gt_path)
            print(f"  GT points: {len(gt_points):,}")
        except Exception as e:
            print(f"  ERROR loading ground truth: {e}")
            continue

        for method in METHODS:
            rec_path = find_file(folder, method, scene)
            if rec_path is None:
                print(f"  [{method}] ⚠️  File not found — skipping")
                missing.append(f"{method}_{scene}")
                continue

            try:
                rec_points = load_point_cloud(rec_path)
                print(f"  [{method}] Points: {len(rec_points):,} — computing metrics...")
                metrics = compute_metrics(rec_points, gt_points)
                print(f"  [{method}] "
                      f"Chamfer: {metrics['Chamfer Distance']:.4f} | "
                      f"Hausdorff: {metrics['Hausdorff Distance']:.4f} | "
                      f"Accuracy: {metrics['Accuracy (%)']:.2f}% | "
                      f"Completeness: {metrics['Completeness (%)']:.2f}% | "
                      f"F-Score: {metrics['F-Score (%)']:.2f}%")
                results.append({
                    "Method": method,
                    "Scene": scene,
                    **metrics
                })
            except Exception as e:
                print(f"  [{method}] ERROR: {e}")

        print()

    # ── Print Summary Table ────────────────────────────────────────────────────
    if results:
        print(f"\n{'='*95}")
        print("  RESULTS SUMMARY")
        print(f"{'='*95}")
        print(f"{'Method':<20} {'Scene':<20} {'Chamfer':>10} {'Hausdorff':>12} {'Acc%':>8} {'Comp%':>8} {'F-Score%':>10}")
        print("-" * 95)
        for r in results:
            print(f"{r['Method']:<20} {r['Scene']:<20} "
                  f"{r['Chamfer Distance']:>10.4f} "
                  f"{r['Hausdorff Distance']:>12.4f} "
                  f"{r['Accuracy (%)']:>8.2f} "
                  f"{r['Completeness (%)']:>8.2f} "
                  f"{r['F-Score (%)']:>10.2f}")

        # ── Save to CSV ────────────────────────────────────────────────────────
        output_csv = os.path.join(folder, "evaluation_results.csv")
        keys = results[0].keys()
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Results saved to: {output_csv}")

    if missing:
        print(f"\n⚠️  Missing/skipped files: {', '.join(missing)}")

    print(f"\nNote: If F-Scores look wrong (all 0% or all 100%), adjust THRESHOLD at")
    print(f"the top of this script. Current threshold: {THRESHOLD} units.")
    print(f"\nDone!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_metrics.py /path/to/3D_Model_Meshes/")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"ERROR: '{folder}' is not a valid folder.")
        sys.exit(1)

    main(folder)
