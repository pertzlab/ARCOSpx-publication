"""Extracts tracks and lineage information from segmented images using ARCOS.px"""

# Import necessary libraries
import numpy as np
import pandas as pd
import os
from skimage import io
from arcos4py.tools import track_events_image
from skimage.filters import rank
from skimage.measure import regionprops_table
from typing import Tuple, Optional, Callable, Union, Literal
from scipy.optimize import curve_fit
import pickle
from tifffile import imwrite

ExpFuncType = Callable[[np.ndarray, float, float], np.ndarray]
BiExpFuncType = Callable[[np.ndarray, float, float, float, float], np.ndarray]
FuncType = Union[ExpFuncType, BiExpFuncType]

# Define paths and directories
IMAGE_INDEX = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))  # this assumes you are using a SLURM cluster
CONDITIONS = ["dmso", "blebbistatin", "latrunculinb", "ycompound"]
CORE_DIRECTORY = "../../../data/3_podosome_tracks/input-data/lineage_4treat_all_rawdata/"
RAW_DIRECTORY = os.path.join(CORE_DIRECTORY, "raw")
SEGMENTATION_DIRECTORY = os.path.join(CORE_DIRECTORY, "seg")

STABILITY_THRESHOLD = 15

OUTPUT_DIRECTORY = os.path.join(CORE_DIRECTORY.replace("input_data", "output_data"), f"results_stability_{STABILITY_THRESHOLD}")

# prepare output directories
output_tracks_dir = os.path.join(OUTPUT_DIRECTORY, "tracks")
output_lineage_dir = os.path.join(OUTPUT_DIRECTORY, "lineage")
output_masks_dir = os.path.join(OUTPUT_DIRECTORY, "masks")
output_bleach_dir = os.path.join(OUTPUT_DIRECTORY, "bleach_corrected")


def _exp(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(-b * x)


def _bi_exp(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return (a * np.exp(-b * x)) + (c * np.exp(-d * x))


def _exponential_correct(
    images: np.ndarray,
    contrast_limits: Optional[Tuple[int, int]] = None,
    method: Literal["mono", "bi"] = "mono",
) -> np.ndarray:
    """Corrects photobleaching in a 3D or 4D image stack using an exponential curve.

    Adapted from:
    https://github.com/marx-alex/napari-bleach-correct/blob/main/src/napari_bleach_correct/modules/exponential.py.
    """
    dtype = images.dtype
    if contrast_limits is None:
        contrast_limits = (np.min(images), np.max(images))
    assert 3 <= len(images.shape) <= 4, f"Expected 3d or 4d image stack, instead got {len(images.shape)} dimensions"

    avail_methods = ["mono", "bi"]
    func: FuncType
    if method == "mono":
        func = _exp
    elif method == "bi":
        func = _bi_exp
    else:
        raise NotImplementedError(f"method must be one of {avail_methods}, instead got {method}")

    axes = tuple(range(len(images.shape)))
    I_mean = np.mean(images, axis=axes[1:])
    x_data = np.arange(images.shape[0])

    with np.errstate(over="ignore"):
        try:
            popt, _ = curve_fit(func, x_data, I_mean)
            f_ = np.vectorize(func)(x_data, *popt)
        except (ValueError, RuntimeError, Warning):
            f_ = np.ones(x_data.shape)

    residuals = I_mean - f_
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((I_mean - np.mean(I_mean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R^2: {r_squared}")

    f = f_ / np.max(f_)
    if len(images.shape) == 3:
        f = f.reshape(-1, 1, 1)
    else:
        f = f.reshape(-1, 1, 1, 1)
    images = images / f

    images[images < contrast_limits[0]] = contrast_limits[0]
    images[images > contrast_limits[1]] = contrast_limits[1]
    return images.astype(dtype)


def main():
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    os.makedirs(output_tracks_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(output_bleach_dir, exist_ok=True)
    os.makedirs(output_lineage_dir, exist_ok=True)
    
    print(IMAGE_INDEX, CONDITIONS)

    # Create a dictionary mapping raw files to segmentation files
    files_dict = {}
    for condition in CONDITIONS:
        raw_condition_dir = os.path.join(RAW_DIRECTORY, condition)
        seg_condition_dir = os.path.join(SEGMENTATION_DIRECTORY, condition)
        
        print(f"Processing condition: {condition}")
        print(f"Raw directory: {raw_condition_dir}")
        print(f"Segmentation directory: {seg_condition_dir}")

        # List all .tif files in raw and segmentation directories
        raw_files = [f for f in os.listdir(raw_condition_dir) if f.endswith(".tif") or f.endswith(".tiff")]
        seg_files = [f for f in os.listdir(seg_condition_dir) if f.endswith(".tif") or f.endswith(".tiff")]

        print(f"Found {len(raw_files)} raw files and {len(seg_files)} segmentation files.")
        print(f"Raw files: {raw_files}")
        print(f"Segmentation files: {seg_files}")
        
        # Create a mapping for segmentation files based on filename
        seg_files_map = {f: os.path.join(seg_condition_dir, f) for f in seg_files}
        print(seg_files_map)
        
        for raw_file in raw_files:
            raw_file_path = os.path.join(raw_condition_dir, raw_file)

            # Find the corresponding segmentation file
            print(f"Looking for segmentation file for {raw_file_path}")
            
            seg_file_path = seg_files_map.get(raw_file)
            
            print(f"Matched segmentation file: {seg_file_path}")
            
            if seg_file_path:
                files_dict[raw_file_path] = (condition, seg_file_path)
            else:
                print(f"Warning: No corresponding segmentation file found for {raw_file_path}")

    # load image and segmentation of index image
    raw_image = io.imread(list(files_dict.keys())[IMAGE_INDEX])
    seg_image = io.imread(list(files_dict.values())[IMAGE_INDEX][1])
    condition = list(files_dict.values())[IMAGE_INDEX][0]

    print(f"Processing {IMAGE_INDEX} image from {condition} condition")

    bleach_corrected = _exponential_correct(raw_image, method="mono")
    all_seg_filtered = rank.majority(seg_image.astype("uint8"), np.ones((2, 2, 2)))

    # track events
    binary_mask = all_seg_filtered > 1
    events, lineage = track_events_image(
        binary_mask,
        eps=1.5,
        eps_prev=2.5,
        min_clustersize=5,
        n_prev=3,
        allow_merges=True,
        allow_splits=True,
        stability_threshold=5,
        min_size_for_split=1,
        remove_small_clusters=True,
        predictor=False,
        show_progress=True,
    )

    all_dfs = []

    # extract metrics for each event
    for frame in range(bleach_corrected.shape[0]):
        props_events = regionprops_table(
            events[frame], bleach_corrected[frame], properties=("label", "area", "centroid", "bbox", "mean_intensity")
        )
        props_df_events = pd.DataFrame(props_events)
        props_df_events["frame"] = frame
        props_df_events["fov"] = IMAGE_INDEX
        props_df_events["condition"] = condition
        props_df_events["file"] = list(files_dict.keys())[IMAGE_INDEX]
        props_df_events["median_image_intensity"] = np.median(bleach_corrected[frame])
        props_df_events["mean_image_intensity"] = np.mean(bleach_corrected[frame])
        all_event_mask = events[frame] > 0
        props_df_events["median_event_intensity"] = np.median(bleach_corrected[frame][all_event_mask])
        props_df_events["mean_event_intensity"] = np.mean(bleach_corrected[frame][all_event_mask])

        all_dfs.append(props_df_events)

    all_dfs = pd.concat(all_dfs)

    lineage_stats = {}

    # Traverse nodes and aggregate stats
    for node in lineage.nodes.values():
        lid = node.lineage_id

        if lid not in lineage_stats:
            lineage_stats[lid] = {
                "num_splits": 0,
                "num_merges": 0,
                "minframe": float("inf"),
                "maxframe": float("-inf"),
                "ended_at_last_frame": False,
                "ended_as_a_merge": False,
            }

        # Update min and max frames
        if node.minframe < lineage_stats[lid]["minframe"]:
            lineage_stats[lid]["minframe"] = node.minframe
        if node.maxframe > lineage_stats[lid]["maxframe"]:
            lineage_stats[lid]["maxframe"] = node.maxframe
            if node.maxframe == 299:
                lineage_stats[lid]["ended_at_last_frame"] = True

        # Check merges/splits at node level
        if len(node.parents) > 1:
            lineage_stats[lid]["num_merges"] += 1
        if len(node.children) > 1:
            lineage_stats[lid]["num_splits"] += 1

        # If this node is a merge child (more than one parent),
        # mark all non-dominant parents as ended_as_a_merge
        if len(node.parents) > 1:
            # The child's lineage_id is node.lineage_id (the "dominant" ID).
            dominant_id = node.lineage_id
            for parent in node.parents:
                parent_id = parent.lineage_id
                if parent_id != dominant_id:
                    # This parent's lineage is lost to merge
                    parent_lid = parent_id
                    # Mark that lineage as ended_as_a_merge
                    if parent_lid in lineage_stats:
                        lineage_stats[parent_lid]["ended_as_a_merge"] = True

        # Convert lineage_stats to a DataFrame
        data = []
        for lid, stats in lineage_stats.items():
            lineage_duration = stats["maxframe"] - stats["minframe"]
            data.append(
                {
                    "lineage_id": lid,
                    "condition": condition,
                    "lineage_duration": lineage_duration,
                    "num_splits": stats["num_splits"],
                    "num_merges": stats["num_merges"],
                    "ended_at_last_frame": stats["ended_at_last_frame"],
                    "ended_as_a_merge": stats["ended_as_a_merge"],  # new column
                }
            )

    lineages_df = pd.DataFrame(data)

    # get lineage for each event
    all_dfs = lineage._add_parents_and_lineage_to_df(all_dfs, "label")
    all_dfs = all_dfs.rename(columns={"lineage": "lineage_id"})

    # merge lineages_df with all_dfs
    all_dfs = all_dfs.merge(lineages_df, on=["lineage_id", "condition"], how="left")
    all_dfs["real_time"] = all_dfs["frame"] * 4  # 4 seconds per frame
    all_dfs["area_microns"] = all_dfs["area"] * 0.12222222222222222  # um per pixel
    all_dfs["unique_lineage"] = (
        all_dfs["condition"] + "_" + all_dfs["fov"].astype(str) + "_" + all_dfs["lineage_id"].astype(str)
    )
    all_dfs["unique_label"] = (
        all_dfs["condition"] + "_" + all_dfs["fov"].astype(str) + "_" + all_dfs["label"].astype(str)
    )

    # save results
    all_dfs.to_csv(os.path.join(OUTPUT_DIRECTORY, f"results_{IMAGE_INDEX}_condition_{condition}.csv"), index=False)
    imwrite(
        os.path.join(output_tracks_dir, f"tracks_{IMAGE_INDEX}_condition_{condition}.tif"), events, compression="zlib"
    )
    imwrite(
        os.path.join(output_masks_dir, f"masks_{IMAGE_INDEX}_condition_{condition}.tif"),
        all_seg_filtered,
        compression="zlib",
    )
    imwrite(
        os.path.join(output_bleach_dir, f"bleach_corrected_{IMAGE_INDEX}_condition_{condition}.tif"),
        bleach_corrected,
        compression="zlib",
    )

    with open(os.path.join(output_lineage_dir, f"lineage_{IMAGE_INDEX}_{condition}.pkl"), "wb") as f:
        pickle.dump(lineage, f)


if __name__ == "__main__":
    main()
