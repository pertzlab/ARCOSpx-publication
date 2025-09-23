"""Compute MOT metrics for the bounding box tracking results of the wave simulation."""

import os
import sys
import numpy as np
import logging
import motmetrics as mm
from numba import njit
import tifffile
import re
import datetime
import pandas as pd
from glob import glob
from natsort import natsorted
from multiprocessing import Pool, cpu_count
import argparse


# variable definitions
OUT_DIR = "evaluation_arcospx_run006/"  # Base directory for the dataset

SIMULATION_FUNCTION_NAMES_TO_EVAL = ["sim_circles", "sim_target_pattern", "sim_directional", "sim_chaotic"]
PATTERN = re.compile(r".*(\d{3}).tif")  # Pattern to extract the frame number from the filename

IOU_THRESHOLD = 0.75  # Intersection over Union threshold for matching bounding boxes
SIZE_THRESHOLD = 1  # Minimum size of a bounding box to be considered a event


@njit(cache=True)
def mask_to_bbox(mask):
    rows, cols = mask.shape
    x_min, y_min, x_max, y_max = cols, rows, 0, 0
    for y in range(rows):
        for x in range(cols):
            if mask[y, x]:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
    return np.array([x_min, y_min, x_max - x_min + 1, y_max - y_min + 1])  # [x, y, width, height]


@njit(cache=True)
def label_image_to_detections(label_image, frame_num, min_size=1):
    unique_labels = np.unique(label_image)
    detections = []
    labels = []
    for label_id in unique_labels:
        if label_id == 0:  # Skip background
            continue
        mask = label_image == label_id
        size = np.sum(mask)
        if size < min_size:
            continue
        bbox = mask_to_bbox(mask)
        detections.append(bbox)
        labels.append(label_id)
    return detections, labels  # Return a list instead of np.array


def process_frame(gt_image, pred_image, frame_num, min_size=1):
    gt_detections, gt_labels = label_image_to_detections(gt_image, frame_num, min_size=min_size)
    pred_detections, pred_labels = label_image_to_detections(pred_image, frame_num, min_size=min_size)

    return gt_detections, pred_detections, gt_labels, pred_labels


def load_single_tif(file_path):
    return tifffile.imread(file_path)


def load_tif_sequence(directory):
    file_pattern = os.path.join(directory, "*.tif")
    files = glob(file_pattern)

    sorted_files = natsorted(files, key=lambda x: int(PATTERN.match(os.path.basename(x)).group(1)))

    # Determine shape of the first image to pre-allocate the array
    sample_image = tifffile.imread(sorted_files[0])
    sequence = np.empty((len(sorted_files),) + sample_image.shape, dtype=sample_image.dtype)

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(load_single_tif, sorted_files)

    for i, img in enumerate(results):
        sequence[i] = img

    return sequence


def find_gt_res_pairs(base_dir: str, tracker_name: str = ""):
    """Find pairs of *_GT and *_RES folders that share the same prefix."""
    gt_res_pairs = []

    directories = os.listdir(base_dir)

    gt_dirs = [d for d in directories if d.endswith("GT")]
    res_dirs = [d for d in directories if d.endswith(f"_RES_{tracker_name}")]

    # Match GT and RES pairs by the same prefix (e.g., '00_GT' and '00_RES')
    for gt_dir in gt_dirs:
        prefix = gt_dir[:-3]
        corresponding_res_dir = f"{prefix}_RES_{tracker_name}"

        if corresponding_res_dir in res_dirs:
            gt_res_pairs.append((os.path.join(base_dir, gt_dir), os.path.join(base_dir, corresponding_res_dir)))

    return gt_res_pairs


def run_evaluation_and_compute_metrics(sim_function_index, signal_to_noise_ratio=np.inf, tracker_name="arcospx", iteration=0):

    sim_function = SIMULATION_FUNCTION_NAMES_TO_EVAL[sim_function_index]
    dir_out = f"{str(sim_function)}_snr_{signal_to_noise_ratio}"
    base_dir = os.path.join(OUT_DIR, dir_out)

    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)

    logger.info("Computing metrics for %s", dir_out)

    all_summaries = []

    # Find valid *_GT and *_RES folder pairs
    gt_res_pairs = find_gt_res_pairs(base_dir, tracker_name)

    if not gt_res_pairs:
        logger.error("No valid GT/RES folder pairs found in the output directory!")
        logger.info(gt_res_pairs)
        return
    
    prefix = str(iteration).zfill(2)  # Ensure prefix is zero-padded to 3 digits
    gt_folder = os.path.join(base_dir, f"{prefix}_GT")
    res_folder = os.path.join(base_dir, f"{prefix}_RES_{tracker_name}")
    prefix = os.path.basename(gt_folder)[:-3]  # Get the common prefix (e.g., '00')

    logger.info(f"Processing folder pair {gt_folder} and {res_folder} of {dir_out}")

    gt_tra_folder = os.path.join(gt_folder, "TRA")
    acc = mm.MOTAccumulator(auto_id=False)

    # Load tracked labels and ground truth tracking data from .tif files
    tracked_labels = load_tif_sequence(res_folder)
    gt_tracking = load_tif_sequence(gt_tra_folder)
    logger.info(f"Loaded {len(tracked_labels)} tracked labels and {len(gt_tracking)} ground truth tracking data")

    # Accumulate metrics for every frame
    for frame_num, (gt_image, pred_image) in enumerate(zip(gt_tracking, tracked_labels)):
        logger.debug(f"Processing frame {frame_num} of folder pair {prefix}")
        gt_detections, pred_detections, gt_labels, pred_labels = process_frame(
            gt_image, pred_image, frame_num, SIZE_THRESHOLD
        )

        distances = mm.distances.iou_matrix(gt_detections, pred_detections, max_iou=IOU_THRESHOLD)

        acc.update(gt_labels, pred_labels, distances, frameid=frame_num)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "num_detections",
            "num_objects",
            "num_predictions",
            "precision",
            "recall",
        ],
        name=prefix,
    )
    summary['tracker_name'] = tracker_name
    summary['sim_function'] = sim_function
    summary['snr'] = signal_to_noise_ratio
    summary['size_threshold'] = SIZE_THRESHOLD
    summary['iou_threshold'] = IOU_THRESHOLD
    summary['iteration'] = iteration

    # Combine all summaries

    # Compute and save metrics summary
    csv_path = os.path.join(base_dir, f"summary_bbox_{IOU_THRESHOLD}_{prefix}_{tracker_name}.csv")
    summary.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics summary to: {csv_path}")

    logger.info(f"Finished computing metrics for {prefix}_{dir_out}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute MOT metrics for bounding box tracking results.")
    parser.add_argument(
        "--sim_function_index", type=int, required=False, help="Index of the simulation function.", default=0
    )
    parser.add_argument("--snr", type=str, help='Signal-to-noise ratio. Use "inf" for infinity.', default="inf")
    parser.add_argument(
        "--tracker", type=str, help="Name of the tracker to evaluate.", default="arcospx"
    )
    parser.add_argument(
        "--iteration", type=int, help="Iteration number for the evaluation.", default=0
    )

    args = parser.parse_args()

    sim_function_index = args.sim_function_index
    snr = args.snr
    tracker_name = args.tracker
    iteration = args.iteration

    # Convert SNR string to appropriate type
    if snr == "inf":
        snr_value = np.inf
    else:
        snr_value = float(snr)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfilename = os.path.join(OUT_DIR, f"evaluation_metrics_{current_date}_snr_{snr}.log")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    try:
        run_evaluation_and_compute_metrics(sim_function_index, snr_value, tracker_name, iteration)
    except Exception as e:
        logger.error("An error occurred during the main execution: %s", str(e))
        logger.exception("Exception traceback:")
