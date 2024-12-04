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
import os
import numpy as np
import tifffile
import re
from glob import glob
from natsort import natsorted
from multiprocessing import Pool, cpu_count

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


# Global variables
OUT_DIR = "evaluation_arcospx_run24/"  # Base directory for the dataset
NUM_STEPS = 500  # Number of frames in the simulation

SIMULATION_FUNCTIONS = ["sim_circles", "sim_target_pattern", "sim_directional", "sim_chaotic"]
# Pattern should match files with 3 digits before the extension e.g mask001.tif or man_track001.tif
PATTERN = re.compile(r".*(\d{3}).tif")


def load_single_tif(file_path):
    return tifffile.imread(file_path)

def load_tif_sequence(directory):
    # Use glob to find all .tif files
    file_pattern = os.path.join(directory, '*.tif')
    files = glob(file_pattern)
    
    # Use natsorted for faster, more intuitive sorting
    sorted_files = natsorted(files, key=lambda x: int(PATTERN.match(os.path.basename(x)).group(1)))
    
    # Determine shape of the first image to pre-allocate the array
    sample_image = tifffile.imread(sorted_files[0])
    sequence = np.empty((len(sorted_files),) + sample_image.shape, dtype=sample_image.dtype)
    
    # Use multiprocessing to load images in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(load_single_tif, sorted_files)
    
    # Fill the pre-allocated array
    for i, img in enumerate(results):
        sequence[i] = img
    
    return sequence


def find_gt_res_pairs(base_dir):
    """Find pairs of *_GT and *_RES folders that share the same prefix."""
    gt_res_pairs = []
    
    # List all directories in base_dir
    directories = os.listdir(base_dir)
    
    # Find all unique prefixes based on *_GT and *_RES naming pattern
    gt_dirs = [d for d in directories if d.endswith('_GT')]
    res_dirs = [d for d in directories if d.endswith('_RES')]
    
    # Match GT and RES pairs by the same prefix (e.g., '00_GT' and '00_RES')
    for gt_dir in gt_dirs:
        prefix = gt_dir[:-3]  # Remove the '_GT' part to get the prefix
        corresponding_res_dir = f"{prefix}_RES"
        
        if corresponding_res_dir in res_dirs:
            gt_res_pairs.append((os.path.join(base_dir, gt_dir), os.path.join(base_dir, corresponding_res_dir)))

    return gt_res_pairs


def run_evaluation_and_compute_metrics(sim_function_index, signal_to_noise_ratio=np.inf):
    sim_function = SIMULATION_FUNCTIONS[sim_function_index]
    dir_out = f'{str(sim_function)}_snr_{signal_to_noise_ratio}'
    base_dir = os.path.join(OUT_DIR, dir_out)

    logger.info("Computing metrics for %s", dir_out)

    # Find valid *_GT and *_RES folder pairs
    gt_res_pairs = find_gt_res_pairs(base_dir)

    if not gt_res_pairs:
        logger.error("No valid GT/RES folder pairs found in the output directory!")
        return

    all_summaries = []
    
    for _, (gt_folder, res_folder) in enumerate(gt_res_pairs):
        prefix = os.path.basename(gt_folder)[:-3]  # Get the common prefix (e.g., '00')

        logger.info(f"Processing folder pair {prefix}_GT and {prefix}_RES of {dir_out}")

        gt_tra_folder = os.path.join(gt_folder, "TRA")
        acc = mm.MOTAccumulator(auto_id=False)

        # Load tracked labels and ground truth tracking data from .tif files
        tracked_labels = load_tif_sequence(res_folder)
        gt_tracking = load_tif_sequence(gt_tra_folder)
        logger.info(f"Loaded {len(tracked_labels)} tracked labels and {len(gt_tracking)} ground truth tracking data")

        # Compute metrics for every frame
        for frame_num, (gt_image, pred_image) in enumerate(zip(gt_tracking, tracked_labels)):
            logger.debug(f"Processing frame {frame_num} of folder pair {prefix}")
            gt_detections, pred_detections, gt_labels, pred_labels = process_frame(gt_image, pred_image, frame_num, 1)

            # # plot detections
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

            # ax1.imshow(gt_image)
            # for bbox in gt_detections:
            #     rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            #     ax1.add_patch(rect)

            # ax2.imshow(pred_image)
            # for bbox in pred_detections:
            #     rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            #     ax2.add_patch(rect)

            # plt.show(block=True)      §     
            
            distances = mm.distances.iou_matrix(gt_detections, pred_detections, max_iou=0.5)
            
            acc.update(
                gt_labels,
                pred_labels,
                distances,
                frameid=frame_num
            )

        # Compute metrics
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'num_switches', 
                                           'num_false_positives', 'num_misses', 'num_detections', 
                                           'num_objects', 'num_predictions', 'precision', 'recall'],
                             name=prefix)
        all_summaries.append(summary)

    # Combine all summaries
    final_summary = pd.concat(all_summaries)
    
    # Compute and save metrics summary
    final_summary.to_csv(f"{base_dir}/summary_bbox_05iou.csv", index=True)

    logger.info(f"Finished computing metrics for {dir_out}")

if __name__ == "__main__":
    try:
        sim_function_index = int(sys.argv[1])
        snr = float(sys.argv[2])
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        logfilename = os.path.join(OUT_DIR, f"evaluation_metrics_{current_date}_snr_{snr}.log")

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        logger = logging.getLogger(__name__)

        run_evaluation_and_compute_metrics(sim_function_index, snr)
    except Exception as e:
        logger.error("An error occurred during the main execution: %s", str(e))
        logger.exception("Exception traceback:")