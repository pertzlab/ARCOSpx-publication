"""
Cellular Automaton Rules

This cellular automaton simulates wave propagation on a 2D grid. The behavior of each cell at every time step is 
determined by its state, the states of its neighboring cells, and a set of probabilistic rules. The automaton 
evolves over discrete time steps, updating the grid according to the following rules:

1. Grid Representation
    - Grid (`grid`): Represents the activity of each cell, with a non-zero value indicating active cells and a zero 
      value indicating inactive cells.
    - Refractory Grid (`refractory_grid`): Tracks the refractory period of each cell, which prevents reactivation 
      during this period.
    - Lifetime Grid (`lifetime_grid`): Records the number of time steps each cell has been continuously active.
    - Wave ID Grid (`wave_id_grid`): Assigns a unique ID to each wave, allowing tracking of individual wave fronts 
      across the grid.

2. Cell State Updates

   Each cell `(i, j)` in the grid is updated according to its current state:

   2.1 Active Cells (`grid[i, j] > 0`)
        - Increment Lifetime: The lifetime of the cell is incremented by 1 (`lifetime_grid[i, j] += 1`).
        - Neighbor Counting: The number of active neighbors is determined by checking cells in the directions specified 
          by `propagation_directions`.
        - Death Probability:
            - The death probability `P_death` for the cell is computed as:
              `P_death = min(base_death_probability * (1 + lifetime_grid[i, j] / 10), max_death_probability)`
            - If the cell has no active neighbors, the death probability is set to `isolated_death_probability`.
            - The cell remains active if a random number exceeds `P_death`; otherwise, it becomes inactive.
        - Activity Decrement: If the cell remains active, its activity level decreases by 1 (`grid[i, j] -= 1`).
        - Refractory Period Update: The cells refractory period is recalculated with a random multiplier, and
          `refractory_grid[i, j]` is updated.

   2.2 Inactive and Non-Refractory Cells (`grid[i, j] == 0` and `refractory_grid[i, j] == 0`)
        - Neighbor Counting and Wave ID Collection:
            - Active neighbors are counted, and their wave IDs are collected.
        - Activation Probability:
            - The probability `P_activate` of the cell becoming active is calculated as:
              `P_activate = min(active_neighbors * propagation_probability * excitability[i, j], 0.8)`
            - The cell activates if a random number is less than `P_activate`.
        - Wave ID Assignment:
            - If the cell activates and has neighbors with wave IDs, it inherits the most common wave ID among its 
              neighbors.
            - If the cell activates but has no active neighbors, it is assigned a new wave ID (`next_wave_id`).
        - Spontaneous Activation:
            - If the cell does not activate through its neighbors, it may activate spontaneously with a probability:
              `P_spontaneous = wave_formation_probability * excitability[i, j]`
            - The process for wave ID assignment follows the same logic as for neighbor-induced activation.

   2.3 Refractory Cells (`refractory_grid[i, j] > 0`)
        - Refractory cells do not activate and remain inactive until their refractory period expires.

3. Propagation Directions
    - The directions in which neighbors are checked (`propagation_directions`) influence the potential paths of wave 
      propagation. Typical directions might include the cardinal directions (up, down, left, right) or may also include 
      diagonals.

4. Wave ID Management
    - Wave ID Inheritance: When a cell becomes active due to its neighbors, it inherits the most common wave ID among 
      those neighbors.
    - New Wave ID: If a cell activates without neighbors or all neighbor wave IDs are inactive, it is assigned a new 
      wave ID, and `next_wave_id` is incremented.

5. Simulation Process
    - The simulation runs for a specified number of steps. At each step, the grid is updated according to the rules 
      above.
    - The state of the grid (`history`) and the wave ID grid (`wave_id_history`) are recorded for each time step, 
      allowing for analysis of wave propagation and tracking.
"""

import psutil
import numpy as np
from numba import njit, prange
from arcos4py.tools import track_events_image
import motmetrics as mm
import tifffile
import os
import logging

from skimage.filters import threshold_otsu
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
from datetime import datetime
import numpy as np
import psutil
import argparse


def create_required_folders(base_dir, sequence_id):
    """
    Create the required folder structure for the dataset.

    Parameters:
    - base_dir: Base directory for the dataset
    - sequence_id: Identifier for the sequence (e.g., "01")
    """
    res_folder = os.path.join(base_dir, f"{sequence_id}_RES")
    gt_seg_folder = os.path.join(base_dir, f"{sequence_id}_GT", "SEG")
    gt_tra_folder = os.path.join(base_dir, f"{sequence_id}_GT", "TRA")
    gt_raw_folder = os.path.join(base_dir, f"{sequence_id}_GT", "RAW")
    os.makedirs(res_folder, exist_ok=True)
    os.makedirs(gt_seg_folder, exist_ok=True)
    os.makedirs(gt_tra_folder, exist_ok=True)
    os.makedirs(gt_raw_folder, exist_ok=True)
    return res_folder, gt_seg_folder, gt_tra_folder, gt_raw_folder


def save_mask_tif_files(data, output_dir, prefix="mask"):
    """
    Save labeled data as maskT.tif files.

    Parameters:
    - data: numpy array of shape (time, height, width)
    - output_dir: directory to save the maskT.tif files
    - prefix: file name prefix ("mask" for RES, "man_seg" or "man_track" for GT)
    """
    time_points = data.shape[0]
    for t in range(time_points):
        mask = data[t]
        t_str = f"{t:03d}"  # Format temporal index with three digits
        file_path = os.path.join(output_dir, f"{prefix}{t_str}.tif")
        tifffile.imwrite(file_path, mask.astype(np.uint16))
    logger.info(f"Saved {time_points} maskT.tif files to {output_dir}")


def generate_tracking_txt(data, output_file):
    """
    Generate and save the tracking txt file.

    Parameters:
    - data: numpy array of shape (time, height, width)
    - output_file: file path to save the tracking txt file
    """
    time_points = data.shape[0]
    tracking_data = []

    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background label

    label_dict = {}
    parent_dict = {label: 0 for label in unique_labels}

    for t in range(time_points):
        labels_t = np.unique(data[t])
        labels_t = labels_t[labels_t != 0]  # Exclude background label
        for label in labels_t:
            if label not in label_dict:
                label_dict[label] = [t, t]  # Initialize with [begin, end]
            else:
                label_dict[label][1] = t  # Update end time

    for label, (begin, end) in label_dict.items():
        parent = parent_dict[label]
        tracking_data.append(f"{label} {begin} {end} {parent}")

    with open(output_file, "w") as f:
        for line in tracking_data:
            f.write(line + "\n")
    logger.info(f"Saved {output_file}")


@njit(cache=True)
def label_image_to_detections(label_image, frame_num, min_size=1):
    unique_labels = np.unique(label_image)
    detections = []
    for label_id in unique_labels:
        if label_id == 0:  # Skip background
            continue
        mask = label_image == label_id
        size = np.sum(mask)
        if size < min_size:
            # Logging is not supported in njit functions
            # logger.info(f"Skipping label {label_id} with size {size}")
            continue
        detections.append((frame_num, label_id, mask))
    return detections


@njit(cache=True)
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    union_area = np.sum(union)
    intersection_area = np.sum(intersection)
    if union_area == 0:
        return 0.0
    iou = intersection_area / union_area
    return iou


@njit(parallel=True, cache=True)
def compute_distance_matrix(gt_masks, pred_masks):
    num_gt = len(gt_masks)
    num_pred = len(pred_masks)
    distance_matrix = np.zeros((num_gt, num_pred))
    for i in prange(num_gt):
        for j in prange(num_pred):
            distance_matrix[i, j] = 1 - compute_iou(gt_masks[i], pred_masks[j])
    return distance_matrix


@njit(cache=True)
def process_frame(gt_image, pred_image, frame_num, threshold=1):
    gt_detections = label_image_to_detections(gt_image, frame_num, min_size=2)
    pred_detections = label_image_to_detections(pred_image, frame_num, min_size=2)

    gt_ids = np.array([d[1] for d in gt_detections], dtype=np.int64)
    pred_ids = np.array([d[1] for d in pred_detections], dtype=np.int64)

    gt_masks = [d[2] for d in gt_detections]
    pred_masks = [d[2] for d in pred_detections]

    distance_matrix = compute_distance_matrix(gt_masks, pred_masks)
    distance_matrix = np.where(distance_matrix > threshold, np.nan, distance_matrix)

    return gt_ids, pred_ids, distance_matrix


@njit(cache=True)
def update_wave(
    grid,
    refractory_grid,
    lifetime_grid,
    wave_id_grid,
    refractory_periods,
    excitability,
    randomness_factor,
    next_wave_id,
    base_activation_value,
    propagation_directions,
    grid_size,
    base_death_probability,
    max_death_probability,
    isolated_death_probability,
    wave_formation_probability,
    propagation_probability,
    rng_state,
):
    new_grid = np.zeros_like(grid)
    new_refractory_grid = np.maximum(refractory_grid - 1, 0)
    new_lifetime_grid = np.copy(lifetime_grid)
    new_wave_id_grid = np.zeros_like(wave_id_grid)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if grid[i, j] > 0:
                new_lifetime_grid[i, j] += 1
                active_neighbors = 0

                for direction in propagation_directions:
                    dx, dy = direction
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < grid_size[0] and 0 <= nj < grid_size[1]:
                        if grid[ni, nj] > 0:
                            active_neighbors += 1

                death_probability = min(
                    base_death_probability * (1 + new_lifetime_grid[i, j] / 10), max_death_probability
                )
                if active_neighbors == 0:
                    death_probability = isolated_death_probability

                if rng_state.random() < death_probability:
                    continue
                new_grid[i, j] = grid[i, j] - 1
                if new_grid[i, j] == 0:
                    new_wave_id_grid[i, j] = 0  # Clear wave ID when cell becomes refractory
                else:
                    new_wave_id_grid[i, j] = wave_id_grid[i, j]
                random_multiplier = 1 + (rng_state.standard_normal() * randomness_factor)
                new_refractory_grid[i, j] = max(0, refractory_periods[i, j] * random_multiplier)
            elif refractory_grid[i, j] == 0:
                active_neighbors = 0
                neighbor_ids = np.zeros(len(propagation_directions), dtype=np.int32)
                id_count = 0

                for direction in propagation_directions:
                    dx, dy = direction
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < grid_size[0] and 0 <= nj < grid_size[1]:
                        if grid[ni, nj] > 0:
                            active_neighbors += 1
                            if wave_id_grid[ni, nj] > 0:
                                neighbor_ids[id_count] = wave_id_grid[ni, nj]
                                id_count += 1

                activation_probability = min(active_neighbors * propagation_probability * excitability[i, j], 0.8)
                if rng_state.random() < activation_probability:
                    random_multiplier = 1 + (rng_state.standard_normal() * randomness_factor)
                    activation_value = max(1, base_activation_value * random_multiplier)
                    new_grid[i, j] = activation_value
                    random_multiplier = 1 + (rng_state.standard_normal() * randomness_factor)
                    new_refractory_grid[i, j] = max(0, refractory_periods[i, j] * random_multiplier)
                    new_lifetime_grid[i, j] = 0
                    if id_count > 0:
                        # Use the most common ID among the neighbors
                        most_common_id = find_most_common_id(neighbor_ids[:id_count])
                        new_wave_id_grid[i, j] = most_common_id
                    else:
                        new_wave_id_grid[i, j] = next_wave_id
                        next_wave_id += 1
                elif rng_state.random() < wave_formation_probability * excitability[i, j] and active_neighbors == 0:
                    random_multiplier = 1 + (rng_state.standard_normal() * randomness_factor)
                    activation_value = max(1, base_activation_value * random_multiplier)
                    new_grid[i, j] = activation_value
                    random_multiplier = 1 + (rng_state.standard_normal() * randomness_factor)
                    new_refractory_grid[i, j] = max(0, refractory_periods[i, j] * random_multiplier)
                    new_lifetime_grid[i, j] = 0
                    new_wave_id_grid[i, j] = next_wave_id
                    next_wave_id += 1

    return new_grid, new_refractory_grid, new_lifetime_grid, new_wave_id_grid, next_wave_id


@njit(cache=True)
def find_most_common_id(ids):
    if len(ids) == 0:
        return -1  # Should not happen, but as a safeguard

    unique_ids = np.unique(ids)
    max_count = 0
    most_common_id = unique_ids[0]

    for uid in unique_ids:
        count = np.sum(ids == uid)
        if count > max_count:
            max_count = count
            most_common_id = uid

    return most_common_id


def run_simulation(
    grid_size,
    num_steps,
    base_death_probability,
    max_death_probability,
    isolated_death_probability,
    wave_formation_probability,
    propagation_probability,
    base_activation_value,
    min_refractory_period,
    max_refractory_period,
    randomness_factor,
    propagation_directions,
    excitability_range,
    seed,
):
    rng = np.random.default_rng(seed)
    excitability = rng.uniform(excitability_range[0], excitability_range[1], grid_size)
    refractory_periods = rng.integers(min_refractory_period, max_refractory_period + 1, size=grid_size)

    history = []
    wave_id_history = []
    grid = np.zeros(grid_size, dtype=int)
    refractory_grid = np.zeros(grid_size, dtype=int)
    lifetime_grid = np.zeros(grid_size, dtype=int)
    wave_id_grid = np.zeros(grid_size, dtype=int)
    next_wave_id = 1

    for step in range(num_steps):
        grid, refractory_grid, lifetime_grid, wave_id_grid, next_wave_id = update_wave(
            grid,
            refractory_grid,
            lifetime_grid,
            wave_id_grid,
            refractory_periods,
            excitability,
            randomness_factor,
            next_wave_id,
            base_activation_value,
            propagation_directions,
            grid_size,
            base_death_probability,
            max_death_probability,
            isolated_death_probability,
            wave_formation_probability,
            propagation_probability,
            rng,
        )
        history.append(grid.copy())
        wave_id_history.append(wave_id_grid.copy())

    return history, wave_id_history


def sim_chaotic(seed, grid_size=(512, 512), num_steps=500):
    # Parameters
    base_death_probability = 0.1
    max_death_probability = 0.9
    isolated_death_probability = 0.25
    wave_formation_probability = 0.00005
    propagation_probability = 0.3
    base_activation_value = 50
    min_refractory_period = 1
    max_refractory_period = 10
    randomness_factor = 10
    excitability_range = (0.5, 0.5)
    propagation_directions = np.array(
        [
            (1, 0),  # Right
            (0, 1),  # Up
            (0, -1),  # Down
            (1, 1),  # Diagonal Down-Right
            (-1, 1),  # Diagonal Up-Right
            (1, -1),  # Diagonal Down-Left
            (-1, -1),  # Diagonal Up-Left
            (-1, 0),  # Left
        ],
        dtype=np.int32,
    )

    history_circular, wave_id_history_circular = run_simulation(
        grid_size,
        num_steps,
        base_death_probability,
        max_death_probability,
        isolated_death_probability,
        wave_formation_probability,
        propagation_probability,
        base_activation_value,
        min_refractory_period,
        max_refractory_period,
        randomness_factor,
        propagation_directions,
        excitability_range,
        seed,
    )

    return history_circular, wave_id_history_circular


def sim_circles(seed, grid_size=(512, 512), num_steps=500):
    # Parameters
    base_death_probability = 0.001
    max_death_probability = 0.99
    isolated_death_probability = 0.99
    wave_formation_probability = 0.0000005
    propagation_probability = 0.5
    base_activation_value = 50
    min_refractory_period = 50
    max_refractory_period = 90
    randomness_factor = 0
    excitability_range = (0.5, 0.5)
    propagation_directions = np.array(
        [
            (1, 0),  # Right
            (0, 1),  # Up
            (0, -1),  # Down
            (1, 1),  # Diagonal Down-Right
            (-1, 1),  # Diagonal Up-Right
            (1, -1),  # Diagonal Down-Left
            (-1, -1),  # Diagonal Up-Left
            (-1, 0),  # Left
        ],
        dtype=np.int32,
    )

    history_circular, wave_id_history_circular = run_simulation(
        grid_size,
        num_steps,
        base_death_probability,
        max_death_probability,
        isolated_death_probability,
        wave_formation_probability,
        propagation_probability,
        base_activation_value,
        min_refractory_period,
        max_refractory_period,
        randomness_factor,
        propagation_directions,
        excitability_range,
        seed,
    )

    return history_circular, wave_id_history_circular


def sim_target_pattern(seed, grid_size=(512, 512), num_steps=500):
    base_death_probability = 0.01
    max_death_probability = 0.9
    isolated_death_probability = 0.9
    wave_formation_probability = 0.000005
    propagation_probability = 0.99
    base_activation_value = 15
    min_refractory_period = 16
    max_refractory_period = 16
    randomness_factor = 0
    excitability_range = (0.25, 0.25)
    propagation_directions = np.array(
        [
            (1, 0),  # Right
            (0, 1),  # Up
            (0, -1),  # Down
            (1, 1),  # Diagonal Down-Right
            (-1, 1),  # Diagonal Up-Right
            (1, -1),  # Diagonal Down-Left
            (-1, -1),  # Diagonal Up-Left
            (-1, 0),  # Left
        ],
        dtype=np.int32,
    )

    history_target, wave_id_history_target = run_simulation(
        grid_size,
        num_steps,
        base_death_probability,
        max_death_probability,
        isolated_death_probability,
        wave_formation_probability,
        propagation_probability,
        base_activation_value,
        min_refractory_period,
        max_refractory_period,
        randomness_factor,
        propagation_directions,
        excitability_range,
        seed,
    )

    return history_target, wave_id_history_target


def sim_directional(seed, grid_size=(512, 512), num_steps=500):
    base_death_probability = 0.01
    max_death_probability = 0.95
    isolated_death_probability = 0.99
    wave_formation_probability = 0.00001
    propagation_probability = 0.8
    base_activation_value = 50
    min_refractory_period = 20
    max_refractory_period = 50
    randomness_factor = 0
    excitability_range = (0, 1)
    propagation_directions = np.array(
        [(1, 0), (1, 1), (-1, 1)], dtype=np.int32  # Right  # Diagonal Down-Right  # Diagonal Up-Right
    )

    history_directional, wave_id_history_directional = run_simulation(
        grid_size,
        num_steps,
        base_death_probability,
        max_death_probability,
        isolated_death_probability,
        wave_formation_probability,
        propagation_probability,
        base_activation_value,
        min_refractory_period,
        max_refractory_period,
        randomness_factor,
        propagation_directions,
        excitability_range,
        seed,
    )

    return history_directional, wave_id_history_directional


def binarize_image(image, threshold=0):
    """Converts an image to binary and extracts coordinates of non-zero pixels."""
    binary_image = np.where(image > threshold, 1, 0)
    coords = np.column_stack(np.where(binary_image > 0))
    return coords


def estimate_eps(
    data: pd.DataFrame = None,
    image: np.ndarray = None,
    method: str = "kneepoint",
    position_columns: list[str] = ["x", "y"],
    frame_column: str = "t",
    n_neighbors: int = 5,
    plot: bool = True,
    plt_size: tuple[int, int] = (5, 5),
    max_samples=50_000,
    binarize_threshold=0,
    **kwargs: dict,
):
    """Estimates eps parameter in DBSCAN, working with both dataframes and images.

    Arguments:
        data (pd.DataFrame): DataFrame containing the data. (Optional if image is provided)
        image (np.ndarray): Image array, either single frame or multi-frame time series. (Optional if data is provided)
        method (str, optional): Method to use for estimating eps. Defaults to 'kneepoint'.
        position_columns (list[str]): List of column names containing the position data.
        frame_column (str, optional): Name of the column containing the frame number. Defaults to 't'.
        n_neighbors (int, optional): Number of nearest neighbors to consider. Defaults to 5.
        plot (bool, optional): Whether to plot the results. Defaults to True.
        plt_size (tuple[int, int], optional): Size of the plot. Defaults to (5, 5).
        max_samples (int, optional): Maximum number of samples to use for estimation. Defaults to 50,000.
        binarize_threshold (int, optional): Threshold for binarizing the image. Defaults to 127.
        kwargs (Any): Keyword arguments for the method.

    Returns:
        eps (float): eps parameter for DBSCAN.
    """
    method_option = ["kneepoint", "mean", "median"]

    if method not in method_option:
        raise ValueError(f"Method must be one of {method_option}")

    if (data is None and image is None) or (data is not None and image is not None):
        raise ValueError("You must provide either a DataFrame or an image, but not both.")

    # Process image if provided
    if image is not None:
        if len(image.shape) == 3:  # Time series of images
            coords = []
            for t, img in enumerate(image):
                img_coords = binarize_image(img, threshold=binarize_threshold)
                if img_coords.size > 0:
                    frame_coords = np.column_stack((np.full(img_coords.shape[0], t), img_coords))
                    coords.append(frame_coords)
            data_np = np.vstack(coords)
        else:  # Single image
            data_np = binarize_image(image, threshold=binarize_threshold)
            data_np = np.column_stack((np.zeros(data_np.shape[0]), data_np))  # Add a dummy frame column for consistency
        # Convert to DataFrame
        data = pd.DataFrame(data_np, columns=[frame_column] + position_columns)

    # Validate DataFrame input
    subset = [frame_column] + position_columns
    for col in subset:
        if col not in data.columns:
            raise ValueError(f"Column {col} not in data")

    # Convert DataFrame to numpy array
    data_np = data[subset].to_numpy(dtype=np.float64)
    data_np = data_np[data_np[:, 0].argsort()]  # Sort by frame

    grouped_array = np.split(data_np[:, 1:], np.unique(data_np[:, 0], axis=0, return_index=True)[1][1:])

    # Calculate nearest neighbors distances
    distances = [_nearest_neighbour_eps(group, n_neighbors) for group in grouped_array if group.shape[0] >= n_neighbors]
    if not distances:
        distances_array = np.array([])
    else:
        distances_array = np.concatenate(distances)

    distances_flat = distances_array.flatten()
    distances_flat = distances_flat[np.isfinite(distances_flat)]
    distances_flat_selection = np.random.choice(
        distances_flat, min(max_samples, distances_flat.shape[0]), replace=False
    )
    distances_sorted = np.sort(distances_flat_selection)

    if distances_sorted.shape[0] == 0:
        raise ValueError("No valid distances found, please check input data.")

    # Estimate eps based on method
    if method == "kneepoint":
        k1 = KneeLocator(
            np.arange(0, distances_sorted.shape[0]),
            distances_sorted,
            S=kwargs.get("S", 1),
            online=kwargs.get("online", True),
            curve=kwargs.get("curve", "convex"),
            interp_method=kwargs.get("interp_method", "polynomial"),
            direction=kwargs.get("direction", "increasing"),
            polynomial_degree=kwargs.get("polynomial_degree", 7),
        )
        eps = distances_sorted[k1.knee]

    elif method == "mean":
        eps = np.mean(distances_sorted) * kwargs.get("mean_multiplier", 1.5)

    elif method == "median":
        eps = np.median(distances_sorted) * kwargs.get("median_multiplier", 1.5)

    if plot:
        fig, ax = plt.subplots(figsize=plt_size)
        ax.plot(distances_sorted)
        ax.axhline(eps, color="r", linestyle="--")
        ax.set_xlabel("Sorted Distance Index")
        ax.set_ylabel("Nearest Neighbour Distance")
        ax.set_title(f"Estimated eps: {eps:.4f}")
        plt.show()

    return eps


def _nearest_neighbour_eps(data: np.ndarray, n_neighbors: int):
    """Helper function to compute nearest neighbor distances."""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    distances, _ = nbrs.kneighbors(data)
    return distances[:, 1:]  # Skip distance to self (which is zero)


def add_noise(image, snr):
    offset = 0.1  # Offset to avoid negative values in the log calculation
    # Convert the image to float and add offset
    image = image.astype(np.float32) + offset

    # Calculate the signal power
    signal_power = np.mean(image**2)

    # Calculate the noise power based on the desired SNR
    noise_power = signal_power / (10 ** (snr / 10))

    # Generate noise
    noise = np.random.normal(scale=np.sqrt(noise_power), size=image.shape)

    # Add the noise to the image
    noisy_image = image + noise

    # Subtract the offset
    noisy_image -= offset

    # shift the image to the range [0, 255]
    noisy_image -= np.min(noisy_image)
    noisy_image /= np.max(noisy_image)
    noisy_image *= 2**16 - 1

    return noisy_image.astype(np.uint16)


def main_alt():
    # custom logging entry
    logger.info("Starting simulation script, skipping eval of size 1 clusters")
    try:
        out_dir = "evaluation_arcospx_run4/"  # Base directory for the dataset
        grid_size = (512, 128)
        num_steps = 500
        min_clustersize = 2
        eps = 2.5
        all_accumulators = []
        all_names = []
        for simulation_function in [sim_circles, sim_target_pattern, sim_directional, sim_chaotic]:
            logger.info("====================================")
            dir_out = str(simulation_function).split(" ")[1]
            base_dir = os.path.join(out_dir, dir_out)
            logger.info("====================================")
            logger.info("running simulation for %s", dir_out)

            for i in range(10):
                logger.info("running simulation for iteration %d", i)
                history_circular, wave_id_history_circular = simulation_function(i, grid_size, num_steps)
                logger.info("====================================")
                logger.info(f"tracking events with min_clustersize={min_clustersize} and eps={eps}")
                events_tracked = track_events_image(
                    np.stack(history_circular) > 0,
                    clustering_method="dbscan",
                    min_clustersize=min_clustersize,
                    eps=eps,
                    downsample=1,
                )
                logger.info("====================================")
                logger.info("saving results for evaluation with cell tracking metrics")
                sequence_id = f"0{i}"  # Sequence identifier
                # make sure to start only from a frame with detected events
                for idx, f in enumerate(wave_id_history_circular):
                    if np.max(f) > 0:
                        break

                tracked_labels = events_tracked[idx:]
                gt_tracking = np.stack(wave_id_history_circular)[idx:]
                gt_segmentation = np.stack(wave_id_history_circular)[idx:]

                # Create the required folder structure
                res_folder, gt_seg_folder, gt_tra_folder = create_required_folders(base_dir, sequence_id)
                res_track_file = os.path.join(res_folder, "res_track.txt")
                gt_track_file = os.path.join(gt_tra_folder, "man_track.txt")

                # Save the results (RES) - output from your algorithm
                save_mask_tif_files(tracked_labels, res_folder, prefix="mask")
                generate_tracking_txt(tracked_labels, res_track_file)

                # Save the ground truth (GT)
                save_mask_tif_files(gt_segmentation, gt_seg_folder, prefix="man_seg")
                save_mask_tif_files(gt_tracking, gt_tra_folder, prefix="man_track")
                generate_tracking_txt(gt_tracking, gt_track_file)
                logger.info("====================================")
                logger.info("evaluation with pymotmetrics")

                acc = mm.MOTAccumulator(auto_id=True)

                for frame_num, (gt_image, pred_image) in enumerate(zip(gt_tracking, tracked_labels)):
                    gt_ids, pred_ids, distance_matrix = process_frame(gt_image, pred_image, frame_num)

                    if frame_num % 100 == 0:
                        logger.info("Frame %d:", frame_num)
                    acc.update(gt_ids.tolist(), pred_ids.tolist(), distance_matrix)
                all_accumulators.append(acc)
                all_names.append(f"{dir_out}_run_{i}")
                logger.info("====================================")
                logger.info("done with iteration %d", i)

            logger.info("====================================")
            logger.info("computing summary")

        mh = mm.metrics.create()
        summary = mh.compute_many(
            all_accumulators,
            metrics=mm.metrics.motchallenge_metrics.extend(["num_detections", "num_objects"]),
            names=all_names,
            generate_overall=True,
        )
        out_summary = os.path.join(out_dir, "summary.csv")
        summary.to_csv(out_summary)
        logger.info("Saved summary to summary.csv")

    except Exception as e:
        logger.error("An error occurred: %s", str(e))


# Global variables
OUT_DIR = "evaluation_arcospx_run24/"  # Base directory for the dataset
GRID_SIZE = (512, 512)
NUM_STEPS = 500
SIMULATION_FUNCTIONS = [sim_circles, sim_target_pattern, sim_directional, sim_chaotic]

# Parameters for each simulation type and SNR
simulation_parameters = {
    'sim_circles': {
        -15.0: {'cluster_size': 80, 'eps': 'auto', 'n_prev': 1},
        -10.0: {'cluster_size': 20, 'eps': 'auto', 'n_prev': 1},
        -5.0: {'cluster_size': 20, 'eps': 'auto', 'n_prev': 1},
        0.0: {'cluster_size': 20, 'eps': 'auto', 'n_prev': 1},
        25.0: {'cluster_size': 5, 'eps': 'auto', 'n_prev': 1},
        np.inf: {'cluster_size': 1, 'eps': 'auto', 'n_prev': 1},
    },
    'sim_target_pattern': {
        -15.0: {'cluster_size': 80, 'eps': 'auto', 'n_prev': 20},
        -10.0: {'cluster_size': 80, 'eps': 'auto', 'n_prev': 20},
        -5.0: {'cluster_size': 10, 'eps': 'auto', 'n_prev': 20},
        0.0: {'cluster_size': 10, 'eps': 'auto', 'n_prev': 20},
        25.0: {'cluster_size': 5, 'eps': 5, 'n_prev': 15},
        np.inf: {'cluster_size': 1, 'eps': 'auto', 'n_prev': 1},
    },
    'sim_directional': {
        -15.0: {'cluster_size': 80, 'eps': 'auto', 'n_prev': 1},
        -10.0: {'cluster_size': 10, 'eps': 'auto', 'n_prev': 1},
        -5.0: {'cluster_size': 10, 'eps': 'auto', 'n_prev': 1},
        0.0: {'cluster_size': 5, 'eps': 5, 'n_prev': 1},
        25.0: {'cluster_size': 4, 'eps': 5, 'n_prev': 1},
        np.inf: {'cluster_size': 1, 'eps': 'auto', 'n_prev': 1},
    },
    'sim_chaotic': {
        -15.0: {'cluster_size': 20, 'eps': 'auto', 'n_prev': 1},
        -10.0: {'cluster_size': 20, 'eps': 'auto', 'n_prev': 1},
        -5.0: {'cluster_size': 20, 'eps': 'auto', 'n_prev': 1},
        0.0: {'cluster_size': 20, 'eps': 'auto', 'n_prev': 1},
        25.0: {'cluster_size': 20, 'eps': 'auto', 'n_prev': 1},
        np.inf: {'cluster_size': 1, 'eps': 'auto', 'n_prev': 1},
    },
}

# Modify the function to use the new data structure
def run_simulation_and_tracking(sim_function_index, signal_to_noise_ratio=np.inf):
    sim_function = SIMULATION_FUNCTIONS[sim_function_index]
    sim_type = sim_function.__name__
    dir_out = f'{sim_type}_snr_{signal_to_noise_ratio}'
    base_dir = os.path.join(OUT_DIR, dir_out)

    # Fetch simulation parameters based on sim_type and snr
    snr_key = signal_to_noise_ratio if signal_to_noise_ratio == np.inf else round(signal_to_noise_ratio, 2)
    sim_params = simulation_parameters.get(sim_type, {}).get(snr_key, {'cluster_size': 1, 'eps': 'auto', 'n_prev': 1})

    cluster_size = sim_params['cluster_size']
    eps = sim_params['eps']
    n_prev = sim_params['n_prev']

    # Loop to repeat each simulation
    for iteration in range(10):
        seed = iteration
        np.random.seed(seed)

        logger.info(
            "Iteration %d: Running simulation for %s, snr %s, cluster_size %d, eps %s, n_prev %d, seed %d",
            iteration,
            dir_out,
            str(signal_to_noise_ratio),
            cluster_size,
            str(eps),
            n_prev,
            seed,
        )

        # Run the simulation with the seed
        history_circular, wave_id_history_circular = sim_function(seed, GRID_SIZE, NUM_STEPS)

        # Add noise to the data
        history_circular = np.stack(history_circular)
        wave_id_history_circular = np.stack(wave_id_history_circular)
        if signal_to_noise_ratio < np.inf:
            history_circular = add_noise(history_circular, signal_to_noise_ratio)

        thresh = np.mean(history_circular[0]) * 1.5 if signal_to_noise_ratio < np.inf else 0

        # If eps is set to 'auto', estimate it
        if eps == 'auto':
            eps = estimate_eps(
                image=history_circular > thresh,
                method="kneepoint",
                max_samples=100_000,
                plot=False,
                n_neighbors=cluster_size + 1,
            )
            eps = max(eps, 1.5)  # Ensure EPS is at least 5

        logger.info(f"Iteration {iteration}: Estimated eps: {eps}")

        # Track events
        events_tracked = track_events_image(
            history_circular > thresh,
            clustering_method="dbscan",
            min_clustersize=cluster_size,
            eps=eps,
            downsample=1,
            n_prev=n_prev,
            show_progress=False,
        )

        # Make sure to start only from a frame with detected events
        for idx, f in enumerate(wave_id_history_circular):
            if np.max(f) > 0:
                break

        sequence_id = f"{iteration:02d}"  # Sequence identifier with leading zeros

        tracked_labels = events_tracked[idx:]
        gt_tracking = wave_id_history_circular[idx:]
        gt_segmentation = wave_id_history_circular[idx:]
        history_circular = history_circular[idx:]

        # Create the required folder structure
        res_folder, gt_seg_folder, gt_tra_folder, gt_raw_folder = create_required_folders(base_dir, sequence_id)
        res_track_file = os.path.join(res_folder, "res_track.txt")
        gt_track_file = os.path.join(gt_tra_folder, "man_track.txt")

        # Save the results (RES) - output from your algorithm
        save_mask_tif_files(tracked_labels, res_folder, prefix="mask")
        generate_tracking_txt(tracked_labels, res_track_file)

        # Save the ground truth (GT)
        save_mask_tif_files(gt_segmentation, gt_seg_folder, prefix="man_seg")
        save_mask_tif_files(gt_tracking, gt_tra_folder, prefix="man_track")
        save_mask_tif_files(history_circular, gt_raw_folder, prefix="raw")
        generate_tracking_txt(gt_tracking, gt_track_file)

        logger.info(f"Iteration {iteration}: Finished simulation and tracking")

    logger.info("Finished all iterations for %s", dir_out)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run simulation and tracking with dynamic parameters.")
    parser.add_argument('--sim_function_index', type=int, required=True, help='Index of the simulation function.', default=2)
    parser.add_argument('--snr', required=True, help='Signal-to-noise ratio. Use "inf" for infinity.', default="25")
    args = parser.parse_args()

    sim_function_index = args.sim_function_index
    snr_arg = args.snr

    if snr_arg.lower() == "inf":
        signal_to_noise_ratio = np.inf
    else:
        signal_to_noise_ratio = float(snr_arg)

    sim_type = SIMULATION_FUNCTIONS[sim_function_index].__name__

    logging.basicConfig(
        filename=os.path.join(
            OUT_DIR, f"simulation_run_{sim_type}_snr_{signal_to_noise_ratio}.log"
        ),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # Run simulation and tracking
        run_simulation_and_tracking(sim_function_index, signal_to_noise_ratio)
    except Exception as e:
        logger.error("An error occurred during the main execution: %s", str(e))
        logger.exception("Exception traceback:")
