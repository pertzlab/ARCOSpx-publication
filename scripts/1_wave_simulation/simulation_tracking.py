""" This script contains the code to run different simulations for emergend dynamics and tracking with ARCOS.px."""

import numpy as np
from arcos4py.tools import track_events_image
import tifffile
import os
import logging

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
import argparse

# Import simulation functions, individual parameters are set in the cellular_automaton.py file
from cellular_automaton import sim_circles, sim_target_pattern, sim_directional, sim_chaotic

# Global variables
OUT_DIR = "evaluation_arcospx_run001/"  # Base directory for the dataset
GRID_SIZE = (512, 512)
NUM_STEPS = 500

SIMULATION_FUNCTIONS_TO_RUN = [sim_circles, sim_target_pattern, sim_directional, sim_chaotic]
# Parameters for each simulation type and SNR
TRACKING_PARAMETERS = {
    "sim_circles": {
        -15.0: {"cluster_size": 80, "eps": "auto", "n_prev": 1},
        -10.0: {"cluster_size": 20, "eps": "auto", "n_prev": 1},
        -5.0: {"cluster_size": 20, "eps": "auto", "n_prev": 1},
        0.0: {"cluster_size": 20, "eps": "auto", "n_prev": 1},
        25.0: {"cluster_size": 5, "eps": "auto", "n_prev": 1},
        np.inf: {"cluster_size": 1, "eps": "auto", "n_prev": 1},
    },
    "sim_target_pattern": {
        -15.0: {"cluster_size": 80, "eps": "auto", "n_prev": 20},
        -10.0: {"cluster_size": 80, "eps": "auto", "n_prev": 20},
        -5.0: {"cluster_size": 10, "eps": "auto", "n_prev": 20},
        0.0: {"cluster_size": 10, "eps": "auto", "n_prev": 20},
        25.0: {"cluster_size": 5, "eps": 5, "n_prev": 15},
        np.inf: {"cluster_size": 1, "eps": "auto", "n_prev": 1},
    },
    "sim_directional": {
        -15.0: {"cluster_size": 80, "eps": "auto", "n_prev": 1},
        -10.0: {"cluster_size": 10, "eps": "auto", "n_prev": 1},
        -5.0: {"cluster_size": 10, "eps": "auto", "n_prev": 1},
        0.0: {"cluster_size": 5, "eps": 5, "n_prev": 1},
        25.0: {"cluster_size": 4, "eps": 5, "n_prev": 1},
        np.inf: {"cluster_size": 1, "eps": "auto", "n_prev": 1},
    },
    "sim_chaotic": {
        -15.0: {"cluster_size": 20, "eps": "auto", "n_prev": 1},
        -10.0: {"cluster_size": 20, "eps": "auto", "n_prev": 1},
        -5.0: {"cluster_size": 20, "eps": "auto", "n_prev": 1},
        0.0: {"cluster_size": 20, "eps": "auto", "n_prev": 1},
        25.0: {"cluster_size": 20, "eps": "auto", "n_prev": 1},
        np.inf: {"cluster_size": 1, "eps": "auto", "n_prev": 1},
    },
}


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
        tifffile.imwrite(file_path, mask.astype(np.uint16), compression="zlib")
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
    return distances[:, 1:]


def add_noise(image, snr):
    # Convert the image to float
    image = image.astype(np.float32)

    # Calculate the signal power
    signal_power = np.mean(image**2)

    # Calculate the noise power based on the desired SNR
    noise_power = signal_power / (10 ** (snr / 10))

    # Generate noise
    noise = np.random.normal(scale=np.sqrt(noise_power), size=image.shape)

    # Add the noise to the image
    noisy_image = image + noise

    # shift the image to 16-bit range
    noisy_image -= np.min(noisy_image)
    noisy_image /= np.max(noisy_image)
    noisy_image *= 2**16 - 1

    return noisy_image.astype(np.uint16)


def run_simulation_and_tracking(sim_function_index, signal_to_noise_ratio=np.inf):
    sim_function = SIMULATION_FUNCTIONS_TO_RUN[sim_function_index]
    sim_type = sim_function.__name__
    dir_out = f"{sim_type}_snr_{signal_to_noise_ratio}"
    base_dir = os.path.join(OUT_DIR, dir_out)

    # Fetch simulation parameters based on sim_type and snr
    snr_key = signal_to_noise_ratio if signal_to_noise_ratio == np.inf else round(signal_to_noise_ratio, 2)
    sim_params = TRACKING_PARAMETERS.get(sim_type, {}).get(snr_key, {"cluster_size": 1, "eps": "auto", "n_prev": 1})

    cluster_size = sim_params["cluster_size"]
    eps = sim_params["eps"]
    n_prev = sim_params["n_prev"]

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
        if eps == "auto":
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
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run simulation and tracking with dynamic parameters.")
    parser.add_argument(
        "--sim_function_index", type=int, required=False, help="Index of the simulation function.", default=0
    )
    parser.add_argument("--snr", required=False, help='Signal-to-noise ratio. Use "inf" for infinity.', default="inf")
    args = parser.parse_args()

    sim_function_index = args.sim_function_index
    snr_arg = args.snr

    if snr_arg.lower() == "inf":
        signal_to_noise_ratio = np.inf
    else:
        signal_to_noise_ratio = float(snr_arg)

    sim_type = SIMULATION_FUNCTIONS_TO_RUN[sim_function_index].__name__

    logging.basicConfig(
        filename=os.path.join(OUT_DIR, f"simulation_run_{sim_type}_snr_{signal_to_noise_ratio}.log"),
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
