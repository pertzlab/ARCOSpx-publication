# Podosome Tracking

This folder contains Python and R scripts to recreate the analysis of podosome tracks in the paper. 
Subfolder 3a contains a Jupyter notebook that can be used to run the segmentation and tracking of podosomes in a single cell.
Subfolder 3b contains scripts and a notebook to analyze lineage statistics of multiple fields of view treated with latrunculin B, blebbistatin, Y-27632, and DMSO.

## Contents

- 3a_single_cell
    - `podosomes_arcos_px_single_fov.ipynb`: Jupyter notebook to run segmentation and tracking of podosomes in a single cell using convpaint and arcos4py.
- 3b_lineage_analysis
    - `convpaint_model`: CatBoost model to predict semantic segmentation of podosomes.
    - `convpaint_batch.py`: Python script to run segmentation.
    - `track_extraction.py`: Python script to track podosomes and extract features and lineage statistics.
    - `podosome-dynamics-4treat-20250212.Rmd`: R notebook to analyze lineage statistics of multiple fields of view treated with latrunculin B, blebbistatin, Y-27632, and DMSO.
    - `slurm_batch_convpaint.sh`: Slurm script to run segmentation on a cluster.
    - `slurm_batch_tracking.sh`: Slurm script to run tracking on a cluster.

## Running Lineage Analysis from Raw Data
1. Run segmentation on raw data using `slurm_batch_convpaint.sh` on a cluster. Images should be saved in the current working directory in a folder per treatment as individual tif stacks per field of view.
2. Run tracking on segmented images using `slurm_batch_tracking.sh`. Make sure to adjust the path to the segmentation results and raw images in the script. Results are saved to the specified output directory. Concatenate the results of all fields of view into a single CSV file.
3. Run the R notebook `podosome-dynamics-4treat-20250212.Rmd` to analyze lineage statistics and generate plots.
