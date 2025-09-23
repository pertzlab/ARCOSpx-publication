# Simulations of Wave Splits and Merges 

This folder contains Python scripts to simulate splits and merges of spatio-temporal events. The results of these simulations are tracked with ARCOS.px at different split/merge stability threshold values. The comparison between the simulated ground truth and ARCOS.px tracking is evaluated using standard metrics such as IoU, MOTA, MOTP, F1 score, etc.


## Contents

- `sim_split_merge.ipynb`: Contains the main functions for running the simulations that generate the ground truth. Saves the results to the `GT` sub-folder of a `sim_seedXXX` folder. The `XXX` is the random seed of the simulation.
- `track_split_merge.ipynb`: The code to track the events using ARCOS.px. Loads the simulated ground truth data and saves predictions in the `PRED` sub-folder. Tracking is run for various lineage stability threshold values.
- `compare_sim_track.ipynb`: Script for comparing simulations to ground truth data using various metrics. Loads the ground truth and the simulation results and saves the metrics for all simulations in a single file `tracking_evaluation_summary.csv`
- `analyze_summary.rmd`: R notebook to calculate plot the results from the `tracking_evaluation_summary.csv` file.