# Wave Simulation

This folder contains Python scripts to simulate various wave patterns using a cellular automaton model. The simulations include circular waves, directional waves, target patterns, and chaotic patterns. The results of these simulations can be tracked with ARCOS.px and subsequently evaluated using py-motmetrics.


## Contents

- `cellular_automaton.py`: Contains the main functions for running the wave simulations.
- `simulation_tracking.py`: The code to run different simulations and track the emergent dynamics using ARCOS.px.
- `evaluation_metrics_bbox.py`: Script for evaluating simulation metrics results using py-motmetrics.
- `submit_simulation_run_snr_batch.sh`: Example shell script for submitting simulations eith different noise levels on a slurm cluster.
- `submit_evaluation_batch_bbox.sh`: Example shell script for submitting batch evaluations with slurm.
- `plot_eval.ipynb`: Jupyter notebook for plotting and evaluating simulation runs.
- `plot_eval_multiple.ipynb`: Jupyter notebook for plotting and evaluating simulation runs from multiple trackers; here from ARCOS.px and TrackMates' LAP.
- `calc_effect_size.Rmd`: R notebook to calculate Cohen's d effect size from calculated MOTA & MOTP metrics.


## Predefined Simulation Functions

The following simulation functions are defined in `cellular_automaton.py`:

- `sim_chaotic(seed, grid_size=(512, 512), num_steps=500)`: Simulates chaotic wave patterns.
- `sim_circles(seed, grid_size=(512, 512), num_steps=500)`: Simulates circular wave patterns.
- `sim_target_pattern(seed, grid_size=(512, 512), num_steps=500)`: Simulates target wave patterns.
- `sim_directional(seed, grid_size=(512, 512), num_steps=500)`: Simulates directional wave patterns.


## Reproducing Fig. S3

Follow these steps to reproduce Figure S3 from the manuscript.

### Running Simulations

Use the `simulation_tracking.py` script to run a simulation and track the results using ARCOS.px. The `sim_function_index` command-line argument specifies the simulation function: 0 for circular waves, 1 for directional waves, 2 for target patterns, 3 for chaotic patterns; the `signal_to_noise_ratio` defines the amount of added noise to the simulation.

For example, to run 10 iterations of the circular wave simulation with an infinite signal-to-noise ratio, use the following command:

```
python simulation_tracking.py --sim_function_index 0 --signal_to_noise_ratio inf --iteration 10
```

This will create a folder `sim_circles_snr_inf` with 10 subfolders corresponding to iterations. Each iteration contains `RAW`, `SEG` and `TRA` folders that include raw simulation results and ground truth segmentation for future comparisons with trackers.


## Calculating metrics

To evaluate the results of the simulations, use the `evaluation_metrics_bbox.py` script, which uses the `py-motmetrics` package to compute metrics such as MOTA, MOTP:

```
python evaluation_metrics_bbox.py --sim_function_index 0 --signal_to_noise_ratio inf --iteration 
```

The script will read the results of the circular wave simulation with an infinite signal-to-noise ratio and will 