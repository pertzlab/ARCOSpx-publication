"""
The Cellular automaton follows the rules outlined below:

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

import numpy as np
from numba import njit


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
