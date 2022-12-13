#!/usr/bin/python3
"""
Salva su file i dataset degli esperimenti.
"""

import argparse
import math

import pandas as pd


def get_subdomains(ni, nj, n_procs):
    """Dato il numero di punti e il numero di processi, restituisce
    un generatore che fornisce i possibili sottodomini di
    dimensione >=5 per ogni lato.
    Ogni elemento è composto da
    - numero di suddivisioni lungo asse X della griglia
    - numero di suddivisioni lungo asse Y della griglia
    - numero di punti per ogni sottodominio lungo asse X
    - numero di punti per ogni sottodominio lungo asse Y
    I sottodomini non devono coprire completamente la griglia: i punti
    rimanenti saranno ripartiti in modo opportuno del modello.
    """
    for ny in range(1, n_procs + 1):
        if n_procs % ny == 0:
            nx = n_procs // ny
            dx = ni // nx
            dy = nj // ny
            if dx >= 5 and dy >= 5:
                yield nx, ny, dx, dy


def calculate_estimated_duration(grid_step, ni, nj, n_procs, single_precision, physics_on):
    return math.prod([
        72,
        20/grid_step,
        ni*nj/403776,
        864 / n_procs,
        1 - 0.3 * single_precision,
        1 - 0.3 * (1 - physics_on)
    ])


def calculate_parallel_duration(df):
    duration = []
    for n_nodes, grp in df.groupby("n_nodes"):
        n_parallel = 60 // n_nodes
        n_groups = math.ceil(len(grp) / n_parallel)
        grp = grp.sort_values("estimated_duration", ascending=False)
        for idx in range(n_groups):
            start = n_parallel * idx
            end = start + n_parallel
            duration.append(grp.iloc[start:end].estimated_duration.max())

    return sum(duration)


def create_dataset(bbox_x_length, bbox_y_length):
    n_nodes_range = range(10, 61)
    n_cores_range = range(10, 37, 2)
    min_grid_step = 100
    grid_step_range = [
        gs for gs in range(10, min_grid_step + 1)
        if bbox_x_length % gs == 0 and bbox_y_length % gs == 0
    ]
    dataset = []
    for grid_step in grid_step_range:
        # number of points along a parallel
        ni = bbox_x_length // grid_step
        # number of points along a meridian
        nj = bbox_y_length // grid_step
        for n_nodes in n_nodes_range:
            for n_cores in n_cores_range:
                n_procs = n_nodes * n_cores
                for nx, ny, dx, dy, in get_subdomains(ni, nj, n_procs):
                    for single_precision in (0, 1):
                        # physics parameter disabled
                        for physics_on in (1,):
                            estimated_duration = calculate_estimated_duration(grid_step, ni, nj, n_procs, single_precision, physics_on)
                            dataset.append([
                                bbox_x_length,
                                bbox_y_length,
                                grid_step,
                                ni,
                                nj,
                                n_nodes,
                                n_cores,
                                n_procs,
                                dx,
                                dy,
                                nx,
                                ny,
                                dx/dy,
                                single_precision,
                                physics_on,
                                estimated_duration,
                            ])
    df = pd.DataFrame(dataset, columns=[
        "x_length",
        "y_length",
        "grid_step",
        "ni",
        "nj",
        "n_nodes",
        "n_cores",
        "n_procs",
        "dx",
        "dy",
        "nx",
        "ny",
        "subdomain_ratio",
        "single_precision",
        "physics_on",
        "estimated_duration"
    ])
    return df


QUERY = " and ".join([
    "subdomain_ratio >= 1",
    "subdomain_ratio <= 5",
    "estimated_duration <= 500",
    "grid_step in [20, 50, 80, 100]",
    "n_cores >= 14"
])


def create_dataset_fixed_area():
    df = create_dataset(bbox_x_length=10000, bbox_y_length=10000)
    df = df.query(QUERY)
    return df


def create_dataset_variable_area():
    n_samples = 1000
    df = pd.concat([
        d.query(QUERY).sample(n_samples, random_state=1)
        for d in [
            create_dataset(bbox_x_length=x, bbox_y_length=y)
            for x, y in (
                (7200, 24000),
                (23200, 18000),
                (30000, 12000),
                (46000, 14400),
            )
        ]
    ] + [
        create_dataset_fixed_area().query(QUERY).sample(n_samples, random_state=1),
    ])
    return df


if __name__ == '__main__':
    create_dataset_fixed_area().to_csv("fixed_area.csv", index=False)
    create_dataset_variable_area().to_csv("variable_area.csv", index=False)
