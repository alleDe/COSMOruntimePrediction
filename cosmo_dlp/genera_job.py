#!/usr/bin/python3
import argparse
import os
from glob import glob
import shutil
import tempfile
from datetime import datetime

import sh
import pandas as pd


def create_int2lm(df, basedir=".", ncdir="."):
    reftime = datetime.now()
    for key, _ in df.groupby(['x_length', 'y_length', 'ni', 'nj', 'grid_step']):
        x_length, y_length, ni, nj, grid_step = key
        ie_ext, je_ext = {
            20: (2401, 1251),
            50: (500, 960),
            80: (601, 313),
            100: (481, 251),
        }[grid_step]
        jobdir = os.path.join(basedir, f"{x_length}_{y_length}", f"{grid_step:03.0f}")
        os.makedirs(jobdir, exist_ok=True)
        params = {
            "start_lat": -12.9 + grid_step/1000,
            "start_lon": -24.8 + grid_step/1000,
            "ni": ni + 1,
            "nj": nj + 1,
            "step": grid_step/1000,
            "ie_ext": ie_ext,
            "je_ext": je_ext,
            "ncdir": ncdir,
            "ncfile": f"extdata_med_{grid_step:03.0f}.nc",
            "reftime": f"{reftime:%Y%m%d}00",
            "day": f"{reftime:%A}"
        }
        for f in glob("template/int2lm/*"):
            with open(f) as fp:
                data = fp.read().format(**params)

            outfile = os.path.join(jobdir, os.path.basename(f))
            with open(outfile, "w") as fp:
                fp.write(data)


def create_jobs(df, basedir=".", cosmo_base_path=".", baseinputdir="."):
    def get_jobdir(row):
        return "_".join([
            f"{row.x_length:.0f}_{row.y_length:.0f}",
            f"{row.grid_step:03.0f}",
            f"{row.n_nodes:02.0f}",
            f"{row.n_cores:02.0f}",
            f"{row.single_precision:01.0f}",
            f"{row.physics_on:01.0f}",
            f"{row.nx:03.0f}",
            f"{row.ny:03.0f}",
        ])

    df["jobdir"] = df.apply(get_jobdir, axis=1)
    reftime = datetime.now()

    def create_dirs(row):
        jobdir = os.path.join(basedir, row.jobdir)
        os.makedirs(jobdir, exist_ok=True)
        for d in ["dataoutput", "dataoutput_light", "dataoutput_lightml"]:
            os.makedirs(os.path.join(jobdir, d), exist_ok=True)

        cosmo_path = os.path.join(
            cosmo_base_path,
            "single_precision" if row.single_precision else "double_precision",
            "cosmo",
        )
        inputdir = os.path.join(
            baseinputdir,
            f"{row.x_length}_{row.y_length}",
            f"{row.grid_step:03d}",
        )
        params = {
            "n_nodes": row.n_nodes,
            "n_cores": row.n_cores,
            "n_cores_half": row.n_cores//2,
            "cosmo_path": cosmo_path,
            "dlon": row.grid_step/1000,
            "dlat": row.grid_step/1000,
            "lphys": ".TRUE." if row.physics_on else ".FALSE.",
            "ni": row.ni + 1,
            "nj": row.nj + 1,
            "nx": row.nx,
            "ny": row.ny,
            "start_lat": -12.9 + row.grid_step/1000,
            "start_lon": -24.8 + row.grid_step/1000,
            "dt": row.grid_step * 18 / 20,
            "inputdir": inputdir,
            "reftime": f"{reftime:%Y%m%d}00",
        }

        for f in glob("template/cosmo/*"):
            with open(f) as fp:
                data = fp.read().format(**params)

            outfile = os.path.join(jobdir, os.path.basename(f))
            with open(outfile, "w") as fp:
                fp.write(data)

        row.to_csv(os.path.join(jobdir, "params.csv"), header=True, index=False)

    df.apply(create_dirs, axis=1)

    os.makedirs(basedir, exist_ok=True)
    df.to_csv(os.path.join(basedir, "jobs.csv"), index=False)


def sync_experiments():
    tempdir = tempfile.mkdtemp(prefix="tesi-")
    for csvfile, dirname in (
        ("fixed_area.csv", "fixed_area"),
        ("variable_area.csv", "variable_area"),
    ):
        df = pd.read_csv(csvfile)
        create_jobs(df, f"{tempdir}/{dirname}", "/galileo/home/userexternal/edigiaco/cosmo", "/galileo/home/userexternal/edigiaco/int2lm")
        sh.rsync([
            "-avzR",
            f"{tempdir}/./{dirname}",
            "edigiaco@login.galileo.cineca.it:~/experiments/",
        ])

    shutil.rmtree(tempdir)


def sync_ncdata():
    tempdir = tempfile.mkdtemp(prefix="tesi-")
    for csvfile, dirname in (
        ("fixed_area.csv", "fixed_area"),
        ("variable_area.csv", "variable_area"),
    ):
        df = pd.read_csv(csvfile)
        create_int2lm(df, basedir=tempdir, ncdir="/galileo/home/userexternal/edigiaco/int2lm")
        sh.rsync([
            "-avzR",
            f"{tempdir}/./",
            "edigiaco@login.galileo.cineca.it:~/int2lm/",
        ])

    shutil.rmtree(tempdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    sync_experiments()
    sync_ncdata()
