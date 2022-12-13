#!/usr/bin/python3
import argparse
from pathlib import Path
import os
import tempfile
import sys

import sh
import pandas as pd


def load_time_factory(basedir):
    def load_time(row):
        values = [None, None, None]
        try:
            with open(os.path.join(basedir, row.jobdir, "YUTIMING")) as fp:
                for line in fp:
                    if line.startswith("Time for the setup"):
                        values[0] = float(line.split()[-1])
                    elif line.startswith("Time for hour      1"):
                        values[1] = float(line.split()[-1])
                    elif line.startswith("Time for hour      2"):
                        values[2] = float(line.split()[-1])
                        break
        except FileNotFoundError:
            pass

        return values

    return load_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cachedir")

    args = parser.parse_args()

    outdir = args.cachedir
    os.makedirs(outdir, exist_ok=True)

    sh.rsync([
        "-avz",
        '--progress',
        '--prune-empty-dirs',
        '--include', '*/',
        '--include', 'YUTIMING',
        '--include', 'jobs.csv',
        '--exclude', '*',
        'edigiaco@login.galileo.cineca.it:~/experiments/',
        outdir,
    ], _out=sys.stdout, _err=sys.stderr)
    for name in (
        "fixed_area",
        "variable_area",
    ):
        try:
            basedir = os.path.join(outdir, name)
            df = pd.read_csv(os.path.join(basedir, "jobs.csv"))
            df[["time0", "time1", "time2"]] = df.apply(load_time_factory(basedir), axis=1, result_type="expand")
            df.to_csv(f"{name}_results.csv", index=False)
            print(f"Updated results for {name}")
        except FileNotFoundError:
            print(f"WARNING: jobs file not found in {name}")
