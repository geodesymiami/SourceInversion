#!/usr/bin/env python3

import os
import sys
import json
import subprocess


def load_template(template_path):
    """
    Load arguments from a JSON template file.
    """
    try:
        with open(template_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Template file not found: {template_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding JSON in template file: {template_path}")
        sys.exit(1)


def run_downsample(args):
    """
    Run the downsample module in the 'base' conda environment.
    """
    try:
        cmd = [
            "bash", "-c",
            f"source {os.getenv('RSMASINSAR_HOME')}/tools/miniforge3/etc/profile.d/conda.sh && conda activate base && python src/downsample/run_downsample.py {args}"
        ]
        print(f"Running downsample command: {' '.join(cmd)}\n")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running downsample: {e}")
        sys.exit(1)


def run_inversion(args):
    """
    Run the inversion module in the 'vsm' conda environment.
    """
    try:
        cmd = [
            "bash", "-c",
            f"source {os.getenv('RSMASINSAR_HOME')}/tools/miniforge3/etc/profile.d/conda.sh && conda activate vsm && python src/inversion/run_inversion.py {args}"
        ]
        print(f"Running inversion command: {' '.join(cmd)}\n")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running inversion: {e}")
        sys.exit(1)


def main():
    # Path to the template file
    template_path = f"{os.getenv('RSMASINSAR_HOME')}/tools/SourceInversion/template.json"

    # Load arguments from the template file
    template = load_template(template_path)

    # Extract arguments for downsample and inversion
    decomp_args = template.get("downsample", "")
    inversion_args = template.get("inversion", "")

    # Run downsample
    print("Running downsampling...\n")
    run_downsample(decomp_args)

    # Run inversion
    print("Running inversion...\n")
    run_inversion(inversion_args)


if __name__ == "__main__":
    main()