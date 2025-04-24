#!/usr/bin/env python3

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


def run_decomposition(args):
    """
    Run the decomposition module in the 'base' conda environment.
    """
    try:
        cmd = [
            "bash", "-c",
            f"source /Users/giacomo/code/rsmas_insar/tools/miniforge3/etc/profile.d/conda.sh && conda activate base && python src/decomposition/run_decomposition.py {args}"
        ]
        print(f"Running decomposition command: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running decomposition: {e}")
        sys.exit(1)


def run_inversion(args):
    """
    Run the inversion module in the 'vsm' conda environment.
    """
    try:
        cmd = [
            "bash", "-c",
            f"source /Users/giacomo/code/rsmas_insar/tools/miniforge3/etc/profile.d/conda.sh && conda activate vsm && python src/inversion/run_inversion.py {args}"
        ]
        print(f"Running inversion command: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running inversion: {e}")
        sys.exit(1)


def main():
    # Path to the template file
    template_path = "/Users/giacomo/code/rsmas_insar/tools/SourceInversion/template.json"

    # Load arguments from the template file
    template = load_template(template_path)

    # Extract arguments for decomposition and inversion
    decomp_args = template.get("decomposition", "")
    inversion_args = template.get("inversion", "")

    # Run decomposition
    print("Running decomposition...")
    run_decomposition(decomp_args)

    # Run inversion
    print("Running inversion...")
    run_inversion(inversion_args)


if __name__ == "__main__":
    main()