#!/bin/bash

import subprocess
import argparse

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_decomposition():
    subprocess.run(["conda", "run", "-n", "base", "python", "src/cli/run_decomp.py"])

def run_inversion():
    subprocess.run(["conda", "run", "-n", "vsm", "python", "src/cli/run_inversion.py"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomp", action="store_true", help="Run decomposition only")
    parser.add_argument("--inversion", action="store_true", help="Run inversion only")
    args = parser.parse_args()

    if args.decomp:
        run_decomposition()
    elif args.inversion:
        run_inversion()
    else:
        print("Running full workflow: decomposition + inversion")
        run_decomposition()
        run_inversion()