#!/bin/bash

if [ "$1" == "decomp" ]; then
  conda run -n decomp_env python src/cli/run_decomp.py
elif [ "$1" == "inversion" ]; then
  conda run -n inversion_env python src/cli/run_inversion.py
else
  echo "Running full workflow..."
  conda run -n decomp_env python src/cli/run_decomp.py
  conda run -n inversion_env python src/cli/run_inversion.py
fi