import os
import re
import csv
import pandas as pd

def displacement_csv(file, x, y, z, err, lose, losn, losz):
    if not file.endswith('.csv'):
        file_name = os.path.join(file + '.csv')
    else:
        file_name = os.path.join(file)

    if z.shape != x.shape:
        z = z.flatten()

    df = pd.DataFrame({
        'xx': x,
        'yy': y,
        'dd': z,
        'ee': err,
        'lx': lose,
        'ly': losn,
        'lz': losz,
    })

    # Remove rows with any NaN or empty values
    df = df.dropna()

    print("-" * 50)
    print(f"Saving {file_name}.\n")

    df.to_csv(file_name, index=False)

    return file_name


def results_csv(file):
    db_sar = pd.read_csv(file)
    d_sar = db_sar.values

    east, north = d_sar[:,0],d_sar[:,1]
    data, synth = d_sar[:,3], d_sar[:,2]

    return east, north, data, synth


def read_csv(file):
    with open(file, mode='r') as f:
        reader = csv.DictReader(f)
        # Read the first (and only) row
        row = next(reader)
    return row

def read_best_values(file):
    df = pd.read_csv(file)
    row = df.iloc[0]  # best-fit row

    sources = {}
    terms = ["xcen", "ycen", "radius", "ytlc", "xtlc", "s_axis_max", "ratio", "strike", "dip", "length", "width"]

    for col in df.columns:
        for term in terms:
            # Match columns with the term and optional "_<n>", excluding those ending with "sigma"
            m = re.match(fr"({term})(?:_(\d+))?(?<!sigma)$", col)

            if m:
                feature = m.group(1)  # Extract the feature name (e.g., "xcen", "s_axis_max")
                idx = m.group(2) if m.group(2) else "1"  # Default index = 1 if no "_<n>"

                # Group by source index
                if idx not in sources:
                    sources[idx] = {}

                sources[idx][feature] = row[col]
                break

    return sources