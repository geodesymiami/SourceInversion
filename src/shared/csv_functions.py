import os
import csv
import pandas as pd

def displacement_csv(file, x, y, z, err, lose, losn, losz):
    if not file.endswith('_test.csv'):
        file_name = os.path.join(file + '_test.csv')
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

    print("#" * 50)
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