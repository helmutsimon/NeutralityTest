# coding=utf-8

""" Calibrate the neutrality test by calculating a threshold for a given false positive
    rate (default is 0.02) against the Wright-Fisher model."""

import os, sys
import numpy as np
import pandas as pd
from selectiontest import selectiontest
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def compute_tajD_threshold(n, seg_sites, reps=10000, fpr=0.05):
    """
    Calculate threshold value of :math:`\\rho` corresponding to a given false positive rate (FPR).
    For values of :math:`\\rho` above the threshold we reject the
    null (by default neutral) hypothesis.

    Parameters
    ----------
    n: int
        Sample size
    seg_sites: int
        Number of segregating sites in sample.
    reps: int
        Number of SFS configs to generate if default is used.
    fpr: float
        Selected FPR tolerance.

    Returns
    -------
    numpy.float64 (2)
        Upper and lower threshold values for Tajima's D

    """
    results = list()
    for sfs in selectiontest.generate_sfs_array(n, seg_sites, reps):
        tajD = selectiontest.calculate_D(sfs)
        results.append(tajD)
    results = np.array(results)
    results = np.sort(results)
    return results[int(len(results) * (fpr/2))], results[int(len(results) * (1 - fpr/2))]



@click.command()
@click.argument('job_no')
@click.argument('seg_sites_values', nargs=4, type=int)
@click.argument('sample_size_values', nargs=-1, type=int)
@click.option('-f', '--fpr', default=0.02, help="False positive rate. Default = 0.02")
@click.option('-r', '--reps', default=100000, help="Number of repetitions to generate sfs samples.")
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, seg_sites_values, sample_size_values, fpr, reps, dirx):
    np.set_printoptions(precision=3)                #
    if not os.path.exists(dirx):
        os.makedirs(dirx)
    LOGGER.log_file_path = dirx + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass
    label = "Imported module".ljust(30)
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__, label=label)
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__, label=label)
    LOGGER.log_message('Name = ' + selectiontest.__name__ + ', version = ' + selectiontest.__version__, label=label)

    start_time = time()
    rows = list()
    print('job = ', job_no)
    print('fpr = ', fpr, '\n')
    for sn in seg_sites_values:
        thresholds = list()
        if sn == 0:
            break
        for n in sample_size_values:
            lower, upper = compute_tajD_threshold(n, sn, reps=reps, fpr=fpr)
            print("%4d" % n, "%3d" % sn, "%.3f" % lower, "%.3f" % upper)
            sys.stdout.flush()
            thresholds.append(lower)
            thresholds.append(upper)
        rows.append(thresholds)
    columns = list()
    for ssv in sample_size_values:
        columns.append(str(ssv) + '_lower')
        columns.append(str(ssv) + '_upper')
    results = pd.DataFrame(rows, index = seg_sites_values, columns=columns)
    fname = dirx + "/calibration_tajD_" + job_no + ".csv"
    results.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()