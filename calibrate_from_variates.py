# coding=utf-8

""" Calibrate the neutrality test by calculating a threshold for a given false positive
    rate (default is 0.02) against the Wright-Fisher model."""

import os, sys
import numpy as np
import pandas as pd
import pickle, gzip
from selectiontest import selectiontest
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)



def compute_threshold(n, seg_sites, variates0, sreps=10000, fpr=0.02):
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
    njobs: int
        Number of parallel joblib processes.
    sreps: int
        Number of SFS configs and of uniform variates to generate if default is used.
    wreps: int
        Number of Wright-Fisher variates to generate if default is used.
    fpr: float
        Selected FPR tolerance.

    Returns
    -------
    numpy.float64
        Threshold value for log odds ratio

    """

    #variates0 = selectiontest.sample_wf_distribution(n, wreps)
    #variates0 = sample_WF_pllel(n, wreps, njobs)
    assert variates0.shape[1] == n - 1, "Variates incorrect shape: " + str(variates0.shape[1])
    variates1 = selectiontest.sample_uniform_distribution(n, sreps)
    sfs_array = selectiontest.generate_sfs_array(n, seg_sites, sreps)
    print('sfs simulation complete')
    sys.stdout.flush()
    num_wf_vars = variates0.shape[0]
    results = list()
    for sfs in sfs_array:
        a = sfs > 0
        b = variates0[:, a] > 0
        c = np.all(b > 0, axis=1)
        compat_vars = variates0[c, :]
        if compat_vars.shape[0] == 0:
            h0 = 0
        else:
            h0 = np.sum(selectiontest.multinomial_pmf(sfs, seg_sites, compat_vars)) / num_wf_vars
        h1 = np.mean(selectiontest.multinomial_pmf(sfs, seg_sites, variates1))
        rho = np.log10(h1) - np.log10(h0)
        results.append(rho)
    print('selectiontest complete')
    sys.stdout.flush()
    results = np.array(results)
    print(np.sum(np.isneginf(results)))
    print(np.sum(np.isinf(results)))
    print(np.sum(np.isnan(results)))
    sys.stdout.flush()
    results = results[~np.isnan(results)]
    results = np.sort(results)
    return results[int(len(results) * (1 - fpr))]

@click.command()
@click.argument('job_no')
@click.argument('sample_size', type=int)
@click.argument('fname')
@click.argument('seg_sites_values', nargs=-1, type=int)
@click.option('-f', '--fpr', default=0.02, help="False positive rate. Default = 0.02")
@click.option('-sr', '--sreps', default=10000, help="Number of repetitions to generate sfs and uniform samples.")
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, seg_sites_values, fname, sample_size, fpr, sreps, dirx):
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
    thresholds = list()
    fname = dirx + '/' + fname + '.csv'
    #variates0 = np.genfromtxt(fname, delimiter=',')
    infile = open(fname, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    with gzip.open(fname, 'rb') as variates0:
        variates0 = pickle.load(variates0)
    for sn in seg_sites_values:
        thr = compute_threshold(sample_size, sn, variates0, sreps=sreps, fpr=fpr)
        duration = time() - start_time
        print("%.2f" % (duration / 60.), "%4d" % sample_size, "%3d" % sn, "%.3f" % thr)
        sys.stdout.flush()
        thresholds.append(thr)
    results = pd.Series(thresholds, index = seg_sites_values, name=sample_size)
    fname = dirx + "/calibration_" + job_no + ".csv"
    results.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()