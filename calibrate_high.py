# coding=utf-8

""" Calibrate the neutrality test by calculating a threshold for a given false positive
    rate (default is 0.02) against the Wright-Fisher model.
    We read a file of W-F variates, which are read in chunks to determine the likelihood of SFS values.
    This version keeps chunk in memory to process all sfs instances."""

import os, sys
import numpy as np
import pandas as pd
from collections import Counter
from scipy.special import binom
from selectiontest import selectiontest
from joblib import Parallel, delayed
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def multinomial_pmf_chunk(variates, mask, sfs, seg_sites):
    b = variates[:, mask] > 0
    c = np.all(b > 0, axis=1)
    compat_vars = variates[c, :]
    if compat_vars.shape[0] == 0:
        return 0
    else:
        return np.sum(selectiontest.multinomial_pmf(sfs, seg_sites, compat_vars))


@click.command()
@click.argument('job_no')
@click.argument('sample_size', type=int)
@click.argument('seg_sites', type=int)
@click.argument('wchunks', type=int)
@click.option('-c', '--chunksize', default=10000, help="Chunk size for reading variates file.")
@click.option('-f', '--fpr', default=0.05, help="False positive rate. Default = 0.05")
@click.option('-sr', '--sreps', default=10000, help="Number of repetitions to generate sfs.")
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, sample_size, wchunks, seg_sites, chunksize, fpr, sreps, dirx):
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
    sfs_array = list(selectiontest.generate_sfs_array(sample_size, seg_sites, sreps))

    results = list()
    linecount = 0
    for j in range(wchunks):
        print(time() - start_time, j)
        sys.stdout.flush()
        chunk = np.empty((chunksize, sample_size - 1), dtype=float)
        for i, q in enumerate(selectiontest.sample_wf_distribution(sample_size, chunksize)):
            chunk[i] = q
        chunk_col = list()
        linecount += chunk.shape[0]
        assert chunk.shape[1] == sample_size - 1, "Sample size does not match variates" + str(chunk.shape[1])
        for i, sfs in enumerate(sfs_array):
            mask = sfs > 0
            h0 = multinomial_pmf_chunk(chunk, mask, sfs, seg_sites)
            chunk_col.append(h0)
        results.append(chunk_col)
    LOGGER.log_message(str(linecount), label="Number of WF variates processed".ljust(50))
    results = pd.DataFrame(results)
    results = results.transpose()
    print(results)
    fname = dirx + "/wf_likelihoods_" + job_no + ".csv"
    results.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    wf_lhoods = results.sum(axis=1) / linecount
    variates1 = selectiontest.sample_uniform_distribution(sample_size, sreps)
    rho_vals = list()
    for h0, sfs in zip(wf_lhoods, sfs_array):
        h1 = np.mean(selectiontest.multinomial_pmf(sfs, seg_sites, variates1))
        rho = np.log10(h1) - np.log10(h0)
        rho_vals.append(rho)
    rho_vals = np.array(rho_vals)
    print('Count of -inf: ', np.sum(np.isneginf(rho_vals)))
    print('Count of +inf: ', np.sum(np.isinf(rho_vals)))
    print('Count of  nan: ', np.sum(np.isnan(rho_vals)))
    rho_vals = rho_vals[~np.isnan(rho_vals)]
    rho_vals = np.sort(rho_vals)
    thr =  rho_vals[int(len(rho_vals) * (1 - fpr))]
    LOGGER.log_message("%.4f" % thr, label="Threshold".ljust(50))
    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()