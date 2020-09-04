# coding=utf-8

"""
Generate a sample of randomly generated tree matrices, using joblib parallel processing.
Returns two dictionaries whose keys are hashed matrices. One dictionary contains the matrices themselves, the other
the counts of each matrix in the sample.
"""

import numpy as np
import os, sys
import csv
#import gzip, pickle
from scipy.special import binom
#from collections import Counter, defaultdict
from joblib import Parallel, delayed
from selectiontest import selectiontest
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def sample_matrix(n, reps):
    for i in range(reps):
        f = list()
        for i in range(1, n):
            f.append(np.random.choice(i))
        f = f[::-1]
        mx = selectiontest.derive_tree_matrix(f)
        yield mx


def sample_branch_lengths(n, reps):
    kvec = np.arange(2, n + 1, dtype=int)
    branch_lengths = np.random.exponential(scale=1 / binom(kvec, 2), size=(reps, n - 1))
    total_branch_lengths = branch_lengths @ kvec
    for row, total_length in zip(branch_lengths, total_branch_lengths):
        rel_row = row / total_length
        yield rel_row


def sample_wf_distribution(n, reps):
    """
    Calculate variates for the probability distribution Q under Wright Fisher model.

    Parameters
    ----------
    n: int
        Sample size
    reps: int
        Number of variates to generate if default is used.

    Returns
    -------
    numpy.ndarray
         Array of variates (reps, n-1)

    """
    for mx, rel_branch_length in zip(sample_matrix(n, reps), sample_branch_lengths(n, reps)):
        variate = (mx.T).dot(rel_branch_length)
        err = 1 - np.sum(variate)
        variate[np.argmax(variate)] += err
        yield(variate)


def sample_WF_pllel(n, size, njobs):
    seeds = np.random.choice(2 * njobs, njobs, replace=False)
    results = Parallel(n_jobs=njobs)(delayed(sample_wf_distribution)(n, size, seed) for seed in seeds)
    return np.vstack(results)


@click.command()
@click.argument('job_no')
@click.argument('n', type=int)
@click.argument('size', type=int)
@click.option('-j', '--njobs', default=10, help="Number of repetitions")
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, n, size, njobs, dirx):
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
    LOGGER.log_message('Name = ' + selectiontest.__name__ + ', version = ' + selectiontest.__version__, label=label)

    start_time = time()
    fname = dirx + "/wf_samples_" + job_no + ".csv"
    count = 0
    with open(fname, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in sample_wf_distribution(n, size):
            count += 1
            if count % 50000 == 0:
                print(count)
                sys.stdout.flush()
            writer.writerow(line)


    #np.savetxt(fname, results, delimiter=",")
    #with gzip.open(fname, 'wb') as outfile:
    #    pickle.dump(results, outfile)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()