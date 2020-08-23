# coding=utf-8

"""
Generate a sample of randomly generated tree matrices, using joblib parallel processing.
Returns two dictionaries whose keys are hashed matrices. One dictionary contains the matrices themselves, the other
the counts of each matrix in the sample.
"""

import numpy as np
import os
import gzip, pickle
from scipy.special import binom
from collections import Counter, defaultdict
from joblib import Parallel, delayed
from selectiontest import selectiontest
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def sample_matrices3(n, size, seed):
    np.random.seed(seed)
    counts, mxs = Counter(), Counter()
    for j in range(size):
        f = list()
        for i in range(1, n):
            f.append(np.random.choice(i))
        f = f[::-1]
        mx = selectiontest.derive_tree_matrix(f)
        hashmx = mx.tostring()
        mxs[hashmx] = mx
        counts[hashmx] += 1
    return mxs, counts


def sample_wf_distribution(n, reps, seed):
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
    matrices, counts = sample_matrices3(n, reps, seed)
    kvec = np.arange(2, n + 1, dtype=int)
    branch_lengths = np.random.exponential(scale=1 / binom(kvec, 2), size=(reps, n - 1))
    total_branch_lengths = branch_lengths @ kvec
    rel_branch_lengths = list()
    for row, total_length in zip(branch_lengths, total_branch_lengths):
        rel_row = row / total_length
        rel_branch_lengths.append(rel_row)
    rel_branch_lengths = np.array(rel_branch_lengths)
    variates = list()
    rbl_count = 0
    for key in matrices.keys():
        for j in range(counts[key]):
            mx = matrices[key]
            variate = (mx.T).dot(rel_branch_lengths[rbl_count])
            rbl_count += 1
            err = 1 - np.sum(variate)
            variate[np.argmax(variate)] += err
            variates.append(variate)
    return np.array(variates)


def sample_WF_pllel(n, size, njobs):
    seeds = np.random.choice(2 * njobs, njobs, replace=False)
    results = Parallel(n_jobs=njobs)(delayed(sample_wf_distribution)(n, size, seed) for seed in seeds)
    return np.vstack(results)


def sample_matrices_pllel(n, size, njobs):
    seeds = np.random.choice(2 * njobs, njobs, replace=False)
    results = Parallel(n_jobs=njobs)(delayed(sample_matrices3)(n, size, seed) for seed in seeds)
    mxs, counts = Counter(), Counter()
    for pair in results:
        mxs = {**mxs, **pair[0]}
        counts = counts + pair[1]
    return mxs, counts

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
    results = sample_WF_pllel(n, size, njobs)
    fname = dirx + "/wf_samples_" + job_no + ".pklz"
    with gzip.open(fname, 'wb') as outfile:
        pickle.dump(results, outfile)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()