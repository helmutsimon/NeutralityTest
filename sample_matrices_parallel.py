# coding=utf-8

import numpy as np
import os
import gzip, pickle
from collections import Counter, defaultdict
from joblib import Parallel, delayed
from selectiontest import selectiontest
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def sample_matrices3(n, size):
    counts, mxs = Counter(), Counter()
    for i in range(size):
        f = list()
        for i in range(1, n):
            f.append(np.random.choice(i))
        f = f[::-1]
        mx = selectiontest.derive_tree_matrix(f)
        hashmx = mx.tostring()
        mxs[hashmx] = mx
        counts[hashmx] += 1
    return mxs, counts


def sample_matrices_pllel(n, size, njobs):
    results = Parallel(n_jobs=njobs)(delayed(sample_matrices3)(n, size) for i in range(1))
    counts, mxs = Counter(), Counter()
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
    mxs, counts = sample_matrices_pllel(n, size, njobs)
    results = [mxs, counts]
    fname = dirx + "/matrix_samples_" + job_no + ".csv"
    with gzip.open(fname, 'wb') as outfile:
        pickle.dump(results, outfile)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()