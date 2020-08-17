# coding=utf-8


"""
A simple model of background selection using fwdpy.

A sample run statement is:

nohup python3 /Users/helmutsimon/repos/NeutralityTest/fwdpy11_background_sel.py 001 1000 0.005 0.04 0.02 20 1500 -nr 500 > fbs_001.txt &
"""

import pandas as pd
import os
import pybind11
from collections import Counter
import attr
from time import time
import numpy as np
import click
import fwdpy11
from selectiontest import selectiontest
from joblib import Parallel, delayed
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)

def tmrca_from_pop(pop, l):
    nodes = np.array(pop.tables.nodes, copy=False)
    a = [node[1] for node in nodes]
    b = Counter(a)
    return l - min([x for x in b.keys() if b[x] == 1])


def evolve_sfs(N, genome_length, params, nsam, l, variates0, variates1, seed):
    """
    Evolve population with neutral mutation rate passed in parameters. Return SFS of random sample of
    specified size.
    """
    pop = fwdpy11.DiploidPopulation(N, genome_length)    # Initializes a population
    rng = fwdpy11.GSLrng(seed)
    # Evolve population, pruning every 50 steps
    fwdpy11.evolvets(rng, pop, params, 50, suppress_table_indexing=True)   #suppress_... because no recombination?
    np.random.seed(seed)
    sample_ix = np.random.choice(N, nsam, replace=False)
    alive_nodes = pop.alive_nodes                                           # the present-day nodes
    sample = [alive_nodes[sample_ix]]
    sfs = pop.tables.fs(sample).compressed() #ignores 1st and last entries (mutations not in sample and fixed)
    rho = selectiontest.test_neutrality(sfs, variates0=variates0, variates1=variates1)
    tajD = selectiontest.calculate_D(sfs)
    return (sfs, tmrca_from_pop(pop, l), rho, tajD)


@click.command()
@click.argument('job_no')
@click.argument('pop_size', type=int)       # population size
@click.argument('un', type=float)    # neutral mutation rate per haploid genome per generation.
@click.argument('us', type=float)    # selective_mutation_rate per haploid genome per generation.
@click.argument('s', type=float)     # absolute value of selection coefficient
@click.argument('n', type=int)       # sample size
@click.argument('l', type=int)       # number of generations to run simulation
@click.option('-h', '--h', default=0.5, help="Dominance coefficient, default = 0.5 for additive effect")
@click.option('-g', '--genome_length', default=100)
@click.option('-nr', '--nreps', default=10000, help="Number of simulations")
@click.option('-s', '--seed', type=int, default=None)
@click.option('-j', '--n_jobs', default=4, type=int, help='Number of parallel jobs. Default is 4.')
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, genome_length, pop_size, un, us, s, h, n, l, nreps, seed, n_jobs, dirx):
    start_time = time()
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
    LOGGER.log_message('Name = ' + fwdpy11.__name__ + ', version = ' + fwdpy11.__version__, label=label)
    LOGGER.log_message('Name = ' + selectiontest.__name__ + ', version = ' + selectiontest.__version__, label=label)
    LOGGER.log_message('Name = ' + pybind11.__name__ + ', version = ' + pybind11.__version__, label=label)
    LOGGER.log_message('Name = ' + attr.__name__ + ', version = ' + attr.__version__, label=label)
    testresults = dict()
    if seed is not None:
        np.random.seed(seed)
    s = -s                    #selection coefficient is negative
    lam = -us / s
    print('Fixation metric: ', -pop_size * s * np.exp(us / s))  # See Cvijovic p.1238 (requires this metric >>1)
    print('Lambda         : ', lam)  # Needs to be > 1 for background selection to be effective (ibid. p1239)
    theta = 4 * pop_size * un
    intervals = str(-np.exp(lam) * n / (pop_size * s)) + ', ' + str((1 + np.exp(lam) / (pop_size * s)) * n)
    LOGGER.log_message(intervals, label="Interval for Charlesworth".ljust(50))
    print('theta= ', theta)
    # With background selection
    pdict = {"gvalue": fwdpy11.Multiplicative(1.0),  # multiplicave effect, s is effect of mutant homozygote
              "rates": (un, us, 0.0),  # neutral mutation rate, selected mutation rate, recombination rate
              "nregions": [fwdpy11.Region(0, 1.0, 1)],  # region for neutral mutations -
              # not needed if using fwdpy11.infinite_sites
              "sregions": [fwdpy11.ConstantS(0, 1.0, 1, s, h)],  # whole genome
              "recregions": [],  # region for recombination (none)
              "demography": fwdpy11.DiscreteDemography(),  # No population structure or growth
              "simlen": l}  # Should >> TMRCA if we are to get full tree
    params = fwdpy11.ModelParams(**pdict)
    np.set_printoptions(precision=3)
    variates0 = selectiontest.sample_wf_distribution(n, nreps)
    variates1 = selectiontest.sample_uniform_distribution(n, nreps)
    seeds = np.random.choice(nreps * 50, size=nreps, replace=False)
    results = Parallel(n_jobs=n_jobs)(delayed(evolve_sfs)(pop_size, genome_length, params, n, l,
                    variates0, variates1, seed) for seed in seeds)
    sfs_list = [item[0] for item in results]
    tmrcas = [item[1] for item in results]
    testresults['rho_true'] = [item[2] for item in results]
    testresults['taj_true'] = [item[3] for item in results]
    LOGGER.log_message("%.2f" % np.mean(tmrcas), label="Mean selection TMRCA".ljust(50))
    sfs_df = pd.DataFrame(sfs_list)
    fname = 'fwdpy_bgrd_sfs' + job_no + '.csv'
    sfs_df.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    wf_sfs = theta / np.arange(1, n + 1)
    l1 = np.mean(sfs_df, axis=0)
    l2 = wf_sfs
    l3 = wf_sfs * np.exp(-lam)
    for x, y, z in zip(l1, l3, l2):
        print("%.3f" % x, "%.3f" % y, "%.3f" % z)

    # Neutral model
    pdict["sregions"] = []
    params = fwdpy11.ModelParams(**pdict)
    np.set_printoptions(precision=3)
    seeds = np.random.choice(nreps * 50, size=nreps, replace=False)
    results = Parallel(n_jobs=n_jobs)(delayed(evolve_sfs)(pop_size, genome_length, params, n, l,
                                                          variates0, variates1, seed) for seed in seeds)
    sfs_list = [item[0] for item in results]
    tmrcas = [item[1] for item in results]
    testresults['rho_false'] = [item[2] for item in results]
    testresults['taj_false'] = [item[3] for item in results]
    LOGGER.log_message("%.2f" % np.mean(tmrcas), label="Mean neutral TMRCA".ljust(50))
    sfs_df = pd.DataFrame(sfs_list)
    fname = 'fwdpy_neutral_sfs' + job_no + '.csv'
    sfs_df.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    duration = time() - start_time

    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()