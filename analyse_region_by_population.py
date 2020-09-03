# coding=utf-8

"""
Script used to analyse the 2q11./1 region.
A sample run statement is:

nohup python3 /Users/helmutsimon/repos/NeutralityTest/analyse_region_by_population.py
004 2 96765244 20000 25 50000 --demog > o.txt &
"""

import os, sys
from time import time
import numpy as np
from vcf import Reader  # https://pypi.org/project/PyVCF/
from selectiontest import selectiontest
import pandas as pd
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def create_heatmap_table(results, panel_all, statistic):
    """
    Reformat data from format produced internally by analyse_region_by_population.py for seaborn heatmap.

    """
    panel2 = panel_all[['pop', 'super_pop']]
    pop_tab = panel2.drop_duplicates(subset=['pop', 'super_pop'])
    heat_table = results.pivot(index='pop', columns='segstart', values=statistic)
    heat_table = heat_table.merge(pop_tab, on='pop')
    heat_table = heat_table.sort_values(['super_pop'])
    heat_table = heat_table.drop(columns='super_pop')
    heat_table.set_index('pop', inplace=True)
    return heat_table


@click.command()
@click.argument('job_no')
@click.argument('chrom')
@click.argument('start', type=int)
@click.argument('interval', type=int)
@click.argument('segments', type=int)
@click.argument('reps', type=int)
@click.option('--demog/--no-demog', default=False,
              help='choose whether to implement non-neutral demographic history for European populations.')
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, chrom, start, interval, segments, reps, demog, dirx):
    start_time = time()
    if not os.path.exists(dirx):
        os.makedirs(dirx)
    LOGGER.log_file_path = dirx + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + selectiontest.__name__ + ', version = ' + selectiontest.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('chr' + str(chrom) + ':' + str(start) + '-' + str(start + segments * interval),
                       label="Genomic region".ljust(30))
    selected_pops = ['YRI', 'LWK', 'GWD', 'MSL', 'ESN', 'ASW', 'ACB', 'CEU', 'TSI', 'FIN', 'GBR', 'IBS']
    fname = '/Users/helmutsimon/Data sets/1KG variants full/integrated_call_samples_v3.20130502.ALL.panel'
    infile = open(fname, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    panel_all = pd.read_csv(fname, sep=None, engine='python', skipinitialspace=True, index_col=0)
    vcf_filename = '/Users/helmutsimon/Data sets/1KG variants full/ALL.chr' + str(chrom) \
                   + '.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz'
    infile = open(vcf_filename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    vcf_file = Reader(filename=vcf_filename, compressed=True, encoding='utf-8')
    rows = list()
    panel_select = panel_all[panel_all['pop'].isin(selected_pops)]
    pops = list(set(panel_select['pop']))
    pop_sizes = [6.6e3, 3.3e3, 1e4]
    LOGGER.log_message(''.join([str(x) + ', ' for x in pop_sizes]), label='Population sizes')
    timepoints = [0, 500, 1500]
    LOGGER.log_message(''.join([str(x) + ', ' for x in timepoints]), label='Time points     ')
    if demog:
        demog_pops = ['CEU', 'TSI', 'FIN', 'GBR', 'IBS']
    else:
        demog_pops = []
    for pop in pops:
        panel = panel_select[panel_select['pop'] == pop]
        n = panel.shape[0]
        if pop in demog_pops:
            variates = selectiontest.piecewise_constant_variates(n, timepoints, pop_sizes, reps)
            LOGGER.log_message(pop.ljust(30), label="Modified demographic history for population ")
        else:
            variates = np.empty((reps, n - 1), dtype=float)
            for i, q in enumerate(selectiontest.sample_wf_distribution(n, reps)):
                variates[i] = q
            LOGGER.log_message(pop.ljust(30), label="Neutral demographic history for population  ")
        for segment in range(segments):
            print('\nPopulation               =', pop)
            # Use GRCh37 coordinates
            seg_start = start + segment * interval
            print('Segment start            =', seg_start)
            seg_end = seg_start + interval
            sfs, n2, non_seg_snps = selectiontest.vcf2sfs(vcf_file, panel, chrom, seg_start, seg_end)
            assert n == n2 , 'Sample size mismatch for ' + pop
            tajd = selectiontest.calculate_D(sfs)
            print('Tajimas D                =', tajd)
            print(sfs)
            print(type(sfs))
            rho = selectiontest.test_neutrality(sfs, variates0=variates)  # variates parameter only if required
            print('\u03C1                        =', rho)
            if len(non_seg_snps) > 0:
                print(non_seg_snps)
            row = [pop, seg_start, tajd, rho]
            rows.append(row)
            sys.stdout.flush()
    results = pd.DataFrame(rows, columns=['pop', 'segstart', 'tajd', 'rlnt'])
    fname = dirx + '/chr' + str(chrom) + 'gene_cluster_results' + job_no + '.csv'
    results.to_csv(fname, sep=',')
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    heatmap_table_r = create_heatmap_table(results, panel_select, 'rlnt')
    fname = dirx + '/chr' + str(chrom) + '_heat_table_rlnt' + job_no + '.csv'
    heatmap_table_r.to_csv(fname, sep=',')
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    heatmap_table_t = create_heatmap_table(results, panel_select, 'tajd')
    fname = dirx + '/chr' + str(chrom) + '_heat_table_tajd' + job_no + '.csv'
    heatmap_table_t.to_csv(fname, sep=',')
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()
