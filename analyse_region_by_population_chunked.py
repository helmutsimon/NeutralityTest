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
import csv
from vcf import Reader  # https://pypi.org/project/PyVCF/
from selectiontest import selectiontest
import pandas as pd
import click
from scitrack import CachingLogger, get_file_hexdigest


LOGGER = CachingLogger(create_dir=True)


def multinomial_pmf_chunk(variates, mask, sfs, seg_sites):
    b = variates[:, mask] > 0
    c = np.all(b > 0, axis=1)
    compat_vars = variates[c, :]
    incompat_count = variates.shape[0] - compat_vars.shape[0]
    if compat_vars.shape[0] == 0:
        h0_sum = 0
        sum_sq = 0
    else:
        probs0 = selectiontest.multinomial_pmf(sfs, seg_sites, compat_vars)
        h0_sum = np.sum(probs0)
        h0 = h0_sum / variates.shape[0]
        sum_sq = np.sum((probs0 - h0) ** 2) + incompat_count * h0 ** 2
    return h0_sum, sum_sq


def multinomial_pmf_chunk_old(variates, mask, sfs, seg_sites):
    b = variates[:, mask] > 0
    c = np.all(b > 0, axis=1)
    compat_vars = variates[c, :]
    if compat_vars.shape[0] == 0:
        return 0
    else:
        return np.sum(selectiontest.multinomial_pmf(sfs, seg_sites, compat_vars))


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
@click.option('-w', '--wreps', default=10000)
@click.option('-c', '--chunksize', default=10000)
@click.option('-u', '--ureps', default=10000)
@click.option('--demog/--no-demog', default=False,
              help='choose whether to implement non-neutral demographic history for European populations.')
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, chrom, start, interval, segments, wreps, chunksize, ureps, demog, dirx):
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
    panel__filename = '/Users/helmutsimon/Data sets/1KG variants full/integrated_call_samples_v3.20130502.ALL.panel'
    infile = open(panel__filename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    panel_all = pd.read_csv(panel__filename, sep=None, engine='python', skipinitialspace=True, index_col=0)
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
        variates1 = selectiontest.sample_uniform_distribution(n, ureps)
        if pop in demog_pops:
            tempfname = 'pcvars_' + pop + '_' + job_no + '.csv'
            with open(tempfname, "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                count1 = 0
                for line, transf_coal_times in selectiontest.piecewise_constant_variates(n, timepoints,\
                                                        pop_sizes, wreps):
                    count1 += 1
                    writer.writerow(line)
                print('Count1: ', count1)
            LOGGER.log_message(pop.ljust(30), label="Modified demographic history for population ")
        else:
            tempfname = 'wfvars_' + pop + '_' + job_no + '.csv'
            with open(tempfname, "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                count1 = 0
                for line in selectiontest.sample_wf_distribution(n, wreps):
                    count1 += 1
                    writer.writerow(line)
                print('Count1: ', count1)
            LOGGER.log_message(pop.ljust(30), label="Neutral demographic history for population  ")
        for segment in range(segments):
            print('\nPopulation               =', pop, (time() - start_time) / 60)
            # Use GRCh37 coordinates
            seg_start = start + segment * interval
            print('Segment start            =', seg_start)
            seg_end = seg_start + interval
            sfs, n2, non_seg_snps = selectiontest.vcf2sfs(vcf_file, panel, chrom, seg_start, seg_end)
            assert n == n2 , 'Sample size mismatch for ' + pop
            tajd = selectiontest.calculate_D(sfs)
            print('Tajimas D                =', tajd)
            print(sfs, n2)
            sys.stdout.flush()
            seg_sites = sum(sfs)
            count = 0
            h0, sumsq = 0, 0
            mask = sfs > 0
            reader = pd.read_csv(tempfname, sep=',', chunksize=chunksize)
            for chunk in reader:
                count += chunksize
                npchunk = chunk.to_numpy()
                assert npchunk.shape[1] == n - 1, "Sample size does not match variates " + str(npchunk.shape[1])
                h0_chunk, sumsq_chunk = multinomial_pmf_chunk(npchunk, mask, sfs, seg_sites)
                h0 += h0_chunk
                sumsq += sumsq_chunk
            assert count == wreps, "Mismatch in number of variates" + str(count)
            h0 = h0 / wreps
            stderr0 = np.sqrt(sumsq) / wreps
            probs1 = selectiontest.multinomial_pmf(sfs, seg_sites, variates1)
            h1 = np.mean(probs1)
            stderr1 = np.std(probs1) / np.sqrt(ureps)
            rho = np.log10(h1) - np.log10(h0)
            print('\u03C1                        =', rho)
            if len(non_seg_snps) > 0:
                print(non_seg_snps)
            row = [pop, seg_start, tajd, rho, h0, stderr0, h1, stderr1]
            rows.append(row)
            sys.stdout.flush()
    results = pd.DataFrame(rows, columns=['pop', 'segstart', 'tajd', 'rlnt', 'lhood0', 'stderr0', 'lhood1', 'stderr01'])
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
