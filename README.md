# NeutralityTest
This repository contains Python code supporting the analyses in the paper *A New Statistical Test Provides Evidence of Selection Against Deleterious Mutations in Genes Promoting Disease Resistance* (in preparation).

The central topic is the calculation of a statistic for selective neutrality, &rho;, which is a relative likelihood of two evolutionary models.

Code in this repository uses the library *selectiontest*, which is intended to make the core functions available to end users for analysis of their own data. Documentation of *selectiontest* can be found at https://readthedocs.org/projects/selectiontest/.

The following is a brief summary of scripts and notebooks in this repository.

*roc_simulation.py* creates synthetic data for a range of scenarios using the MSMS program (*Ewing, G. and Hermisson, J. (2010). MSMS: a coalescent simulation program including recombination, demographic structure and selection at a single locus. Bioinformatics, 26(16):2064–2065*). It  calculates &rho; and Tajima's D for these data sets. This data can be used to generate receiver operating characteristic (ROC) curves to compare the performance of these two statistics. MSMS is Java code, which we call from within a Python script.

*Plot_roc_curves.ipynb* plots output from *roc_simulation.py*.

*generate_calibration_table.ipynb* generates a table of calibration values used in the paper.

*analyse_region_by_population.py* supports the analysis of the 2q11.1 region of the human chromosome contained in the paper 'A New Statistical Test Provides Evidence of Selection Against Deleterious Mutations in Genes Promoting Disease Resistance'. The analysis uses data in VCF format from the 1000 Genomes Project. See *Auton, A. et al. (2015). A global reference for human genetic variation. Nature, 526:68–74* ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/2013050

*vcf_1KG.p* contains functions supporting the analysis of the 1000 Genomes Project data.

*plot_heatmap.ipynb* is used to plot output from *analyse_region_by_population.py*.

*Chromosome 2q11.1-analyse 20k segment.ipynb* supports further analysis of the 2q11.1 region of the human chromosome.

*ACKR1 Gene (FY, DARC, Duffy).ipynb* supports the analysis of the ACKR1 gene contained in the paper 'A New Statistical Test Provides Evidence of Selection Against Deleterious Mutations in Genes Promoting Disease Resistance'. This uses the same 1000 Genomes Project data set as above.
