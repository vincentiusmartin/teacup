code
=========

parsing the input files and analyze::

    infile = "data/dataset/all_Myc_Mad_2nd_myc_log_bound2.txt"
    negctrlfile = "data/dataset/all_Myc_Mad_2nd_myc_log_negctrl.txt"
    logsetting= True
    pvalthres = .05

    tbl = read_tbl(infile)
    med = get_median_orientation(tbl)

    cutoff = cutoff_from_negctrl(negctrlfile)

    indivsum,twosites =  make_permutation(tbl)
    classification = classify_probes(med,indivsum,twosites,cutoff,pvalthres)

    classification2file(classification)

    ax = plot_median_binding_sum(med,classification,0,log=logsetting,plotname="plot_overlap")
    ax_o1 = plot_median_binding_sum(med,classification,1,log=logsetting,plotname="plot_o1")
    ax_o2 = plot_median_binding_sum(med,classification,2,log=logsetting,plotname="plot_o2")

    scatter_boxplot(indivsum,twosites,2001)

get pwm distance::

    pwm = read_dna_pwm("data/pwm/MYC_M4532_1.02.txt",1,7)
    categories = read_classification_file("data/generated/classification.txt")

    seq = get_seq(tbl,"wt",categories,"coop_overlap")
    dist = get_distances_pwm(seq,pwm)

classify all data::

    classify_all(tbl,cutoff,pvalthres)
    plot_training("training.csv")

get scatter for all permutations in a specific probe::

    scatter_boxplot_probe(tbl,2014)

make distance plot using pwm and e-score data::

    pwm = read_dna_pwm("data/pwm/MYC-MAX_8mers_pwm.txt",8,15)
    escore = read_escore("data/escore/MYC-MAX_8mers_11111111.txt")

    categories = read_classification_file("classification.txt")

    mutpos = get_mutpos(tbl,categories["coop_overlap"])
    seqdict = get_seq(tbl,"wt",categories["coop_overlap"])
    coop_overlap = get_distances_pwm(seqdict,pwm)

    lineplot_pwm_escore(seqdict,pwm,escore,mutpos,startoffset=1,endoffset=1)

filter sequence using pwm and escore::

    filter_sequences(seqdict,pwm,escore,category="additive")
    filter_sequences(seqdict,pwm,escore,category="coop")

to plot after filter::

    # read classification file so we don't need to run the function again
    categories = read_classification_file("data/generated/classification.txt")

    # this read from the output of filter_sequences
    filteredcategories = read_classification_file("data/generated/classification-filtered.txt")

    ax = plot_median_binding_sum(med,categories,0,log=logsetting,plotname="plot_overlap",
    subset_coop=filteredcategories["coop_filtered"],
    subset_additive=filteredcategories["additive_filtered"])

UPDATED

to initialize::
    infile = "data/dataset/all_Myc_Mad_2nd_myc_log_bound2.txt"
    negctrlfile = "data/dataset/all_Myc_Mad_2nd_myc_log_negctrl.txt"
    logsetting= True
    pvalthres = .05

    probes = probeparser.ProbeParser(infile,negctrlfile)

    probes.scatter_boxplot(2000)

    classification = classifier.classify_probes(probes,pvalthres)

    utils.dictlist2file(classification,"test.txt")

    classification = utils.read_dictlist_file("test.txt")

    ax = classifier.plot_median_binding_sum(probes,classification,0,log=logsetting,plotname="plot_overlap")
    #ax_o1 = plot_median_binding_sum(med,classification,1,log=logsetting,plotname="plot_o1")
    #ax_o2 = plot_median_binding_sum(med,classification,2,log=logsetting,plotname="plot_o2")
