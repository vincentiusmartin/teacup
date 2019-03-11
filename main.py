import sys
sys.path.insert(0, 'src')

from teacup import utils
from teacup.probes import probeparser
from teacup.probes import classifier
from teacup.probes import sitesfinder
from teacup.training import trainingparser
from teacup.training import dnashape

if __name__=="__main__":
    infile = "data/dataset/all_Myc_Mad_2nd_myc_log_bound2.txt"
    negctrlfile = "data/dataset/all_Myc_Mad_2nd_myc_log_negctrl.txt"
    pwmpath = "data/pwm/MYC-MAX_8mers_pwm.txt"
    escorepath = "data/escore/MYC-MAX_8mers_11111111.txt"
    trainingpath = "data/generated/training.csv"
    logsetting= True
    pvalthres = .05
    shapepath = "data/dnashape"
    distances = range(10,21)

    #tp = trainingparser.TrainingParser(trainingpath,6)
    #tp.test_model(["dist-numeric","linker_2mer"], testing_type="cv", outpath="roc.png")
    #tp.visualize_random_forest(['dist-numeric','linker-1mer','linker-2mer'])
    #tp.compare_distance_features(iter=1,fpr_lim=100)
    #tp.compare_dist_linker_features(iter=10, fpr_lim=20)
    #tp.compare_dist_pos_features(iter=10,fpr_lim=20)

    ds = dnashape.plot_average_all(trainingpath,shapepath,distances)
