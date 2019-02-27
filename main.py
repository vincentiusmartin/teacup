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

    train = trainingparser.TrainingParser(trainingpath,motiflen=6)
    train.compare_distance_features(iter=100,fpr_lim=20)
