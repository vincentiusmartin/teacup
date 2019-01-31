import sys
sys.path.insert(0, 'src')

from teacup import probeparser
from teacup import classifier
from teacup import utils
from teacup import sitesfinder
from teacup import trainingparser

if __name__=="__main__":
    infile = "data/dataset/all_Myc_Mad_2nd_myc_log_bound2.txt"
    negctrlfile = "data/dataset/all_Myc_Mad_2nd_myc_log_negctrl.txt"
    pwmpath = "data/pwm/MYC-MAX_8mers_pwm.txt"
    escorepath = "data/escore/MYC-MAX_8mers_11111111.txt"
    trainingpath = "data/generated/training.csv"
    logsetting= True
    pvalthres = .05

    train = trainingparser.TrainingParser(trainingpath)
    train.scatter_boxplot_col("distance")
    train.train_from_distance(categorical=True)
