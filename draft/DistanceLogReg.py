import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

DATA_PATH = "../data/generated/training.csv"

def prepare_data(data_path):
    # Prepare the data
    df = pd.read_csv(data_path)

    # shuffle the data
    for i in range(10):
        df = shuffle(df)
        
    # Import data
    # x is all columns but the last
    x =  df.iloc[:, 1].values
    # y is the last column
    y = df.iloc[:, -1].values

    # set cooperative as 0, additive as 1
    y_bin = [0 if y[i] == "cooperative" else 1 for i in range(len(y))]
    y_bin = np.array(y_bin)
    return x.reshape(-1,1), y_bin.reshape(-1,1)

def get_parameter(x, y):
    # get 12 values for K

    K = [10000,5000,1000,100,10, 1, 0.1, 0.01, 0.001, 0.0001,0.00001, 0.000001]
    cv = KFold(n_splits=10)
    output = []
    for k in K:
        acc = 0
        i = 1
        # Train with the 9 cross validation, test with 1, for all splits
        for train,test in cv.split(x,y):
            clf = LogisticRegression(C=k)
            y_score = clf.fit(x[train], y[train]).predict(x[test])
            i += 1
            acc += accuracy_score(y[test].ravel(),y_score.ravel())
        # get the average
        output.append(acc/10)
    # index of max accuracy
    index = output.index(max(output))
    return K[index], output

def train(k):
    clf = LogisticRegression(C=k)
    cv = KFold(n_splits=10,shuffle=False)
    # Compute ROC curve and ROC area
    i = 1
    fpr_list = []
    tpr_list = []
    print("Beginning our 10 fold cross validation")
    for train,test in cv.split(x,y):
        y_score = clf.fit(x[train], y[train]).predict(x[test])
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y[test].ravel(), y_score.ravel())
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        i += 1
    return fpr_list, tpr_list

def display_output(fpr_list, tpr_list):
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i], lw=2, alpha=0.3, label='ROC fold %d' % (i+1))

    # Show the ROC curves for all 10 test folds on the same plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All 10 Test Folds')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')

if __name__=="__main__":
    x,y = prepare_data(DATA_PATH)
    cv_parameter, acc_list = get_parameter(x,y)
    fpr, tpr = train(cv_parameter)
    display_output(fpr, tpr)