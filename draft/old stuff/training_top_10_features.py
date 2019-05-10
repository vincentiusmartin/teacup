
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve,roc_auc_score, accuracy_score
from sklearn.model_selection import KFold

DATA_PATH = "../data/generated/training_numerical_1mer_2mer_3mer_fracs.csv"

def prepare_data(data_path, features):
    # Prepare the data
    df = pd.read_csv(data_path)

    # shuffle the data
    for i in range(10):
        df = shuffle(df)

    # Import data
    x = []
    for col_name in features:
        x.append(df[col_name].values.tolist())
    x = np.array(x).T
    print(x.shape)

    # y is the last column
    y = df['label'].values

    # set cooperative as 0, additive as 1
    y_bin = [0 if y[i] == "cooperative" else 1 for i in range(len(y))]
    y_bin = np.array(y_bin)

    # return our data
    return x, y_bin


def train(x ,y):
    #clf = LogisticRegression(C=k)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
    cv = KFold(n_splits=10,shuffle=True)

     # Compute ROC curve and ROC area with averaging
    tprs = []
    aucs_val = []
    # up to 0.2 fpr
    base_fpr = np.linspace(0, 1, 101)
    i = 1
    for train, test in cv.split(x,y):
        model = clf.fit(x[train], y[train])
        y_score = model.predict_proba(x[test])
        fpr, tpr, _ = roc_curve(y[test], y_score[:, 1])
        auc = roc_auc_score(y[test], y_score[:,1])
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        print("fold " + str(i) + " AUC: " + str(auc))
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        aucs_val.append(auc)
        i += 1

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b', label = "Average AUC:" + str(sum(aucs_val)/len(aucs_val)))
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim((0, 1))
    plt.ylim(0, 1)
    plt.title("Random Forest ROC Curve with 10 Fold Cross Validation for Top 10 Features")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend(loc="lower right")
    plt.savefig('Avg_ROC_Top_10_Features.png')
    

if __name__=="__main__":
    # get data for all combinations
    features = ["distance", "2mer_AA_count", "3mer_CAT_count", "3mer_CAG_count", "3mer_CCA_count", "3mer_CCG_count", "2mer_CA_count", "3mer_AAA_count", "3mer_GCG_count", "1mer_A_count"]
    X,Y = prepare_data(DATA_PATH, features)
    train(X,Y)




