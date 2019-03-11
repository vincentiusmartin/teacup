import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve,auc, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

DATA_PATH = "../data/generated/training_distance_1mer_2mer.csv"

def prepare_data(data_path, prefix):
    # Prepare the data
    df = pd.read_csv(data_path)

    # shuffle the data
    for i in range(10):
        df = shuffle(df)

    # Import data
    x = []
    for label in prefix:
        filter_col = [col for col in df.columns.tolist() if col.startswith(label)]
        for col_name in filter_col:
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
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    cv = KFold(n_splits=10,shuffle=True)

    # Compute ROC curve and ROC area with averaging
    aucs_val = []
    base_fpr = np.linspace(0, 1, 101)[:21]
    # perform 10 fold cross validation
    for train, test in cv.split(x,y):
        # train the model
        model = clf.fit(x[train], y[train])
        y_score = model.predict_proba(x[test])[:,1]
        # get fpr and tpr
        fpr, tpr, _ = roc_curve(y[test], y_score)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        res_auc = auc(base_fpr, tpr)
        aucs_val.append(res_auc)

    aucs_val = np.array(aucs_val)
    return aucs_val.mean(axis=0)

def display_output(output, names):
    # create boxplots for the lists in output
    plt.boxplot(output, labels=names)

    # Show the ROC curves for all 10 test folds on the same plot
    plt.title('Boxplot for Feature Combinations')
    plt.savefig('Boxplot_feature_comb.png')

if __name__=="__main__":
    # get data for all 7 combinations
    output = []
    comb_name = []
    prefix = ["one_hot", "1mer", "2mer"]

    # get every feature combination
    for i in range(3):
        print(i)
        for comb in itertools.combinations(prefix, i+1):
            print(comb)
            x, y = prepare_data(DATA_PATH, comb)
            comb_name.append(comb)
            # run 10 times
            auc_list = []
            for j in range(10):
                auc_res = train(x, y)
                auc_list.append(auc_res)
            output.append(auc_list)

    display_output(output, comb_name)
