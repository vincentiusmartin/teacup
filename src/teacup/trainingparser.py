import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm
from sklearn import linear_model

class TrainingParser:

    def __init__(self, trainingpath):
        self.training = pd.read_csv(trainingpath)

    # ======== Modifier to training data ========

    def get_numeric_label(self):
        train = self.training['label'].map({'cooperative': 1, 'additive': 0})
        return train

    # ======== Plot related ========

    def scatter_boxplot_col(self, colname, filepath="scatterbox.png"):
        groupdict = self.training.groupby(['label'])[colname].apply(list).to_dict()

        keys = groupdict.keys()
        listrep = [groupdict[key] for key in keys]

        pos = np.linspace(1,1+len(listrep)*0.5-0.5,len(listrep))
        bp = plt.boxplot(listrep,positions=pos,widths=0.4)
        plt.xticks(pos, keys)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['caps'], color='black')

        for i in range(0,len(listrep)):
            y = listrep[i]
            x = np.random.normal(1+i*0.5, 0.02, size=len(y))
            plt.plot(x, y, 'r.', alpha=0.4,c='red')

        #print("Save distribution of row %s to %s" % (rownum,plotfilename))
        plt.savefig(filepath,positions=[0, 1])
        plt.clf() # clear canvas

    # ======== Training and testing modelss ========

    def train_from_distance(self,categorical=False):
        """
        model:
            numeric: distance as it is
            categorical: use one hot encoder
        """

        if categorical:
            one_hot = pd.get_dummies(self.training['distance'])
            x_train = one_hot.values.tolist()
        else:
            x_train = self.training["distance"].values.reshape((-1,1))

        y_train = self.get_numeric_label().values

        clfs = {
                "decision tree":tree.DecisionTreeClassifier(),
                "random forest":ensemble.RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0),
                "SVM":svm.SVC(kernel="rbf",gamma=1.0/5),
                "log regression":linear_model.LogisticRegression()
               }

        for key in clfs:
            clf = clfs[key].fit(x_train, y_train)
            y_pred = clf.predict(x_train)

            # https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
            print("Accuracy %s: %f" % (key,metrics.accuracy_score(y_train, y_pred)))

            # ROC curve
            fpr, tpr, _ = metrics.roc_curve(y_train, y_pred)
            auc = metrics.roc_auc_score(y_train, y_pred)
            plt.plot(fpr,tpr,label="%s, training auc=%f" % (key,auc))

        plt.legend(loc=4)
        plt.show()

    #def training_with_seqfeatures():
