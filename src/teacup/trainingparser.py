import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm
from sklearn import linear_model
from sklearn import naive_bayes

class Simple1DClassifier:
    """
    input table needs to have column label on it
    """

    def __init__(self):
        self.label_gt = -1 # greater than
        self.label_lt = -1 # less than
        self.threshold = 0

    def train(self,xtrain,ytrain,threshold):
        index = [ytrain[i] for i in range(len(xtrain)) if xtrain[i] >= threshold]
        self.label_gt = max(index,key=index.count)
        if self.label_gt == 1:
            self.label_lt = 0
        else:
            self.label_lt = 1
        self.threshold = threshold

    def test(self,xtest):
        predictions = []
        for x in xtest:
            if x >= self.threshold:
                predictions.append(self.label_gt)
            else:
                predictions.append(self.label_lt)
        return predictions

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

    def extract_kmer_feature(self,seq):
        nucleotides = ['A','C','G','T']
        feature = []
        for k in range(1,4):
            perm = ["".join(p) for p in itertools.product(nucleotides, repeat=k)]
            for i in range(len(seq)):
                print(seq[i:i+k])
                for kmer in perm:
                    if seq[i:i+k] == kmer:
                        feature.append(1)
                    else:
                        feature.append(0)
                print(feature)
        return feature

    def extract_kmer_features_bpos(self,seq,bpos1,bpos2):
        span = 5
        bpos = [bpos1 - 1, bpos2 - 1] # adjustment -1 for programming
        nucleotides = ['A','C','G','T']
        features = []

        start = bpos1 - span
        end = bpos2 + span + 1

        feature = []
        for pos in bpos:
            spanseq = seq[pos-span:pos+span+1]
            feature += self.extract_kmer_feature(spanseq)
        return feature

    def get_features(self,type="distance-numeric"):
        """
        type:
            distance-numeric
            distance-categorical
            sites-centered
            linker
        """
        if type == "distance-numeric":
            return self.training["distance"].values.reshape((-1,1))
        elif type == "distance-categorical":
            one_hot = pd.get_dummies(self.training['distance'])
            return  one_hot.values.tolist()
        elif type == "sites-centered":
            features = []
            for idx,row in self.training.iterrows():
                rowfeature = self.extract_kmer_features_bpos(row["sequence"],row["bpos1"],row["bpos2"])
                features.append(rowfeature + [row["distance"]])
            return features
        #elif type == "linker":


    def calculate_fpr_tpr(self,ytrue,ypred):
        if len(ytrue) != len(ypred):
            print("the length of y-true and y-pred differ")
            return 0
        fp_count = 0
        tp_count = 0
        pos_count = 0
        neg_count = 0
        for i in range(len(ytrue)):
            if ytrue[i] == 1:
                pos_count += 1
                if ypred[i] == 1:
                    tp_count += 1
            elif ytrue[i] == 0:
                neg_count += 1
                if ypred[i] == 1:
                    fp_count += 1
        fpr = float(fp_count)/neg_count
        tpr = float(tp_count)/pos_count
        return fpr,tpr

    def roc_simple_clf(self):
        # still numeric for now

        xtrain = self.training["distance"].values
        ytrain = self.get_numeric_label().values
        distances = self.training['distance'].unique()

        fpr_list = np.array([0])
        tpr_list = np.array([0])
        for dist in sorted(distances):
            scf = Simple1DClassifier()
            scf.train(xtrain,ytrain,dist)
            ypred = scf.test(xtrain)
            #print("Accuracy %f" % metrics.accuracy_score(ytrain, ypred))
            fpr,tpr = self.calculate_fpr_tpr(ytrain, ypred)
            fpr_list = np.append(fpr_list,fpr)
            tpr_list = np.append(tpr_list,tpr)

        fpr_list = np.append(fpr_list,1)
        tpr_list = np.append(tpr_list,1)

        return fpr_list,tpr_list

    def test_model(self,featuretype):
        """
        model:
            numeric: distance as it is
            categorical: use one hot encoder
        """

        x_train = self.get_features(featuretype)
        '''y_train = self.get_numeric_label().values

        clfs = {
                "decision tree":tree.DecisionTreeClassifier(),
                "random forest":ensemble.RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0),
                "SVM":svm.SVC(kernel="rbf",gamma=1.0/5,probability=True),
                "log regression":linear_model.LogisticRegression(),
                "simple":Simple1DClassifier(),
                "gradient boosting":ensemble.GradientBoostingClassifier(),
                "naive bayes":naive_bayes.GaussianNB()
               }

        auc_total = 0
        for key in clfs:
            if key == "simple":
                fpr,tpr = self.roc_simple_clf()
                auc = metrics.auc(fpr,tpr)
                plt.plot(fpr,tpr,label="distance threshold, training auc=%f" % auc,linestyle=":", color="orange")
            else:
                clf = clfs[key].fit(x_train, y_train)
                y_pred = clf.predict_proba(x_train)[:, 1]

                # https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
                # print("Accuracy %s: %f" % (key,metrics.accuracy_score(y_train, y_pred)))

                # ROC curve
                fpr, tpr, _ = metrics.roc_curve(y_train, y_pred)
                auc = metrics.roc_auc_score(y_train, y_pred)
                plt.plot(fpr,tpr,label="%s, training auc=%f" % (key,auc))
            auc_total += auc
        print(len(clfs))
        print("Average AUC %f"%(auc_total/len(clfs)))

        plt.plot([0, 1], [0, 1], linestyle="--", color="red")
        plt.legend(loc=4)
        plt.show()
        '''
