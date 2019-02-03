import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools
import scipy

from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import model_selection

from teacup.training import simpleclassifier

def calculate_fpr_tpr(ytrue,ypred):
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
                for kmer in perm:
                    if seq[i:i+k] == kmer:
                        feature.append(1)
                    else:
                        feature.append(0)
        return np.asarray(feature)

    def extract_kmer_features_bpos(self,seq,bpos1,bpos2):
        span = 4
        bpos = [bpos1 - 1, bpos2 - 1] # adjustment -1 for programming
        nucleotides = ['A','C','G','T']

        start = bpos1 - span
        end = bpos2 + span + 1

        feature = []
        for pos in bpos:
            if pos - span < 0:
                start = 0
            else:
                start = pos - span
            spanseq = seq[pos-span:pos+span+1]
            feature.extend(self.extract_kmer_feature(spanseq))
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
                features.append(rowfeature) # + [row["distance"]]
            return features
        elif type == "sites-linker":
            features = []
            for idx,row in self.training.iterrows():
                # since the binding pos is one index, we need to -1
                midpos = row["bpos2"] - row["bpos1"] - 1
                seq = row["sequence"][midpos-8:midpos+8]
                features.append(self.extract_kmer_feature(seq))
            return features

    def roc_simple_clf(self,n_splits=1):
        # still numeric for now
        x_train = self.training["distance"].values
        y_train = self.get_numeric_label().values
        distances = self.training['distance'].unique()

        if n_splits > 1:
            cv = model_selection.KFold(n_splits=n_splits,shuffle=True)
            split = cv.split(x_train,y_train)
        else:
            split = [(range(len(x_train)),range(len(y_train)))]

        fpr_all = []
        tpr_all = []
        auc_all = []

        for train, test in split:
            fpr_list = [0]
            tpr_list = [0]
            for dist in sorted(distances):
                scf = simpleclassifier.Simple1DClassifier()
                scf.fit_on_thres(x_train[train],y_train[train],dist)
                y_pred = scf.test(x_train[test])
                #print("Accuracy %f" % metrics.accuracy_score(ytrain, ypred))
                fpr,tpr = calculate_fpr_tpr(y_train[test], y_pred)
                fpr_list.append(fpr)
                tpr_list.append(tpr)


            fpr_list.append(1)
            tpr_list.append(1)

            auc = metrics.auc(fpr_list,tpr_list)
            auc_all.append(auc)
            fpr_all.append(fpr_list)
            tpr_all.append(tpr_list)
        return fpr_all,tpr_all,auc_all

    def test_model(self,feature_type, testing_type="cv"):
        """
        model:
            numeric: distance as it is
            categorical: use one hot encoder
        """

        x_train = self.get_features(feature_type)
        y_train = self.get_numeric_label().values
        #print(len(x_train),len(y_train))

        clfs = {
                "decision tree":tree.DecisionTreeClassifier(),
                "random forest":ensemble.RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0),
                "SVM":svm.SVC(kernel="rbf",gamma=1.0/5,probability=True),
                #"log regression":linear_model.LogisticRegression(),
                "simple":simpleclassifier.Simple1DClassifier(),
                #"gradient boosting":ensemble.GradientBoostingClassifier(),
                #"naive bayes":naive_bayes.GaussianNB()
               }

        if testing_type == "cv":
            fpr_list, tpr_list, auc_list = self.test_with_cv(clfs, x_train, y_train)
        else:
            fpr_list, tpr_list, auc_list = self.test_on_train(clfs,x_train,y_train)

        self.display_output(fpr_list, tpr_list, auc_list, list(clfs.keys()))

    def test_with_cv(self,clfs,x_train,y_train,fold=10):
        fpr_list = []
        tpr_list = []
        auc_list = []
         # Compute ROC curve and ROC area with averaging for each classifier
        for key in clfs:
            base_fpr = np.linspace(0, 1, 101)
            tprs = []
            aucs_val = []
            if key == "simple":
                fprs_simple,tprs_simple,aucs_val = self.roc_simple_clf(n_splits=fold)
                for i in range(0,len(fprs_simple)):
                    tpr = scipy.interp(base_fpr, fprs_simple[i], tprs_simple[i])
                    tprs.append(tpr)
            else:
                cv = model_selection.KFold(n_splits=fold,shuffle=True)
                # initialize a list to store the average fpr, tpr, and auc
                print("Cross validation on %s" % key)
                i = 1
                for train_idx,test_idx in cv.split(x_train,y_train):
                    # need to convert this with index, somehow cannot do
                    # x_train[train_idx] for multi features
                    data_train = [x_train[i] for i in train_idx]
                    data_test = [x_train[i] for i in test_idx]
                    lbl_train = [y_train[i] for i in train_idx]
                    lbl_test = [y_train[i] for i in test_idx]

                    model = clfs[key].fit(data_train, lbl_train)
                    y_score = model.predict_proba(data_test)
                    fpr, tpr, _ = metrics.roc_curve(lbl_test, y_score[:, 1])
                    auc = metrics.roc_auc_score(lbl_test, y_score[:,1])
                    print("fold " + str(i) + " AUC: " + str(auc))
                    # vmartin: please have the package name instead of using
                    # the function directly so we know where does the function
                    # come from :)
                    tpr = scipy.interp(base_fpr, fpr, tpr)
                    tprs.append(tpr)
                    aucs_val.append(auc)
                    i += 1

            # calculate mean true positive rate
            tprs = np.array(tprs)
            mean_tprs = tprs.mean(axis=0)

            # calculate mean auc
            aucs_val = np.array(aucs_val)
            mean_aucs = aucs_val.mean(axis=0)

            fpr_list.append(base_fpr)
            tpr_list.append(mean_tprs)
            auc_list.append(mean_aucs)

        return fpr_list, tpr_list, auc_list

    def test_on_train(self,clfs,x_train,y_train):
        auc_total = 0
        fpr_list = []
        tpr_list = []
        auc_list = []
        for key in clfs:
            if key == "simple":
                fpr,tpr,auc = self.roc_simple_clf()
                fpr = fpr[0]
                tpr = tpr[0]
                auc = auc[0]
                #plt.plot(fpr,tpr,label="distance threshold, training auc=%f" % auc,linestyle=":", color="orange")
            else:
                print("key is:", key)
                clf = clfs[key].fit(x_train, y_train)
                y_pred = clf.predict_proba(x_train)[:, 1]

                # https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
                # print("Accuracy %s: %f" % (key,metrics.accuracy_score(y_train, y_pred)))

                # ROC curve
                fpr, tpr, _ = metrics.roc_curve(y_train, y_pred)
                auc = metrics.roc_auc_score(y_train, y_pred)
                #plt.plot(fpr,tpr,label="%s, training auc=%f" % (key,auc))

            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(auc)
            auc_total += auc
        print("Average AUC %f"%(auc_total/len(clfs)))

        return fpr_list, tpr_list, auc_list


    def display_output(self, fpr_list, tpr_list, auc_list, classifier_names):
        """
            This plots the average ROC curve of all the classifiers in a single plot
        """
        plt.plot([0, 1], [0, 1], linestyle="--", color="red", alpha=0.1)
        for i in range(len(fpr_list)):
            plt.plot(fpr_list[i], tpr_list[i], lw=2, alpha=0.4, label='%s, AUC %f' % (classifier_names[i], auc_list[i]))

        # Show the ROC curves for all classifiers on the same plot
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Average ROC Curves for All Classifiers')
        plt.legend(loc="lower right")
        plt.savefig('Avg_ROC.png')
        plt.show()
