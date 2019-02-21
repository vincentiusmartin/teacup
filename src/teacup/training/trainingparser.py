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
from sklearn import preprocessing
from matplotlib.backends.backend_pdf import PdfPages

from teacup.training import simpleclassifier
from teacup import utils

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

    def __init__(self, trainingdata,motiflen):
        if type(trainingdata) == str: # input path
            self.training = pd.read_csv(trainingdata)
        elif type(trainingdata) == pd.core.frame.DataFrame: # from an existing data frame
            self.training = trainingdata[['sequence', 'bpos1', 'bpos2', 'distance', 'label']]
        self.motiflen = motiflen

    # ===== Getter part ====

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

                linker = row["sequence"][row["bpos1"] + self.motiflen // 2 : row["bpos2"] - self.motiflen // 2]
                ratio = self.extract_kmer_ratio(linker)

                all = np.concatenate((rowfeature,ratio,[self.training['distance'][idx]]))
                #features.append(preprocessing.normalize([all])[0])
                features.append(all)
            return features
        elif type == "sites-linker":
            features = []
            for idx,row in self.training.iterrows():
                numericdist = self.training["distance"].values.reshape((-1,1))
                # since the binding pos is one index, we need to -1
                midpos = (row["bpos2"] + row["bpos1"] - 1)//2
                seq = row["sequence"][midpos-13:midpos+13]
                features.append(self.extract_kmer_binary(seq) + [self.training['distance'][idx]])
            return features

    # ======== Modifier to training data ========

    def get_numeric_label(self):
        train = self.training['label'].map({'cooperative': 1, 'additive': 0})
        return train

    # ======= For simple model that is based on distance only =======
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

    # ====== Processing part ======

    def compare_distance_features(self, iter=10, fpr_lim=100):
        clfs = {
            #"decision tree":tree.DecisionTreeClassifier(),
            "random forest":ensemble.RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0),
            #"SVM":svm.SVC(kernel="rbf",gamma=1.0/5,probability=True),
            #"log regression":linear_model.LogisticRegression(),
            "simple":simpleclassifier.Simple1DClassifier(),
            #"gradient boosting":ensemble.GradientBoostingClassifier(),
            #"naive bayes":naive_bayes.GaussianNB()
           }

        dists = ["distance-numeric","distance-categorical"]

        auc_dict = {}
        fpr_dict = {}
        tpr_dict = {}
        for dist_type in dists:
            auc_dict[dist_type] = []
            for i in range(iter):
                print("Processing using %s, iteration %d" % (dist_type,i+1))
                x_train = self.get_features(dist_type)
                y_train = self.get_numeric_label().values
                fpr_list, tpr_list, auc_list = self.test_with_cv(clfs, x_train, y_train,fpr_lim=fpr_lim)
                auc_dict[dist_type].append(auc_list['random forest'])


        print("Making scatter boxplot for each feature...")
        utils.scatter_boxplot_dict(auc_dict,ylabel="AUC")

        print("Two sided wilcox test, pval: %.4f" % utils.wilcox_test(auc_dict["distance-numeric"],auc_dict["distance-categorical"]))
        print("Numeric > Categorical test, pval: %.4f" % utils.wilcox_test(auc_dict["distance-numeric"],auc_dict["distance-categorical"],alternative="greater"))
        print("Numeric < Categorical test, pval: %.4f" % utils.wilcox_test(auc_dict["distance-numeric"],auc_dict["distance-categorical"],alternative="less"))

    def test_with_cv(self,clfs,x_train,y_train,fold=10,fpr_lim=100):
        fpr_dict = {}
        tpr_dict = {}
        auc_dict = {}
         # Compute ROC curve and ROC area with averaging for each classifier
        for key in clfs:
            # we limit this to get roc curve / auc until the fpr that we want
            base_fpr = np.linspace(0, 1, 101)[:fpr_lim+1]
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
                    #auc = metrics.roc_auc_score(lbl_test, y_score[:,1])
                    tpr = scipy.interp(base_fpr, fpr, tpr)
                    res_auc = metrics.auc(base_fpr, tpr)
                    tprs.append(tpr)
                    aucs_val.append(res_auc)
                    i += 1

            # calculate mean true positive rate
            tprs = np.array(tprs)
            mean_tprs = tprs.mean(axis=0)

            # calculate mean auc
            aucs_val = np.array(aucs_val)
            mean_aucs = aucs_val.mean(axis=0)

            fpr_dict[key] = base_fpr
            tpr_dict[key] = mean_tprs
            auc_dict[key] = mean_aucs

        return fpr_dict, tpr_dict, auc_dict

    # ========= Plotting =======

    def display_output(self, fpr_dict, tpr_dict, auc_dict, path):
        """
            This plots the average ROC curve of all the classifiers in a single plot
        """
        plt.clf() # first, clear the canvas

        plt.plot([0, 1], [0, 1], linestyle="--", color="red", alpha=0.1)
        for key in fpr_dict:
            plt.plot(fpr_dict[key], tpr_dict[key], lw=2, alpha=0.4, label='%s, AUC %f' % (key, auc_dict[key]))

        # Show the ROC curves for all classifiers on the same plot
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Average ROC Curves for All Classifiers')
        plt.legend(loc="lower right")
        plt.savefig(path)
