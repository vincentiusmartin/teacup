import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve,auc, accuracy_score
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import train_test_split

DATA_PATH = "../data/generated/training_numerical_1mer_2mer_3mer_fracs.csv"
i = 1
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
        #print(filter_col)
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

def train_with_tune(x ,y):
    print("in training.....")
    
    cv = KFold(n_splits=3,shuffle=False)

    # Compute ROC curve and ROC area with averaging
    aucs_val = []
    #-----------------20 percent fpr--------------
    #base_fpr = np.linspace(0, 1, 101)[:21]
    #-----------------Full AUC------------------
    base_fpr = np.linspace(0, 1, 101)

    # perform 10 fold nested cross validation
    print("In cv....")
    for train, test in cv.split(x,y):
        print(str(i/300) + "%...")
        i += 1
        print("Performing parameter tuning...")
        rfc = RandomForestClassifier(n_estimators=100, max_depth=10) 

        # param_grid = { 
        #     'n_estimators': [10, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000],
        #     'max_features': ['sqrt', 'log2'],
        #     'max_depth' : [x for x in range(1,21)],
        #     'min_samples_leaf' : [1, 2, 4, 8, 16, 32, 64],
        #     'criterion' :['gini', 'entropy'],

        #     #'bootstrap': [True,False]
        # }
        param_grid = { 
            'n_estimators': [10, 100, 200, 300, 500, 1000, 2000],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': sp_randint(2,11),
            'min_samples_leaf': sp_randint(1,11),
            'max_depth': [x for x in range(1,21)]
            #'min_samples_leaf' : [1, 2, 4, 8, 16, 32, 64],
            #'criterion' :['gini', 'entropy'],

            #'bootstrap': [True,False]
        }
        # 10 fold cross validation
        CV_rfc = RandomizedSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        CV_rfc.fit(x[train], y[train])
        #print(CV_rfc.best_params_)
        param = CV_rfc.best_params_
        #clf = LogisticRegression(C=k)
        clf = RandomForestClassifier(n_estimators=param['n_estimators'], min_samples_split=param['min_samples_split'], min_samples_leaf=param['min_samples_leaf'], max_depth=param['max_depth'], max_features = param['max_features'],  bootstrap = True)
        #clf = RandomForestClassifier(n_estimators=param['n_estimators'], max_depth=param['max_depth'], max_features = param['max_features'], criterion = param['criterion'], min_samples_leaf=param['min_samples_leaf'], bootstrap = True)

        print("Done tuning parameters....")
        
        # train the model
        model = clf.fit(x[train], y[train])

        # get test accuracy 
        y_score = model.predict_proba(x[test])[:,1]
        # get fpr and tpr
        fpr, tpr, _ = roc_curve(y[test], y_score)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        res_auc = auc(base_fpr, tpr)
        aucs_val.append(res_auc)

    #return aucs_val
    aucs_val = np.array(aucs_val)

    return aucs_val.mean(axis=0)


def train(x ,y, clf):
    
    cv = KFold(n_splits=10,shuffle=False)

    # Compute ROC curve and ROC area with averaging
    aucs_val = []
    acc_val = []
    #--------------------get full roc----------------
    base_fpr = np.linspace(0, 1, 101)
    # --------------------get up to 0.2 fpr only--------------
    #base_fpr = np.linspace(0, 1, 101)[:21]

    # perform 10 fold cross validation
    for train, test in cv.split(x,y):
        # train the model
        model = clf.fit(x[train], y[train])
        #y_score = model.predict_proba(x[test])[:,1]
        y_pred = model.predict(x[test])
        # get fpr and tpr
        # fpr, tpr, _ = roc_curve(y[test], y_score)
        # tpr = interp(base_fpr, fpr, tpr)
        # tpr[0] = 0.0
        #res_auc = auc(base_fpr, tpr)
        res_acc = accuracy_score(y_pred, y[test])
        #aucs_val.append(res_auc)
        acc_val.append(res_acc)

    #aucs_val = np.array(aucs_val)
    acc_val = np.array(acc_val)
    # return the average auc and accuracy from cross validation
    #return aucs_val.mean(axis=0)
    return acc_val.mean(axis=0)

def display_output(output, names):
    # create boxplots for the lists in output
    plt.boxplot(output, labels=names)
    #plt.xticks(rotation=25)
    #plt.tight_layout()
    plt.xlabel("Combination index")
    plt.ylabel("Accuracy")
    # Show the ROC curves for all 10 test folds on the same plot
    plt.title('Accuracy Comparison Using Best Parameters')
    plt.gcf().subplots_adjust(top=0.85)
    plt.savefig('Boxplot_feature_comb_best_param_accuracy.png')

if __name__=="__main__":
    # get data for all combinations
    prog = 1
    output = []
    comb_name = []
    prefix = ["distance", "1mer", "2mer", "3mer"]
    #--------------------------uncomment this if we're using best parameters----------------------
    # for auc's best param
    # comb_dict_auc = {('distance',): [700,2], ('1mer',):[500,20], ('2mer',):[2000,20], ('3mer',):[700,20],\
    #              ('distance', '1mer'):[700,5], ('distance', '2mer'):[500,10],\
    #              ('distance', '3mer'):[500,15], ('1mer', '2mer'):[500,20], ('1mer', '3mer'):[1000,15],\
    #              ('2mer', '3mer'):[1000,15],('distance', '1mer', '2mer'):[500,10], \
    #              ('distance', '1mer', '3mer'):[1000,10], ('distance', '2mer', '3mer'):[2000,10],\
    #              ('1mer', '2mer', '3mer'):[1000,15], ('distance', '1mer', '2mer', '3mer'): [2000,10]}

    # for acc best param
    comb_dict_acc = {('distance',): [100,2], ('1mer',):[1000,15], ('2mer',):[2000,5], ('3mer',):[2000,20],\
                 ('distance', '1mer'):[1000,5], ('distance', '2mer'):[500,10],\
                 ('distance', '3mer'):[1000,15], ('1mer', '2mer'):[500,20], ('1mer', '3mer'):[2000,20],\
                 ('2mer', '3mer'):[2000,15],('distance', '1mer', '2mer'):[2000,15], \
                 ('distance', '1mer', '3mer'):[2000,15], ('distance', '2mer', '3mer'):[500,20],\
                 ('1mer', '2mer', '3mer'):[500,15], ('distance', '1mer', '2mer', '3mer'): [2000,20]}

    #prefix = ["1mer"]
    f = open("compare_cv_output_best_param_accuracy.txt", 'w')
    # get every feature combination
    # the number inside range will be the number of prefix we have
    for i in range(len(prefix)):
        print("Number of prefixes we're working on right now: ", (i+1))
        for comb in itertools.combinations(prefix, i+1):
            f.write("Current combination is " + str(comb))
            x, y = prepare_data(DATA_PATH, comb)
            comb_name.append(comb)
            # run 100 times
            auc_list = []
            for j in range(100):
                print(prog/(15))
                prog += 1
                #------------------if using best parameters------------------
                # n_est = comb_dict_acc[comb][0]
                # max_depth = comb_dict_acc[comb][1]

                # #-----------------if n_est and max_depth are set---------------
                n_est = 1000
                max_depth = 10

                rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth)

                auc_res = train(x, y, rf)
                auc_list.append(auc_res)
            #output.append(auc_res)
            output.append(auc_list)
            f.write("The average AUC for " + str(comb)+ " is: " + str(sum(auc_list)/float(len(auc_list))))

    comb_name = [i+1 for i in range(15)]
    display_output(output, comb_name)
    f.close()
