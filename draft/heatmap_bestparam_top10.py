import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import itertools
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve,auc, accuracy_score
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import seaborn as sns;


DATA_PATH = "../data/generated/training_numerical_1mer_2mer_3mer_fracs.csv"
OUTPUT_PATH = ""

def prepare_data(data_path, prefix):
    # Prepare the data
    df = pd.read_csv(data_path)


    df = shuffle(df)

     # Import data
    x = []
    col = []
    # for each label in prefix, get the columns associated with it
    for label in prefix:

        filter_col = [column for column in df.columns.tolist() if column.startswith(label)]

        for col_name in filter_col:
            x.append(df[col_name].values.tolist())
            col.append(col_name)

    x = np.array(x).T
    print(x.shape)

    # y is the last column
    y = df['label'].values

    # set cooperative as 0, additive as 1
    y_bin = [0 if y[i] == "cooperative" else 1 for i in range(len(y))]
    y_bin = np.array(y_bin)

    return x, y_bin

def train_with_tune(x ,y):
    global prog
    print("in training.....")
    
    cv = KFold(n_splits=5,shuffle=False)

    # auc and accuracy score
    aucs_val = []
    acc_val  =[]
    
    #-----------------20 percent fpr--------------
    #base_fpr = np.linspace(0, 1, 101)[:21]
    
    #-----------------Full AUC------------------
    base_fpr = np.linspace(0, 1, 101)

    # perform 5 fold nested cross validation
    print("In cv....")
    for train, test in cv.split(x,y):

        print("Performing parameter tuning...")
        
        # parameters to tune - reduced
        n_estimators= [100, 500, 700, 1000, 2000]
        max_depth= [2, 5, 10, 15, 20]
        

        # 3 fold cross validation
        # scoring: accuracy
        # choose the parameter with the highest average validation performance
        cv_nested = KFold(n_splits=3,shuffle=False)
        for train_nested, test_nested in cv_nested.split(x[train], y[train]):
            res_nested = {}
            for est in n_estimators:
                for depth in max_depth:
                    #----------progress bar----------
                    print(str(prog/90) + "%...")
                    prog += 1
                    
                    clf = RandomForestClassifier(n_estimators=est, max_depth=depth)
                    model = clf.fit(x[train][train_nested], y[train][train_nested])
                    y_pred = model.predict(x[train][test_nested])
                    # store validation accuracy
                    if (est, depth) not in res_nested:
                        res_nested[(est, depth)] = []
                    res_nested[(est, depth)].append(accuracy_score(y[train][test_nested], y_pred))
            # get the average average validation accuracy for each tuple
            for key, value in res_nested.items():
                res_nested[key] = float(sum(res_nested[key]))/len(res_nested[key])
            
            # get the tuple with the largest average
            best_param = max(res_nested.items(), key=operator.itemgetter(1))[0]
            
        
        print("Done tuning parameters....")
        
        clf = RandomForestClassifier(n_estimators=best_param[0], max_depth=best_param[1])
        
        # train the model
        model = clf.fit(x[train], y[train])

        # get test auc
        y_score = model.predict_proba(x[test])[:,1]
        # get fpr and tpr
        fpr, tpr, _ = roc_curve(y[test], y_score)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        res_auc = auc(base_fpr, tpr)
        aucs_val.append(res_auc)
        
        # get test accuracy
        y_pred = model.predict(x[test])
        res_acc = accuracy_score(y[test], y_pred)
        acc_val.append(res_acc)
    
    #return aucs_val and acc_val
    #aucs_val = np.array(aucs_val)
    #acc_val = np.array(acc_val)
    
    return aucs_val, acc_val

def train_manual(x,y):
    res_nested = {}
    # n_estimators = [100, 500, 700, 1000, 2000]
    # max_depth = [2, 5, 10, 15, 20]
    n_estimators = [100]
    max_depth = [2]
    cv = KFold(n_splits=3,shuffle=False)
    # perform parameter tuning using cross validation
    for train, test in cv.split(x, y):
        for est in n_estimators:
            for depth in max_depth:            
                clf = RandomForestClassifier(n_estimators=est, max_depth=depth)
                model = clf.fit(x[train], y[train])
                y_pred = model.predict(x[train])
                # store validation accuracy
                if (est, depth) not in res_nested:
                    res_nested[(est, depth)] = []
                res_nested[(est, depth)].append(accuracy_score(y[train], y_pred))
    # get the average average validation accuracy for each tuple
    for key, value in res_nested.items():
        res_nested[key] = float(sum(res_nested[key]))/len(res_nested[key])

    # get the 2d array of avg validation
    scores = []
    
    # get the tuple with the largest average
    best_param = max(res_nested.items(), key=operator.itemgetter(1))[0]
        
    
    print("Done tuning parameters in train_manual....")
    
    clf = RandomForestClassifier(n_estimators=best_param[0], max_depth=best_param[1])

    return clf


def train_gridsearchcv(x, y):
    rf = RandomForestClassifier()
    # n_estimators = [100, 500, 700, 1000, 2000]
    # max_depth = [2, 5, 10, 15, 20]
    n_estimators = [100]
    max_depth = [2]
    parameters = {'n_estimators': n_estimators, 'max_depth': max_depth}
    # uses accuracy as the scoring system by default
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3)
    CV_rfc.fit(x, y)
    print("The best parameters are...\n")
    print(CV_rfc.best_params_)
    print("\n\n")
    rfc_best=RandomForestClassifier(n_estimators= best_params_['n_estimators'], max_depth=best_params_['max_depth'])

    # get the grid of the acc scores
    scores = grid_obj.cv_results_['mean_test_score'].reshape(len(n_estimators),len(min_samples_leaf))
    
    return rfc_best, scores

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
        y_score = model.predict_proba(x[test])[:,1]
        y_pred = model.predict(x[test])
        # get fpr and tpr
        fpr, tpr, _ = roc_curve(y[test], y_score)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        res_auc = auc(base_fpr, tpr)
        res_acc = accuracy_score(y_pred, y[test])
        aucs_val.append(res_auc)
        acc_val.append(res_acc)

    aucs_val = np.array(aucs_val)
    acc_val = np.array(acc_val)
    # return the average auc and accuracy from cross validation
    return aucs_val.mean(axis=0), acc_val.mean(axis=0)

if __name__=="__main__":

    prog = 1

    comb_list = [('distance'), ('1mer'), ('2mer'), ('3mer'), ('distance', '1mer'), ('distance', '2mer'),\
                 ('distance', '3mer'), ('1mer', '2mer'), ('1mer', '3mer'), ('2mer', '3mer'),\
                 ('distance', '1mer', '2mer'), ('distance', '1mer', '3mer'), ('distance', '2mer', '3mer'),\
                 ('1mer', '2mer', '3mer'), ('distance', '1mer', '2mer', '3mer')]

    # get data for all combinations
    prefix = ["distance", "1mer", "2mer", "3mer"]
    
    # get every feature combination
    # the number inside range will be the number of prefix we have
    # parameters to tune - reduced
    n_estimators= [100, 500, 700, 1000, 2000]
    max_depth= [2, 5, 10, 15, 20]
    #n_estimators=[100]
    #max_depth=[2]
    
    # for each feature combination, get the acc and auc

    f = open("heatmap_bestparam_top10_output.txt", 'w')
    output_auc = []
    output_acc = []
    # for this combination, get avg auc and avg acc for every parameter combination
    for n_est in n_estimators:
        for depth in max_depth:
            f.write("Current parameter combination is n_estimators: " + str(n_est) + " and max_depth: " + str(depth) + "\n")
            print("Current parameter combination is n_estimators: " + str(n_est) + " and max_depth: " + str(depth) + "\n")
            auc_list = []
            acc_list = []
            # get data for the current combination
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth)

            #----------------------Prepare the data----------------------------
            df = pd.read_csv(DATA_PATH)


            df = shuffle(df)

             # Import data
            x = []
            col = []
            # for each label in prefix, get the columns associated with it
            for label in prefix:

                filter_col = [column for column in df.columns.tolist() if column.startswith(label)]

                for col_name in filter_col:
                    x.append(df[col_name].values.tolist())
                    col.append(col_name)

            x = np.array(x).T

            # y is the last column
            y = df['label'].values

            # set cooperative as 0, additive as 1
            y_bin = [0 if y[i] == "cooperative" else 1 for i in range(len(y))]
            y = np.array(y_bin)

            rf.fit(x,y)
            feature_importances = pd.DataFrame(rf.feature_importances_,
                               index = col,
                                columns=['importance']).sort_values('importance',ascending=False)

            # list of x locations for plotting
            importances = list(rf.feature_importances_)
            feature_list = col
            # List of tuples with variable and importance
            feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
            # List of features sorted from most to least important
            sorted_features = [importance[0] for importance in feature_importances]
            # get the top 10 only
            sorted_features = sorted_features[:10]

            f.write('The top 10 features are:\n')
            # print the top 10 features
            for feature in sorted_features:
                f.write(feature)
                f.write('\n')

            # Import data
            x = []
            col = []
            # for each label in prefix, get the columns associated with it
            for label in sorted_features:

                filter_col = [column for column in df.columns.tolist() if column.startswith(label)]

                for col_name in filter_col:
                    x.append(df[col_name].values.tolist())
                    col.append(col_name)

            x = np.array(x).T

            # y is the last column
            y = df['label'].values

            # set cooperative as 0, additive as 1
            y_bin = [0 if y[i] == "cooperative" else 1 for i in range(len(y))]
            y = np.array(y_bin)

            # get the average of 100 runs
            for j in range(100):
                print(str(float(prog)/(25)) + "%")
                # get cross validation auc and acc output
                auc_res, acc_res = train(x, y, rf)
                auc_list.append(auc_res)
                acc_list.append(acc_res)

                # for progress bar
                prog += 1


            # print the average auc and acerage accuracy to file
            auc_avg = np.array(auc_list).mean(axis=0)
            acc_avg = np.array(acc_list).mean(axis=0)
            output_auc.append(auc_avg)
            output_acc.append(acc_avg)
            f.write("Average AUC: " + str(auc_avg) + "\n")
            f.write("Average accuracy: " + str(acc_avg))


    f.write("\n\n\n\n")

    # create heatmaps using seaborn
    sns.set()
    rename_ind = {k:v for k,v in enumerate(n_estimators)}
    rename_col = {k:v for k,v in enumerate(max_depth)}

    # heatmap for auc
    plt.figure()
    auc_arr = ((np.asarray(output_auc)).reshape(len(n_estimators),len(max_depth)))
    df_auc = pd.DataFrame(auc_arr)
    df_auc = df_auc.rename(index=rename_ind, columns=rename_col)
    sns_plot_auc = sns.heatmap(df_auc, annot=True, fmt=".3f",cmap="YlGnBu", xticklabels=max_depth,\
                                yticklabels=n_estimators)
    plt.title("AUC Score Heatmap Top 10 Features")
    plt.xlabel("max_depth")
    plt.ylabel("n_estimators")
    plt.savefig("AUC_heatmap_bestparam_top10.png")


    # heatmap for acc
    plt.figure()
    acc_arr = ((np.asarray(output_acc)).reshape(len(n_estimators),len(max_depth)))
    df_acc = pd.DataFrame(acc_arr)
    df_acc = df_acc.rename(index=rename_ind, columns=rename_col)
    sns_plot_acc = sns.heatmap(df_acc, annot=True, fmt=".3f",cmap="YlGnBu",xticklabels=max_depth,\
                                yticklabels=n_estimators)
    plt.title("Accuracy Score Heatmap Top 10 Features")
    plt.xlabel("max_depth")
    plt.ylabel("n_estimators")
    plt.savefig("Acc_heatmap_bestparam_top10.png")
