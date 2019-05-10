import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "../data/generated/training_numerical_1mer_2mer_3mer_fracs.csv"
df = pd.read_csv(DATA_PATH)
col = []
# shuffle the data
for i in range(10):
    df = shuffle(df)
prefix = ["distance", "1mer", "2mer", "3mer"]
# Import data
x = []
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

rf = RandomForestClassifier(n_estimators=1000, max_depth=10)
rf.fit(x, y_bin)

# count_dict = defaultdict(int)
# index = 2
# for val in rf.feature_importances_:
# 	if df.columns[index].startswith("2mer"):
# 		count_dict["2mer"] += val
# 	elif df.columns[index].startswith("1mer"):
# 		count_dict["1mer"] += val
# 	else:
# 		count_dict[df.columns[index]] += val
# 	index += 1
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = col,
                                    columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)

# list of x locations for plotting
importances = list(rf.feature_importances_)
feature_list = col
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
x_values = list(range(len(importances)))
# Make a bar chart
plt.figure(0)
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
plt.tight_layout()
# Tick labels for x axis
#plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
plt.savefig("Variable Importances.png")

plt.figure(1)
# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
# Make a line graph
plt.plot(x_values, cumulative_importances, 'g-')
#plt.tight_layout()
# Draw line at 95% of importance retained
#plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
# Format x ticks and labels
#plt.xticks(x_values, sorted_features, rotation = 'vertical')
# Axis labels and title
plt.xlabel('Number of Features'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');
plt.savefig("Cumulative Importances.png")

# # Find number of features for cumulative importance of 95%
# # Add 1 because Python is zero-indexed
# print('Number of features for 95% importance:', np.where(cumulative_importances > 0.95)[0][0] + 1)

# # Extract the names of the most important features
# important_feature_names = [feature[0] for feature in feature_importances[0:5]]
# # Find the columns of the most important features
# important_indices = [feature_list.index(feature) for feature in important_feature_names]
# # Create training and testing sets with only the important features
# train_features, test_features, train_labels, test_labels = train_test_split(x, y_bin, test_size = 0.25, random_state=42)
# important_train_features = train_features[:, important_indices]
# important_test_features = test_features[:, important_indices]
# # Sanity check on operations
# print('Important train features shape:', important_train_features.shape)
# print('Important test features shape:', important_test_features.shape)

# rf.fit(important_train_features, train_labels);
# # Make predictions on test data
# predictions = rf.predict(important_test_features)
# # Performance metrics
# errors = abs(predictions - test_labels)
# print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')
# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / len(test_labels))
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')
# print(predictions)
# print('\n\n')
# print(test_labels)
# print("Accuracy using accuracy score: ", accuracy_score(test_labels, predictions))



# #-------------------------------Get cumulative accuracy---------------------

# # List of features sorted from most to least important
# sorted_importances = [importance[1] for importance in feature_importances]
# sorted_features = [importance[0] for importance in feature_importances]
# cumulative_accuracies = []
# max_acc = 0
# num_features = 0
# num_run = 100
# tot_num_features = 0
# tot_max_acc = 0
# cumulative_accuracies = [0 for _ in range(len(sorted_importances))]
# for run in range(num_run):
# 	for num in range(len(sorted_importances)):
# 		x = []
# 		for i in range(num+1):
# 			x.append(df[sorted_features[i]].values.tolist())
# 		x = np.array(x).T
# 		x_train, x_test, y_train, y_test = train_test_split(x, y_bin, test_size = 0.33, random_state=0)
# 		rf = RandomForestClassifier(n_estimators=1000, max_depth=10)
# 		rf.fit(x_train, y_train)
# 		y_pred = rf.predict(x_test)
# 		acc = accuracy_score(y_test, y_pred)
# 		cumulative_accuracies[num] += acc

# 		if acc > max_acc:
# 			max_acc = acc
# 			tot_max_acc += acc
# 			num_features = num+1
# 			tot_num_features += num_features

# cumulative_accuracies = [float(cumulative_accuracies[i])/num_run for i in range(len(cumulative_accuracies))]


# print("The average maximum accuracy is ", float(tot_max_acc)/num_run)
# print("The average top ", float(tot_num_features)/num_run, " features achieved this accuracy")
# # Make a line graph
# plt.figure(1)
# plt.plot(x_values, cumulative_accuracies, 'g-')
# plt.tight_layout()
# # Format x ticks and labels
# #plt.xticks(x_values, sorted_features, rotation = 'vertical')
# # Axis labels and title
# plt.xlabel('Variable'); plt.ylabel('Accuracy'); plt.title('Average Cumulative Accuracy for 100 Runs');
# plt.gcf.subplots_adjust(top=0.85)
# plt.savefig("Average Cumulative Accuracy.png")