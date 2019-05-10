import sys
sys.path.insert(0,"..")
from src.teacup import utils
import pandas as pd
import itertools

DATA_NUMERIC_FRAC = "numeric_fracs_auc.csv"
#DATA_DISTANCE_COUNT = "../data/generated/training_distance_1mer_2mer_counts.csv"
DATA_DISTANCE_COUNT = "categorical_counts_auc.csv"

# print("------------Hypothesis testing using distance categorical and fraction counts-------------")
# # Prepare the data
# df = pd.read_csv(DATA_DISTANCE_FRAC)
# # get number of columns
# num_col = len(df.columns)
# # compare every pair
# for comb in itertools.combinations(range(num_col), 2):
# 	arr1 = df.iloc[:, comb[0]].values.tolist()
# 	print(arr1)
# 	arr2 = df.iloc[:, comb[1]].values.tolist()
# 	print(arr2)
# 	arr1_name = df.columns[comb[0]]
# 	arr2_name = df.columns[comb[1]]
# 	print(arr1_name, "<", arr2_name, ":", utils.wilcox_test(arr1, arr2, "less"))
# 	print(arr1_name, ">", arr2_name, ":", utils.wilcox_test(arr1, arr2, "greater"))
# for every pair 
print("------------Hypothesis testing using distance categorical and counts-------------")
df = pd.read_csv(DATA_DISTANCE_COUNT)
df = df.drop(df.columns[0], axis=1)
# get number of columns
num_col = len(df.columns)
# compare every pair
for comb in itertools.combinations(range(num_col), 2):
	arr1 = df.iloc[:, comb[0]].values.tolist()
	arr2 = df.iloc[:, comb[1]].values.tolist()
	arr1_name = df.columns[comb[0]]
	arr2_name = df.columns[comb[1]]
	print(arr1_name, "<", arr2_name, ":", utils.wilcox_test(arr1, arr2, "less"))
	print(arr1_name, ">", arr2_name, ":", utils.wilcox_test(arr1, arr2, "greater"))

print("------------Hypothesis testing using distance numerical and fraction counts-------------")
df = pd.read_csv(DATA_NUMERIC_FRAC)
df = df.drop(df.columns[0], axis=1)
# get number of columns
num_col = len(df.columns)
# compare every pair
for comb in itertools.combinations(range(num_col), 2):
	arr1 = df.iloc[:, comb[0]].values.tolist()
	arr2 = df.iloc[:, comb[1]].values.tolist()
	arr1_name = df.columns[comb[0]]
	arr2_name = df.columns[comb[1]]
	print(arr1_name, "<", arr2_name, ":", utils.wilcox_test(arr1, arr2, "less"))
	print(arr1_name, ">", arr2_name, ":", utils.wilcox_test(arr1, arr2, "greater"))