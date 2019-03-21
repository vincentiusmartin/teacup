import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from sklearn.utils import shuffle
import numpy as np

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

    filter_col = [col for col in df.columns.tolist() if col.startswith(label)]

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

rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
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
#print(count_dict)
