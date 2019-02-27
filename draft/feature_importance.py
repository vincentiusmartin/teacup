import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

DATA_PATH = "../data/generated/training_numeric_1mer_2mer_fracs.csv"
df = pd.read_csv(DATA_PATH)
X = df.iloc[:, 2:-1]
Y = df.iloc[:, -1]
rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
rf.fit(X, Y)

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
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)
#print(count_dict)
