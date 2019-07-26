import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('../input/train.csv')
df.fillna(-1, inplace=True)
df["annotated"] = (df["EncodedPixels"] != -1).astype("int8")

df["ImageId"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[0])
df["class"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[1])
df = df.set_index(["ImageId", "class"])
df.drop("ImageId_ClassId", axis=1, inplace=True)
df = df.unstack()
df.columns = ["_".join(c) for c in df.columns]
df = df.reset_index()
df["sum_target"] = df.loc[:, ["annotated_{}".format(i) for i in range(1, 5)]].sum(1)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
df["fold_id"] = np.nan
for i, (train_index, valid_index) in enumerate(folds.split(df["ImageId"], df["sum_target"])):
    df.loc[valid_index, "fold_id"] = i

df[["ImageId", "fold_id"]+["EncodedPixels_{}".format(i) for i in range(1, 5)]].to_csv("severstal_folds01.csv", index=False)
