import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

rles = pd.read_csv('../input/train.csv')
rles.fillna(-1, inplace=True)
# images can have multiple annotations
rles_ = defaultdict(list)
for image_id, rle in zip(rles['ImageId_ClassId'], rles['EncodedPixels']):
    rles_[image_id].append(rle)
annotated = {k: v for k, v in rles_.items() if v[0] != -1}

df = pd.DataFrame({"ImageId_ClassId": rles["ImageId_ClassId"].unique()})
df["annotated"] = 0
df.loc[df["ImageId_ClassId"].isin(list(annotated.keys())), "annotated"] = 1

df["ImageId"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[0])
df["class"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[1])
df = df.set_index(["ImageId", "class"])
df.drop("ImageId_ClassId", axis=1, inplace=True)
df = df.unstack()
df.columns = [1, 2, 3, 4]
df = df.reset_index()
df["sum_target"] = df.iloc[:, 1:].sum(1)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
df["fold_id"] = np.nan
for i, (train_index, valid_index) in enumerate(folds.split(df["ImageId"], df["sum_target"])):
    df.loc[valid_index, "fold_id"] = i

df[["ImageId", "fold_id"]].to_csv("severstal_folds01.csv", index=False)
