import pandas as pd

df = pd.read_csv("results.csv")  # no header=None
start, end = 110, 122 #110-122

df["Unnamed: 3"] = pd.to_numeric(df["Unnamed: 3"], errors="coerce")  # the blank column = frame_num

mask = (
    (df["VIDEO"] == 3)
    & (df["MODEL_TYPE"] == "YOLOv11s")
    & (df["object_id"] == 0)
    & (df["Unnamed: 3"].between(start, end))
)

print("Matching rows:", mask.sum())
df.loc[mask, "litter_present"] = False

df.to_csv("results.csv", index=False)