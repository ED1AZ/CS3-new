import pandas as pd

L_df = pd.read_csv("data_analysis/litternet.csv")
noL_df = pd.read_csv("data_analysis/results.csv")

df = L_df.groupby('litter_present')['conf'].mean()
df2 = noL_df.groupby('litter_present')['conf'].mean()
df3 = noL_df.groupby(['MODEL_TYPE', 'litter_present'])['conf'].mean()


print(df) 
print(df2) 
print(df3)