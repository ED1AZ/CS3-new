import pandas as pd

L_df = pd.read_csv("data_analysis/litternet.csv")
noL_df = pd.read_csv("data_analysis/results.csv")

df = L_df.groupby('litter_present')['conf'].mean()
df2 = noL_df.groupby('litter_present')['conf'].mean()
df3 = noL_df.groupby(['MODEL_TYPE', 'litter_present'])['conf'].mean()

cool = pd.read_csv("results.csv")
cooldf = cool.groupby('litter_present')['conf'].mean()
tool = pd.read_csv("litternet.csv")
tooldf = cool.groupby('litter_present')['conf'].mean()

print(cooldf) 
print(tooldf) 

