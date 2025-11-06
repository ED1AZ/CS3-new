import pandas as pd

df = pd.read_csv("data_analysis/YOLOv11_no_litternet1_results.csv")

print(df['confidence'].mean())