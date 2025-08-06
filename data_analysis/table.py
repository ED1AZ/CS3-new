import pandas as pd

# file to make csv
"""
data = {
    'VIDEO': [],         
    'MODEL_TYPE': [],            
    'LitterNET': [],
    'frame_num': [],
    'object_id': [],
    'litter_present': [],
    'litter_detected': [],   
    'confidence': []
}
"""                                                                                                                                         
"""
df = pd.read_csv("results.csv")
filtered = df[(df['litter_present'] == True) & (df['litter_detected'] == True)]
mean_conf = filtered.groupby('MODEL_TYPE')['conf'].mean()
print(mean_conf)
no_alg_mean_conf_list = mean_conf.tolist()
print(no_alg_mean_conf_list)
"""

df = pd.read_csv("litternet.csv")
filtered = df[(df['litter_present'] == True) & (df['litter_detected'] == True)]
mean_conf_2 = filtered.groupby('MODEL_TYPE')['conf'].mean()
print(mean_conf_2)
alg_mean_conf_list = mean_conf_2.tolist()






    
