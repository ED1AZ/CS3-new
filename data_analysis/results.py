import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from table import no_alg_mean_conf_list
#from table import

# model types
categories = ["YOLOv11", "YOLOv9", "RF-DETR"]
yolov11_withoutL_conf = 0.5653278226261347
# average confidences

#RF-DETRs    0.762855    # NO LITTERNET
#YOLOv9s     0.435772

#RF-DETR     0.906489
#YOLOv11s    0.863864     # WITH LITTERNET
#YOLOv9s     0.891197
values1 = [85, 89, 91] #with LitterNET
values2 = [65, 44, 76] #without 


x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

# plot bars
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, values1, width, label='With LitterNET algorithm')
bars2 = ax.bar(x + width/2, values2, width, label='Without LitterNET')

# labels
ax.set_ylabel('Average Confidence (%)')
ax.set_title('Confidence on True Positives')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.savefig("plot.png")