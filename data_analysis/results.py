import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# model types
categories = ["YOLOv11", "YOLOv9", "Faster R-CNN", "RF-DETR"]

# average confidences
values1 = [20, 40, 45, 50] #with LitterNET
values2 = [20, 45, 60, 30] #without 


x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

# plot bars
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, values1, width, label='With LitterNET algorithm')
bars2 = ax.bar(x + width/2, values2, width, label='Without LitterNET')

# labels
ax.set_ylabel('Average Confidence (%)')
ax.set_title('Confidence on Detected Litter')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.show()