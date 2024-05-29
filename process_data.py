import numpy as np
import csv
import ast
import pandas as pd

v_list = []
a_list = []
d_list = []

data = pd.read_csv('data/iemocap_new.csv')
vad = data["[V, A, D]"].values
for row in vad:
    row = eval(row)
    v_list.append(row[0])
    a_list.append(row[1])
    d_list.append(row[2])
data.drop("[V, A, D]", axis=1, inplace=True)
data["valence"] = v_list
data["arousal"] = a_list
data["dominance"] = d_list
df = pd.DataFrame(data=data,
                      columns=['FileName', 'Label', 'valence', 'arousal', 'dominance'])
df.to_csv('data.csv')
