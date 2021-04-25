import os
from pathlib import Path
import numpy as np
import pandas
import matplotlib.pyplot as plt

def load_material_data(path):

    df = pandas.read_csv(path)

    freq_dict_key = [key for key in df.keys() if "freq" in key][0]
    ref_ind_key = [key for key in df.keys() if "ref_ind" in key][0]

    frequencies = np.array(df[freq_dict_key])
    ref_ind = np.array(df[ref_ind_key])

    return frequencies, ref_ind

base = Path('CeramicTeraLyzerResult')

# find result files
resultfiles = [os.path.join(root, name)
     for root, dirs, files in os.walk(base)
        for name in files
            if name.endswith((".csv"))]

materials = ['Al2O3_1', 'Al2O3_2', 'Al2O3_D', 'QMQM', 'ZrO3', 'Req1', 'Req2', 'Req3', 'Req4', 'Req5']

for material in materials:
    for resultfile in resultfiles:
        if material in str(resultfile):
            deg = str(resultfile).split('_')[-2]
            freq, ref_ind = load_material_data(resultfile)
            plt.plot(freq, ref_ind, label=f'{material}_{deg}')

    plt.legend()
    plt.show()

