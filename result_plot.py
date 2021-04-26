import os
from pathlib import Path
import numpy as np
import pandas
import matplotlib.pyplot as plt


def load_material_data(path):
    df = pandas.read_csv(path)

    freq_dict_key = [key for key in df.keys() if 'freq' in key][0]
    ref_ind_key = [key for key in df.keys() if 'ref_ind' in key][0]
    # dn_key = [key for key in df.keys() if "delta_N" in key][0]

    frequencies = np.array(df[freq_dict_key])
    ref_ind = np.array(df[ref_ind_key])
    # dn = np.array(df[dn_key])

    return frequencies, ref_ind


base = Path('CeramicTeraLyzerResult')

# find result files
resultfiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(base)
               for name in files
               if name.endswith('.csv')]

materials = ['Al2O3_1', 'Al2O3_2', 'Al2O3_D', 'QMQM', 'ZrO3', 'Req1', 'Req2', 'Req3', 'Req4', 'Req5']
d = {'Al2O3_1': 2702, 'Al2O3_2': 3091, 'Al2O3_D': 2819, 'QMQM': 5035, 'ZrO3': 4928,
     'Req1': 2481, 'Req2': 1499, 'Req3': 1738, 'Req4': 3089, 'Req5': 2717}

dd = 5


def ref_ind_err(ref_ind, d):
    return dd * (ref_ind - 1) / (d + dd)


fig = plt.figure()

for i, material in enumerate(materials):
    for resultfile in resultfiles:
        if material in str(resultfile):
            deg = str(resultfile).split('_')[-2]
            freq, ref_ind = load_material_data(resultfile)
            dn = ref_ind_err(ref_ind, d[material])

            plt.subplot(2, 5, i+1)
            plt.plot(freq, ref_ind, label=f'{material}_{deg}')
            plt.fill_between(freq, ref_ind - dn, ref_ind + dn, alpha=0.5)
    plt.legend()

plt.show()
