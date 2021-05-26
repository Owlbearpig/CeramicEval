import os
from pathlib import Path
import numpy as np
import pandas
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from pathlib import Path

def export_csv(data, path):
    df = pd.DataFrame(data=data)
    df.to_csv(path)


def load_material_data(path):
    df = pandas.read_csv(path)

    freq_dict_key = [key for key in df.keys() if 'freq' in key][0]
    ref_ind_key = [key for key in df.keys() if 'ref_ind' in key][0]
    alpha_key = [key for key in df.keys() if 'alpha' in key][0]
    alpha_err_key = [key for key in df.keys() if 'delta_A' in key][0]
    ref_ind_err_key = [key for key in df.keys() if "delta_N" in key][0]

    frequencies = np.array(df[freq_dict_key])
    ref_ind = np.array(df[ref_ind_key])
    alpha = np.array(df[alpha_key])
    alpha_err = np.array(df[alpha_err_key])
    ref_ind_err = np.array(df[ref_ind_err_key])

    data = {'freq': frequencies, 'ref_ind': ref_ind, 'alpha': alpha, 'alpha_err': alpha_err, 'ref_ind_err': ref_ind_err}

    return data


base = Path('CeramicTeraLyzerResult')

# find result files
resultfiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(base)
               for name in files
               if name.endswith('.csv')]

materials = ['Al2O3_1', 'Al2O3_2', 'Al2O3_D', 'QMQM', 'ZrO3', 'Req1', 'Req2', 'Req3', 'Req4', 'Req5', 'R2shift']
#materials = ['Al2O3_1', 'Al2O3_2', 'Al2O3_D', 'QMQM', 'ZrO3', 'Req1', 'Req2', 'Req3', 'Req4', 'Req5']
d = {'Al2O3_1': 2702, 'Al2O3_2': 3091, 'Al2O3_D': 2819, 'QMQM': 5035, 'ZrO3': 4928,
     'Req1': 2481, 'Req2': 1499, 'Req3': 1738, 'Req4': 3089, 'Req5': 2717, 'R2shift': 1499}

def ref_ind_err(ref_ind, d):
    dd = 5
    return dd * (ref_ind - 1) / (d + dd)

material = 'Req2'
for resultfile in resultfiles:
    if material in str(resultfile):
        deg = str(resultfile).split('_')[-2]
        zro3_res = load_material_data(resultfile)
        freq, alpha, alpha_err = zro3_res['freq'], zro3_res['alpha'], zro3_res['alpha_err']
        ref_ind, ref_ind_err = zro3_res['ref_ind'], zro3_res['ref_ind_err']

        plt.plot(freq, ref_ind, label=f'{Path(resultfile).name}')
        plt.fill_between(freq, ref_ind - ref_ind_err, ref_ind + ref_ind_err, alpha=0.5)
plt.legend()
plt.show()


fig = plt.figure()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

for i, material in enumerate(materials):
    export_data = {}
    for resultfile in resultfiles:
        if '1482' in str(resultfile):
            continue
        if '1420' in str(resultfile):
            continue
        if '1474' in str(resultfile):
            continue

        if material in str(resultfile):
            deg = str(resultfile).split('_')[-2]
            res_data = load_material_data(resultfile)
            freq, ref_ind, alpha = res_data['freq'], res_data['ref_ind'], res_data['alpha']

            #ref_ind = moving_average(ref_ind, 10)

            dn = res_data['ref_ind_err']
            an = res_data['alpha_err']
            #dn = ref_ind_err(ref_ind, d[material]) # 'error from thickness uncertainty'

            plt.subplot(2, 5, i+1)
            plt.plot(freq[:len(alpha)], alpha, label=f'{material}_{deg}')
            plt.fill_between(freq[:len(alpha)], alpha - an[:len(alpha)], alpha + an[:len(alpha)], alpha=0.5)
            if 'freq' not in export_data.keys():
                export_data['freq'] = freq[:len(alpha)]
            export_data['alpha_' + deg] = alpha
            export_data['alpha_err_' + deg] = an[:len(alpha)]
    plt.legend()

    if material == 'Req2':
        export_data['freq'] = export_data['freq'][::2]
        export_data['freq'] = export_data['freq'][:452]
        export_data['alpha_0deg'] = export_data['alpha_0deg'][::2]
        export_data['alpha_0deg'] = export_data['alpha_0deg'][:452]
        export_data['alpha_err_0deg'] = export_data['alpha_err_0deg'][::2]
        export_data['alpha_err_0deg'] = export_data['alpha_err_0deg'][:452]
    else:
        for key in export_data:
            if len(export_data['freq']) != len(export_data[key]):
                export_data[key] = export_data[key][::2]
                export_data[key] = export_data[key][:len(export_data['freq'])]

    print(material)
    for key in export_data:
        print(key)
        print(len(export_data[key]))

    #export_csv(export_data, str(material)+'_abs.csv')

plt.show()
