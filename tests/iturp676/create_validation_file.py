"""Creates .csv file with values from ITU-Rpy.

The file is to be used for comparing crc-covlib's partial implementation of ITU-R P.676 with a
third-party library (ITU-Rpy).
"""
from os.path import join, dirname, abspath
import csv
import itertools
import numpy as np
import itur # ITU-Rpy


if __name__ == '__main__':
    script_dir = dirname(abspath(__file__))

    f_range = np.arange(1, 1000+0.1, (1000-1)/9)
    P_range = np.arange(950, 1050+0.1, (1050-950)/9)
    rho_range = np.arange(3.5, 15+0.1, (15-3.5)/9)
    T_range = np.arange(233.15, 313.15+0.1, (313.15-233.15)/9)

    itur.models.itu676.change_version(12)

    with open(join(script_dir, 'validation.csv'), 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['f (GHz)', 'P (hPa)', 'rho (g/m3)', 'T (k)', 'gamma0 (dB/km)', \
                             'gammaw (dB/km)'])
        for f, P, rho, T in itertools.product(f_range, P_range, rho_range, T_range):
            csv_writer.writerow([f, P, rho, T, \
                                 itur.models.itu676.gamma0_exact(f, P, rho, T).value, \
                                 itur.models.itu676.gammaw_exact(f, P, rho, T).value])
