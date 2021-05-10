#!/usr/bin/env python
# coding: utf-8

# Analyza knižnica na prácu so signalom pre Kopija_Finder_6.0

import numpy as np
# from scipy import signal


def info():
    print("Analyza knižnica na prácu so signalom pre Kopija_Finder_6.0 ")

    
# Znormovanie všetkých dát    
    
def norm(zaznam):
    zaznam_max = np.nanmax(np.abs(zaznam))
    return np.divide(zaznam, zaznam_max)  


# Moving average

def mova(zaznam, rozpetie=1):
    vychylka = (rozpetie - 1) // 2
    gauss_zaznam = np.zeros(len(zaznam), dtype=np.float64)
    for i in range(vychylka, len(zaznam) - vychylka):
        gauss_zaznam[i] = np.sum(zaznam[i - vychylka:i + vychylka]) / rozpetie
    return gauss_zaznam

#     vychylka = (rozpetie - 1) // 2
#     answ = data = np.zeros(len(a), dtype=float)
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     sup = ret[n - 1:] / n
#     answ[vychylka:-vychylka] = sup
#     return answ


# Numerické zderivovanie

def der(time, zaznam): 
    delta = (time[1] - time[0])
    der_zaznam = (np.roll(zaznam, -1) - zaznam) / delta
    return der_zaznam


# Nadefinovanie nastroja na binovanie, vsetko v array-i zaokruhluje po desiatkach,

def binn(zaznam, roun=8, multi=1):
    return np.round(multi * zaznam, roun) / multi
