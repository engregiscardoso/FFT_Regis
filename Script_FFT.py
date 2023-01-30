# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:19:57 2023

@author: REGIS CARDOSO
"""

######################################################################################################
## ANÁLISE DE VIBRAÇÃO REAL DE UM MOTOR - MUITO UTILIZADO PARA MANUTENÇÃO PREDITIVA ###
######################################################################################################

## IMPORTAR AS BIBLIOTECAS UTILIZADAS ###

import pandas as pd
import numpy as np
import statistics
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
from scipy.fftpack import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler


## FUNÇÕES

# FUNÇÃO PARA FEATURE ENGINEERING VIA TRANSFORMADA RÁPIDA DE FOURIER - FFT


def FFT(df):

    df_Final = df

    graf_x = df_Final['Tempo'].values
    graf_y = df_Final['Valor'].values

    x = graf_x

    y = graf_y

    N = len(graf_x)

    T = x[1] - x[0]

    Fs = 1 / T

    yf = 2.0 / N * np.abs(fft(y)[0:N // 2])

    xf = fftfreq(N, T)[:N // 2]

    verX = []
    verY = []

    obs = len(yf)

    for i in range(1, obs, 1):
        verX.insert(i, xf[i])
        verY.insert(i, yf[i])
        
    df_FFT = []
    df_FFT = pd.DataFrame(df_FFT)

    df_FFT['Frequencia'] = verX
    df_FFT['Amplitude'] = verY

    return (df_FFT)



df = pd.read_csv('Dado_Vibracao.csv', sep=';')

df.columns = ['Tempo', 'Valor']

columns = ['Tempo', 'Valor']


## CONVERTE OS DADOS EM FLOAT E ADICIONA O PONTO COMO SEPARADOR DECIMAL ###

df[columns] = df[columns].apply(lambda x: x.str.replace(',', '.').astype('float'))


df_FFT = FFT(df)

