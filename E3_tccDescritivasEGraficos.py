# -*- coding: utf-8 -*-
# E3_tccDescritivasEGraficos.py
# Autor: Luis C Buratini
# Script para Análise exploratória
# Entrega como saída gráficos e tabelas de apoio

"""
Created on Sat Aug 30 11:31:56 2025

@author: lcbur
"""

#%% carregas Libs
import pandas as pd
from E0_tccFuncoesApoioF import descritivas,mapaCalor,graficoFrequencia

#%% importar dados
df = pd.read_excel('Entregas.xlsx') #gerado por RomaneioExplore.py


#%% Descritivas

descritivas(df)
mapaCalor(df)
graficoFrequencia(df['Custo_Total'],"Frequência dos Custos em R$","Custo (R$)","Frequência")
