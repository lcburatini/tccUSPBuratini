# -*- coding: utf-8 -*-
# E2_tccComplementarDadosFicticios.py
# Autor: Luis C Buratini
# Script para adicionar variáveis explicativas Fictícias ao arquivo transformado RomaneioT.xlsx
# Recebe como entrada o arquivo com a base full dos romaneios - RomaneioT.xlsx 
# Entrega como saída o arquivo de entrada para as análises - Entregas.xlsx
"""
Created on Sat Aug 30 09:59:44 2025

@author: lcbur
"""

import pandas as pd
import numpy as np

# Carregar a planilha base
romaneio = pd.read_excel("RomaneioT.xlsx")
romaneio = romaneio.dropna()

# Gerar colunas fictícias
np.random.seed(42)

n = len(romaneio)

romaneio["Tempo_Entrega_h"] = romaneio["DISTANCIA"] / np.random.uniform(40, 80, size=n)
romaneio["Idade_Caminhao"] = np.random.randint(16, 25, size=n)

# Custo de combustível e manutenção
romaneio["Custo_Combustivel"] = romaneio["DISTANCIA"] * np.random.uniform(0.8, 3.1, size=n)
romaneio["Custo_Manutencao"] = romaneio["Idade_Caminhao"] * np.random.uniform(10, 30, size=n)

# Custo total (dependente)
romaneio["Custo_Total"] = (
    romaneio["Custo_Combustivel"] +
    romaneio["Custo_Manutencao"] +
    np.random.uniform(100, 500, size=n) # custo extra para garantir que seja maior
)

romaneio.drop(columns=["ID_ROMANEIO","DATA_ENTREGA","FORNECEDOR","ESCOLA",
                        "MUNICIPIO","Idade_Caminhao"],inplace=True)


# Salvar nova planilha
romaneio.to_excel("Entregas.xlsx", index=False)

















