# -*- coding: utf-8 -*-
# E1_tccDescaracterizar.py
# Autor: Luis C Buratini
# Script para descaracterizar os dados para fins de preservação de identidade
# Recebe como entrada o arquivo com a base full dos romaneios - RomaneioF.xlsx 
# Entrega como saída o arquivo descaracterizado - RomaneioT.xlsx ( T = transformado )

"""
Created on Sun Aug 24 12:37:28 2025
@author: lcbur
"""

import pandas as pd

# Carrega o arquivo Excel
df = pd.read_excel("RomaneioF.xlsx")

# Remove espaços em branco antes e depois de cada observação
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) 

# Verifica se a coluna DS_NOME existe
coluna_nome = "MUNICIPIO"
if coluna_nome in df.columns:
    # Cria os nomes anonimizados
    nomes_unicos = df[coluna_nome].dropna().unique()
    mapeamento = {nome: f"municipio_{str(i+1).zfill(3)}" for i, nome in enumerate(nomes_unicos)}

    # Cria nova coluna com os nomes descaracterizados
    df["MUNICIPO_DESCARACTERIZADA"] = df[coluna_nome].map(mapeamento)


coluna_nome = 'FORNECEDOR'
if coluna_nome in df.columns:
    # Cria os nomes anonimizados
    nomes_unicos = df[coluna_nome].dropna().unique()
    mapeamento = {nome: f"fornecedor_{str(i+1).zfill(3)}" for i, nome in enumerate(nomes_unicos)}

    # Cria nova coluna com os nomes descaracterizados
    df["FORNECEDOR_DESCARACTERIZADA"] = df[coluna_nome].map(mapeamento)


coluna_nome = 'ESCOLA'
if coluna_nome in df.columns:
    # Cria os nomes anonimizados
    nomes_unicos = df[coluna_nome].dropna().unique()
    mapeamento = {nome: f"escola_{str(i+1).zfill(3)}" for i, nome in enumerate(nomes_unicos)}

    # Cria nova coluna com os nomes descaracterizados
    df["ESCOLA_DESCARACTERIZADA"] = df[coluna_nome].map(mapeamento)

#%% Incluir coluna com as distancias da empresa até o Municipio

distancias = {
            'São Bernardo do Campo': 8 ,
            'Santo André': 1,
            'São Sebastião': 210,
            'Diadema':16 ,
            'São José dos campos':107 ,
            'Ribeirão Pires': 14,
            'Piracaia': 110,
            'Santos': 80
}
   
df['DISTANCIA'] = df['MUNICIPIO'].map(distancias)

#%% Tratar Peso
df = df[df['PESO'] != 0] #elimina observações com peso nulo

#%% Tratar a data da entrega
df["DATA_ENTREGA_ANO"] = df["DATA_ENTREGA"].dt.year
df["DATA_ENTREGA_MES"] = df["DATA_ENTREGA"].dt.month
df["DATA_ENTREGA_DIA"] = df["DATA_ENTREGA"].dt.day

#%% Limpeza

df.drop(columns=['ESCOLA','FORNECEDOR','MUNICIPIO'],inplace=True)
             
df = df.rename(columns={
        'ESCOLA_DESCARACTERIZADA': 'ESCOLA',
        'FORNECEDOR_DESCARACTERIZADA': 'FORNECEDOR',
        'MUNICIPO_DESCARACTERIZADA': 'MUNICIPIO',
        }
    )

df = df[['ID_ROMANEIO','DATA_ENTREGA','FORNECEDOR','PESO','ESCOLA', 'MUNICIPIO','DISTANCIA']]


#%% Salva no mesmo arquivo (sobrescrevendo)
df.to_excel("RomaneioT.xlsx", index=False)
