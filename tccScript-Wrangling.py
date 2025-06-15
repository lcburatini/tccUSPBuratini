# -*- coding: utf-8 -*-
"""
Created on Thu May  1 09:04:46 2025
author: Luis C Buratini
TCC - Projeto para Determinar Custos através de Técnicas de Regressão 
Script para tratativas preliminares dos dados recebidos
"""

#%% Importação e configuração das Libs
import pandas as pd
pd.set_option("display.max.columns",None)

import numpy as np
#%%

#%% Importação dos Dados
df = pd.read_csv("romaneios3.csv", 
                 delimiter=";", 
                 on_bad_lines='skip',     # ignora linhas problemáticas
                 engine='python')         # necessário com on_bad_lines)
df_romaneio = df #preserva os dados originais


df_custos = pd.read_csv("custos.csv", 
                 delimiter=";", 
                 on_bad_lines='skip',     # ignora linhas problemáticas
                 engine='python')         # necessário com on_bad_lines)
df_custos = df_custos.set_index('DATA_ENTREGA_ANO')

#%%

#%% Tratativa das linhas e colunas da base de romaneios recebida
df_romaneio.columns = df_romaneio.columns.str.replace(' ', '', regex=False) #remove espaços em branco no nome das colunas
df_romaneio['DS_PESO'] = df_romaneio['DS_PESO'].str.replace(',', '.', regex=False) #substitui "," por "."
df_romaneio['DS_PESO'] = pd.to_numeric(df['DS_PESO'])
df_romaneio = df_romaneio[df_romaneio['DS_PESO'] != 0] #elimina observações com peso nulo
df_romaneio['DATA_ENTREGA'] = pd.to_datetime(df_romaneio['DATA_ENTREGA'])

df_romaneio['DS_PRODUTO'] = df_romaneio['DS_PRODUTO'].str.strip() #remove espaços no inicio e fim dos valores da coluna

#Excluir produtos expressos em litros
produtos_exclusao = [
    'Óleo*',
    '#Leite UHT Desnatado (1 litro)',                                                                                               
    '*Óleo de soja',                                                                                        
    '#Leite UHT Desnatado (1 litro)',                                                                       
    '#Leite integral',                                                                                      
    '#Leite UHT sem lactose (1 litro)',                                                                     
    '#Bebida Láctea choc/malte',                                                                            
    '#Bebida Láctea sabor variados',
    'Bebida Chocolate',                                                                        
    'Óleo Composto (500 ml)',                                                                              
    'Óleo de Soja (900 ml)',                                                                               
    'Vinagre maça',                                                                                        
    '#Bebida de Arroz',
    'Iogurte morango',                                                                                    
    ]
df_romaneio = df_romaneio[~df_romaneio['DS_PRODUTO'].isin(produtos_exclusao)]
#%%

#%% Criar Dataframe de Municipios com as distancias partindo da empresa
#Criar dataframe Municipio
municipios = {
    'DS_MUNICIPIO': ['São Bernardo do Campo',
             'Santo André',
             'São Sebastião',
             'Diadema',
             'São José dos campos',
             'Ribeirão Pires',
             'Piracaia',
             'Santos'
             ],
    'DISTANCIA': [8,1,210,16,107,14,110,80] #Distancia Escola - Empresa
}
df_municipios = pd.DataFrame(municipios)
#%%


#%% Cálculo do peso total por romaneio,quantidade de carros utilizado. Cada carro suporta até 4.000kg
#Um romaneio compreende a distribuição de produto para vávias escolas


#Cálculo do peso ( volume) por romaneio
df_romaneio_agrupamentos = df_romaneio.groupby('ID_ROMANEIO')['DS_PESO'].sum()  #soma o peso total por romaneio
df_romaneio_agrupamentos = df_romaneio_agrupamentos.to_frame()

#Calculo da Qtde de Carros
df_romaneio_agrupamentos['DS_QTDE_CARROS_POR_ROMANEIO'] = round(df_romaneio_agrupamentos['DS_PESO']/4000) # considera o limite de 4 toneladas por carro
df_romaneio_agrupamentos.loc[df_romaneio_agrupamentos['DS_QTDE_CARROS_POR_ROMANEIO'] < 1, 'DS_QTDE_CARROS_POR_ROMANEIO'] = 1 # considera pelo menos 1 carro para volume menor que 4000 kg

#Calculo da Qtde de Escolas
df_romaneio_agrupamentos['DS_QTDE_ESCOLAS'] = df_romaneio.groupby('ID_ROMANEIO')['DS_NOME'].count()

#Calculo da distancia percorrida entre escolas - Média de 2 km
df_romaneio_agrupamentos['DS_DISTANCIA_PERCORRIDA_DENTRO_MUNICIPIO'] = df_romaneio_agrupamentos['DS_QTDE_ESCOLAS'] * 2

#Agregar a distancia entre Empresa e o Municipio
df_romaneio_municipios = df_romaneio[['ID_ROMANEIO','DS_MUNICIPIO','DATA_ENTREGA','DS_FORNECEDOR2']]
df_romaneio_municipios = df_romaneio_municipios.drop_duplicates(subset='ID_ROMANEIO')
df_romaneio_municipios['DS_MUNICIPIO'] = df_romaneio_municipios['DS_MUNICIPIO'].str.strip() #remove espaços no inicio e fim dos valores da coluna
df_romaneio_municipios.set_index('ID_ROMANEIO', inplace=True)
d_empresa_municipio =[
    df_romaneio_municipios['DS_MUNICIPIO'] == 'São Bernardo do Campo',
    df_romaneio_municipios['DS_MUNICIPIO'] == 'Santo André',
    df_romaneio_municipios['DS_MUNICIPIO'] == 'São Sebastião',
    df_romaneio_municipios['DS_MUNICIPIO'] == 'Diadema',
    df_romaneio_municipios['DS_MUNICIPIO'] == 'São José dos campos',
    df_romaneio_municipios['DS_MUNICIPIO'] == 'Ribeirão Pires',
    df_romaneio_municipios['DS_MUNICIPIO'] == 'Piracaia',
    df_romaneio_municipios['DS_MUNICIPIO'] == 'Santos'
]
distancia = [8,1,210,16,107,14,110,80]
df_romaneio_municipios['DS_DISTANCIA_EMPRESA_MUNICIPIO'] = np.select(d_empresa_municipio,distancia)

#Cálculo da Distância Total percorrida por romaneio
df_romaneio_agrupamentos = pd.merge(df_romaneio_agrupamentos,df_romaneio_municipios[['DS_DISTANCIA_EMPRESA_MUNICIPIO','DATA_ENTREGA','DS_FORNECEDOR2']], on='ID_ROMANEIO',how='left')
df_romaneio_agrupamentos['DS_DISTANCIA_TOTAL_PERCORRIDA_ROMANEIO'] = df_romaneio_agrupamentos['DS_DISTANCIA_EMPRESA_MUNICIPIO'] + df_romaneio_agrupamentos['DS_DISTANCIA_PERCORRIDA_DENTRO_MUNICIPIO']

#Agregar custo por veículo
df_romaneio_agrupamentos['DATA_ENTREGA_ANO'] = df_romaneio_agrupamentos['DATA_ENTREGA'].dt.year
df_romaneio_agrupamentos = df_romaneio_agrupamentos.set_index('DATA_ENTREGA_ANO')
df_romaneio_agrupamentos = pd.merge(df_romaneio_agrupamentos, df_custos, on='DATA_ENTREGA_ANO', how='left')

#Cálculo do Custo Total
df_romaneio_agrupamentos['CUSTO_ROMANEIO'] = df_romaneio_agrupamentos['DS_DISTANCIA_TOTAL_PERCORRIDA_ROMANEIO'] * df_romaneio_agrupamentos['CUSTO_VEIC_KM']

#%%


#%% Exportar dataframes para planilha excel
df_romaneio_agrupamentos.to_csv('df_romaneio_agrupamentos.csv')

#%%

