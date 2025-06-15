# -*- coding: utf-8 -*-
"""
Created on Thu May  1 09:04:46 2025
author: Luis C Buratini
TCC - Projeto para Determinar Custos através de Técnicas de Regressão 
"""

#%% Importação e configuração das Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm # estimação de modelos
from statsmodels.stats.anova import anova_lm
from scipy.stats import f
from scipy import stats # utilizado na definição da função 'breusch_pagan_test'


#%%

#%% Criação da função 'breusch_pagan_test' - Professor Fávero

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value
#%%


#%% Importação dos Dados
df = pd.read_csv("df_romaneio_agrupamentos.csv", 
                 delimiter=",", 
                 on_bad_lines='skip',     # ignora linhas problemáticas
                 engine='python')         # necessário com on_bad_lines)

df_dados = df.drop(columns=['DS_FORNECEDOR2','DATA_ENTREGA','DATA_ENTREGA_ANO'])
#%%

#%% Estudo da Correlação
corr = df_dados.corr()
corr['CUSTO_ROMANEIO']
#%%

#%%Mapa de distribuição dos preços
sns.displot(df_dados['CUSTO_ROMANEIO'],kde=True, color='green')
plt.title('Distribuição do Custo por Romaneio')
plt.show()
#%%


#%%Mapa de Calor
# Gerar uma máscara para o triângulo superior
mascara = np.zeros_like(corr, dtype=bool)
mascara[np.triu_indices_from(mascara)] = True

# Configurar a figura do matplotlib
f, ax = plt.subplots(figsize=(11, 9))

# Gerar o mapa de calor (heatmap)
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mascara, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})

# Exibir o mapa de calor (heatmap)
plt.show()
#%%

#%% Gráfico de dispersão com o ajuste linear 
plt.figure(figsize=(15,10))
sns.regplot(data=df_dados, x='DS_DISTANCIA_TOTAL_PERCORRIDA_ROMANEIO', y='CUSTO_ROMANEIO', marker='o', ci=False,
            scatter_kws={"color":'navy', 'alpha':0.9, 's':220},
            line_kws={"color":'grey', 'linewidth': 5})
plt.title('Valores Reais e Fitted Values (Modelo de Regressão)', fontsize=30)
plt.xlabel('Distancia total percorrida', fontsize=24)
plt.ylabel('Custo do romaneio', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(3, 1644)
plt.ylim(7, 5000)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24, loc='upper left')
plt.show
#%% Estimação do modelo de regressão linear simples

#%% Estimação do modelo
modelo = sm.OLS.from_formula('CUSTO_ROMANEIO ~ DS_DISTANCIA_TOTAL_PERCORRIDA_ROMANEIO', df_dados).fit()
modelo.summary() #parâmetros resultantes da estimação

modelo.rsquared #R2
modelo.ess #somatório dos quadrados do modelo
modelo.ssr #somatório dos erros ao quadrado
modelo.params # parâmetros do modelo
modelo.nobs # quantidade de observações
modelo.df_model #graus de liberdade do modelo
modelo.df_resid #graus de liberdade dos resíduos
modelo.fittedvalues #fitted values do modelo

anova = anova_lm(modelo)


#%%Adicionando fitted values e resíduos no modelo
df_dados['fitted'] = modelo.fittedvalues
df_dados['residuos'] = modelo.resid

#Grafico do fitted values x residuos
plt.figure(figsize=(15,10))

sns.regplot(x='fitted', y='residuos', data=df_dados,
            marker='o', fit_reg=False,
            scatter_kws={"color":'red', 'alpha':0.2, 's':150})


plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo', fontsize=20)
plt.ylabel('Resíduos do Modelo', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%%


#%%Diagnóstico de heterocedasticidade com Breusch-Pagan
breusch_pagan_test(modelo)
# Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

# H0 do teste: ausência de heterocedasticidade.
# H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de variável relevante!

# Interpretação
teste_bp = breusch_pagan_test(modelo) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

#%%


#%% Predição
modelo.predict(pd.DataFrame({'DS_DISTANCIA_TOTAL_PERCORRIDA_ROMANEIO':[436]}))















