# -*- coding: utf-8 -*-
# E4_tccRegressaoLinear.py
# Autor: Luis C Buratini
# Script para execução da Regreção Linear. Recebe de entrada o arquivo "Entregas.xlsx"
# Entrega como saída gráficos e saida do modelo
"""
Created on Tue Sep  2 13:52:29 2025

@author: lcbur
"""

#%% carregar libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#import statsmodels.formula.api as smf
from scipy.stats import shapiro, probplot, boxcox
from E0_tccFuncoesApoioF import impModelo, resumo_modelo,homoscedasticidade,calcular_vif

#%% Carregar os dados
df = pd.read_excel('Entregas.xlsx')
#df = df.sample(n=500, random_state=42)

#%%
# Separar variável dependente e explicativas
y = df['Custo_Total'] # variável dependente
#X = df.drop(columns=['Custo_Total']) # variáveis explicativas
X = df[['Custo_Manutencao','Custo_Combustivel']]


#%% Verificar a existenca de multicolinearidade
# VIF
vif_sem = calcular_vif(X)
print("\n=== VIF (sem transformação) ===")
print(vif_sem.to_string(index=False))


#%% Executar Modelo Regressão Linear
#Adicionar constante (intercepto)

X = sm.add_constant(X)

# Regressão linear
model = sm.OLS(y, X).fit()
residuos = model.resid

impModelo(model.summary())

resumo_modelo("Sem Transformação", model)

#%%Teste de homocedasticidade de breuschpagan
homoscedasticidade("sem transformação", model, X, model.resid)

#%% Verificar normalidade com teste de Shapiro-Wilk nos resíduos
# Aplicar Shapiro dos resíduos do modelo
stat, p_value = shapiro(residuos)

print("=== Teste de Shapiro-Wilk nos Resíduos ===")
print("Estatística W:", stat)
print("p-valor:", p_value)

alpha = 0.05 # nível de significancia
if p_value > alpha:
    print("Não rejeitamos H0 → resíduos seguem distribuição normal.")
else:
    print("Rejeitamos H0 → resíduos não seguem distribuição normal.")

# === Visualizações dos resíduos ===
plt.figure(figsize=(14,5))

# Histograma + KDE
plt.subplot(1,2,1)
sns.histplot(residuos, kde=True, bins=30, color="blue")
plt.title("Histograma dos Resíduos")
plt.tight_layout()

# Q-Q Plot
plt.subplot(1,2,2)
probplot(residuos, dist="norm", plot=plt)
plt.title("Q-Q Plot dos Resíduos")

plt.tight_layout()
plt.show()

#%% Aplicar Transformação Box-Cox com objetivo de reduzir a assimetria e trazer a normalidade

y_boxcox, lambda_bc = boxcox(y)
print("\nLambda ótimo para Box-Cox:", lambda_bc)


#%% Repetir a estimação do modelo com a variavel transformada
# Regressão com Y transformado
modelo_bc = sm.OLS(y_boxcox, X).fit()

impModelo(modelo_bc.summary())

resumo_modelo("Após Transformação", modelo_bc)

#%% Estudo dos residuos após a transformação
residuos_bc = modelo_bc.resid

#%%Aplicação de Shapiro-Wilk após transformação
print("\n=== Teste de Shapiro-Wilk (Resíduos com Box-Cox) ===")
stat_bc, p_value_bc = shapiro(residuos_bc)
print("Estatística W:", stat_bc, " | p-valor:", p_value_bc)

#%%Visualização após transformação
# === Visualizações ===
plt.figure(figsize=(14,8))

# Histograma resíduos originais
plt.subplot(2,2,1)
sns.histplot(residuos, kde=True, bins=30, color="blue")
plt.title("Histograma dos Resíduos (sem transformação)")

# Q-Q Plot resíduos originais
plt.subplot(2,2,2)
probplot(residuos, dist="norm", plot=plt)
plt.title("Q-Q Plot Resíduos (sem transformação)")

# Histograma resíduos com Box-Cox
plt.subplot(2,2,3)
sns.histplot(residuos_bc, kde=True, bins=30, color="green")
plt.title("Histograma dos Resíduos (Box-Cox)")

# Q-Q Plot resíduos com Box-Cox
plt.subplot(2,2,4)
probplot(residuos_bc, dist="norm", plot=plt)
plt.title("Q-Q Plot Resíduos (Box-Cox)")

plt.tight_layout()
plt.show()






