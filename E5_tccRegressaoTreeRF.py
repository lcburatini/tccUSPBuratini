# -*- coding: utf-8 -*-
# E5_tccRegressaoTreeRF.py
# Autor: Luis C Buratini
# Script para execução da Regreção utilizando Random Forest. Recebe de entrada o arquivo "Entregas.xlsx"
# Entrega como saída gráficos e saida do modelo
#%% Importar os pacotes
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#%% Importar o banco de dados

dados = pd.read_excel('Entregas.xlsx')

#%% Separando as variáveis Y e X

X = dados.drop(columns=['Custo_Total'])
y = dados['Custo_Total']

#%% Separando as amostras de treino e teste

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.20, random_state=42
     
     )   

#%% 
#Hiperparâmetros do modelo
param_grid_tree = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [5, 10]
}


# Identificar o algoritmo em uso
tree_grid = RandomForestRegressor(random_state=100)

# Treinar os modelos para o grid search
tree_grid_model = GridSearchCV(estimator = tree_grid, 
                               param_grid = param_grid_tree,
                               scoring='neg_mean_squared_error', 
                               cv=5, verbose=2)

tree_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
tree_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros
tree_best = tree_grid_model.best_estimator_

# Predict do modelo
tree_grid_pred_train = tree_best.predict(X_train)
tree_grid_pred_test = tree_best.predict(X_test)

#%% Avaliando o novo modelo (base de treino)

mse_train_tree_grid = mean_squared_error(y_train, tree_grid_pred_train)
mae_train_tree_grid = mean_absolute_error(y_train, tree_grid_pred_train)
r2_train_tree_grid = r2_score(y_train, tree_grid_pred_train)

print("Avaliação do Modelo (Base de Treino)")
print(f"MSE: {mse_train_tree_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_train_tree_grid):.1f}")
print(f"MAE: {mae_train_tree_grid:.1f}")
print(f"R²: {r2_train_tree_grid:.1%}")

#%% Avaliando o novo modelo (base de teste)

mse_test_tree_grid = mean_squared_error(y_test, tree_grid_pred_test)
mae_test_tree_grid = mean_absolute_error(y_test, tree_grid_pred_test)
r2_test_tree_grid = r2_score(y_test, tree_grid_pred_test)

print("Avaliação do Modelo (Base de Teste)")
print(f"MSE: {mse_test_tree_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_test_tree_grid):.1f}")
print(f"MAE: {mae_test_tree_grid:.1f}")
print(f"R²: {r2_test_tree_grid:.1%}")

#%% Importância das variáveis preditoras

tree_features = pd.DataFrame({'features':X.columns.tolist(),
                              'importance':tree_best.feature_importances_}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print(tree_features)

#%% Graficos
# ------------------------------
# Gráfico 1: Real vs. Previsto
# ------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, tree_grid_pred_test, alpha=0.5)
plt.title(f"Random Forest - Real vs. Previsto (Teste)\nR²={r2_test_tree_grid:.3f} | RMSE={np.sqrt(mse_test_tree_grid):.2f} | MAE={mae_test_tree_grid:.2f}")
plt.xlabel("Custo_Total (real)")
plt.ylabel("Custo_Total (previsto)")
mse_test_tree_grid = mean_squared_error(y_test, tree_grid_pred_test)
mae_test_tree_grid = mean_absolute_error(y_test, tree_grid_pred_test)
r2_test_tree_grid = r2_score(y_test, tree_grid_pred_test)

min_val = float(min(y_test.min(),tree_grid_pred_test.min()))
max_val = float(max(y_test.max(), tree_grid_pred_test.max()))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.tight_layout()
plt.show()

#%%
# ------------------------------
# Gráfico 2: Importância das variáveis
# ------------------------------
importances = pd.Series(tree_best.feature_importances_, index=X.columns).sort_values(ascending=False)

fi_dados = importances.reset_index()
fi_dados.columns = ["Variavel", "Importancia"]

topn = min(10, len(fi_dados))
plt.figure(figsize=(8, 5))
plt.barh(fi_dados["Variavel"].head(topn)[::-1], fi_dados["Importancia"].head(topn)[::-1])
plt.title("Random Forest - Importância das Variáveis (Top 10)")
plt.xlabel("Importância (Ganho de impureza)")
plt.ylabel("Variável")
plt.tight_layout()
plt.show()
