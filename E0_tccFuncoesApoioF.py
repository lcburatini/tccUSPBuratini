# -*- coding: utf-8 -*-
# E0_tccFuncoesApoio
# Autor: Luis C Buratini
# Script com funções de apoio para desenvolvimento do tcc
# Entrega como saída gráficos e tabelas de apoio
"""
Created on Sun Aug 17 09:35:19 2025

@author: lcbur
"""

#%% carregas Libs
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go
import io
import pandas as pd
import numpy as np
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity #esfericidade de bartlett
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress,shapiro, probplot
from tabulate import tabulate
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


#%% Descritivas impressas formatada na tela 

def descritivaT(df):
    desc = df.describe().round(2)
    
    # Transpor para que cada estatística seja uma linha
    desc_transposed = desc.T
    
    # Exibir como tabela formatada
    print("\n Estatísticas descritivas formatadas:")
    print(tabulate(desc_transposed, headers='keys', tablefmt='fancy_grid'))

#%%Descritivas formatada para o Documento
# === Estrutura do DataFrame (INFO) ===
def descritivas(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue().splitlines()
    
    # Transformar a saída do info em uma tabela
    info_rows = []
    for line in info_text[5:]:  # linhas onde começam as colunas
        parts = line.split()
        if len(parts) >= 3:
            coluna = parts[1]
            nao_nulos = parts[2]
            tipo = parts[-1]
            info_rows.append([coluna, nao_nulos, tipo])
    
    df_info = pd.DataFrame(info_rows, columns=["Coluna", "Valores não nulos", "Tipo de dado"])
    
    # === Estatísticas Descritivas (DESCRIBE) ===
    desc = df.describe(include="all").transpose()
    
    # Melhorar nomes das colunas
    desc = desc.rename(columns={
        "count": "Contagem",
        "mean": "Média",
        "std": "Desvio Padrão",
        "min": "Mínimo",
        "25%": "1º Quartil",
        "50%": "Mediana",
        "75%": "3º Quartil",
        "max": "Máximo"
    })
    
    # Arredondar valores numéricos para 2 casas decimais
    desc = desc.round(2)
    
    print(df_info)
    
    # Exportar tabelas formatadas
    df_info.to_excel("estrutura_dados_tcc.xlsx", index=False)
    desc.to_excel("estatisticas_descritivas_tcc.xlsx")
    
    print("Tabelas geradas:")
    print("- estrutura_dados_tcc.xlsx")
    print("- estatisticas_descritivas_tcc.xlsx")
    
#%% Mapa de Calor
def mapaCalor(df):
    corr = df.corr()
    
    # Gráfico interativo
    fig = go.Figure()
    
    fig.add_trace(
        go.Heatmap(
            x = corr.columns,
            y = corr.index,
            z = np.array(corr),
            text=corr.values,
            texttemplate='%{text:.4f}',
            colorscale='viridis'))
    
    fig.update_layout(
        height = 700,
        width = 1000,
        yaxis=dict(autorange="reversed"))
    
    fig.show()

#%% Teste de Esfericidade de Bartlett
def barlett(df):
    bartlett, p_value = calculate_bartlett_sphericity(df)
    
    print(f'Qui² Bartlett: {round(bartlett, 2)}')
    print(f'p-valor: {round(p_value, 4)}')
    
#%% Teste de heterocedasticidade: Breusch-Pagan

# bp_test = het_breuschpagan(model.resid, model.model.exog)
# labels = ['LM Statistic', 'LM p-value', 'F-Statistic', 'F p-value']
# bp_results = dict(zip(labels, bp_test))

# print("\n Teste de Heterocedasticidade (Breusch-Pagan):")
# for k, v in bp_results.items():
#     print(f"{k}: {v:.4f}")

# # Visualização: dispersão e linha de regressão ajustada
# sns.pairplot(df, x_vars=['DISTANCIA_TOTAL_PERCORRIDA', 'PESO'], y_vars='CUSTO_TOTAL', kind='reg', height=5)
# plt.suptitle('Regressão Linear: CUSTO_TOTAL vs DISTÂNCIA/PESO', y=1.02)
# plt.show()

#%% Gráfico de dispersão com o ajuste linear
def graficoDispersao(df, var_x, var_y, titulo=None):
    """
    Gera um gráfico de dispersão com linha de tendência para duas variáveis numéricas.

    Parâmetros:
    - df: pandas.DataFrame contendo os dados
    - var_x: nome da coluna para o eixo X (str)
    - var_y: nome da coluna para o eixo Y (str)
    - titulo: título opcional para o gráfico (str)
    """
    if var_x in df.columns and var_y in df.columns:
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Gráfico de dispersão
        sns.scatterplot(
            data=df,
            x=var_x,
            y=var_y,
            color='steelblue',
            s=60,
            edgecolor='black'
        )

        # Linha de tendência
        slope, intercept, _, _, _ = linregress(df[var_x], df[var_y])
        x_vals = df[var_x]
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, color='darkred', linestyle='--', linewidth=2, label='Linha de Tendência')

        # Títulos e rótulos
        plt.title(titulo or f'Relação entre {var_y} e {var_x}', fontsize=16, fontweight='bold')
        plt.xlabel(var_x, fontsize=14)
        plt.ylabel(var_y, fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"Colunas '{var_x}' e/ou '{var_y}' não encontradas no DataFrame.")

#%% Gráfico de Frequencia
def graficoFrequencia(obs,titulo,xtitulo,ytitulo):
    # Estilo visual
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(obs, bins=30, kde=True, color='green')
   
    # Títulos e rótulos
    plt.title(titulo, fontsize=16, fontweight='bold')
    plt.xlabel(xtitulo, fontsize=14)
    plt.ylabel(ytitulo, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    
#%%Gráfico boxplot
def graficoBoxplot(obs):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=obs, color='skyblue')
    plt.title('Boxplot ' + obs)
    plt.xlabel(obs)

    
#%% Imprime Summary do modelo
def impModelo(summary_str):
    # Cria figura
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')  # Remove os eixos
    
    # Adiciona o texto do resumo
    ax.text(0, 1, summary_str, fontsize=10, fontfamily='monospace', verticalalignment='top')
    
    # Salva como imagem
    plt.savefig('resumo_regressao.png', bbox_inches='tight', dpi=300)
    plt.show()
    
#%%Resumo do modelo
    
def resumo_modelo(nome, modelo):
    print(f"\n=== {nome} ===")
    print(f"R²: {modelo.rsquared:.4f} | R² ajust.: {modelo.rsquared_adj:.4f}")
    print(f"Estatística F: {modelo.fvalue:.4f} | p-valor do F: {modelo.f_pvalue:.3e}")
    print(f"df_model: {int(modelo.df_model)} | df_resid: {int(modelo.df_resid)}")


#%%
def shapiro_residuos(rotulo, residuos):
    w, p = shapiro(residuos)
    print(f"\n=== Teste de Shapiro-Wilk ({rotulo}) ===")
    print(f"Estatística W: {w:.6f} | p-valor: {p:.3e}")
    return w, p

def homoscedasticidade(rotulo, modelo, X, residuos):
    # Usar valores ajustados como regressor auxiliar no teste
    y_fitted = modelo.fittedvalues

    # Breusch-Pagan
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(residuos, sm.add_constant(y_fitted))
    print(f"\n=== Homoscedasticidade ({rotulo}) ===")
    print("Breusch–Pagan:")
    print(f"  LM stat: {lm_stat:.4f} | LM p-valor: {lm_pvalue:.3e} | F stat: {f_stat:.4f} | F p-valor: {f_pvalue:.3e}")
    # White (opcional, mais geral)
    white_stat, white_pvalue, _, _ = het_white(residuos, X)
    print("White:")
    print(f"  Stat: {white_stat:.4f} | p-valor: {white_pvalue:.3e}")

def calcular_vif(X_com_const):
    # VIF não deve incluir a constante; criaremos uma cópia sem 'const'
    cols = [c for c in X_com_const.columns if c != 'const']
    X_noconst = X_com_const[cols]
    vif_data = []
    for i, col in enumerate(X_noconst.columns):
        vif = variance_inflation_factor(X_noconst.values, i)
        vif_data.append((col, vif))
    vif_df = pd.DataFrame(vif_data, columns=["Variável", "VIF"])
    return vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

def plot_residuos(rotulo, residuos):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(residuos, kde=True, bins=30, edgecolor="black", alpha=0.7)
    plt.title(f"Histograma dos Resíduos ({rotulo})")

    plt.subplot(1,2,2)
    probplot(residuos, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot dos Resíduos ({rotulo})")
    plt.tight_layout()
    plt.show()

def plot_residuos_vs_ajustados(rotulo, modelo):
    residuos = modelo.resid
    ajustados = modelo.fittedvalues
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=ajustados, y=residuos, edgecolor=None)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Valores ajustados")
    plt.ylabel("Resíduos")
    plt.title(f"Resíduos vs Ajustados ({rotulo})")
    plt.tight_layout()
    plt.show()

