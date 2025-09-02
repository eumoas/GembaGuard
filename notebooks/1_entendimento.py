#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import os
import seaborn as sns

def carregar_dados():
    """Carrega o dataset e retorna um DataFrame."""
    print("--- INICIANDO ETAPA 1: ENTENDIMENTO DOS DADOS ---")

    caminho_arquivo = Path(__file__).parent / "bootcamp_train.csv"

    if not caminho_arquivo.exists():
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None
    
    df = pd.read_csv(caminho_arquivo)
    print(f"Dados carregados com sucesso. Dimensões: {df.shape}")
    return df

def converter_colunas_falhas(df):
    """
    Converte colunas de falha para formato numérico (0 ou 1)
    e garante que os nomes estejam corretos.
    """
    print("\n--- CONVERTENDO COLUNAS DE FALHA ---")
    
    falhas_esperadas = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
    
    mapeamento_nomes = {
        'FDF (Falha Desgaste Ferramenta)': 'FDF',
        'FDC (Falha Dissipacao Calor)': 'FDC',
        'FP (Falha Potencia)': 'FP',
        'FTE (Falha Tensao Excessiva)': 'FTE',
        'FA (Falha Aleatoria)': 'FA',
        'falha_maquina': 'falha_maquina'
    }
    
    colunas_para_renomear = {
        col: mapeamento_nomes[col] for col in df.columns if col in mapeamento_nomes
    }
    df.rename(columns=colunas_para_renomear, inplace=True)
    
    print("   -> Colunas de falha renomeadas para nomes curtos.")
    
    for falha in falhas_esperadas + ['falha_maquina']:
        if falha in df.columns:
            if df[falha].dtype in ['object', 'bool']:
                df[falha] = df[falha].astype(str).str.lower()
                mapping = {'true': 1, 'false': 0, '1': 1, '0': 0}
                df[falha] = df[falha].map(mapping).fillna(0).astype(int)
                print(f"   -> Coluna '{falha}' convertida para tipo numérico.")
    
    return df

def analise_inicial(df):
    """Realiza uma análise exploratória inicial e salva um relatório."""
    print("\n--- ANÁLISE INICIAL DO DATASET ---")
    relatorio = []
    relatorio.append("Resumo da Análise Inicial")
    relatorio.append("="*50)
    relatorio.append(f"Dimensões do Dataset: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    tipos_dados = df.dtypes.value_counts().to_dict()
    relatorio.append("\nContagem de Tipos de Dados:")
    for tipo, count in tipos_dados.items():
        relatorio.append(f"  - {tipo}: {count} colunas")
        
    faltantes = df.isnull().sum()
    faltantes = faltantes[faltantes > 0].sort_values(ascending=False)
    
    relatorio.append("\nValores Faltantes por Coluna:")
    if faltantes.empty:
        relatorio.append("  - Nenhum valor faltante. ✅")
    else:
        for coluna, count in faltantes.items():
            relatorio.append(f"  - {coluna}: {count} ({count/len(df):.2%})")

    print("\n".join(relatorio))
    with open('1_relatorio_entendimento.txt', 'w') as f:
        f.write("\n".join(relatorio))
    print("Relatório de entendimento salvo em '1_relatorio_entendimento.txt'")
    
    return df

def visualizar_distribuicao_features(df):
    """
    Gera histogramas e boxplots para as features numéricas.
    """
    print("\n--- VISUALIZANDO DISTRIBUIÇÃO DAS FEATURES ---")
    features_numericas = df.select_dtypes(include=np.number).columns.tolist()
    
    features_a_plotar = [f for f in features_numericas if f not in ['id', 'falha_maquina', 'FDF', 'FDC', 'FP', 'FTE', 'FA']]
    
    n_features = len(features_a_plotar)
    cols = 3
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_features > 0:
        axes = axes.ravel() if n_features > 1 else [axes]
    
        for i, feature in enumerate(features_a_plotar):
            sns.histplot(df[feature], bins=50, kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribuição de {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequência')
        
    plt.tight_layout()
    plt.savefig('1_distribuicao_features.png')
    plt.close(fig)
    print("Gráfico de distribuição salvo em '1_distribuicao_features.png'")

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_features > 0:
        axes = axes.ravel() if n_features > 1 else [axes]
    
        for i, feature in enumerate(features_a_plotar):
            sns.boxplot(x=df[feature], ax=axes[i], color='lightcoral')
            axes[i].set_title(f'Boxplot de {feature}')
            axes[i].set_xlabel(feature)
        
    plt.tight_layout()
    plt.savefig('1_boxplots_features.png')
    plt.close(fig)
    print("Boxplots de features salvos em '1_boxplots_features.png'")

def visualizar_matriz_correlacao_bruta(df):
    """
    Gera uma matriz de correlação das features numéricas.
    """
    print("\n--- VISUALIZANDO MATRIZ DE CORRELAÇÃO BRUTA ---")
    features_numericas = df.select_dtypes(include=np.number).columns.tolist()
    features_a_plotar = [f for f in features_numericas if f not in ['id']]
    
    corr_matrix = df[features_a_plotar].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de Correlação das Features Brutas')
    plt.tight_layout()
    plt.savefig('1_matriz_correlacao_bruta.png')
    plt.close()
    print("Matriz de correlação salva em '1_matriz_correlacao_bruta.png'")

def visualizar_distribuicao_falhas(df):
    """
    Gera um gráfico de barras da distribuição dos tipos de falha.
    """
    print("\n--- VISUALIZANDO DISTRIBUIÇÃO DAS FALHAS ---")
    falhas = ['FDF', 'FDC', 'FP', 'FTE', 'FA', 'falha_maquina']
    falhas_existentes = [f for f in falhas if f in df.columns]
    
    if not falhas_existentes:
        print("Nenhuma coluna de falha encontrada para plotar.")
        return
        
    df_falhas = df[falhas_existentes].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df_falhas.plot(kind='bar', ax=ax, color='lightcoral')
    
    ax.set_title('Distribuição de Casos Positivos por Tipo de Falha')
    ax.set_xlabel('Tipo de Falha')
    ax.set_ylabel('Número de Ocorrências')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('1_distribuicao_falhas.png')
    plt.close(fig)
    print("Gráfico de distribuição de falhas salvo em '1_distribuicao_falhas.png'")

def main():
    df = carregar_dados()
    if df is not None:
        df_com_falhas = converter_colunas_falhas(df)
        df_analisado = analise_inicial(df_com_falhas)
        
        visualizar_distribuicao_features(df_analisado)
        visualizar_matriz_correlacao_bruta(df_analisado)
        visualizar_distribuicao_falhas(df_analisado)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        caminho_saida = os.path.join(script_dir, '1_df_analise_inicial.pkl')
        df_analisado.to_pickle(caminho_saida)
        print(f"DataFrame inicial salvo em '{caminho_saida}'")
        print("\n--- ETAPA 1 CONCLUÍDA! PRÓXIMO PASSO: '2_preparacao.py' ---")

        profile = ProfileReport(
            df_analisado,
            title='EDA Report - Sistema de Manutenção Preditiva',
            explorative=True,
            minimal=False
        )

        caminho_report = os.path.join(script_dir, "1_profiling_report.html")
        profile.to_file(caminho_report)
        print(f"Relatório de profiling salvo em '{caminho_report}'")

if __name__ == "__main__":
    main()
    
'''

### Considerações do dataset

- Correlação e Redundância: Observamos uma alta correlação entre pares de variáveis operacionais, 
como temperatura_ar e temperatura_processo, e torque e velocidade_rotacional. Isso indica uma possível 
multicolinearidade, um desafio estatístico que exige atenção na seleção de features para garantir 
que os modelos não sejam prejudicados por informações redundantes.

- Desbalanceamento Extremo: O conjunto de dados apresenta um desbalanceamento severo, 
onde as classes de falha (FDF, FDC, FP, FTE, FA) são eventos extremamente raros, com taxas de 
ocorrência inferiores a 6%. Tal característica requer a aplicação de técnicas avançadas de balanceamento 
de classes e métricas de avaliação robustas para evitar que o modelo classifique todas as amostras como 0 (sem falha).

- Multi-Label: A natureza multi-label do problema, em que várias falhas podem ocorrer simultaneamente,
exige uma abordagem cuidadosa na modelagem. Técnicas como Classifier Chains ou algoritmos específicos para exige a construção de modelos especializados por label ou o uso de arquiteturas de modelos multi-saída para capturar
a interdependência entre os diferentes tipos de falha.

- Qualidade dos Dados: A presença de valores ausentes em variáveis-chave como desgaste_da_ferramenta,
 velocidade_rotacional e as variáveis de temperatura, embora em pequena proporção (1.7% a 2.7%), é um ponto crítico. 
O tratamento adequado desses valores é fundamental para evitar vieses no treinamento do modelo e garantir a 
integridade dos dados.

- Distribuição de Dados (Skewed Data): A variável umidade_relativa mostra uma distribuição altamente enviesada (Skewed). 
Em um contexto de modelagem, essa assimetria pode afetar a performance de algoritmos sensíveis à distribuição de dados, 
como regressão linear e alguns modelos de detecção de anomalias.

- Variáveis Categóricas e a Unicidade do ID: A variável id possui valores únicos e não contribui para a predição. 
Ela deve ser tratada como um identificador e removida antes da modelagem para evitar que o modelo memorize a 
amostra em vez de aprender os padrões.

- Z-scores e Anomalias: A alta importância das features com z-score (como temperatura_ar_zscore e torque_zscore) 
sugere que o modelo está aprendendo a detectar falhas com base no desvio padrão dos dados. 

- Feature desgaste_da_ferramenta: A presença de 758 valores zero nesta variável requer uma análise mais aprofundada 
para determinar se o valor zero representa ausência de desgaste ou erro de medição. A maneira como esses valores 
são tratados pode impactar a precisão do modelo na detecção de falhas por desgaste.
'''
