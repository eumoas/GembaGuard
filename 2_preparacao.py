#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def carregar_dados_etapa_anterior(caminho):
    """Carrega o DataFrame da etapa anterior."""
    print("--- INICIANDO ETAPA 2: PREPARAÇÃO DOS DADOS ---")
    if not Path(caminho).exists():
        print(f"Erro: O arquivo '{caminho}' não foi encontrado.")
        print("Certifique-se de executar a Etapa 1 primeiro.")
        return None
    
    df = pd.read_pickle(caminho)
    print(f"Dados carregados com sucesso. Dimensões: {df.shape}")
    return df

def limpar_dados(df):
    """Limpeza básica dos dados: nulos, outliers e inconsistências físicas."""
    print("\n--- LIMPEZA DOS DADOS ---")
    df_limpo = df.copy()

    features_numericas = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa',
                          'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']
    
    print("1. Tratando valores nulos...")
    for feature in features_numericas:
        if df_limpo[feature].isnull().sum() > 0:
            mediana = df_limpo[feature].median()
            df_limpo[feature].fillna(mediana, inplace=True)
            print(f"   -> Nulos em '{feature}' preenchidos com a mediana ({mediana:.2f})")
    print("   -> Tratamento de nulos concluído.")

    print("\n2. Tratando outliers extremos...")
    for feature in features_numericas:
        Q1 = df_limpo[feature].quantile(0.25)
        Q3 = df_limpo[feature].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 3 * IQR
        limite_superior = Q3 + 3 * IQR
        outliers_antes = ((df_limpo[feature] < limite_inferior) | (df_limpo[feature] > limite_superior)).sum()
        if outliers_antes > 0:
            df_limpo[feature] = np.clip(df_limpo[feature], limite_inferior, limite_superior)
            print(f"   -> {outliers_antes} outliers em '{feature}' tratados.")
    print("   -> Tratamento de outliers concluído.")

    print("\n3. Validando consistência física...")
    inconsistencias = (df_limpo['temperatura_processo'] < df_limpo['temperatura_ar']).sum()
    if inconsistencias > 0:
        mask = df_limpo['temperatura_processo'] < df_limpo['temperatura_ar']
        df_limpo.loc[mask, 'temperatura_processo'] = df_limpo.loc[mask, 'temperatura_ar'] + 1
        print(f"   -> {inconsistencias} inconsistências de temperatura corrigidas.")
    
    negativos_desgaste = (df_limpo['desgaste_da_ferramenta'] < 0).sum()
    if negativos_desgaste > 0:
        df_limpo['desgaste_da_ferramenta'] = np.maximum(df_limpo['desgaste_da_ferramenta'], 0)
        print(f"   -> {negativos_desgaste} valores negativos de desgaste corrigidos.")
    
    print("   -> Validação física concluída.")
    print(f"\nLimpeza e tratamento concluídos. Dimensões finais: {df_limpo.shape}")
    return df_limpo

def criar_features_avancadas(df):
    """Cria features avançadas baseadas no conhecimento do domínio."""
    print("\n--- ENGENHARIA DE FEATURES ---")
    df_features = df.copy()

    def divisao_segura(numerador, denominador, default=0.001):
        denominador_safe = np.where(denominador == 0, default, denominador)
        return numerador / denominador_safe

    df_features['potencia_estimada'] = df_features['torque'] * df_features['velocidade_rotacional']
    df_features['delta_temperatura'] = df_features['temperatura_processo'] - df_features['temperatura_ar']
    df_features['densidade_potencia'] = divisao_segura(df_features['potencia_estimada'], df_features['temperatura_ar'])

    df_features['taxa_desgaste'] = divisao_segura(df_features['desgaste_da_ferramenta'], df_features['velocidade_rotacional'])
    df_features['fadiga_ferramenta'] = np.power(df_features['desgaste_da_ferramenta'] + 1, 1.2)
    df_features['indice_calor'] = df_features['delta_temperatura'] * (1 + df_features['umidade_relativa'] / 100)
    df_features['stress_mecanico'] = divisao_segura(df_features['torque'], df_features['velocidade_rotacional']) * 1000

    features_base = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa', 'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']
    for feature in features_base:
        if feature in df_features.columns:
            media_por_tipo = df_features.groupby('tipo')[feature].transform('mean')
            std_por_tipo = df_features.groupby('tipo')[feature].transform('std')
            std_por_tipo = np.where(std_por_tipo == 0, 1, std_por_tipo)
            df_features[f'{feature}_zscore'] = ((df_features[feature] - media_por_tipo) / std_por_tipo)
    
    zscore_cols = [col for col in df_features.columns if '_zscore' in col]
    if zscore_cols:
        df_features['indice_anomalia'] = np.sqrt((df_features[zscore_cols] ** 2).sum(axis=1) / len(zscore_cols))

    print(f"Engenharia de features concluída. Novas colunas: {df_features.shape[1] - df.shape[1]}")
    return df_features

def visualizar_features_criadas(df_original, df_features):
    """
    Compara a distribuição de features originais e criadas.
    """
    print("\n--- VISUALIZANDO FEATURES CRIADAS ---")
    
    features_criadas = [col for col in df_features.columns if col not in df_original.columns and col not in ['id', 'id_produto', 'falha_maquina', 'tipo']]
    features_a_plotar = features_criadas[:6]
    
    if not features_a_plotar:
        print("Nenhuma feature nova encontrada para plotar.")
        return
        
    n_features = len(features_a_plotar)
    cols = 3
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_features > 0:
        axes = axes.ravel() if n_features > 1 else [axes]
    
        for i, feature in enumerate(features_a_plotar):
            sns.histplot(df_features[feature], bins=50, kde=True, ax=axes[i], color='teal')
            axes[i].set_title(f'Distribuição da Feature: {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequência')
        
    plt.tight_layout()
    plt.savefig('2_features_criadas.png')
    plt.close(fig)
    print("Gráfico de features criadas salvo em '2_features_criadas.png'")

def visualizar_matriz_correlacao_final(df):
    """
    Gera uma matriz de correlação das features finais.
    """
    print("\n--- VISUALIZANDO MATRIZ DE CORRELAÇÃO FINAL ---")
    features_numericas = df.select_dtypes(include=np.number).columns.tolist()
    features_a_plotar = [f for f in features_numericas if f not in ['id'] and 'falha' not in f]
    
    corr_matrix = df[features_a_plotar].corr()
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de Correlação das Features Finais')
    plt.tight_layout()
    plt.savefig('2_matriz_correlacao_final.png')
    plt.close()
    print("Matriz de correlação final salva em '2_matriz_correlacao_final.png'")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    caminho_entrada = os.path.join(script_dir, "1_df_analise_inicial.pkl")
    caminho_saida = os.path.join(script_dir, "2_df_preparado.pkl")
    
    df = carregar_dados_etapa_anterior(caminho_entrada)
    if df is not None:
        df_limpo = limpar_dados(df)
        df_final = criar_features_avancadas(df_limpo)
        
        visualizar_features_criadas(df, df_final)
        visualizar_matriz_correlacao_final(df_final)

        df_final.to_pickle(caminho_saida)
        print(f"\n DataFrame preparado salvo em '{caminho_saida}'")
        print("\n--- ETAPA 2 CONCLUÍDA! PRÓXIMO PASSO: '3_modelagem.py' ---")

if __name__ == "__main__":
    main()




    '''
### Considerações da preparação


- Tratamento de Dados Faltantes: A decisão de preencher os valores ausentes com a mediana foi estratégica. 
Em um conjunto de dados com outliers, a mediana é uma medida mais robusta que a média, pois não é sensível a 
valores extremos. Isso garante que a imputação dos dados não introduza um viés significativo nos modelos.

- Detecção e Ajuste de Outliers: A análise de outliers foi realizada utilizando o Intervalo Interquartil (IQR), 
com um fator de 3x para identificar e ajustar apenas os valores mais extremos. Esta abordagem, mais conservadora 
que o fator padrão de 1.5x, evita a perda de dados raros, que podem ser eventos importantes, como picos de temperatura
 associados a uma falha.

- Validação de Consistência Física: A correção de inconsistências lógicas, como garantir que a temperatura_processo 
seja sempre maior que a temperatura_ar, demonstra um cuidado fundamental com a qualidade dos dados. Esta validação 
previne que o modelo aprenda padrões fisicamente impossíveis, o que aumenta a sua confiabilidade.

- Engenharia de Features - Potência Estimada: A criação da potencia_estimada a partir do torque e da velocidade_rotacional
 transformou duas variáveis correlacionadas em uma única feature de alto valor preditivo. Isso reduz a multicolinearidade
e fornece ao modelo um indicador direto do esforço da máquina, o que é crucial para a detecção de falhas de potência.

- Criação de Features de Z-Score: A normalização das principais features em z-scores permite que o modelo interprete 
cada valor não pelo seu valor absoluto, mas pelo seu desvio em relação à média do tipo de máquina. Isso é particularmente
 útil para a detecção de anomalias, pois um z-score alto sinaliza um comportamento atípico para um tipo específico de 
 equipamento.

- Indicadores de Stress e Fadiga: A criação de features como stress_mecanico e fadiga_ferramenta mostra uma aplicação 
do conhecimento de domínio. Em vez de deixar o modelo descobrir essas relações por conta própria, fornecemos a ele 
indicadores diretos de tensão e desgaste, o que acelera o aprendizado e aumenta a precisão.

- Análise de Multi-Label e Features: A etapa de feature engineering foi projetada para apoiar a natureza multi-label do 
problema. Ao criar features como indice_calor e taxa_desgaste, o sistema pode isolar e aprender os padrões que são 
únicos a cada tipo de falha, permitindo que os modelos especializados na Etapa 3 sejam mais eficazes.

'''