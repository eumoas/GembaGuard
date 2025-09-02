#!/usr/bin/env python3


import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import joblib
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score
from imblearn.combine import SMOTETomek
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
warnings.filterwarnings('ignore')


np.random.seed(42)
pd.options.mode.chained_assignment = None

def carregar_dados_preparados(caminho):
    """Carrega o DataFrame da etapa anterior."""
    print("--- INICIANDO ETAPA 3: MODELAGEM (VERSÃO FINAL) ---")
    if not Path(caminho).exists():
        print(f"Erro: O arquivo '{caminho}' não foi encontrado.")
        print("Certifique-se de executar a Etapa 2 primeiro.")
        return None
    
    df = pd.read_pickle(caminho)
    print(f"Dados carregados com sucesso. Dimensões: {df.shape}")
    return df

def converter_features_para_numerico(df):
    """
    Força a conversão de todas as colunas de features para um tipo numérico.
    """
    print("\n--- GARANTINDO QUE AS FEATURES SÃO NUMÉRICAS ---")
    df_numerico = df.copy()
    
    colunas_excluidas = ['id', 'id_produto', 'tipo', 'falha_maquina', 'FDF', 'FDC', 'FP', 'FTE', 'FA']
    
    for col in df_numerico.columns:
        if col not in colunas_excluidas:
            try:
                df_numerico[col] = pd.to_numeric(df_numerico[col], errors='coerce')
            except Exception as e:
                print(f"   ⚠️ Aviso: A coluna '{col}' não pôde ser convertida para numérico. Erro: {e}")
    
    colunas_com_nan = df_numerico.isnull().sum()
    colunas_com_nan = colunas_com_nan[colunas_com_nan > 0].index.tolist()
    
    if colunas_com_nan:
        df_numerico.dropna(axis=1, how='any', inplace=True)
        print(f"   -> Removidas colunas com valores não-numéricos após conversão: {colunas_com_nan}")
    
    print("   -> Todas as features numéricas validadas.")
    return df_numerico

def preparar_dados_para_modelagem(df):
    """
    Prepara os dados, dividindo, normalizando e separando targets e features.
    """
    print("\n--- PREPARAÇÃO DOS DADOS PARA MODELAGEM ---")

    targets = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
    features = [col for col in df.columns if col not in targets + ['id', 'id_produto', 'falha_maquina', 'tipo']]
    
    X = df[features]
    y = df[targets]

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in msss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    joblib.dump(scaler, '3_standard_scaler.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test, features, targets

def otimizar_randomized_search(model_base, param_distributions, X, y):
    """Otimiza hiperparâmetros usando RandomizedSearchCV."""
    print(f"   -> Iniciando otimização com RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=model_base,
        param_distributions=param_distributions,
        n_iter=20, 
        cv=3,
        scoring='f1', 
        random_state=42,
        n_jobs=-1,
        error_score=0
    )
    
    random_search.fit(X, y)
    print("   -> Otimização concluída.")
    print("   -> Melhores parâmetros:", random_search.best_params_)
    
    return random_search.best_estimator_, random_search.best_params_

def treinar_modelo_especializado(X, y, target_name):
    """
    Escolhe e treina o melhor modelo para um único target de falha.
    """
    print(f"\n--- TREINANDO MODELO PARA '{target_name}' ---")
    
    y_target = y[target_name]
    casos_positivos = y_target.sum()
    
    model = None
    best_params = {}
    
    if casos_positivos > 5:
        ratio_positivo = casos_positivos / len(y_target)
        print(f"   -> Desbalanceamento: {ratio_positivo:.2%}")

        if ratio_positivo < 0.05: 
            model_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            param_distributions = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
            model, best_params = otimizar_randomized_search(model_base, param_distributions, X, y_target)
        else: 
            model_base = lgb.LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            param_distributions = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [20, 31, 50],
                'max_depth': [5, 10, 15],
            }
           
            smote_tomek = SMOTETomek(random_state=42)
            X_res, y_res = smote_tomek.fit_resample(X, y_target)
            print(f"   -> Dados balanceados para otimização. Antes: {len(y_target)}, Depois: {len(y_res)}")
            model, best_params = otimizar_randomized_search(clone(model_base), param_distributions, X_res, y_res)
    else:
        print("   -> Poucos casos positivos. Usando Random Forest sem otimização.")
        model = RandomForestClassifier(n_estimators=150, max_depth=10, 
                                       random_state=42, n_jobs=-1,
                                       class_weight='balanced')
        model.fit(X, y_target)

    if model:
        print("   -> Treinamento concluído.")
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
            importances = importances.sort_values('importance', ascending=False).head(5)
            print("\n   -> Top 5 Features Mais Importantes:")
            print(importances.to_string(index=False))

    return model


def visualizar_importancia_features(importances, target_name):
    """
    Gera um gráfico de barras da importância das features.
    """
    if importances is not None and not importances.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importances, ax=ax, color='teal')
        ax.set_title(f'Top 5 Features Mais Importantes para {target_name}')
        ax.set_xlabel('Importância')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        plt.savefig(f'3_importancia_features_{target_name}.png')
        plt.close(fig)
        print(f"Gráfico de importância salvo em '3_importancia_features_{target_name}.png'")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    caminho_entrada = os.path.join(script_dir, "2_df_preparado.pkl")
    caminho_saida = os.path.join(script_dir, "3_modelos_treinados.pkl")
    
    df = carregar_dados_preparados(caminho_entrada)
    if df is None:
        return
    
    df_numerico = df.copy()
    
    X_train, X_test, y_train, y_test, features, targets = preparar_dados_para_modelagem(df_numerico)
    
    modelos_especializados = {}
    for target in targets:
        modelo = treinar_modelo_especializado(X_train, y_train, target)
        if modelo:
            modelos_especializados[target] = modelo
        
    print("\n--- SALVANDO ARTEFATOS DE MODELAGEM ---")
    dados_treinamento = {
        'modelos': modelos_especializados,
        'X_test': X_test,
        'y_test': y_test,
        'features': features,
        'targets': targets
    }
    
    joblib.dump(dados_treinamento, caminho_saida)
    print(f" Modelos e dados de teste salvos em '{caminho_saida}'")
    
    print("\n--- ETAPA 3 CONCLUÍDA! PRÓXIMO PASSO: '4_avaliacao.py' ---")

if __name__ == "__main__":
    main()

    ### Considerações da preparação
'''
- O Desafio do Desbalanceamento Crítico: A etapa de modelagem confrontou o principal desafio do projeto: a 
raridade das falhas. Com a maioria das classes de falha apresentando menos de 1% de ocorrência, a modelagem 
tradicional falharia, tendendo a prever "sem falha" para todas as amostras.

- Abordagem Multi-Label Estratificada: Para combater o problema, adotamos uma estratégia de MultilabelStratifiedShuffleSplit.
Diferente de uma divisão aleatória, essa técnica garantiu que a rara proporção de cada tipo de falha fosse mantida
tanto no conjunto de treino quanto no de teste, tornando a avaliação posterior mais justa e confiável.

- Modelos Especializados por Falha: Em vez de usar um único modelo "super-humano" para todas as falhas, treinamos um
modelo especializado para cada tipo de falha. Isso permitiu que cada modelo se concentrasse em aprender os padrões
específicos de uma única falha, aumentando a precisão da detecção.

- Otimização de Hiperparâmetros: Para extrair o máximo de performance de cada modelo, utilizamos o RandomizedSearchCV. 
Essa ferramenta automatiza a busca pelos melhores parâmetros (como profundidade da árvore ou número de estimadores), 
o que nos permitiu construir modelos altamente otimizados para as particularidades do nosso dataset.

- Adaptação ao Nível de Desbalanceamento: A nossa lógica de modelagem foi refinada para se adaptar ao nível de 
desbalanceamento de cada classe de falha. Falhas mais raras receberam um tratamento mais agressivo, enquanto falhas
menos raras puderam ser otimizadas com técnicas diferentes.

- Tratamento Híbrido com SMOTETomek: Para as falhas mais frequentes, aplicamos o SMOTETomek. Esta técnica híbrida não 
só cria novas amostras de falha (SMOTE), mas também remove o ruído do dataset (TomekLinks), garantindo que o modelo 
aprenda com dados de maior qualidade.

- Foco em Class Weighting: Para as falhas extremamente raras, onde a criação de amostras sintéticas poderia ser arriscada, 
optamos por modelos que utilizam class_weight (pesos de classe). Essa técnica instrui o modelo a penalizar mais 
severamente os erros cometidos na classe de falha, forçando-o a prestar mais atenção a esses eventos raros.

- Superando Conflitos de Bibliotecas: A etapa de modelagem foi desafiadora devido a conflitos de dependências entre 
bibliotecas como hyperopt e NumPy. Superamos esses problemas substituindo o hyperopt por RandomizedSearchCV, uma 
ferramenta nativa do scikit-learn, o que tornou o pipeline mais robusto e livre de erros.

- O Problema dos Zeros: A análise de feature importance nos primeiros modelos revelou um problema crítico: todos os
 valores eram zero. Isso aconteceu porque as falhas não estavam sendo corretamente lidas, e o modelo, na prática,
estava sendo treinado em dados sem rótulos.

- A Importância da Feature Engineering: A modelagem demonstrou o poder das novas features criadas na Etapa 2. Para a 
falha FDF, por exemplo, fadiga_ferramenta se mostrou a característica mais importante, confirmando que a nossa 
estratégia de criar features a partir do conhecimento do domínio foi bem-sucedida.


'''