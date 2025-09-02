#!/usr/bin/env python3
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score, 
    recall_score, hamming_loss
)
from pathlib import Path
import joblib
import os
import warnings
from sklearn.metrics import roc_curve, precision_recall_curve, auc
warnings.filterwarnings('ignore')

# Configurar visualizações
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def carregar_modelos_e_dados(caminho_modelos, caminho_scaler):
    """Carrega modelos e scaler da etapa anterior."""
    print("--- INICIANDO ETAPA 4: AVALIAÇÃO ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    caminho_modelos_completo = os.path.join(script_dir, caminho_modelos)
    caminho_scaler_completo = os.path.join(script_dir, caminho_scaler)
    caminho_novos_dados = os.path.join(script_dir, "novos_dados_teste.csv")

    if not Path(caminho_modelos_completo).exists() or not Path(caminho_scaler_completo).exists():
        print(f"Erro: Arquivo '{caminho_modelos_completo}' ou '{caminho_scaler_completo}' não encontrado.")
        print("Certifique-se de executar a Etapa 3 primeiro.")
        return None, None, None, None, None
        
    dados_treinamento = joblib.load(caminho_modelos_completo)
    scaler = joblib.load(caminho_scaler_completo)
    
    modelos = dados_treinamento['modelos']
    targets = dados_treinamento['targets']
    
    if Path(caminho_novos_dados).exists():
        print("Carregando novos dados para avaliação...")
        df_novos_dados = pd.read_csv(caminho_novos_dados)

        falhas_presentes = [t for t in targets if t in df_novos_dados.columns]
        if falhas_presentes:
            for falha in falhas_presentes:
                if df_novos_dados[falha].dtype in ['object', 'bool']:
                    df_novos_dados[falha] = df_novos_dados[falha].astype(int)

        X_test = df_novos_dados.drop(columns=falhas_presentes, errors='ignore')
        y_test = df_novos_dados[falhas_presentes]
    else:
        print("Arquivo 'novos_dados_teste.csv' não encontrado. Usando dados originais de teste.")
        X_test = dados_treinamento['X_test']
        y_test = dados_treinamento['y_test']

    print(f"Modelos e dados carregados com sucesso.")
    return X_test, y_test, modelos, targets, scaler

def gerar_predicoes_com_probabilidade(X_test, modelos, scaler):
    """Gera predições de probabilidade para cada modelo especializado."""
    print("\n--- GERANDO PREDIÇÕES DE PROBABILIDADE ---")
    
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    y_proba = pd.DataFrame(index=X_test.index)
    
    for target, modelo in modelos.items():
        if hasattr(modelo, 'predict_proba'):
            y_proba[target] = modelo.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba[target] = modelo.predict(X_test_scaled)
        print(f"   -> Probabilidades para '{target}' geradas.")
        
    print(f"Predições de probabilidade geradas para todos os targets.")
    return y_proba

def otimizar_thresholds(y_test, y_proba, targets):
    """Otimiza o threshold para cada target para maximizar o F1-Score."""
    print("\n--- OTIMIZANDO THRESHOLDS ---")
    thresholds_otimizados = {}
    
    for target in targets:
        y_true = y_test[target]
        y_prob = y_proba[target]
        
        # Gerar a série de thresholds apenas se houverem duas classes
        if len(y_true.unique()) > 1:
            thresholds = np.arange(0.01, 0.99, 0.01)
            f1_scores = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
            
            melhor_f1_score = max(f1_scores)
            melhor_threshold = thresholds[np.argmax(f1_scores)]
            
            thresholds_otimizados[target] = melhor_threshold
            print(f"   -> '{target}': Melhor Threshold = {melhor_threshold:.2f} (F1-Score: {melhor_f1_score:.4f})")
        else:
            thresholds_otimizados[target] = 0.5 # Default
            print(f"   -> '{target}': Apenas uma classe encontrada, usando threshold padrão de 0.5.")
    
    return thresholds_otimizados

def gerar_predicoes_binarias(y_proba, thresholds_otimizados):
    """Converte probabilidades em predições binárias usando os thresholds otimizados."""
    print("\n--- GERANDO PREDIÇÕES BINÁRIAS COM THRESHOLDS OTIMIZADOS ---")
    y_pred = pd.DataFrame(index=y_proba.index)
    
    for target, threshold in thresholds_otimizados.items():
        y_pred[target] = (y_proba[target] > threshold).astype(int)
    
    print("Predições binárias geradas.")
    return y_pred

def avaliar_metrica_multilabel(y_test, y_pred, targets):
    """Calcula e exibe métricas de avaliação multi-label."""
    print("\n--- AVALIANDO MÉTRICAS MULTI-LABEL ---")
    
    print("\n📋 Relatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred, target_names=targets, zero_division=0))
    
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    hamming = hamming_loss(y_test, y_pred)
    
    print("Métricas Globais:")
    print(f"   -> F1-Score (Macro): {f1_macro:.4f}")
    print(f"   -> F1-Score (Micro): {f1_micro:.4f}")
    print(f"   -> Hamming Loss: {hamming:.4f}")
    
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'hamming_loss': hamming
    }

def analisar_matriz_confusao(y_test, y_pred, targets, script_dir):
    """Gera e salva matrizes de confusão para cada target."""
    print("\n--- ANALISANDO MATRIZES DE CONFUSÃO ---")
    n_targets = len(targets)
    cols = min(3, n_targets)
    rows = (n_targets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_targets > 0:
        axes = axes.ravel() if n_targets > 1 else [axes]
    
        for i, target in enumerate(targets):
            y_true_target = y_test[target]
            y_pred_target = y_pred[target]
            
            if len(y_true_target.unique()) > 1:
                cm = confusion_matrix(y_true_target, y_pred_target)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Não', 'Sim'], yticklabels=['Não', 'Sim'], ax=axes[i])
                axes[i].set_title(f'Matriz de Confusão - {target}')
                axes[i].set_xlabel('Predito')
                axes[i].set_ylabel('Real')
            else:
                axes[i].set_title(f'Matriz de Confusão - {target}')
                axes[i].text(0.5, 0.5, 'Dados sem falha', ha='center', va='center')
                axes[i].set_xticks([])
                axes[i].set_yticks([])

    plt.tight_layout()
    caminho_imagem = os.path.join(script_dir, '4_matrizes_confusao.png')
    plt.savefig(caminho_imagem)
    print(f"✅ Matrizes de confusão salvas em '{caminho_imagem}'")
    plt.close(fig) 


def visualizar_metricas_por_falha(metricas, targets, script_dir):
    """
    Gera um gráfico de barras comparando as métricas por tipo de falha.
    """
    print("\n--- VISUALIZANDO MÉTRICAS POR FALHA ---")
    
    metricas_df = pd.DataFrame(columns=['Falha', 'Métrica', 'Valor'])
    for target in targets:
        precision = precision_score(y_test[target], y_pred[target], zero_division=0)
        recall = recall_score(y_test[target], y_pred[target], zero_division=0)
        f1 = f1_score(y_test[target], y_pred[target], zero_division=0)
        
        metricas_df.loc[len(metricas_df)] = [target, 'Precision', precision]
        metricas_df.loc[len(metricas_df)] = [target, 'Recall', recall]
        metricas_df.loc[len(metricas_df)] = [target, 'F1-Score', f1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Falha', y='Valor', hue='Métrica', data=metricas_df, ax=ax)
    
    ax.set_title('Comparativo de Métricas por Tipo de Falha')
    ax.set_xlabel('Tipo de Falha')
    ax.set_ylabel('Valor da Métrica')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('4_metricas_por_falha.png')
    plt.close(fig)
    print("Gráfico de métricas por falha salvo em '4_metricas_por_falha.png'")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    caminho_modelos = "3_modelos_treinados.pkl"
    caminho_scaler = "3_standard_scaler.pkl"

    X_test, y_test, modelos, targets, scaler = carregar_modelos_e_dados(caminho_modelos, caminho_scaler)
    if X_test is None or y_test is None:
        return

    y_proba = gerar_predicoes_com_probabilidade(X_test, modelos, scaler)
    
    thresholds_otimizados = otimizar_thresholds(y_test, y_proba, targets)
    
    y_pred = gerar_predicoes_binarias(y_proba, thresholds_otimizados)
    
    metricas = avaliar_metrica_multilabel(y_test, y_pred, targets)
    
    analisar_matriz_confusao(y_test, y_pred, targets, script_dir)
    
    print("\n--- ETAPA 4 CONCLUÍDA! PRÓXIMO PASSO: '5_deploy.py' ---")

if __name__ == "__main__":
    main()