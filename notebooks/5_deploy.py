#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

def carregar_artefatos_deploy():
    """Carrega modelos e scaler necessários para o deploy."""
    print("--- INICIANDO ETAPA 5: DEPLOY E PREDIÇÃO ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    caminho_modelos = os.path.join(script_dir, "3_modelos_treinados.pkl")
    caminho_scaler = os.path.join(script_dir, "3_standard_scaler.pkl")

    if not Path(caminho_modelos).exists() or not Path(caminho_scaler).exists():
        print("Erro: Arquivos de modelo não encontrados.")
        print("Certifique-se de executar as Etapas 1, 2 e 3 em ordem.")
        return None, None
    
    modelos_e_dados = joblib.load(caminho_modelos)
    scaler = joblib.load(caminho_scaler)
    
    modelos = modelos_e_dados['modelos']
    targets = modelos_e_dados['targets']
    features = modelos_e_dados['features']
    
    print(f"Artefatos de deploy carregados com sucesso.")
    return modelos, scaler, features, targets

def simular_novos_dados(features):
    """Simula a entrada de novos dados de sensores para teste."""
    print("\n--- SIMULANDO NOVOS DADOS ---")
    
    novos_dados = pd.DataFrame([
        {
            'temperatura_ar': 301.5,
            'temperatura_processo': 310.2,
            'umidade_relativa': 55.0,
            'velocidade_rotacional': 1500,
            'torque': 45.0,
            'desgaste_da_ferramenta': 110,
            'potencia_estimada': 67500,
            'delta_temperatura': 8.7,
            'densidade_potencia': 224.0,
            'taxa_desgaste': 0.073,
            'fadiga_ferramenta': 1.15,
            'indice_calor': 13.4,
            'stress_mecanico': 30.0,
            'indice_vibracao': 821.6,
            'tensao_estrutural': 1.25,
            'instabilidade': 0.0006,
            'termo_mecanico': 261.0,
            'eficiencia_global': 5000,
            'fator_risco': 1.5,
            'indice_anomalia': 2.0
        }
    ])
    
    for feature in features:
        if feature not in novos_dados.columns:
            novos_dados[feature] = 0
            
    print("Novos dados simulados com sucesso.")
    return novos_dados[features]

def prever_falhas(modelos, scaler, novos_dados):
    """Faz a predição usando os modelos treinados."""
    print("\n--- FAZENDO PREDIÇÕES ---")
    
    novos_dados_scaled = pd.DataFrame(scaler.transform(novos_dados), columns=novos_dados.columns)
    
    predicoes = {}
    
    for target, modelo in modelos.items():
        if hasattr(modelo, 'predict_proba'):
            proba = modelo.predict_proba(novos_dados_scaled)[:, 1][0]
        else:
            proba = modelo.predict(novos_dados_scaled)[0]
        
        predicoes[target] = proba
        
    print("Predições de probabilidade geradas.")
    return predicoes

def interpretar_predicoes(predicoes, targets):
    """Interpreta as predições e exibe um relatório."""
    print("\n--- RELATÓRIO DE PREDIÇÃO ---")
    
    tem_falha = False
    
    for target in targets:
        probabilidade = predicoes.get(target, 0)
        
        if probabilidade > 0.5:
            status = "ATENÇÃO"
            tem_falha = True
        else:
            status = "OK"
            
        print(f"[{status}] - Falha de {target}: Probabilidade = {probabilidade:.2f}%")
        
    if tem_falha:
        print("\nO sistema detectou uma ou mais falhas potenciais. Recomenda-se uma verificação imediata.")
    else:
        print("\n O sistema está operando normalmente. Nenhuma falha detectada.")

def main():
    modelos, scaler, features, targets = carregar_artefatos_deploy()
    
    if modelos is not None:
        novos_dados = simular_novos_dados(features)
        predicoes = prever_falhas(modelos, scaler, novos_dados)
        interpretar_predicoes(predicoes, targets)
        
    print("\n--- ETAPA 5 CONCLUÍDA! FIM DO PROJETO. ---")

if __name__ == "__main__":
    main()

    ## Conclusão e orientações para o Futuro

'''
- Para o futuro, o foco deve ser em estratégias que melhorem a precisão e permitam que o modelo aprenda 
com as falhas mais raras. Para tal, as seguintes orientações podem ser consideradas:

- Enriquecimento do Dataset: A maior limitação é a escassez de dados de falha. Para melhorar a performance,
 seria fundamental buscar mais informações de outras manutenções, como dados históricos de falhas preventivas
e corretivas, e a sua relação com os dados de sensores.

- Engenharia de Features Baseada em Tempo: É possível aprimorar os dados com a criação de features baseadas em
 tempo. Por exemplo, pode-se criar uma feature que mostre a quantidade de horas que a máquina operou desde a 
 última manutenção. Isso forneceria ao modelo um contexto de tempo mais robusto para previsões mais precisas.

Abordagem de Detecção de Anomalias: Para as falhas FP e FA, que o modelo não consegue prever, a classificação 
tradicional não é a melhor abordagem. Uma estratégia seria redefinir o problema para essas falhas como uma detecção 
de anomalias. Para isso, é possível treinar um modelo para reconhecer o que é uma operação "normal" e, em seguida, 
sinalizar qualquer comportamento que seja fora desse padrão como uma possível falha.

-Ensemble e Stacking de Modelos: Uma outra melhoria significativa seria o uso de um Ensemble, combinando as previsões 
de múltiplos modelos. Para isso, seria necessário treinar vários modelos para a mesma falha e, em seguida,
 usar um modelo de stacking para combinar as previsões e gerar um resultado final mais robusto e menos propenso a erros.


'''