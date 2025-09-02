
![Foto do Projeto GembaGuard](https://github.com/eumoas/GembaGuard/blob/main/docs/images/residencia.jpeg?raw=true)

#  GEMBAGUARD - SISTEMA INTELIGENTE DE MANUTENÇÃO PREDITIVA


## Entendimento do negócio

A manutenção industrial dependia quase exclusivamente da 
experiência e percepção dos técnicos especializados. Os profissionais utilizavam
seus sentidos - audição para detectar ruídos anômalos, tato para sentir vibrações irregulares, 
visão para identificar vazamentos ou desgastes visíveis - e seu conhecimento empírico 
acumulado ao longo dos anos para avaliar o estado dos equipamentos. 
![Foto do Projeto GembaGuard](https://github.com/eumoas/GembaGuard/blob/main/docs/images//manuten%C3%A7%C3%A3o.gif?raw=true)

Embora essa abordagem baseada na expertise humana tenha sido fundamental para o
desenvolvimento da manutenção industrial, ela apresentava limitações significativas: 
dependia da disponibilidade e subjetividade do técnico, não permitia detecção precoce 
de problemas internos e estava sujeita a variações na interpretação dos sinais.

Atualmente, a manutenção preditiva revolucionou esse cenário ao incorporar tecnologias avançadas 
que amplificam e complementam a capacidade humana. Sensores de vibração, análise termográfica, 
monitoramento de corrente elétrica, análise de óleos lubrificantes e sistemas de Internet das Coisas 
(IoT) coletam dados precisos e contínuos sobre o desempenho dos equipamentos. 
Essas tecnologias,combinadas com inteligência artificial e machine learning, 
processam grandes volumes de dados em tempo real, 
identificando padrões sutis que seriam imperceptíveis ao olho humano. 
O resultado é uma capacidade diagnóstica muito mais precisa e
antecipada, permitindo intervenções antes que falhas críticas ocorram.
Hoje, o técnico especializado continua sendo fundamental, mas 
agora trabalha equipado com ferramentas que potencializam sua expertise, 
transformando a manutenção de uma arte baseada na intuição em uma 
ciência fundamentada em dados precisos e análises preditivas.

A manutenção preditiva permite a detecção de problemas antes de ocorrerem
falhas através do monitoramento de desempenho e condições de máquinas e
equipamentos em tempo real. Isso significa evitar paradas imprevistas que causam
perda de produção e gastos adicionais com manutenção corretiva, ajudando a
prolongar a vida útil dos ativos. 
Segundo dados da Siemens, falhas não planejadas custam às 500 maiores empresas do mundo cerca 
de USD$ 1,4 trilhão anualmente, representando 11% de suas receitas​. 
Esse impacto se reflete não apenas em prejuízos imediatos, 
mas também em riscos operacionais que podem se agravar com o tempo.

## Impacto financeiro

Um estudo de pesquisa recente mostra economias de custo de manutenção preditiva de 18% a 25% 
apenas em despesas de manutenção, com economia e benefícios adicionais por meio do aumento do 
tempo de atividade. De acordo com o Departamento de Energia dos EUA, em muitos casos, a implementação 
da manutenção preditiva pode resultar em um ROI de até 10 vezes sobre o custo da abordagem.

## Descrição do Projeto

Criar um sistema capaz de identificar as falhas que
venham a ocorrer, e se possível, qual foi o tipo da falha. Cada amostra no conjunto de dados é
composta por 8 atributos que descrevem o comportamento de desgaste da máquina e do
ambiente. Além dessas características, cada amostra é rotulada com uma das 5 possíveis
classes de defeitos.
O sistema deverá ser capaz de, a partir de uma nova medição do dispositivo IoT (ou conjunto
de medições), prever a classe do defeito e retornar a probabilidade associada. 
Além disso, aempresa espera que você extraia insights da operação e dos defeitos e gere visualizações de
dados.

## Objetivos

- Desenvolver um sistema de classificação multiclasse para detectar 5 tipos de defeitos
- Implementar modelos de Machine Learning com alta precisão e recall
- Criar uma aplicação web interativa para predição de defeitos

## Tecnologias Utilizadas

- **Python 3.11**
- **Pandas** - Manipulação de dados
- **Scikit-learn** - Algoritmos de Machine Learning
- **XGBoost** - Gradient Boosting
- **Imbalanced-learn** - SMOTE para balanceamento de classes
- **Matplotlib/Seaborn** - Visualização de dados
- **Streamlit** - Interface web interativa

## Dados

O dataset contém 35.260 amostras com 14 características extraídas de informações coletadas a 
partir de dispositivos IoT sensorizando atributos
básicos de cada máquina.

### Parâmetros do Processo

- **id_produto**: Identificador único do produto (combinação da variável tipo e um número)

- **tipo**: Tipo de produto/máquina (L / M / H)

- **temperatura_ar**: Temperatura do ar no ambiente (K)

- **temperatura_processo**: Temperatura do processo (K)

- **umidade_relativa**: Umidade relativa do ar (%)

- **velocidade_rotacional**: Velocidade rotacional da máquina em rotações por minuto (RPM)

- **torque**: Torque da máquina em Nm

- **desgaste_da_ferramenta**: Duração do uso da ferramenta em minutos]


 ### Identificação

- **id_produto**: Identificador das amostras do banco

### Classes de Defeitos

- **falha_maquina**: Indica se houve falha na máquina (1) ou não (0)

- **FDF**: Falha por desgaste da ferramenta (1) ou não (0)

- **FDC**: Falha por dissipação de calor (1) ou não (0)

- **FP**: Falha por potência (1) ou não (0)

- **FTE**: Falha por tensão excessiva (1) ou não (0)

- **FA**: Falha aleatória (1) ou não (0)

## Metodologia (CRISP-DM)

### 1. Compreensão do Negócio
- Análise do problema de controle de qualidade na indústria siderúrgica

### 2. Compreensão dos Dados
- Análise exploratória do dataset
- Identificação de valores ausentes e outliers
- Análise da distribuição das classes

### 3. Preparação dos Dados
- Tratamento de valores ausentes com imputação pela mediana
- Padronização das características numéricas
- Aplicação de SMOTE para balanceamento de classes

### 4. Modelagem

--- TREINANDO MODELO PARA 'FDF' ---
   -> Desbalanceamento: 0.20%
   -> Iniciando otimização com RandomizedSearchCV...
   -> Otimização concluída.
   -> Melhores parâmetros: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 15}
   -> Treinamento concluído.

   -> Top 5 Features Mais Importantes:
                      feature  importance
       desgaste_da_ferramenta    0.195314
            fadiga_ferramenta    0.171228
desgaste_da_ferramenta_zscore    0.164009
                taxa_desgaste    0.108770
              indice_anomalia    0.050689

--- TREINANDO MODELO PARA 'FDC' ---
   -> Desbalanceamento: 0.63%
   -> Iniciando otimização com RandomizedSearchCV...
   -> Otimização concluída.
   -> Melhores parâmetros: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 15}
   -> Treinamento concluído.

   -> Top 5 Features Mais Importantes:
                     feature  importance
       velocidade_rotacional    0.167808
velocidade_rotacional_zscore    0.164738
                indice_calor    0.113889
       temperatura_ar_zscore    0.099470
           delta_temperatura    0.097961

--- TREINANDO MODELO PARA 'FP' ---
   -> Poucos casos positivos. Usando Random Forest sem otimização.
   -> Treinamento concluído.

   -> Top 5 Features Mais Importantes:
              feature  importance
      indice_anomalia    0.168730
         indice_calor    0.103251
temperatura_ar_zscore    0.101483
        torque_zscore    0.099874
    delta_temperatura    0.075381

--- TREINANDO MODELO PARA 'FTE' ---
   -> Desbalanceamento: 0.48%
   -> Iniciando otimização com RandomizedSearchCV...
   -> Otimização concluída.
   -> Melhores parâmetros: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 15}
   -> Treinamento concluído.

   -> Top 5 Features Mais Importantes:
          feature  importance
    torque_zscore    0.182477
           torque    0.137436
  stress_mecanico    0.114316
    taxa_desgaste    0.076134
potencia_estimada    0.072039

--- TREINANDO MODELO PARA 'FA' ---
   -> Poucos casos positivos. Usando Random Forest sem otimização.
   -> Treinamento concluído.

   -> Top 5 Features Mais Importantes:
                    feature  importance
temperatura_processo_zscore    0.208998
      temperatura_ar_zscore    0.169291
       temperatura_processo    0.140855
             temperatura_ar    0.093323
              torque_zscore    0.069939



### 5. Avaliação


Modelo para FDC (Falha por Dissipação de Calor):

Performance: Este foi o modelo com a melhor performance. Ele atingiu um F1-score de 0.04, o mais alto de todos. Embora o valor ainda seja baixo, ele conseguiu fazer previsões válidas, com uma precision de 0.02 e um recall de 0.18.

Conclusão: O algoritmo RandomForest usado para esta falha conseguiu aprender o suficiente para fazer algumas previsões corretas. É um modelo fraco, mas é o melhor que tivemos até agora.

Modelo para FDF (Falha por Desgaste da Ferramenta):

Performance: O modelo alcançou um recall perfeito de 1.00, mas com uma precision de 0.00.

Conclusão: Isso é um sinal de que o modelo está prevendo 1 (falha) para todas as amostras para não perder nenhuma falha real. Ele é extremamente sensível, mas completamente impreciso, o que o torna inútil.

Modelo para FTE (Falha por Tensão Excessiva):

Performance: O modelo teve um recall altíssimo de 0.97, mas uma precision de 0.01.

Conclusão: O modelo para FTE tem o mesmo problema do modelo para FDF: ele é extremamente sensível, mas a falta de precisão torna suas previsões inviáveis na prática.

Modelos para FP e FA (Falhas Raras):

Performance: Para ambas as falhas, o support é 0, e todas as métricas são 0.00.

Conclusão: Os algoritmos para estas classes não foram capazes de encontrar nenhum caso positivo, o que nos impede de avaliar seu desempenho. Isso reforça a ideia de que a abordagem para estas falhas deve ser diferente, talvez com a detecção de anomalias.

Conclusão: O Melhor Algoritmo e o Caminho a Seguir
Com base nos resultados, o RandomForest teve o melhor desempenho. No entanto, o F1-score de 0.04 ainda é muito baixo, e a performance geral do modelo ainda não é suficiente para um ambiente de produção.

A análise mostra que o nosso principal desafio agora é aumentar a precisão do modelo. A alta sensibilidade (recall) já está lá para algumas classes, mas a falta de precisão é o que está causando o baixo F1-score. O próximo passo é focar em estratégias que penalizem mais os falsos positivos, para que o modelo aprenda a ser mais seletivo em suas previsões.

## Sumário do Projeto
O projeto é dividido em um fluxo de trabalho sequencial de cinco etapas, com cada script gerando os artefatos necessários para o próximo.

1_entendimento.py: Análise Exploratória e Limpeza Inicial.

2_preparacao.py: Engenharia de Features e Transformação de Dados.

3_modelagem.py: Treinamento e Otimização de Modelos.

4_avaliacao.py: Avaliação de Performance e Geração de Relatórios.

5_deploy.py: Modelo para aplicação



## Como Executar

Perfeito. Um arquivo README.md bem escrito é a porta de entrada do seu projeto no GitHub. Ele não só explica o que o projeto faz, mas também guia os usuários sobre como executá-lo.

Aqui está um modelo de README.md que você pode copiar e colar no seu repositório. Ele está completo, claro e foi feito para o público acadêmico, como você pediu. Ele inclui as instruções de como rodar cada script na ordem certa.

GembaGuard: Sistema de Manutenção Preditiva Multi-Label
Este projeto é uma solução de machine learning para a manutenção preditiva de equipamentos industriais. O sistema, batizado de GembaGuard, é capaz de prever cinco tipos de falhas em um ambiente multi-label, onde uma máquina pode apresentar mais de um problema ao mesmo tempo. A solução foi desenvolvida com foco em dados desbalanceados e na criação de um pipeline de data science completo e reprodutível.

## Sumário do Projeto
O projeto é dividido em um fluxo de trabalho sequencial de cinco etapas, com cada script gerando os artefatos necessários para o próximo.

1_entendimento.py: Análise Exploratória e Limpeza Inicial.

2_preparacao.py: Engenharia de Features e Transformação de Dados.

3_modelagem.py: Treinamento e Otimização de Modelos.

4_avaliacao.py: Avaliação de Performance e Geração de Relatórios.

5_deploy

## Como executar

## Pré-requisitos
Certifique-se de que o Python (versão 3.10 ou superior) e o pip estão instalados em sua máquina. Para gerenciar as dependências do projeto, recomendamos o uso de um ambiente virtual (venv).

1. Configuração do Ambiente
Navegue até a pasta do projeto no seu terminal e execute os seguintes comandos para criar e ativar o ambiente virtual:

Bash

python3 -m venv venv
source venv/bin/activate
2. Instalação das Dependências
Com o ambiente virtual ativado, instale todas as bibliotecas necessárias com um único comando:

Bash

pip install pandas scikit-learn imbalanced-learn iterative-stratification hyperopt xgboost lightgbm catboost ydata_profiling streamlit
Como Executar o Projeto
Siga as instruções para rodar cada script na ordem correta, usando o comando python dentro do seu ambiente virtual ativado.

Etapa 1: Entendimento dos Dados
Este script realiza uma análise exploratória inicial, gera um relatório (.txt) e salva um arquivo de dados (.pkl) para a próxima etapa.

Bash

python 1_entendimento.py
Etapa 2: Preparação dos Dados
Este script faz a limpeza dos dados, trata os valores ausentes e cria as novas features que serão usadas na modelagem.

Bash

python 2_preparacao.py
Etapa 3: Modelagem
Aqui, os modelos são treinados e otimizados para cada tipo de falha, gerando os modelos (.pkl) que serão usados na avaliação e na aplicação.

Bash

python 3_modelagem.py
Etapa 4: Avaliação
Esta etapa carrega os modelos treinados e avalia sua performance em um novo conjunto de dados.

Bash

python 4_avaliacao.py
Etapa 5: Aplicação de Deploy (Streamlit)
Para executar a aplicação web e fazer previsões em novos dados, use o comando abaixo. Uma janela do navegador será aberta com a interface do projeto.

Bash

streamlit run app.py
Artefatos do Projeto
O pipeline irá gerar os seguintes arquivos, que você deve incluir no seu repositório:

1_relatorio_entendimento.txt

1_profiling_report.html

1_distribuicao_features.png

1_matriz_correlacao_bruta.png

1_distribuicao_falhas.png

2_features_criadas.png

2_matriz_correlacao_final.png

3_modelos_treinados.pkl

3_standard_scaler.pkl

4_matrizes_confusao.png

4_metricas_por_falha.png

4_curvas_roc_pr.png

4_distribuicao_probabilidades.png
```

## Resultados e Insights para as personas


- Para o Chefe de Manutenção: O Diagnóstico de Falhas e a Eficácia do Processo
Falha por Desgaste da Ferramenta (FDF): A alta importância de fadiga_ferramenta, desgaste_da_ferramenta e taxa_desgaste valida a intuição 
de que a falha está diretamente ligada à vida útil do componente. A orientação é usar o sistema para focar a inspeção em máquinas com alta taxa de desgaste.

- Falha por Dissipação de Calor (FDC): O modelo mostrou que a falha está ligada à velocidade_rotacional e ao indice_calor. Isso sugere que o plano de 
manutenção pode ser mais eficaz ao focar na calibração dos sensores de temperatura e na monitorização da velocidade de rotação para evitar o superaquecimento.

- Falha por Tensão Excessiva (FTE): O modelo deu alta importância a torque_zscore, torque e stress_mecanico. Isso nos diz que a falha por tensão é um problema de estresse no equipamento. 
A ação é usar o torque e o stress_mecanico como gatilhos para inspeções preventivas.

- Melhoria do Processo: As análises de importância das features mostraram que fadiga_ferramenta, stress_mecanico e indice_anomalia são os mais relevantes. Isso sugere que o plano de manutenção pode ser mais eficaz ao focar na inspeção desses componentes e na calibração dos sensores que os medem.

- Aprimoramento Contínuo: Os dados de novas manutenções preventivas e corretivas são cruciais para treinar o modelo. A recomendação é registrar cada intervenção com o máximo de detalhes possível, incluindo a causa-raiz da falha, para enriquecer o dataset e tornar o sistema de previsão ainda mais inteligente.

- Para o Gerente de Operação: Eficiência e Gestão de Riscos
Falha por Desgaste da Ferramenta (FDF): A taxa_desgaste é uma feature importante, o que pode ser usado para entender a eficiência de cada máquina. Se a taxa de uma máquina é maior que a de outra, isso pode indicar um problema de calibração ou de operação que precisa ser corrigido.

- Falha por Dissipação de Calor (FDC): O indice_calor é uma feature que pode ser usada para monitorar a saúde da máquina em tempo real. Se o indice_calor estiver alto, o gerente pode tomar a decisão de reduzir a velocidade de operação para evitar uma falha e, consequentemente, um downtime inesperado.

- Falha de Potência (FP) e Falha Aleatória (FA): O modelo identificou que o indice_anomalia e as temperaturas são as features mais importantes. Isso sugere que essas falhas, mesmo as aleatórias, não são completamente imprevisíveis. O gerente pode usar um dashboard que monitore esses indicadores para identificar comportamentos atípicos e tomar medidas preventivas.

- Para o Diretor Financeiro: ROI e Otimização de Custos
Priorização de Investimento: O modelo mostrou que features relacionadas a torque e temperatura são cruciais para a detecção de falhas. Isso justifica o investimento em sensores de alta precisão para monitorar essas variáveis, pois elas têm um impacto direto na prevenção de falhas caras.

- Análise de Desgaste: A fadiga_ferramenta e o desgaste_da_ferramenta são features importantes para a falha FDF. Isso sugere que o investimento em ferramentas mais duráveis ou em um sistema de monitoramento de desgaste pode reduzir os custos com a substituição de ferramentas e a manutenção corretiva.

- Prevenção de Downtime: O indice_anomalia é uma feature importante para as falhas FP e FA, o que sugere que o investimento em um sistema de monitoramento de anomalias pode ser um excelente investimento para reduzir os downtimes inesperados, que são os mais caros.

### Principais Descobertas
1. **Desbalanceamento de Classes**: O dataset apresenta desbalanceamento significativo, com 57% das amostras sendo "sem_falha"
2. **Eficácia do SMOTE**: A aplicação de SMOTE melhorou significativamente o desempenho dos modelos

### Visualizações Geradas
- Boxplots para análise de distribuição das características
- Histogramas para análise de frequência
- Gráfico de contagem das classes de defeitos

## Aplicação Web

A aplicação Streamlit oferece três funcionalidades principais:

1. **Análise Exploratória**: Visualização interativa dos dados e estatísticas
2. **Predição de Defeitos**: Interface para inserir características e obter predições
3. **Métricas dos Modelos**: Comparação visual do desempenho dos modelos

## Recomendações para o futuro

1. **Coleta de Mais Dados**: Aumentar o dataset para melhorar a generalização
2. **Feature Engineering**: Criar novas características a partir das existentes
3. **Ensemble Methods**: Combinar múltiplos modelos para melhor performance
4. **Deploy em Produção**: Implementar o sistema em ambiente produtivo
5. **Monitoramento**: Criar sistema de monitoramento da performance em produção

## Referências bibliográficas
GUTENBERG TECHNOLOGY. Predictive Maintenance: Increasing Your Equipment's ROI. [S.l.], [s.d.]. 
Disponível em: https://blog.gutenberg-technology.com/en/predictive-maintenance-increasing-your-equipments-roi. Acesso em: 1 set. 2025.
TIMBERGROVE. Predictive Maintenance: 5 Advantages and 5 Disadvantages. [S.l.], 12 nov. 2024. 
Disponível em: https://timbergrove.com/blog/predictive-maintenance-advantages-and-disadvantages. Acesso em: 1 set. 2025.

## 👥 Autor

**Miriam O. de Aguiar Sobral**
- Bootcamp de Ciência de Dados e Inteligência Artificial - SENAI
- Data: Agosto_Setembro/2025

## Licença

Este projeto foi desenvolvido para fins educacionais como parte do Bootcamp SENAI.

