
![Foto do Projeto GembaGuard](https://github.com/eumoas/GembaGuard/blob/main/docs/images/residencia.jpeg?raw=true)

#  GEMBAGUARD - SISTEMA INTELIGENTE DE MANUTENÇÃO PREDITIVA


## 📋 Entendimento do negócio

A manutenção preditiva permite a detecção de problemas antes de ocorrerem
falhas através do monitoramento de desempenho e condições de máquinas e
equipamentos em tempo real. Isso significa evitar paradas imprevistas que causam
perda de produção e gastos adicionais com manutenção corretiva, ajudando a
prolongar a vida útil dos ativos. 

## 📋 Descrição do Projeto


Este projeto implementa um sistema inteligente fazendo uso de técnicas de machine learning. Foi desenvolvido como parte do Bootcamp de Ciência de Dados e Inteligência Artificial do UniSenai. 



![Foto do Projeto GembaGuard](https://github.com/eumoas/GembaGuard/blob/main/docs/images//manuten%C3%A7%C3%A3o.gif?raw=true)



.

## 🎯 Objetivos

- Desenvolver um sistema de classificação multiclasse para detectar 7 tipos de defeitos
- Implementar modelos de Machine Learning com alta precisão e recall
- Criar uma aplicação web interativa para predição de defeitos
- Seguir a metodologia CRISP-DM para desenvolvimento estruturado

## 🔧 Tecnologias Utilizadas

- **Python 3.11**
- **Pandas** - Manipulação de dados
- **Scikit-learn** - Algoritmos de Machine Learning
- **XGBoost** - Gradient Boosting
- **Imbalanced-learn** - SMOTE para balanceamento de classes
- **Matplotlib/Seaborn** - Visualização de dados
- **Streamlit** - Interface web interativa

## 📊 Dataset

O dataset contém 35.260 amostras com 14 características extraídas de informações coletadas a partir de dispositivos IoT sensorizando atributos
básicos de cada máquina.

- **Características geométricas**: coordenadas, área, perímetro
- **Características de luminosidade**: valores mínimos, máximos e soma
- **Índices calculados**: orientação, bordas, variação
- **Parâmetros do processo**: temperatura, tipo de aço, espessura

### Classes de Defeitos
- **falha_3**: 649 amostras
- **falha_6**: 806 amostras  
- **sem_falha**: 1.935 amostras


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

- **id**: Identificador das amostras do banco

### Classes de Defeitos

- **falha_maquina**: Indica se houve falha na máquina (1) ou não (0)

- **FDF**: Falha por desgaste da ferramenta (1) ou não (0)

- **FDC**: Falha por dissipação de calor (1) ou não (0)

- **FP**: Falha por potência (1) ou não (0)

- **FTE**: Falha por tensão excessiva (1) ou não (0)

- **FA**: Falha aleatória (1) ou não (0)

## 🚀 Metodologia (CRISP-DM)

### 1. Compreensão do Negócio
- Análise do problema de controle de qualidade na indústria siderúrgica
- Definição de critérios de sucesso técnicos e de negócio

### 2. Compreensão dos Dados
- Análise exploratória do dataset
- Identificação de valores ausentes e outliers
- Análise da distribuição das classes

### 3. Preparação dos Dados
- Criação da coluna `falha_principal` para problema multiclasse
- Tratamento de valores ausentes com imputação pela mediana
- Padronização das características numéricas
- Aplicação de SMOTE para balanceamento de classes

### 4. Modelagem
Três algoritmos foram implementados e comparados:

#### Logistic Regression
- Accuracy: 73.01%
- Precision (Macro): 75.19%
- Recall (Macro): 77.87%
- F1-Score (Macro): 75.35%

#### Random Forest
- Accuracy: 78.47%
- Precision (Macro): 78.29%
- Recall (Macro): 78.01%
- F1-Score (Macro): 78.15%

#### XGBoost ⭐ **Melhor Modelo**
- Accuracy: 79.35%
- Precision (Macro): 79.10%
- Recall (Macro): 79.11%
- F1-Score (Macro): 79.10%

### 5. Avaliação
- Validação cruzada com 5 folds
- Foco em métricas Precision e Recall conforme solicitado
- XGBoost apresentou melhor desempenho geral

## 📁 Estrutura do Projeto

```
├── data_preparation.py          # Script de preparação dos dados
├── model_training.py           # Treinamento dos modelos ML
├── data_visualization.py       # Geração de visualizações
├── streamlit_app.py           # Aplicação web interativa
├── bootcamp_train.csv         # Dataset original
├── bootcamp_train_prepared.csv # Dataset processado
├── *.pkl                      # Modelos treinados salvos
├── *.png                      # Visualizações geradas
└── README.md                  # Este arquivo
```

## 🖥️ Como Executar

### 1. Preparação dos Dados
```bash
python data_preparation.py
```

### 2. Treinamento dos Modelos
```bash
python model_training.py
```

### 3. Geração de Visualizações
```bash
python data_visualization.py
```

### 4. Executar Aplicação Streamlit
```bash
streamlit run streamlit_app.py
```

## 📈 Resultados e Insights

### Principais Descobertas
1. **Desbalanceamento de Classes**: O dataset apresenta desbalanceamento significativo, com 57% das amostras sendo "sem_falha"
2. **Eficácia do SMOTE**: A aplicação de SMOTE melhorou significativamente o desempenho dos modelos
3. **Superioridade do XGBoost**: O modelo XGBoost apresentou melhor desempenho em todas as métricas

### Visualizações Geradas
- Boxplots para análise de distribuição das características
- Histogramas para análise de frequência
- Gráfico de contagem das classes de defeitos

## 🌐 Aplicação Web

A aplicação Streamlit oferece três funcionalidades principais:

1. **Análise Exploratória**: Visualização interativa dos dados e estatísticas
2. **Predição de Defeitos**: Interface para inserir características e obter predições
3. **Métricas dos Modelos**: Comparação visual do desempenho dos modelos

## 🔮 Próximos Passos

1. **Coleta de Mais Dados**: Aumentar o dataset para melhorar a generalização
2. **Feature Engineering**: Criar novas características a partir das existentes
3. **Ensemble Methods**: Combinar múltiplos modelos para melhor performance
4. **Deploy em Produção**: Implementar o sistema em ambiente produtivo
5. **Monitoramento**: Criar sistema de monitoramento da performance em produção

## 👥 Autor

**Miriam O. de Aguiar Sobral**
- Bootcamp de Ciência de Dados e Inteligência Artificial - SENAI
- Data: Agosto/2025

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais como parte do Bootcamp SENAI.

