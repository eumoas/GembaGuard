
![Foto do Projeto GembaGuard](https://github.com/eumoas/GembaGuard/blob/main/docs/images/residencia.jpeg?raw=true)

#  SENTINELA - SISTEMA INTELIGENTE DE MANUTENÇÃO PREDITIVA


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

#### XGBoost**Melhor Modelo**
- Accuracy: 79.35%
- Precision (Macro): 79.10%
- Recall (Macro): 79.11%
- F1-Score (Macro): 79.10%

### 5. Avaliação
- Validação cruzada com 5 folds
- Foco em métricas Precision e Recall conforme solicitado
- XGBoost apresentou melhor desempenho geral

## Estrutura do Projeto

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

## Como Executar

### Preparação dos Dados
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

## Resultados e Insights

### Principais Descobertas
1. **Desbalanceamento de Classes**: O dataset apresenta desbalanceamento significativo, com 57% das amostras sendo "sem_falha"
2. **Eficácia do SMOTE**: A aplicação de SMOTE melhorou significativamente o desempenho dos modelos
3. **Superioridade do XGBoost**: O modelo XGBoost apresentou melhor desempenho em todas as métricas

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

