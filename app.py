import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import warnings
import base64
from PIL import Image

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO DO LAYOUT ---
st.set_page_config(
    page_title="GembaGuard - Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOMIZADO ---
st.markdown("""
<style>
    /* Tema principal - Laranja e Azul */
    .main-header {
        background: linear-gradient(135deg, #FF6B35 0%, #1E3A8A 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-title {
        color: white;
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: #F0F8FF;
        font-size: 1.5rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Cards personalizados */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FF6B35;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .alert-card {
        background: linear-gradient(145deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #ef4444;
    }
    
    .success-card {
        background: linear-gradient(145deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 5px solid #22c55e;
    }
    
    /* Botões customizados */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #e55a2b 0%, #e67e36 100%);
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
        transform: translateY(-2px);
    }
    
    /* Sidebar customizada */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E3A8A 0%, #3B82F6 100%);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #FF6B35 0%, #1E3A8A 100%);
        color: white;
        border-radius: 8px;
    }
    
    /* Métricas */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Upload area */
    .uploadedFile {
        border: 2px dashed #FF6B35;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(145deg, #fff7ed 0%, #fed7aa 100%);
    }
    
    /* Tabelas */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Alertas personalizados */
    .custom-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .custom-error {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .custom-success {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 5px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Imagem de fundo para hero section */
    .hero-section {
        background-image: linear-gradient(rgba(30, 58, 138, 0.8), rgba(255, 107, 53, 0.8)), 
                          url('https://images.unsplash.com/photo-1581094794329-c8112a89af12?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        padding: 3rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE CARREGAMENTO E PREPARAÇÃO ---
@st.cache_resource
def load_artifacts():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "3_modelos_treinados.pkl")
        scaler_path = os.path.join(script_dir, "3_standard_scaler.pkl")

        models_and_data = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        models = models_and_data['modelos']
        features = models_and_data['features']
        targets = models_and_data['targets']

        return models, scaler, features, targets
    except FileNotFoundError:
        st.error("Erro: Arquivos de modelo não encontrados. Execute as etapas 1, 2 e 3.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Erro ao carregar artefatos do modelo: {e}")
        return None, None, None, None

def verificar_colunas_necessarias(df):
    """Verifica quais colunas básicas estão disponíveis no dataset."""
    colunas_basicas = [
        'temperatura_ar', 'temperatura_processo', 'umidade_relativa', 
        'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta', 'tipo'
    ]
    
    colunas_presentes = []
    colunas_faltando = []
    
    for col in colunas_basicas:
        if col in df.columns:
            colunas_presentes.append(col)
        else:
            colunas_faltando.append(col)
    
    return colunas_presentes, colunas_faltando

def criar_features_avancadas_robusta(df):
    """Versão robusta da criação de features que lida com colunas ausentes."""
    print("--- INICIANDO ENGENHARIA DE FEATURES ROBUSTA ---")
    df_features = df.copy()
    
    # Verificar colunas disponíveis
    colunas_presentes, colunas_faltando = verificar_colunas_necessarias(df)
    
    if colunas_faltando:
        print(f"⚠️ Colunas não encontradas: {colunas_faltando}")
        st.warning(f"⚠️ Algumas colunas não foram encontradas: {', '.join(colunas_faltando)}")
    
    def divisao_segura(numerador, denominador, default=0.001):
        denominador_safe = np.where(denominador == 0, default, denominador)
        return numerador / denominador_safe

    try:
        # Features que dependem de múltiplas colunas
        if 'torque' in df_features.columns and 'velocidade_rotacional' in df_features.columns:
            df_features['potencia_estimada'] = df_features['torque'] * df_features['velocidade_rotacional']
            df_features['stress_mecanico'] = divisao_segura(df_features['torque'], df_features['velocidade_rotacional']) * 1000
        
        if 'temperatura_processo' in df_features.columns and 'temperatura_ar' in df_features.columns:
            df_features['delta_temperatura'] = df_features['temperatura_processo'] - df_features['temperatura_ar']
            
            # Densidade de potência (se potência foi calculada)
            if 'potencia_estimada' in df_features.columns:
                df_features['densidade_potencia'] = divisao_segura(df_features['potencia_estimada'], df_features['temperatura_ar'])
        
        if 'desgaste_da_ferramenta' in df_features.columns:
            df_features['fadiga_ferramenta'] = np.power(df_features['desgaste_da_ferramenta'] + 1, 1.2)
            
            if 'velocidade_rotacional' in df_features.columns:
                df_features['taxa_desgaste'] = divisao_segura(df_features['desgaste_da_ferramenta'], df_features['velocidade_rotacional'])
        
        if 'delta_temperatura' in df_features.columns and 'umidade_relativa' in df_features.columns:
            df_features['indice_calor'] = df_features['delta_temperatura'] * (1 + df_features['umidade_relativa'] / 100)

        # Z-scores apenas para colunas que existem e se 'tipo' existe
        if 'tipo' in df_features.columns:
            features_base = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa', 
                           'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']
            
            zscore_cols = []
            for feature in features_base:
                if feature in df_features.columns:
                    try:
                        media_por_tipo = df_features.groupby('tipo')[feature].transform('mean')
                        std_por_tipo = df_features.groupby('tipo')[feature].transform('std')
                        std_por_tipo = np.where(std_por_tipo == 0, 1, std_por_tipo)
                        zscore_col = f'{feature}_zscore'
                        df_features[zscore_col] = ((df_features[feature] - media_por_tipo) / std_por_tipo)
                        zscore_cols.append(zscore_col)
                    except Exception as e:
                        print(f"Erro ao calcular z-score para {feature}: {e}")
            
            # Índice de anomalia baseado nos z-scores calculados
            if zscore_cols:
                df_features['indice_anomalia'] = np.sqrt((df_features[zscore_cols] ** 2).sum(axis=1) / len(zscore_cols))
        else:
            st.warning("⚠️ Coluna 'tipo' não encontrada. Z-scores e índice de anomalia não serão calculados.")

        print(f"✅ Engenharia de features concluída. Total de colunas: {df_features.shape[1]}")
        print(f"Novas features criadas: {df_features.shape[1] - df.shape[1]}")
        
    except Exception as e:
        print(f"❌ Erro durante a engenharia de features: {e}")
        st.error(f"Erro na engenharia de features: {e}")
    
    return df_features

def preencher_features_faltando(df, features_necessarias):
    """Preenche features que não puderam ser criadas com valores padrão."""
    df_completo = df.copy()
    
    for feature in features_necessarias:
        if feature not in df_completo.columns:
            # Valores padrão baseados no tipo da feature
            if 'zscore' in feature:
                df_completo[feature] = 0.0  # Z-score neutro
            elif feature == 'indice_anomalia':
                df_completo[feature] = 0.5  # Valor médio de anomalia
            elif 'potencia' in feature:
                df_completo[feature] = 1000.0  # Potência padrão
            elif 'temperatura' in feature:
                df_completo[feature] = 0.0  # Delta padrão
            elif 'taxa' in feature or 'densidade' in feature:
                df_completo[feature] = 1.0  # Razão padrão
            elif 'fadiga' in feature:
                df_completo[feature] = 1.0  # Fadiga mínima
            elif 'stress' in feature:
                df_completo[feature] = 100.0  # Stress padrão
            elif 'indice_calor' in feature:
                df_completo[feature] = 0.0  # Índice neutro
            else:
                df_completo[feature] = 0.0  # Valor padrão genérico
            
            st.warning(f"⚠️ Feature '{feature}' foi preenchida com valor padrão devido à ausência de dados necessários.")
    
    return df_completo

# --- CABEÇALHO PRINCIPAL COM IMAGEM ---
st.markdown(
    """
    <div class="hero-section">
        <h1 class="main-title">🔧 GembaGuard</h1>
        <h3 class="main-subtitle">Sistema Inteligente de Manutenção Preditiva</h3>
        <p style="color: white; font-size: 1.1rem; margin-top: 1rem; opacity: 0.9;">
            Monitore, analise e previna falhas antes que aconteçam
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- SEÇÃO DE INFORMAÇÕES ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #FF6B35; margin: 0;">🎯 Precisão</h4>
        <p style="margin: 0.5rem 0 0 0;">Algoritmos de ML avançados para detecção precisa de anomalias</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #1E3A8A; margin: 0;">⚡ Velocidade</h4>
        <p style="margin: 0.5rem 0 0 0;">Análise em tempo real de milhares de pontos de dados</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #FF6B35; margin: 0;">💰 Economia</h4>
        <p style="margin: 0.5rem 0 0 0;">Reduza custos de manutenção e tempo de inatividade</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- AVISO ESTILIZADO ---
st.markdown("""
<div class="custom-warning">
    <strong>⚠️ Aviso Importante:</strong> Este é um modelo <strong>experimental</strong> com alta sensibilidade. 
    Use para demonstração e validação com especialistas antes de tomar decisões críticas.
</div>
""", unsafe_allow_html=True)

# --- CARREGAR ARTEFATOS E DEFINIR VARIÁVEIS ---
models, scaler, features, targets = load_artifacts()

if models:
    # --- SEÇÃO DE FEATURES ESPERADAS ---
    with st.expander("🔍 **Features Esperadas pelo Modelo**", expanded=False):
        st.markdown("**O modelo foi treinado com as seguintes características:**")
        
        # Dividir features em colunas para melhor visualização
        num_cols = 3
        features_per_col = len(features) // num_cols + 1
        cols = st.columns(num_cols)
        
        for i, feature in enumerate(features):
            col_idx = i // features_per_col
            with cols[col_idx]:
                st.markdown(f"✓ `{feature}`")
    
    # --- SEÇÃO DE UPLOAD ---
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h2 style="color: #1E3A8A; text-align: center; margin-bottom: 1rem;">
            📊 Análise de Dados de Manutenção
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload com estilo personalizado
    uploaded_file = st.file_uploader(
        "**Carregue seu arquivo CSV para análise preditiva**",
        type=['csv'],
        help="💡 Dica: O arquivo deve conter dados de sensores como temperatura, torque, velocidade, etc.",
        key="main_uploader"
    )
    
    # Exemplo de formato esperado
    with st.expander("📋 **Exemplo de Formato de Dados**"):
        exemplo_df = pd.DataFrame({
            'temperatura_ar': [295.3, 298.1, 296.7],
            'temperatura_processo': [308.6, 312.4, 310.1], 
            'umidade_relativa': [38.4, 40.2, 39.1],
            'velocidade_rotacional': [1551, 1602, 1578],
            'torque': [42.8, 45.1, 43.9],
            'desgaste_da_ferramenta': [108, 115, 112],
            'tipo': ['M', 'M', 'H']
        })
        st.dataframe(exemplo_df, use_container_width=True)

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Sucesso estilizado
            st.markdown(f"""
            <div class="custom-success">
                <strong>✅ Arquivo Carregado com Sucesso!</strong><br>
                📁 <strong>{uploaded_file.name}</strong> • 📊 <strong>{df.shape[0]:,}</strong> amostras • 📋 <strong>{df.shape[1]}</strong> colunas
            </div>
            """, unsafe_allow_html=True)
            
            # Informações do dataset em cards
            with st.expander("📊 **Informações Detalhadas do Dataset**"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📏 Dimensões**")
                    st.info(f"• **Linhas:** {df.shape[0]:,}\n• **Colunas:** {df.shape[1]}")
                    
                with col2:
                    st.markdown("**🔍 Prévia dos Dados**")
                    st.dataframe(df.head(3), use_container_width=True)
                
                st.markdown("**📋 Colunas Disponíveis**")
                cols_display = st.columns(4)
                for i, col in enumerate(df.columns):
                    with cols_display[i % 4]:
                        st.markdown(f"• `{col}`")

            # Verificar colunas disponíveis
            colunas_presentes, colunas_faltando = verificar_colunas_necessarias(df)
            
            if colunas_faltando:
                st.markdown(f"""
                <div class="custom-warning">
                    <strong>⚠️ Colunas Básicas Não Encontradas:</strong><br>
                    {', '.join([f'<code>{col}</code>' for col in colunas_faltando])}<br><br>
                    <strong>💡 Não se preocupe!</strong> O sistema usará valores padrão inteligentes para continuar a análise.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="custom-success">
                    <strong>✅ Perfeito!</strong> Todas as colunas básicas foram encontradas.
                </div>
                """, unsafe_allow_html=True)

            # Barra de progresso para o processamento
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Feature Engineering
            status_text.text("🔧 Aplicando engenharia de features...")
            progress_bar.progress(25)
            df_with_features = criar_features_avancadas_robusta(df.copy())
            
            # 2. Preenchimento de dados
            status_text.text("📊 Preenchendo dados faltantes...")
            progress_bar.progress(50)
            df_completo = preencher_features_faltando(df_with_features, features)
            
            # 3. Verificação final
            status_text.text("🔍 Verificando compatibilidade...")
            progress_bar.progress(75)
            features_ainda_faltando = [f for f in features if f not in df_completo.columns]
            
            if features_ainda_faltando:
                st.markdown(f"""
                <div class="custom-error">
                    <strong>❌ Erro Crítico:</strong> Não foi possível criar as features: 
                    {', '.join([f'<code>{f}</code>' for f in features_ainda_faltando])}
                </div>
                """, unsafe_allow_html=True)
                st.stop()
            else:
                # 4. Preparação e predição
                status_text.text("🤖 Gerando predições...")
                progress_bar.progress(90)
                
                df_to_predict = df_completo[features]
                
                # Verificar valores nulos
                if df_to_predict.isnull().any().any():
                    st.markdown("""
                    <div class="custom-warning">
                        <strong>🔧 Valores nulos detectados.</strong> Preenchendo automaticamente...
                    </div>
                    """, unsafe_allow_html=True)
                    df_to_predict = df_to_predict.fillna(df_to_predict.mean())
                
                # Escalar dados
                try:
                    df_scaled = pd.DataFrame(
                        scaler.transform(df_to_predict),
                        columns=df_to_predict.columns
                    )
                except Exception as e:
                    st.markdown(f"""
                    <div class="custom-error">
                        <strong>❌ Erro na normalização:</strong> {e}
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()

                # Fazer predições
                predictions = {}
                for target, model in models.items():
                    try:
                        if hasattr(model, 'predict_proba'):
                            predictions[target] = model.predict_proba(df_scaled)[:, 1]
                        else:
                            predictions[target] = model.predict(df_scaled)
                    except Exception as e:
                        st.error(f"Erro na predição para {target}: {e}")
                        predictions[target] = np.zeros(len(df))
                
                df_predictions = pd.DataFrame(predictions, index=df.index)
                
                # Finalizar progresso
                progress_bar.progress(100)
                status_text.text("✅ Análise completa!")
                
                # Limpar barra de progresso após um momento
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()

                # --- DASHBOARD DE RESULTADOS ---
                st.markdown("""
                <div style="margin: 2rem 0;">
                    <h2 style="color: #1E3A8A; text-align: center;">
                        🎯 Dashboard de Análise Preditiva
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Métricas principais em cards estilizados
                st.markdown("### 📊 **Resumo Executivo**")
                
                metric_cols = st.columns(len(targets))
                
                for i, target in enumerate(targets):
                    falhas = (df_predictions[target] > 0.5).sum()
                    prob_media = df_predictions[target].mean()
                    prob_max = df_predictions[target].max()
                    
                    with metric_cols[i]:
                        # Determinar cor do card baseado no nível de alerta
                        if falhas > 0:
                            card_class = "alert-card"
                            icon = "🚨"
                            status = "ALERTA"
                            color = "#ef4444"
                        else:
                            card_class = "success-card"
                            icon = "✅"
                            status = "OK"
                            color = "#22c55e"
                        
                        st.markdown(f"""
                        <div class="metric-card {card_class}">
                            <div style="text-align: center;">
                                <h2 style="margin: 0; color: {color};">{icon}</h2>
                                <h4 style="margin: 0.5rem 0; color: #374151;">
                                    {target.replace('_', ' ').title()}
                                </h4>
                                <h3 style="margin: 0; color: {color}; font-weight: bold;">
                                    {falhas} Falhas
                                </h3>
                                <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-size: 0.9rem;">
                                    Prob. Média: {prob_media:.1%}<br>
                                    Prob. Máxima: {prob_max:.1%}
                                </p>
                                <div style="background: {color}; color: white; padding: 0.25rem 0.75rem; 
                                           border-radius: 20px; font-size: 0.8rem; font-weight: bold; margin-top: 0.5rem;">
                                    {status}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Gráfico de distribuição de probabilidades
                st.markdown("### 📈 **Distribuição de Probabilidades**")
                
                chart_data = pd.DataFrame()
                for target in targets:
                    chart_data[f'{target.replace("_", " ").title()}'] = df_predictions[target]
                
                st.bar_chart(chart_data, use_container_width=True)

                # Alertas críticos
                alertas_criticos = []
                for target in targets:
                    indices_alta_prob = df_predictions[df_predictions[target] > 0.7].index
                    if len(indices_alta_prob) > 0:
                        alertas_criticos.extend([(idx, target, df_predictions.loc[idx, target]) for idx in indices_alta_prob])
                
                if alertas_criticos:
                    st.markdown("### 🚨 **Alertas Críticos (>70% probabilidade)**")
                    for idx, target, prob in alertas_criticos[:5]:  # Mostrar apenas os 5 primeiros
                        st.markdown(f"""
                        <div class="custom-error">
                            <strong>⚠️ ATENÇÃO IMEDIATA:</strong> 
                            Amostra #{idx} • <strong>{target.replace('_', ' ').title()}</strong> • 
                            Probabilidade: <strong>{prob:.1%}</strong>
                        </div>
                        """, unsafe_allow_html=True)

                # Tabela de resultados estilizada
                st.markdown("### 📋 **Relatório Detalhado**")
                
                df_results = df.copy()
                for target in targets:
                    df_results[f'🎯 Prob_{target}'] = df_predictions[target].apply(lambda x: f"{x:.1%}")
                    df_results[f'🚨 Alert_{target}'] = df_predictions[target].apply(
                        lambda x: "🚨 ALERTA" if x > 0.5 else "✅ OK"
                    )
                
                # Selecionar colunas importantes para exibição
                cols_to_display = []
                if 'id' in df_results.columns:
                    cols_to_display.append('id')
                if 'id_produto' in df_results.columns:
                    cols_to_display.append('id_produto')
                
                # Colunas técnicas importantes
                important_cols = ['tipo', 'temperatura_processo', 'torque', 'desgaste_da_ferramenta']
                for col in important_cols:
                    if col in df_results.columns and col not in cols_to_display:
                        cols_to_display.append(col)
                
                # Adicionar colunas de análise
                for target in targets:
                    cols_to_display.extend([f'🎯 Prob_{target}', f'🚨 Alert_{target}'])
                
                # Destacar linhas com alertas
                def highlight_alerts(row):
                    colors = []
                    for col in row.index:
                        if '🚨 Alert_' in col and row[col] == "🚨 ALERTA":
                            colors.append('background-color: #fee2e2; font-weight: bold;')
                        elif '🚨 Alert_' in col and row[col] == "✅ OK":
                            colors.append('background-color: #dcfce7;')
                        else:
                            colors.append('')
                    return colors
                
                st.dataframe(
                    df_results[cols_to_display].style.apply(highlight_alerts, axis=1),
                    use_container_width=True,
                    height=400
                )

                # Seção de download e ações
                st.markdown("### 💾 **Exportar Resultados**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV completo
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV Completo",
                        data=csv,
                        file_name=f"analise_preditiva_{uploaded_file.name}",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Apenas alertas
                    df_alertas = df_results[df_results[[col for col in df_results.columns if '🚨 Alert_' in col]].apply(
                        lambda row: any("ALERTA" in str(val) for val in row), axis=1
                    )]
                    if len(df_alertas) > 0:
                        csv_alertas = df_alertas.to_csv(index=False)
                        st.download_button(
                            label="🚨 Download Apenas Alertas",
                            data=csv_alertas,
                            file_name=f"alertas_{uploaded_file.name}",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("Nenhum alerta para exportar")
                
                with col3:
                    # Relatório resumido
                    resumo = f"""
# Relatório de Manutenção Preditiva - {uploaded_file.name}

## Resumo Executivo
- **Total de amostras analisadas:** {len(df):,}
- **Data da análise:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}

## Resultados por Tipo de Falha
"""
                    for target in targets:
                        falhas = (df_predictions[target] > 0.5).sum()
                        prob_media = df_predictions[target].mean()
                        resumo += f"- **{target.replace('_', ' ').title()}:** {falhas} alertas (prob. média: {prob_media:.1%})\n"
                    
                    resumo += f"\n## Status Geral\n{'🚨 ATENÇÃO NECESSÁRIA' if any((df_predictions[target] > 0.5).sum() > 0 for target in targets) else '✅ SISTEMA OPERANDO NORMALMENTE'}"
                    
                    st.download_button(
                        label="📊 Download Relatório",
                        data=resumo,
                        file_name=f"relatorio_{uploaded_file.name.replace('.csv', '.md')}",
                        mime="text/markdown",
                        use_container_width=True
                    )

                # Recomendações baseadas nos resultados
                st.markdown("### 💡 **Recomendações Inteligentes**")
                
                total_alertas = sum((df_predictions[target] > 0.5).sum() for target in targets)
                
                if total_alertas == 0:
                    st.markdown("""
                    <div class="custom-success">
                        <strong>🎉 Excelente!</strong> Nenhum alerta crítico detectado.<br>
                        <strong>Recomendações:</strong><br>
                        • Continue o monitoramento regular<br>
                        • Mantenha os planos de manutenção preventiva<br>
                        • Considere análises mensais para monitoramento contínuo
                    </div>
                    """, unsafe_allow_html=True)
                elif total_alertas <= 5:
                    st.markdown("""
                    <div class="custom-warning">
                        <strong>⚠️ Atenção Moderada</strong> - Poucos alertas detectados.<br>
                        <strong>Recomendações:</strong><br>
                        • Priorize a inspeção dos equipamentos com alerta<br>
                        • Agende manutenção preventiva nas próximas 48h<br>
                        • Monitore mais frequentemente estes equipamentos
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="custom-error">
                        <strong>🚨 Ação Imediata Necessária</strong> - Múltiplos alertas detectados.<br>
                        <strong>Recomendações:</strong><br>
                        • Pare a operação dos equipamentos com maior probabilidade de falha<br>
                        • Chame a equipe de manutenção especializada<br>
                        • Implemente plano de contingência<br>
                        • Revise os procedimentos de manutenção preventiva
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="custom-error">
                <strong>❌ Erro no Processamento</strong><br>
                Ocorreu um problema ao analisar o arquivo. Verifique o formato dos dados.
            </div>
            """, unsafe_allow_html=True)
            
            # Seção de debug expandível
            with st.expander("🔧 **Informações Técnicas de Debug**"):
                st.markdown("**Erro detalhado:**")
                st.code(str(e), language="python")
                
                if 'df' in locals():
                    st.markdown("**Estrutura do arquivo carregado:**")
                    st.json({
                        "colunas": df.columns.tolist(),
                        "tipos": df.dtypes.to_dict(),
                        "dimensoes": f"{df.shape[0]} x {df.shape[1]}"
                    })
                
                if 'features' in locals():
                    st.markdown("**Features esperadas pelo modelo:**")
                    st.json(features)
                    
                st.markdown("**Dicas para resolução:**")
                st.info("""
                • Verifique se o arquivo CSV está bem formatado
                • Confirme se as colunas têm os nomes corretos
                • Certifique-se de que não há caracteres especiais nos dados
                • Tente com um arquivo menor para teste
                """)

# --- SIDEBAR COM INFORMAÇÕES ADICIONAIS ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #FF6B35 0%, #1E3A8A 100%); 
                border-radius: 10px; color: white; margin-bottom: 1rem;">
        <h3 style="margin: 0;">🔧 GembaGuard</h3>
        <p style="margin: 0; opacity: 0.9;">Manutenção Inteligente</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📚 **Como Usar**")
    st.info("""
    1. **Carregue** seu arquivo CSV
    2. **Aguarde** o processamento
    3. **Analise** os resultados
    4. **Baixe** o relatório
    5. **Implemente** as recomendações
    """)
    
    st.markdown("### 📊 **Tipos de Análise**")
    st.success("✅ **Detecção de Anomalias**\nIdentifica padrões anômalos nos dados")
    st.warning("⚠️ **Predição de Falhas**\nPrevê falhas antes que aconteçam")
    st.info("📈 **Análise de Tendências**\nMonitora evolução dos parâmetros")
    
    st.markdown("### 🛠️ **Suporte Técnico**")
    st.markdown("""
    **Em caso de dúvidas:**
    - 📧 suporte@gembaguard.com
    - 📱 (11) 9999-9999
    - 💬 Chat online: 24/7
    """)
    
    st.markdown("### ⚙️ **Configurações do Modelo**")
    with st.expander("Parâmetros Avançados"):
        threshold = st.slider("Limite de Alerta", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"Atual: {threshold:.0%} - Probabilidades acima deste valor geram alertas")
        
        show_debug = st.checkbox("Modo Debug", value=False)
        st.caption("Exibe informações técnicas detalhadas")

# --- FOOTER ESTILIZADO COM SEU LOGO ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #054BA6 0%, #F28500 100%); 
            border-radius: 15px; color: white; margin-top: 3rem;">
    <div class="logo-container">
        <svg class="logo-svg" style="width: 50px; height: 50px;" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="45" fill="none" stroke="white" stroke-width="3"/>
            <circle cx="50" cy="35" r="6" fill="none" stroke="white" stroke-width="2"/>
            <ellipse cx="50" cy="50" rx="8" ry="12" fill="none" stroke="white" stroke-width="2"/>
            <circle cx="30" cy="25" r="4" fill="none" stroke="white" stroke-width="2"/>
            <circle cx="70" cy="25" r="4" fill="none" stroke="white" stroke-width="2"/>
            <circle cx="25" cy="65" r="4" fill="none" stroke="white" stroke-width="2"/>
            <circle cx="75" cy="65" r="4" fill="none" stroke="white" stroke-width="2"/>
            <line x1="42" y1="40" x2="30" y2="25" stroke="white" stroke-width="2"/>
            <line x1="58" y1="40" x2="70" y2="25" stroke="white" stroke-width="2"/>
            <line x1="42" y1="55" x2="25" y2="65" stroke="white" stroke-width="2"/>
            <line x1="58" y1="55" x2="75" y2="65" stroke="white" stroke-width="2"/>
        </svg>
    </div>
    <h4 style="margin: 0.5rem 0;">🏭 GEMBAGUARD - Manutenção Preditiva Inteligente</h4>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
        Desenvolvido com ❤️ para a Indústria 4.0 • © 2024 • Todos os direitos reservados
    </p>
    <div style="margin-top: 1rem;">
        <span style="margin: 0 1rem;">🔧 Tecnologia ML</span>
        <span style="margin: 0 1rem;">📊 Analytics Avançado</span>
        <span style="margin: 0 1rem;">⚡ Tempo Real</span>
    </div>
</div>
""", unsafe_allow_html=True)