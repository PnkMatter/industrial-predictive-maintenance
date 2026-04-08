import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Resolve o caminho do modelo relativo a este arquivo (funciona de qualquer diretório)
MODEL_PATH = Path(__file__).parent / 'maintenance_model.pkl'

# Configuração da página
st.set_page_config(page_title="AI Manutenção Preditiva", page_icon="⚙️", layout="centered")

# Título e descrição
st.title("⚙️ Monitoramento de Saúde da Máquina")
st.markdown("""
Este painel utiliza Machine Learning para analisar as leituras dos sensores em tempo real 
e prever se a máquina corre risco de falha iminente.
""")

# Função para carregar o modelo em cache (para não recarregar a cada interação)
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Erro: O arquivo 'maintenance_model.pkl' não foi encontrado. Execute o 'main.py' primeiro.")
    st.stop()

# Criando uma barra lateral para os inputs simularem sensores
st.sidebar.header("Leituras Atuais dos Sensores")

# Ajuste os limites (min_value e max_value) de acordo com a realidade dos seus dados sintéticos
temperature = st.sidebar.slider("Temperatura (°C)", min_value=20.0, max_value=120.0, value=65.0)
vibration = st.sidebar.slider("Vibração (mm/s)", min_value=0.0, max_value=30.0, value=5.0)
pressure = st.sidebar.slider("Pressão (PSI)", min_value=50.0, max_value=250.0, value=120.0)
working_hours = st.sidebar.slider("Horas de Operação (h)", min_value=0, max_value=10000, value=3000)

# Estruturando os dados de input no mesmo formato que o modelo foi treinado
input_data = pd.DataFrame({
    'temperature': [temperature],
    'vibration': [vibration],
    'pressure': [pressure],
    'working_hours': [working_hours]
})

st.subheader("Valores Monitorados")
st.dataframe(input_data, hide_index=True)

# Botão de Ação
if st.button("Executar Diagnóstico", type="primary"):
    # Faz a predição (0 = Saudável, 1 = Falha)
    prediction = model.predict(input_data)[0]
    
    # Pega a probabilidade de ser falha (classe 1)
    probability = model.predict_proba(input_data)[0][1] 

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ **ALERTA: Risco de Falha Detectado!**")
        st.write(f"Probabilidade de Quebra: **{probability:.1%}**")
        st.write("**Ação Recomendada:** Interromper a operação e agendar inspeção imediata na máquina.")
    else:
        st.success(f"✅ **Status: Operação Normal**")
        st.write(f"Risco de Falha: **{probability:.1%}**")
        st.write("**Ação Recomendada:** Seguir com o cronograma padrão de produção.")