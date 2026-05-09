import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Resolve o caminho do modelo relativo a este arquivo (funciona de qualquer diretório)
MODEL_PATH = Path(__file__).parent / 'maintenance_model.pkl'

# Configuração da página
st.set_page_config(page_title="AI Predictive Maintenance", page_icon="⚙️", layout="centered")

# Título e descrição
st.title("⚙️ Machine Health Monitoring")
st.markdown("""
This dashboard uses Machine Learning to analyze sensor readings in real-time
and predict if the machine is at risk of imminent failure.
""")

# Função para carregar o modelo em cache (para não recarregar a cada interação)
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'maintenance_model.pkl' not found. Run 'main.py' first.")
    st.stop()

# Criando uma barra lateral para os inputs simularem sensores
st.sidebar.header("Current Sensor Readings")

# Ajuste os limites (min_value e max_value) de acordo com a realidade dos seus dados sintéticos
temperature = st.sidebar.slider("Temperature (°C)", min_value=20.0, max_value=120.0, value=65.0)
vibration = st.sidebar.slider("Vibration (mm/s)", min_value=0.0, max_value=30.0, value=5.0)
pressure = st.sidebar.slider("Pressure (PSI)", min_value=50.0, max_value=250.0, value=120.0)
working_hours = st.sidebar.slider("Working Hours (h)", min_value=0, max_value=10000, value=3000)

# Estruturando os dados de input no mesmo formato que o modelo foi treinado
input_data = pd.DataFrame({
    'temperature': [temperature],
    'vibration': [vibration],
    'pressure': [pressure],
    'working_hours': [working_hours]
})

st.subheader("Monitored Values")
st.dataframe(input_data, hide_index=True)

# Botão de Ação
if st.button("Run Diagnostics", type="primary"):
    # Faz a predição (0 = Saudável, 1 = Falha)
    prediction = model.predict(input_data)[0]
    
    # Pega a probabilidade de ser falha (classe 1)
    probability = model.predict_proba(input_data)[0][1] 

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ **ALERT: Risk of Failure Detected!**")
        st.write(f"Failure Probability: **{probability:.1%}**")
        st.write("**Recommended Action:** Stop operation and schedule an immediate inspection.")
    else:
        st.success(f"✅ **Status: Normal Operation**")
        st.write(f"Failure Risk: **{probability:.1%}**")
        st.write("**Recommended Action:** Proceed with the standard production schedule.")