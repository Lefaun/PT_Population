import streamlit as st
import time
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.stats import linregress

st.markdown(
    """
    <style>
    /* Make text larger on small screens */
    @media (max-width: 600px) {
        .stTextInput, .stNumberInput, .stSlider {
            font-size: 1.2rem;
        }
        .stButton button {
            font-size: 1.2rem;
            padding: 10px;
        }
        .stPlotlyChart {
            height: 300px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Função para simular o passo da população
def simulate_population_step(population, birth_rate, death_rate):
    births = int(population * birth_rate)
    deaths = int(population * death_rate)
    population += (births - deaths)
    return population, births, deaths

# Função para calcular as estatísticas da simulação
def compute_statistics(population_data):
    mean_population = np.mean(population_data)
    std_dev_population = np.std(population_data)
    variance_population = np.var(population_data)
    return mean_population, std_dev_population, variance_population

# Configuração da página do Streamlit
st.set_page_config(page_title="Simulação de População", layout="wide", page_icon="🌍")

# Título da página
st.title("Simulação de População")

# Formulário de entrada para os parâmetros da simulação
with st.form(key='simulation_form'):
    initial_population = st.number_input("População Inicial", min_value=0, value=1000)
    birth_rate = st.slider("Taxa de Nascimento", min_value=0.0, max_value=1.0, value=0.05)
    death_rate = st.slider("Taxa de Mortalidade", min_value=0.0, max_value=1.0, value=0.02)
    simulation_time = st.number_input("Tempo de Simulação (segundos)", min_value=1, value=10)
    submit_button = st.form_submit_button(label="Iniciar Simulação")

# Executar a simulação se o botão for pressionado
if submit_button:
    # Inicializar variáveis
    population = initial_population
    time_data = []
    population_data = []
    births_data = []
    deaths_data = []

    # Placeholder para atualizações em tempo real
    placeholder = st.empty()

    # Loop de simulação
    for second in range(simulation_time):
        # Simular o passo da população
        population, births, deaths = simulate_population_step(population, birth_rate, death_rate)

        # Adicionar os resultados às listas
        time_data.append(second)
        population_data.append(population)
        births_data.append(births)
        deaths_data.append(deaths)

        # Atualizar a exibição da simulação
        with placeholder.container():
            st.metric(label="População Atual", value=f"{population}")
            st.metric(label="Nascimentos Totais", value=f"{sum(births_data)}")
            st.metric(label="Mortes Totais", value=f"{sum(deaths_data)}")

            # Mostrar o gráfico de população se o espaço for suficiente
            if st.beta_expander("Ver Gráfico de População", expanded=False):
                population_chart = px.line(
                    x=time_data, 
                    y=population_data,
                    labels={'x': 'Tempo (segundos)', 'y': 'População'},
                    title="População ao Longo do Tempo"
                )
                st.plotly_chart(population_chart)

        # Simular efeito de tempo real
        time.sleep(1)

    # Mostrar estatísticas finais após a simulação
    mean_population, std_dev_population, variance_population = compute_statistics(population_data)
    
    st.write("### Estatísticas da Simulação:")
    st.write(f"- Média da População: {mean_population}")
    st.write(f"- Desvio Padrão da População: {std_dev_population}")
    st.write(f"- Variância da População: {variance_population}")
