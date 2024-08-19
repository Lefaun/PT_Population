#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

#######################
# Page configuration
st.set_page_config(
    page_title="Portugal Population Dashboard",
    page_icon="🏂",
    layout="wide",  # Layout is wide to allow more flexibility on desktop, while components will stack on mobile.
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################
# Load data
df_reshaped = pd.read_csv('data/us-population-2010-2019-reshaped.csv')

#######################
# Sidebar
with st.sidebar:
    st.title('🏂 Portugal Population Dashboard')
    
    year_list = list(df_reshaped.year.unique())[::-1]
    
    selected_year = st.selectbox('Selecione o ano', year_list)
    df_selected_year = df_reshaped[df_reshaped.year == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="population", ascending=False)

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Selecione a cor do tema', color_theme_list)

#######################
# Plots

# Function to create heatmap
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Ano", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color(f'max({input_color}):Q',
                        legend=None,
                        scale=alt.Scale(scheme=input_color_theme)),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(
        width='container'  # Adjust width based on the container to make it responsive
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    )
    return heatmap

# Function to create donut chart
def make_donut(input_response, input_text, input_color):
    # Adjust colors based on the selected color theme
    chart_color = {
        'blue': ['#29b5e8', '#155F7A'],
        'green': ['#27AE60', '#12783D'],
        'orange': ['#F39C12', '#875A12'],
        'red': ['#E74C3C', '#781F16']
    }[input_color]

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None),
    ).properties(
        width=130, height=130  # The chart size should be relatively small for mobile compatibility
    )

    text = plot.mark_text(align='center', color=chart_color[0], font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(
        text=alt.value(f'{input_response} %')
    )

    return plot + text

# Function to simulate population step
def simulate_population_step(population, birth_rate, death_rate):
    births = np.random.poisson(birth_rate)
    deaths = np.random.poisson(death_rate)
    net_change = births - deaths
    new_population = population + net_change
    
    return new_population, births, deaths

# Function to compute statistics
def compute_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    variance = np.var(data)
    
    return mean, std_dev, variance

# Function to perform regression
def perform_regression(time, population):
    X = sm.add_constant(time)
    model = sm.OLS(population, X).fit()
    return model

#######################
# Main Panel

col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:
    st.markdown('#### Gains/Losses')

    df_population_difference_sorted = calculate_population_difference(df_reshaped, selected_year)

    if selected_year > 2010:
        first_state_name = df_population_difference_sorted.states.iloc[0]
        first_state_population = format_number(df_population_difference_sorted.population.iloc[0])
        first_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[0])
    else:
        first_state_name = '-'
        first_state_population = '-'
        first_state_delta = ''
    st.metric(label=first_state_name, value=first_state_population, delta=first_state_delta)

    if selected_year > 2010:
        last_state_name = df_population_difference_sorted.states.iloc[-1]
        last_state_population = format_number(df_population_difference_sorted.population.iloc[-1])
        last_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[-1])
    else:
        last_state_name = '-'
        last_state_population = '-'
        last_state_delta = ''
    st.metric(label=last_state_name, value=last_state_population, delta=last_state_delta)

    st.markdown('#### States Migration')

    if selected_year >= 2010:
        df_greater_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference > 50000]
        df_less_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference < -50000]

        states_migration_greater = round((len(df_greater_50000)/df_population_difference_sorted.states.nunique())*100)
        states_migration_less = round((len(df_less_50000)/df_population_difference_sorted.states.nunique())*100)
        donut_chart_greater = make_donut(states_migration_greater, 'Births', 'green')
        donut_chart_less = make_donut(states_migration_less, 'Deaths', 'red')
    else:
        states_migration_greater = 0
        states_migration_less = 0
        donut_chart_greater = make_donut(states_migration_greater, 'Births', 'green')
        donut_chart_less = make_donut(states_migration_less, 'Deaths', 'red')

    migrations_col = st.columns((0.2, 1, 0.2))
    with migrations_col[1]:
        st.write('Births')
        st.altair_chart(donut_chart_greater)
        st.write('Deaths')
        st.altair_chart(donut_chart_less)

with col[1]:
    st.map(pd.DataFrame({
        'awesome cities': ['Portugal','Algarve','Guimarães','Coimbra'],
        'lat': [38.44, 37.01, 41.44, 40.21],
        'lon': [-9.8, -7.9, -8.29, -8.22]
    }))

    st.title("Simulação de População com Atualizações ao Vivo")

    initial_population = st.number_input("População Inicial", value=1000, min_value=1)
    birth_rate = st.slider("Taxa de Nascimento (por segundo)", 0.0, 5.0, 1.0)
    death_rate = st.slider("Taxa de Mortalidade (por segundo)", 0.0, 5.0, 0.5)
    seconds = st.number_input("Duração da Simulação (segundos)", value=100, min_value=1)
    
    if st.button("Iniciar Simulação"):
        time_data, population_data, births_data, deaths_data = [], [], [], []
        population = initial_population

        placeholder = st.empty()

        for second in range(seconds):
            #population, births, deaths = simulate_population_step(pop 

            # Simulate the population step
            population, births, deaths = simulate_population_step(population, birth_rate, death_rate)
            
            # Append the results to lists
            time_data.append(second)
            population_data.append(population)
            births_data.append(births)
            deaths_data.append(deaths)
            
            # Update the simulation display
            with placeholder.container():
                st.metric(label="População Atual", value=f"{population}")
                st.metric(label="Nascimentos Totais", value=f"{sum(births_data)}")
                st.metric(label="Mortes Totais", value=f"{sum(deaths_data)}")

                # Line chart for population over time
                population_chart = px.line(
                    x=time_data, 
                    y=population_data,
                    labels={'x': 'Tempo (segundos)', 'y': 'População'},
                    title="Evolução da População ao Longo do Tempo"
                )
                st.plotly_chart(population_chart, use_container_width=True)
            
            # Wait for one second before the next iteration
            time.sleep(1)

        # Once the simulation is complete, display final results
        st.success(f"Simulação concluída! População final: {population}")

        # Display statistical data
        mean, std_dev, variance = compute_statistics(population_data)
        st.write(f"Média da População: {mean:.2f}")
        st.write(f"Desvio Padrão da População: {std_dev:.2f}")
        st.write(f"Variância da População: {variance:.2f}")

        # Perform and display regression analysis
        model = perform_regression(time_data, population_data)
        st.write("Análise de Regressão Linear")
        st.write(model.summary())

with col[2]:
    st.markdown("#### Map & Heatmap")

    st.altair_chart(make_heatmap(df_selected_year_sorted, 'year', 'states', 'population', selected_color_theme))

    st.write("Mapa Interativo")
    st.map(df_selected_year_sorted[['latitude', 'longitude']].rename(columns={"latitude": "lat", "longitude": "lon"}))
    
    # Add an additional placeholder for potential future charts or information
    placeholder_col2 = st.empty()

# Footer
st.markdown("""
    <hr style="border:1px solid gray"> </hr>
    <p style="text-align: center;">Desenvolvido por <strong>Seu Nome</strong></p>
    """, unsafe_allow_html=True)
