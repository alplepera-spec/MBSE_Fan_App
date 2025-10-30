import streamlit as st
import pandas as pd
import itertools
import plotly.express as px
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import plotly.graph_objects as go

# ============================================================
# MBSE Fan System Configurator (Example of application)
# Version: 2025 — Full Illustrated Edition
# ============================================================

st.set_page_config(page_title="MBSE Fan System Configurator (Example of application)", layout="wide", page_icon="🧠")

# --- Custom Style ---
st.markdown("""

<style>
    body {
        background-color: #FFFFFF;
        color: #2B2B2B;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #B22222;
    }
    .icon-header {
        font-size: 40px;
        line-height: 1;
    }
    .subtext {
        color: #555;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header & Introduction
# ============================================================

st.title("🧠 MBSE Fan System Configurator (Example of application)")
st.markdown("**Apply a model-based methodology to design and optimise fan systems based on performance, cost, and time.**")

st.header("💡 Introduction — What is Model-Based Systems Engineering (MBSE)?")

st.markdown("""
Model-Based Systems Engineering (**MBSE**) is an **innovative approach** that replaces traditional, document-based engineering  
with a **model-centric** process, integrating design, simulation, validation, and lifecycle management.

Instead of isolated spreadsheets and reports, MBSE uses a **connected network of digital models** to describe system behaviour,  
requirements, and interactions — enabling teams to **simulate and optimise before building**.

""")

st.image(
    "https://upload.wikimedia.org/wikipedia/chttps:/commons.wikimedia.org/wiki/File:MBSE-Vorteile.jpg",
    caption="🔄 The MBSE Workflow: From Requirements to Validation",
    use_container_width=True
)
st.markdown("""
### 🚀 Why MBSE is Transformative for Companies
- 🧩 **Integrates disciplines** — mechanical, electrical, and control engineers work in a single ecosystem.  
- 📉 **Reduces errors and rework** through early virtual validation and model consistency.  
- ⏱️ **Shortens development cycles** with model reuse and digital twins.  
- 💡 **Improves innovation** by simulating "what-if" scenarios before prototypes exist.  
- 📈 **Supports decision-making** using data-driven system-level insights.

### 🏭 Impact on Industry
MBSE brings a **digital transformation** to engineering organisations, enabling them to:
- Enhance **collaboration** across teams and suppliers.  
- Reduce **development costs** and time-to-market.  
---

""")

# ============================================================
# Step 1 — Define System Requirements
# ============================================================

st.header("📘 Step 1 — Define System Requirements")
st.markdown("""
In MBSE, the process begins with **formalising system requirements** —  
quantitative targets that the design must meet, such as flow rate, efficiency, noise level, or current draw.  
These requirements guide every subsequent modelling and testing activity.
""")

requirements = pd.DataFrame({
    'ID': ['RQ-001', 'RQ-002', 'RQ-003', 'RQ-004'],
    'Requirement': ['Minimum airflow', 'Efficiency', 'Noise level', 'Current draw'],
    'Target': [2000, 60, 45, 2.0],
    'Unit': ['m³/h', '%', 'dB(A)', 'A']
})
st.dataframe(requirements, hide_index=True)

# ============================================================
# Step 2 — Available Modules
# ============================================================

st.header("⚙️ Step 2 — Available Modules")
st.markdown("""
Here, all **available physical components** (motors, impellers, grilles) are catalogued.  
Each has defined performance parameters, costs, and delivery times —  
forming the **building blocks** of the system architecture.
""")

col1, col2, col3 = st.columns(3)
with col1:
    motors = pd.DataFrame({
        'ID': ['M01', 'M02'],
        'Type': ['EC 230V', 'BLDC 48V'],
        'Eff(%)': [85, 80],
        'Max RPM': [3000, 3200],
        'Cost (€)': [55, 42],
        'Lead Time (weeks)': [3, 2]
    })
    st.subheader("🔋 Motors")
    st.dataframe(motors, hide_index=True)

with col2:
    impellers = pd.DataFrame({
        'ID': ['I01', 'I02'],
        'Type': ['Axial', 'Mixed Flow'],
        'Diameter (mm)': [450, 420],
        'Noise (dB)': [46, 49],
        'Cost (€)': [30, 45],
        'Lead Time (weeks)': [2, 4]
    })
    st.subheader("🌪️ Impellers")
    st.dataframe(impellers, hide_index=True)

with col3:
    grilles = pd.DataFrame({
        'ID': ['G01', 'G02'],
        'Type': ['Linear', 'Diffuser'],
        'ΔP (Pa)': [10, 20],
        'NoiseAdd (dB)': [0.5, 1.5],
        'Cost (€)': [10, 15],
        'Lead Time (weeks)': [1, 2]
    })
    st.subheader("🌀 Grilles")
    st.dataframe(grilles, hide_index=True)

# ============================================================
# Step 3 — Generate All Configurations
# ============================================================

st.header("🔀 Step 3 — Generate All Configurations")
st.markdown("""
MBSE enables **automated combination and evaluation** of components  
to generate all possible system configurations.  
This systematic exploration ensures **no potential design option is missed**.
""")

configs = []
for m, i, g in itertools.product(motors['ID'], impellers['ID'], grilles['ID']):
    total_cost = motors.loc[motors['ID']==m, 'Cost (€)'].values[0] + \
                 impellers.loc[impellers['ID']==i, 'Cost (€)'].values[0] + \
                 grilles.loc[grilles['ID']==g, 'Cost (€)'].values[0]
    total_lead = max(
        motors.loc[motors['ID']==m, 'Lead Time (weeks)'].values[0],
        impellers.loc[impellers['ID']==i, 'Lead Time (weeks)'].values[0],
        grilles.loc[grilles['ID']==g, 'Lead Time (weeks)'].values[0]
    )
    configs.append({
        'Config ID': f'CFG-{len(configs)+1:02d}',
        'Motor': m,
        'Impeller': i,
        'Grille': g,
        'Total Cost (€)': total_cost,
        'Lead Time (weeks)': total_lead
    })
configs_df = pd.DataFrame(configs)
st.dataframe(configs_df, hide_index=True)

# ============================================================
# Step 4 — Test & Validation (linked to Step 5)
# ============================================================

st.header("🧪 Step 4 — Test & Validation")
st.markdown("""
Once configurations are generated, **virtual or physical tests** are conducted to assess their performance.  
Results are compared against system requirements to validate design compliance.
""")

# Dati di test
test_data = pd.DataFrame({
    'Config ID': [f'CFG-{i:02d}' for i in range(1, 9)],
    'Flow rate (m³/h)': [1950, 2050, 2100, 2000, 2150, 2200, 2100, 2250],
    'Efficiency (%)': [63, 59, 61, 64, 66, 62, 65, 68],
    'Noise (dB)': [46, 47, 44, 45, 44, 48, 45, 43],
    'Current (A)': [1.9, 2.1, 2.0, 1.8, 1.9, 2.2, 1.7, 1.9]
})
test_data = pd.merge(test_data, configs_df[['Config ID','Total Cost (€)','Lead Time (weeks)']], on='Config ID', how='left')

# ============================================================
# Step 5 — Dynamic Compliance Visualization & Pareto Analysis
# ============================================================

st.header("📊 Step 5 — Dynamic Compliance Visualization & Pareto Analysis")
st.markdown("""
Here, MBSE shows its strength through **interactive analysis**:  
designers can dynamically vary targets and instantly visualise which configurations still comply.  
This allows **trade-off evaluation** between cost, efficiency, and other parameters.
""")

cols = st.columns(5)
with cols[0]:
    req_flow = st.slider('Minimum Flow (m³/h)', 1500, 2500, 2000, step=50)
with cols[1]:
    req_eff = st.slider('Minimum Efficiency (%)', 50, 80, 60, step=1)
with cols[2]:
    req_noise = st.slider('Maximum Noise (dB)', 40, 55, 45, step=1)
with cols[3]:
    req_curr = st.slider('Maximum Current (A)', 1.5, 2.5, 2.0, step=0.1)
with cols[4]:
    req_cost = st.slider('Maximum Cost (€)', 20, 200, 100, step=5)

# Calcolo dinamico dei requisiti
test_data['Meets Flow'] = test_data['Flow rate (m³/h)'] >= req_flow
test_data['Meets Eff'] = test_data['Efficiency (%)'] >= req_eff
test_data['Meets Noise'] = test_data['Noise (dB)'] <= req_noise
test_data['Meets Curr'] = test_data['Current (A)'] <= req_curr
test_data['Meets Cost'] = test_data['Total Cost (€)'] <= req_cost
test_data['OK count'] = test_data[['Meets Flow','Meets Eff','Meets Noise','Meets Curr','Meets Cost']].sum(axis=1)
test_data['Status'] = test_data['OK count'].apply(lambda x: '✅ OK' if x==5 else '⚠️ NOK')

# Mostra tabella aggiornata (Step 4)
st.subheader("Validation Table — Updated with Current Requirements")
st.dataframe(test_data, hide_index=True)

# Visualizzazione Pareto (Step 5)
st.markdown("""
Pareto analysis identifies **optimal trade-offs** — solutions that cannot be improved in one aspect without  
compromising another. This helps decision-makers choose the best configurations.
""")

numeric_columns = ['Flow rate (m³/h)','Efficiency (%)','Noise (dB)','Current (A)','Total Cost (€)','Lead Time (weeks)']
x_var = st.selectbox("X-axis variable", numeric_columns, index=4)
y_var = st.selectbox("Y-axis variable", numeric_columns, index=1)

fig_pareto = px.scatter(
    test_data,
    x=x_var,
    y=y_var,
    size='OK count',
    color='Status',
    text='Config ID',
    color_discrete_map={'✅ OK': '#2E8B57', '⚠️ NOK': '#B22222'}
)
fig_pareto.update_traces(textposition='top center', textfont=dict(size=11))
st.plotly_chart(fig_pareto, use_container_width=True)

st.success(f"{(test_data['Status']=='✅ OK').sum()} out of {len(test_data)} configurations meet all requirements.")

# ============================================================
# Step 8 — Virtual Exploration (2-variable impact + Performance Zone)
# ============================================================

st.header("🧠 Step 8 — Virtual Exploration — Multivariable Impact & Performance Zone")
st.markdown("""
In this advanced step, MBSE integrates **machine learning surrogate models**  
to explore the effect of **two design variables simultaneously** (e.g., flow & noise, flow & cost).  
The visualization now includes a **Performance Zone**, showing where configurations meet all system requirements.  
""")

# Variabili disponibili
features = ['Flow rate (m³/h)', 'Efficiency (%)', 'Noise (dB)', 'Total Cost (€)', 'Current (A)']
targets = ['Efficiency (%)', 'Noise (dB)', 'Total Cost (€)', 'Flow rate (m³/h)', 'Current (A)']

colA, colB = st.columns(2)
with colA:
    var_x = st.selectbox("Select first variable (X)", features, index=0)
with colB:
    var_y = st.selectbox("Select second variable (Y)", [f for f in features if f != var_x], index=2)

# Variabili rimanenti da predire
remaining_targets = [v for v in targets if v not in [var_x, var_y]]

# Addestra modelli RandomForest
X = test_data[[var_x, var_y]]
models = {}
for target in remaining_targets:
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(X, test_data[target])
    models[target] = model

# Slider per impostare i valori correnti
col1, col2 = st.columns(2)
with col1:
    x_value = st.slider(f"{var_x}", float(test_data[var_x].min()), float(test_data[var_x].max()),
                        float(test_data[var_x].mean()), step=1.0)
with col2:
    y_value = st.slider(f"{var_y}", float(test_data[var_y].min()), float(test_data[var_y].max()),
                        float(test_data[var_y].mean()), step=1.0)

# Predizioni per i target rimanenti
predictions = {target: models[target].predict([[x_value, y_value]])[0] for target in remaining_targets}

# Mostra i risultati predetti
st.subheader("🔮 Predicted System Responses")
cols = st.columns(len(predictions))
for idx, (target, value) in enumerate(predictions.items()):
    with cols[idx]:
        st.metric(f"{target}", f"{value:.2f}")

# ============================================================
# Visualizzazione superfici con Performance Zone
# ============================================================

st.markdown("### 🌈 Response Surface Visualization + Performance Zone")

response_target = st.selectbox("Select parameter to visualize", remaining_targets, index=0)

# Crea griglia di punti
x_range = np.linspace(test_data[var_x].min(), test_data[var_x].max(), 40)
y_range = np.linspace(test_data[var_y].min(), test_data[var_y].max(), 40)
xx, yy = np.meshgrid(x_range, y_range)
grid = np.c_[xx.ravel(), yy.ravel()]

# Predici tutte le grandezze
Z_pred = {target: models[target].predict(grid).reshape(xx.shape) for target in remaining_targets}
Z = Z_pred[response_target]

# Calcola la performance zone (basata sui requisiti attuali Step 5)
perf_mask = np.ones_like(xx, dtype=bool)

if 'Flow rate (m³/h)' in Z_pred:
    perf_mask &= (Z_pred['Flow rate (m³/h)'] >= req_flow)
if 'Efficiency (%)' in Z_pred:
    perf_mask &= (Z_pred['Efficiency (%)'] >= req_eff)
if 'Noise (dB)' in Z_pred:
    perf_mask &= (Z_pred['Noise (dB)'] <= req_noise)
if 'Total Cost (€)' in Z_pred:
    perf_mask &= (Z_pred['Total Cost (€)'] <= req_cost)
if 'Current (A)' in Z_pred:
    perf_mask &= (Z_pred['Current (A)'] <= req_curr)

# --- Grafico 3D con Performance Zone ---
fig_3d = go.Figure()

# Superficie principale
fig_3d.add_trace(go.Surface(
    x=xx, y=yy, z=Z,
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(title=response_target),
    opacity=0.9
))

# Superficie trasparente per zona conforme
fig_3d.add_trace(go.Surface(
    x=xx, y=yy, z=np.where(perf_mask, Z, np.nan),
    colorscale=[[0, 'rgba(0,255,0,0.5)'], [1, 'rgba(0,255,0,0.5)']],
    showscale=False,
    name='Performance Zone',
    opacity=0.4
))

# Punto selezionato
fig_3d.add_trace(go.Scatter3d(
    x=[x_value], y=[y_value], z=[predictions[response_target]],
    mode='markers+text',
    text=['Selected Point'],
    textposition='top center',
    marker=dict(color='red', size=6, symbol='circle')
))

fig_3d.update_layout(
    title=f"3D Surface — {response_target} vs {var_x} & {var_y}",
    scene=dict(
        xaxis_title=var_x,
        yaxis_title=var_y,
        zaxis_title=response_target
    ),
    height=650,
    margin=dict(l=0, r=0, t=40, b=0)
)

# --- Heatmap 2D con Performance Zone ---
fig_2d = go.Figure()

# Contour base
fig_2d.add_trace(go.Contour(
    z=Z,
    x=x_range,
    y=y_range,
    colorscale='Viridis',
    contours_coloring='heatmap',
    colorbar_title=response_target
))

# Overlay zona conforme (verde trasparente)
fig_2d.add_trace(go.Contour(
    z=np.where(perf_mask, 1, np.nan),
    x=x_range,
    y=y_range,
    showscale=False,
    colorscale=[[0, 'rgba(0,255,0,0.3)'], [1, 'rgba(0,255,0,0.3)']],
    hoverinfo='skip',
    name='Performance Zone'
))

# Punto selezionato
fig_2d.add_trace(go.Scatter(
    x=[x_value],
    y=[y_value],
    mode='markers+text',
    text=['Selected Point'],
    textposition='top center',
    marker=dict(color='red', size=10, symbol='x'),
    name='Selected'
))

fig_2d.update_layout(
    title=f"2D Heatmap — {response_target} vs {var_x} & {var_y}",
    xaxis_title=var_x,
    yaxis_title=var_y,
    height=600,
    margin=dict(l=0, r=0, t=40, b=0)
)

# Mostra i due grafici in colonne
col3d, col2d = st.columns(2)
with col3d:
    st.plotly_chart(fig_3d, use_container_width=True)
with col2d:
    st.plotly_chart(fig_2d, use_container_width=True)

st.info(f"""
Use the sliders above to vary **{var_x}** and **{var_y}**.  
✅ The **Performance Zone** (green area) highlights the region where all requirements are met  
(flow ≥ {req_flow}, eff ≥ {req_eff}, noise ≤ {req_noise}, cost ≤ {req_cost}, current ≤ {req_curr}).
""")


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown("<center><i>Model-Based Systems Engineering applied to fan system optimisation — 2025 Edition</i></center>", unsafe_allow_html=True)
