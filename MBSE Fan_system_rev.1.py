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
# Step 4 — Test & Validation
# ============================================================

st.header("🧪 Step 4 — Test & Validation")
st.markdown("""
Once configurations are generated, **virtual or physical tests** are conducted to assess their performance.  
Results are compared against system requirements to validate design compliance.
""")

test_data = pd.DataFrame({
    'Config ID': [f'CFG-{i:02d}' for i in range(1, 9)],
    'Flow rate (m³/h)': [1950, 2050, 2100, 2000, 2150, 2200, 2100, 2250],
    'Efficiency (%)': [63, 59, 61, 64, 66, 62, 65, 68],
    'Noise (dB)': [46, 47, 44, 45, 44, 48, 45, 43],
    'Current (A)': [1.9, 2.1, 2.0, 1.8, 1.9, 2.2, 1.7, 1.9]
})
test_data = pd.merge(test_data, configs_df[['Config ID','Total Cost (€)','Lead Time (weeks)']], on='Config ID', how='left')

# Baseline compliance
test_data['Meets Flow'] = test_data['Flow rate (m³/h)'] >= 2000
test_data['Meets Eff'] = test_data['Efficiency (%)'] >= 60
test_data['Meets Noise'] = test_data['Noise (dB)'] <= 45
test_data['Meets Curr'] = test_data['Current (A)'] <= 2.0
test_data['OK count'] = test_data[['Meets Flow','Meets Eff','Meets Noise','Meets Curr']].sum(axis=1)
test_data['Status'] = test_data['OK count'].apply(lambda x: '✅ OK' if x==4 else '⚠️ NOK')
st.dataframe(test_data, hide_index=True)

# ============================================================
# Step 5 — Dynamic Compliance Visualization
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

test_data['Meets Flow'] = test_data['Flow rate (m³/h)'] >= req_flow
test_data['Meets Eff'] = test_data['Efficiency (%)'] >= req_eff
test_data['Meets Noise'] = test_data['Noise (dB)'] <= req_noise
test_data['Meets Curr'] = test_data['Current (A)'] <= req_curr
test_data['Meets Cost'] = test_data['Total Cost (€)'] <= req_cost
test_data['OK count'] = test_data[['Meets Flow','Meets Eff','Meets Noise','Meets Curr','Meets Cost']].sum(axis=1)
test_data['Status'] = test_data['OK count'].apply(lambda x: '✅ OK' if x==5 else '⚠️ NOK')

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
# change position of the text labels (options: 'top left', 'top center', 'top right', 
# 'middle left', 'middle center', 'middle right', 'bottom left', 'bottom center', 'bottom right')
fig_pareto.update_traces(textposition='top center', textfont=dict(size=11))
st.plotly_chart(fig_pareto, use_container_width=True)

st.success(f"{(test_data['Status']=='✅ OK').sum()} out of {len(test_data)} configurations meet all requirements.")

# ============================================================
# Step 8 — Virtual Exploration
# ============================================================

st.header("🧠 Step 8 — Virtual Exploration")
st.markdown("""
In this final step, MBSE integrates **machine learning surrogate models** to virtually explore new designs.  
Even untested configurations can be predicted — saving time, cost, and enabling innovation.
""")

# Prepare surrogate models
X = test_data[['Flow rate (m³/h)']]
y_eff = test_data['Efficiency (%)']
y_noise = test_data['Noise (dB)']
y_cost = test_data['Total Cost (€)']

model_eff = RandomForestRegressor(n_estimators=200, random_state=0)
model_noise = RandomForestRegressor(n_estimators=200, random_state=0)
model_cost = RandomForestRegressor(n_estimators=200, random_state=0)

model_eff.fit(X, y_eff)
model_noise.fit(X, y_noise)
model_cost.fit(X, y_cost)

flow_input = st.slider("Select Flow rate to explore (m³/h)", 1800, 2400, 2100, step=10)
eff_pred = model_eff.predict([[flow_input]])[0]
noise_pred = model_noise.predict([[flow_input]])[0]
cost_pred = model_cost.predict([[flow_input]])[0]

colA, colB, colC = st.columns(3)
with colA:
    st.metric("Predicted Efficiency (%)", f"{eff_pred:.2f}")
with colB:
    st.metric("Predicted Noise (dB)", f"{noise_pred:.2f}")
with colC:
    st.metric("Predicted Cost (€)", f"{cost_pred:.2f}")

flow_range = np.linspace(1800, 2400, 100).reshape(-1, 1)
eff_curve = model_eff.predict(flow_range)
noise_curve = model_noise.predict(flow_range)
cost_curve = model_cost.predict(flow_range)

fig_virtual = go.Figure()
fig_virtual.add_trace(go.Scatter(
    x=test_data['Flow rate (m³/h)'], y=test_data['Efficiency (%)'],
    mode='markers', name='Test Efficiency', marker=dict(color='red', size=8)
))
fig_virtual.add_trace(go.Scatter(
    x=flow_range.flatten(), y=eff_curve,
    mode='lines', name='Predicted Efficiency', line=dict(color='red', width=2)
))
fig_virtual.add_trace(go.Scatter(
    x=test_data['Flow rate (m³/h)'], y=test_data['Noise (dB)'],
    mode='markers', name='Test Noise', marker=dict(color='blue', size=8, symbol='square')
))
fig_virtual.add_trace(go.Scatter(
    x=flow_range.flatten(), y=noise_curve,
    mode='lines', name='Predicted Noise', line=dict(color='blue', dash='dash')
))
fig_virtual.add_trace(go.Scatter(
    x=test_data['Flow rate (m³/h)'], y=test_data['Total Cost (€)'],
    mode='markers', name='Test Cost', marker=dict(color='green', size=8, symbol='triangle-up')
))
fig_virtual.add_trace(go.Scatter(
    x=flow_range.flatten(), y=cost_curve,
    mode='lines', name='Predicted Cost', line=dict(color='green', dash='dot')
))

fig_virtual.add_vline(x=flow_input, line_color="black", line_dash="dot",
                      annotation_text="Selected Flow", annotation_position="top left")

fig_virtual.update_layout(
    title="Predicted Efficiency, Noise & Cost vs Flow rate",
    xaxis_title="Flow rate (m³/h)",
    yaxis_title="Value",
    height=600,
    margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig_virtual, use_container_width=True)

st.info("Use the slider above to explore predicted performance and cost of virtual (not-tested) configurations.")


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown("<center><i>Model-Based Systems Engineering applied to fan system optimisation — 2025 Edition</i></center>", unsafe_allow_html=True)
