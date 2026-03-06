import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import time 

# --- 1. GLOBAL MODEL LOADING ---
rf_model = joblib.load('models/randomforest_model.pkl')
xgb_model = joblib.load('models/xgboost_model.pkl')
lgbm_model = joblib.load('models/lightgbm_model.pkl')
scaler_p1 = joblib.load('models/scaler.pkl')

# Phase 2 Models
model_p2 = joblib.load('models/phase2_forecaster.pkl')
scaler_p2 = joblib.load('models/scaler_phase2.pkl')

features_p1 = [
    'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light',
    'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound', 'S5_CO2',
    'S5_CO2_Slope', 'S6_PIR', 'S7_PIR'
]

# --- 2. SETTINGS & NAVIGATION ---
st.set_page_config(page_title="Occupancy Edge-AI", layout="wide")

st.sidebar.title("🏢 Edge-AI Controller")
page = st.sidebar.radio("Select Dashboard:", ["Phase 1: Real-Time Inference", "Phase 2: Predictive Forecasting"])

# ==============================================================================
# DASHBOARD 1: PHASE 1 (INCLUDING ALL RESEARCH & WRITING)
# ==============================================================================
if page == "Phase 1: Real-Time Inference":
    st.title("Phase 1: Real-Time Occupancy Inference")
    st.markdown("*Designed for estimating the exact number of people occupying a room using several sensors, separated by feature importance*")

    # --- SIDEBAR: LIVE INPUTS (P1) ---
    st.sidebar.header("📡 Live Sensor Feeds")
    input_data = {}
    for feat in features_p1:
        if 'Temp' in feat:
            input_data[feat] = st.sidebar.slider(f"{feat} (°C)", 18.0, 32.0, 24.0)
        elif 'Light' in feat:
            input_data[feat] = st.sidebar.slider(f"{feat} (Lux)", 0, 1000, 400)
        elif 'Sound' in feat:
            input_data[feat] = st.sidebar.slider(f"{feat} (dB)", 0.0, 5.0, 0.5)
        elif 'CO2' in feat:
            input_data[feat] = st.sidebar.slider(f"{feat}", 350, 1500, 500) if 'Slope' not in feat else st.sidebar.slider(f"{feat}", -2.0, 5.0, 0.1)
        else: # PIR
            input_data[feat] = st.sidebar.selectbox(f"{feat} (Motion)", [0, 1])

    # --- MAIN PAGE: PREDICTION ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🤖 AI Prediction Engine")
        raw_vector = np.array([list(input_data.values())])
        scaled_vector = scaler_p1.transform(raw_vector)
        prediction = rf_model.predict(scaled_vector)[0]
        
        st.metric(label="Predicted Room Occupancy", value=f"{int(round(prediction))} People")
        
        fig_gauge = px.bar(x=["Occupancy"], y=[prediction], range_y=[0, 5], 
                           title="Live Capacity Monitor", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("🔍 Feature Importance (Winning Model)")
        rf_imp = pd.DataFrame({'Sensor': features_p1, 'Importance': rf_model.feature_importances_.round(3)}).sort_values(by='Importance', ascending=True)
        top8 = rf_imp.tail(8).copy()

        def imp_color(val):
            if val >= 0.15: return '#00CC96'   # green
            elif val >= 0.05: return '#FFA500' # orange
            else: return '#EF553B'             # red

        top8['Band'] = top8['Importance'].apply(imp_color)

        fig_imp = px.bar(
            top8, x='Importance', y='Sensor', orientation='h',
            title="Random Forest Logic", color='Importance', color_continuous_scale='Viridis'
        )
        fig_imp.update_traces(hovertemplate="<b>%{y}</b><br>Importance: <b>%{x:.3f}</b><extra></extra>")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""<div style='display:flex; gap:18px; font-size:0.82rem; margin-top:-10px;'>
            <span style='color:#00CC96'>&#9632; High (&ge;0.150)</span>
            <span style='color:#FFA500'>&#9632; Medium (&ge;0.050)</span>
            <span style='color:#EF553B'>&#9632; Low (&lt;0.050)</span>
        </div>""", unsafe_allow_html=True)

    # log and purpose
    st.markdown("---")
    st.header("📋 Project Documentation")

    tab_purpose, tab_tech, tab_tournament = st.tabs(["Project Purpose", "Technical Methodology", "Tournament Comparison"])

    with tab_purpose:
        st.subheader("Objective: Smart Building Occupancy Edge-AI")
        st.markdown("""
        ### The Problem
        Traditional building management systems (BMS) often rely on high-cost thermal cameras or intrusive PIR sensors that fail to detect stationary occupants. This leads to energy waste in HVAC and lighting systems in Melbourne's commercial office spaces.

        ### The Goal of this Dashboard
        The purpose of this dashboard is to demonstrate an AI system that can infer room occupancy using a variety of  non-intrusive environmental data (CO2, Sound, Light, and Temperature).

        * **Privacy First:** No cameras are used, ensuring compliance with workplace privacy standards.
        * **Edge-Ready:** The models (Random Forest, XGBoost) are lightweight enough to run on local IoT gateways (Edge Computing), reducing the need for expensive cloud data transfer.
        * **Data-Driven HVAC:** By providing an accurate occupancy count, this system can be integrated into smart vents to optimize airflow based on real-time human presence.
        """)
        st.info("💡 **Project Note:** This work has been completed by Joel Kidagan as part of a technical review for Monash DeepNeuron to showcase end-to-end ML pipeline development.")

    with tab_tech:
        st.markdown(f"""
        ### How Random Forest Works
        Works by taking random rows and creating new data sets (**Bootstrapping**). We can also choose the specific independent variables and use them only in certain data sets. Then, we pick values and then pass them through each decision tree created for each data set.
        
        * **Aggregation:** Since this is a regression task to predict the number of people, the results are averaged across all trees to provide the final output. Combining results is known as aggregation. **Bootstrapping + Aggregation = Bagging**.
        * **Redundancy:** Decision trees are susceptible to changes in the data set. Creating 50-100 different decision trees using random forest we take into account redundancy created by certain decision trees and this approach is robust to noise in the data set. 
        * **Stability:** Whilst some of the trees will make bad decisions, this is averaged out through aggregation.

        ### Scrutinising the Model
        I initially used the default `max_features = 1.0` (all of the features), but considering the score I received from Random Forest I looked into ways that I could use to scrutinise the model further. 

        * **The Monopoly:** I noticed that the S5_CO2_Slope was so powerful in prediction (**72% importance**) that it was likely that every single tree was going to start with it. 
        * **The Goal:** I wanted to make sure that at least some of the trees would not be able to see the CO2 slope and find patterns elsewhere. 

        ### Feature Bagging Implementation
        One feature that I looked at was **Feature Bagging**. I researched the standard practice of taking the Square Root ($m = \\sqrt{{p}}$) or **Log** of the total features and using that number of features in the decision tree. I found this helped the model be more stable and so I altered this.

        ### The Results
        Though I was expecting the accuracy to drop slightly due to reduced training data, it improved to 0.9930. 
        
        * This suggested that **Sound** was a powerful secondary variable that was previously ignored due to the dominance of the CO2 slope. 
        * As shown in the chart, S5_CO2_Slope now stands at **22%**, followed by S2_Sound **(16%)** and S1_Light **(13%)**.
        * The reliability of the sensor configuration is higher now even with a potential malfunctioning of a CO2 sensor.
        """)

        bagging_df = pd.DataFrame({
            "Metric": ["Accuracy (R²)", "Primary Sensor", "Secondary Sensor", 'Model "Logic"', "Reliability"],
            "With Feature Bagging (Researcher-Correct)": ["0.9930", "S5_CO2_Slope (22%)", "S2_Sound (16%)", "Diversified: Uses Sound, Light, and CO2.", "High: If the CO2 sensor breaks, the model survives."],
            "Without Feature Bagging (Standard Default)": ["0.9812", "S5_CO2_Slope (72%)", "S1_Light (14%)", "Specialized: Relies almost entirely on CO2.", "Low: If the CO2 sensor fails, the model crashes."]
        })

        def style_bagging_table(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            styles["With Feature Bagging (Researcher-Correct)"] = "background-color: #1a7a4a; color: #ffffff; font-weight: bold;"
            styles["Without Feature Bagging (Standard Default)"] = "background-color: #a02020; color: #ffffff;"
            styles["Metric"] = "font-weight: bold;"
            return styles

        st.markdown("#### Feature Bagging: Impact Comparison")
        st.dataframe(bagging_df.style.apply(style_bagging_table, axis=None), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Data Dictionary & Feature Engineering")
        data_dict = {
            "Feature Group": ["S1-S4 Temp", "S1-S4 Light", "S1-S4 Sound", "S5 CO2", "S5 CO2 Slope", "S6-S7 PIR"],
            "Description": ["Ambient temperature readings across 4 zones in degrees Celsius.", "Light intensity measured in Lux.", "Microphone-based sound pressure levels.", "Carbon Dioxide concentration in PPM.", "The calculated rate of change of CO2.", "Passive Infrared sensors detecting motion."],
            "Importance": ["Low", "Medium", "High", "Low", "Critical (22%)", "Low"]
        }
        st.table(pd.DataFrame(data_dict))
        st.markdown("""
        ### How S5_CO2_Slope was calculated
        The slope is calculated as the **First Derivative** of the CO2 concentration over time:
        $$Slope = \\frac{CO_2(t) - CO_2(t-n)}{\\Delta t}$$
        """)

    with tab_tournament:
        st.markdown("### Model Tournament: Accuracy & Logic Comparison")
        accuracy_results = {
            "Model": ["Random Forest", "XGBoost", "LightGBM"],
            "R² Accuracy Score": [0.9930, 0.9797, 0.9789],
            "Methodology": ["Bagging (Parallel)", "Boosting (Sequential)", "Boosting (Leaf-wise)"]
        }
        accuracy_df = pd.DataFrame(accuracy_results)

        def rank_row_color(row):
            rank = accuracy_df["R² Accuracy Score"].rank(ascending=False)
            r = rank[row.name]
            if r == 1: color = "background-color: #1a7a4a; color: #ffffff; font-weight: bold;"
            elif r == 2: color = "background-color: #b8860b; color: #ffffff; font-weight: bold;"
            else: color = "background-color: #a02020; color: #ffffff; font-weight: bold;"
            return [color] * len(row)

        st.dataframe(accuracy_df.style.apply(rank_row_color, axis=1), use_container_width=True)

        importance_df = pd.DataFrame({
            'Feature': features_p1,
            'RandomForest': rf_model.feature_importances_,
            'XGBoost': xgb_model.feature_importances_,
            'LightGBM': lgbm_model.feature_importances_ / lgbm_model.feature_importances_.sum()
        }).sort_values(by='RandomForest', ascending=False)
        
        st.markdown("#### Feature Weighting across Architectures")
        st.dataframe(importance_df.style.highlight_max(axis=1, subset=['RandomForest', 'XGBoost', 'LightGBM']))
        st.markdown(f"""
    ### Discussion: Why the Scores Differ
    
    As shown in the tournament results, **Random Forest** achieved the highest accuracy (**0.9930**). Based on the feature importance data, here is the technical breakdown:
    
    * **XGBoost's Sensitivity:** In my `train_tournament.py` output, XGBoost showed a **73.4%** reliance on the CO2 Slope. While powerful, this "greedy" approach makes the model more sensitive to specific sensor noise.
    * **The Random Forest Advantage:** By using `max_features='sqrt'`, I forced the Random Forest to explore secondary variables like **S2_Sound** and **S1_Light**. The tournament proves that this diversification led to a higher R² score than the default boosting configurations.
    * **Reliability:** The comparison shows that LightGBM and XGBoost largely ignored individual sound sensors, whereas Random Forest integrated them into a more stable, holistic prediction.
    """)

    # --- DATA ACCESS ---
    st.markdown("---")
    st.subheader("Data Transparency")
    try:
        source_csv = pd.read_csv('data/phase1_room_occupancy.csv')
        csv_bytes = source_csv.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Phase 1 CSV Data", data=csv_bytes, file_name='occupancy_data_monash_review.csv', mime='text/csv')
    except: st.error("Data file not found.")


# ==============================================================================
# DASHBOARD 2: PHASE 2 (PREDICTIVE FORECASTING)
# ==============================================================================
else:
    st.title("Phase 2: Predictive Occupancy Forecasting")
    st.markdown("*Optimized for low-latency proactive forecasting on Intel NUC edge hardware.*")

    # 1. DATA & ACCURACY ENGINE
    p2_data = pd.read_csv('data/phase2_room_occupancy.csv')
    shift_minutes = 15
    p2_data['Future_Occupancy'] = p2_data['Occupancy'].shift(-shift_minutes)
    df_val = p2_data.dropna().copy()
    
    # Calculate Global Accuracy for display
    X_val = df_val[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
    y_val = df_val['Future_Occupancy']
    X_val_scaled = scaler_p2.transform(X_val)
    global_accuracy = model_p2.score(X_val_scaled, y_val)

    # 2. SIMULATION CONTROLS
    st.sidebar.header("🕹️ Edge Simulation")
    sim_temp = st.sidebar.slider("Ambient Temp", float(df_val['Temperature'].min()), float(df_val['Temperature'].max()), float(df_val['Temperature'].mean()))
    sim_co2 = st.sidebar.slider("Ambient CO2", float(df_val['CO2'].min()), float(df_val['CO2'].max()), float(df_val['CO2'].mean()))

    # 3. LIVE INFERENCE (Active Slider + Derived Means)
    active_v = np.array([[
        sim_temp, 
        df_val['Humidity'].mean(), 
        df_val['Light'].mean(), 
        sim_co2, 
        df_val['HumidityRatio'].mean()
    ]])

    start_t = time.perf_counter()
    active_scaled = scaler_p2.transform(active_v)
    pred_p2 = model_p2.predict(active_scaled)[0]
    proba_p2 = model_p2.predict_proba(active_scaled)[0][1]
    lat_ms = (time.perf_counter() - start_t) * 1000

    # 4. PRIMARY KPI DISPLAY
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("15-Min Forecast", "OCCUPIED" if pred_p2 == 1 else "VACANT")
    with m2:
        st.metric("NUC Latency", f"{lat_ms:.4f} ms")
    with m3:
        st.metric("Model Confidence", f"{proba_p2:.1%}")
    with m4:
        st.metric("Forecast Accuracy", f"{global_accuracy:.4f}")

    # 5. INTEGRATED DISCUSSION & DOCUMENTATION
    st.markdown("---")
    st.header("📋 Phase 2 Discussion: Edge-Optimized Forecasting")
    
    tab_strat, tab_logic, tab_performance = st.tabs(["Operational Strategy", "Forecasting Logic", "Hardware Scalability"])

    with tab_strat:
        st.subheader("Objective: Proactive Load Management")
        st.markdown(f"""
        This phase was designed to proactively forecast occupancy with minimal latency. This model provides a {shift_minutes}-minute lead time for HVAC and lighting orchestration.

        * In this dataset, only 5 variables were used to calculate occupancy, as opposed to the 19 used in Phase 1. These reduced variables were intentionally chosen to cater for a low-latency NUC edge device. With 5 variables, the system is also easier to debug when a single sensor fails, requires less power and generates less heat on the edge gateway.
        * Two primary metrics are taken into account for the practical use of the model: **NUC Latency** and **Model Confidence**.
        * The use of these two variables enables the device to shut off at custom settings. For example, if the model calculates the probability of the room being occupied at **less than 10%**, the lights can be automatically shut off to maximize energy savings without risking occupant comfort.
        """)
        st.info("💡 **Business Verdict:** Reducing the feature set by 73% ensures the model remains lightweight enough for real-time deployment in high-density Melbourne commercial assets.")

    with tab_logic:
        st.subheader("Statistical Derivation")
        st.markdown(f"""
        ### Temporal Shift Calculation
        The accuracy metric of **{global_accuracy:.4f}** is derived by validating the model against a 15-minute future target. 
        
        * **Phase 1 comparison:** In phase 1, there was a focus on the AI taking a look at the sensors and current values and telling you the current state. In phase 2, there is a focus on forecasting. Practically speaking, this meant that I took the sensor readings from, say, 3:00pm but told the AI this was the occupancy at 3:15pm. In phase 2, the data was shifted so that the model could learn patterns that lead up to a change, rather than predict the current state.
        * **Input Isolation:** Inputs were isolated to ensure that the most crucial factors were being considered. I allowed temperature and CO2 to be altered as they change quickly when people enter or leave, but humidity generally stays within a certain range and I believe if I allowed alteration of this - this would result in rampant temperature/humidity combinations that would negatively impact the model's confidence score because it wasn't a range the model was trained for. As such, the historical means were taken for these secondary variables.
        * **Leakage Prevention:** Data leakage was another issue that I looked into, when an AI model sees the answer during training. I disregarded  "Current Occupancy" because if the AI knows the lights are on right now, it will guess the room is occupied. In doing so, it is forced to look at environmental trends which are relevant in cases where people do not turn on the lights.
        """)
        # Show the derived means used
        st.table(df_val[['Humidity', 'Light', 'HumidityRatio']].mean().to_frame().T.rename(index={0: "Derived Mean"}))

    with tab_performance:
        st.subheader("NUC Edge Performance")
        st.markdown(f"""
        ### Latency vs. Throughput
        To be viable for a commercial BMS, the inference must not bottleneck the local gateway hardware.
        
        * **Observed Latency:** {lat_ms:.4f} ms per prediction.
        * **Theoretical Throughput:** {int(1000/lat_ms):,} rooms per second.
        
        This level of performance confirms that a single low-power Intel NUC can manage the predictive load of an entire Melbourne CBD office tower.
        """)
    # --- DATA ACCESS ---
    st.markdown("---")
    st.subheader("Data Transparency")
    try:
        source_csv = pd.read_csv('data/phase2_room_occupancy.csv')
        csv_bytes = source_csv.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Phase 2 CSV Data", data=csv_bytes, file_name='occupancy_data_monash_review.csv', mime='text/csv')
    except: st.error("Data file not found.")


# ==============================================================================
# FINAL FOOTER: SHARED ACROSS ALL PAGES
# ==============================================================================
st.markdown("---")
st.header("📄 Project Planning & Strategy")

col_button, col_text = st.columns([1, 2])
plan_path = "MDN_Project_Plan_JoelKidagan.pdf"

with col_button:
    try:
        with open(plan_path, "rb") as f:
            st.download_button(
                label="Download Project Plan (PDF)",
                data=f,
                file_name="MDN_Project_Plan_JoelKidagan.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        st.error("PDF not found. Check the 'Occupancy_Edge' folder.")

with col_text:
    st.write(f"**File:** {plan_path.split('/')[-1]}")
    st.markdown("""
    This document captures the initial architectural planning, including 
    the shift to different phases and the purpose of each phase.
    """)
