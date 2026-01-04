import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Feature Lab", layout="wide")

st.title("Feature Engineering")

tab1, tab2 = st.tabs(["🔬 Experiment Lab", "🔮 Live Prediction"])

with tab1:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("Config")
        use_binning = st.toggle("Bin Tenure", value=False)
        use_poly = st.toggle("Polynomial Features", value=False)
        use_inter = st.toggle("Interaction Features", value=False)
        
        if st.button("🚀 Run Experiment", type="primary"):
            payload = {
                "use_polynomials": use_poly,
                "polynomial_cols": ["tenure", "MonthlyCharges"],
                "use_interaction": use_inter,
                "use_binning": use_binning
            }
            with st.spinner("Training..."):
                resp = requests.post(f"{API_URL}/run_experiment", json=payload)
                if resp.status_code == 200:
                    st.session_state['experiment_results'] = resp.json()
                    st.success("Model Trained & Saved!")
                else:
                    st.error("Training Failed")

    with col2:
        if 'experiment_results' in st.session_state:
            data = st.session_state['experiment_results']
            m = data['metrics']
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{m['accuracy']:.2%}")
            c2.metric("F1 Score", f"{m['f1_score']:.3f}")
            c3.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
            
            df_imp = pd.DataFrame(data['top_features'])
            fig = px.bar(df_imp, x="importance", y="feature", orientation='h', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Test the Model on New Data")
    st.write("Enter customer details below. The model will use the features you selected in Tab 1.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        p_tenure = st.number_input("Tenure (Months)", 0, 100, 12)
        p_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with c2:
        p_monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
        p_internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with c3:
        p_total = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
        p_payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    if st.button("🔮 Predict Churn Risk"):
        pred_payload = {
            "tenure": p_tenure,
            "MonthlyCharges": p_monthly,
            "TotalCharges": p_total,
            "Contract": p_contract,
            "InternetService": p_internet,
            "PaymentMethod": p_payment
        }
        
        try:
            resp = requests.post(f"{API_URL}/predict_single", json=pred_payload)
            if resp.status_code == 200:
                result = resp.json()
                prob = result['churn_probability']
                
                st.metric("Churn Probability", f"{prob:.1%}")
                
                if prob > 0.5:
                    st.error(f"⚠️ High Risk! This customer is likely to leave.")
                else:
                    st.success(f"✅ Safe. This customer is likely to stay.")
            else:
                st.warning("Please run an experiment in Tab 1 first to train a model!")
        except Exception as e:
            st.error(f"Connection Error: {e}")