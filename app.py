"""
AI-Powered Personal Finance Advisor
Streamlit App - Windows Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Configure page
st.set_page_config(
    page_title="AI Finance Advisor",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("💰 AI-Powered Personal Finance Advisor")
st.markdown("**Get personalized savings predictions powered by AI**")
st.divider()

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load models and preprocessors"""
    try:
        from pytorch_tabular import TabularModel
        
        model = TabularModel.load_model("models/tabtransformer_savings_predictor")
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)
        
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('models/feature_config.json', 'r') as f:
            feature_config = json.load(f)
        
        return model, scaler, target_scaler, label_encoders, feature_config
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def engineer_features(user_data):
    """Apply feature engineering"""
    
    # Total expenses
    expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 
                    'Transport', 'Eating_Out', 'Entertainment', 'Utilities',
                    'Healthcare', 'Education', 'Miscellaneous']
    
    user_data['Total_Expenses'] = sum([user_data.get(col, 0) for col in expense_cols])
    
    # Ratios
    user_data['Expense_to_Income_Ratio'] = (user_data['Total_Expenses'] / user_data['Income']) * 100
    user_data['Income_per_Dependent'] = user_data['Income'] / (user_data['Dependents'] + 1)
    user_data['Expenses_per_Dependent'] = user_data['Total_Expenses'] / (user_data['Dependents'] + 1)
    user_data['Income_per_Age'] = user_data['Income'] / (user_data['Age'] + 1)
    
    # Age group
    if user_data['Age'] <= 25:
        user_data['Age_Group'] = '18-25'
    elif user_data['Age'] <= 35:
        user_data['Age_Group'] = '26-35'
    elif user_data['Age'] <= 45:
        user_data['Age_Group'] = '36-45'
    elif user_data['Age'] <= 55:
        user_data['Age_Group'] = '46-55'
    else:
        user_data['Age_Group'] = '55+'
    
    # Spending categories
    user_data['Discretionary_Spending'] = user_data.get('Eating_Out', 0) + user_data.get('Entertainment', 0)
    user_data['Essential_Spending'] = user_data.get('Rent', 0) + user_data.get('Utilities', 0) + user_data.get('Groceries', 0)
    
    # Financial indicators
    user_data['Financial_Cushion'] = user_data['Income'] - user_data['Total_Expenses']
    user_data['Can_Save'] = 1 if user_data['Financial_Cushion'] > 0 else 0
    user_data['Over_Spending'] = 1 if user_data['Expense_to_Income_Ratio'] > 100 else 0
    
    # Income bracket
    if user_data['Income'] < 30000:
        user_data['Income_Bracket'] = 'Very Low'
    elif user_data['Income'] < 50000:
        user_data['Income_Bracket'] = 'Low'
    elif user_data['Income'] < 75000:
        user_data['Income_Bracket'] = 'Medium'
    elif user_data['Income'] < 100000:
        user_data['Income_Bracket'] = 'High'
    else:
        user_data['Income_Bracket'] = 'Very High'
    
    # Placeholder values for training features
    user_data['Savings_Ratio'] = 0
    user_data['Discretionary_Ratio'] = 0
    user_data['Essential_Ratio'] = 0
    
    # Expense percentages (top 5)
    for col in expense_cols[:5]:
        if user_data['Total_Expenses'] > 0:
            user_data[f'{col}_Pct'] = (user_data.get(col, 0) / user_data['Total_Expenses']) * 100
        else:
            user_data[f'{col}_Pct'] = 0
    
    return user_data

def preprocess_input(user_data, scaler, label_encoders, feature_config):
    """Preprocess for model"""
    
    user_data = engineer_features(user_data)
    df = pd.DataFrame([user_data])
    
    # Encode categorical
    categorical_features = feature_config['categorical_features']
    for col in categorical_features:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            value = str(df[col].iloc[0])
            if value in le.classes_:
                df[col] = le.transform([value])[0]
            else:
                df[col] = le.transform([le.classes_[0]])[0]
    
    # Scale numerical
    numerical_features = feature_config['numerical_features']
    num_cols = [col for col in numerical_features if col in df.columns]
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Select features
    required_features = categorical_features + numerical_features
    return df[required_features]

# ============================================================================
# MAIN APP
# ============================================================================

# Load models
with st.spinner("🔄 Loading AI models..."):
    model, scaler, target_scaler, label_encoders, feature_config = load_models()

if model is None:
    st.error("❌ Failed to load models. Check that models/ folder is present.")
    st.stop()

st.success("✅ AI models loaded!")

# Sidebar inputs
st.sidebar.header("📋 Your Financial Profile")

# Demographics
st.sidebar.subheader("👤 Demographics")
age = st.sidebar.slider("Age", 18, 65, 30)
income = st.sidebar.number_input("Monthly Income (₹)", 10000, 500000, 50000, step=5000)
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3, 4, 5])
occupation = st.sidebar.selectbox("Occupation", 
    ["Engineer", "Doctor", "Teacher", "Businessman", "Government Employee", "Private Employee"])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])

# Expenses
st.sidebar.subheader("💳 Monthly Expenses")
rent = st.sidebar.number_input("Rent (₹)", 0, 100000, 15000, step=1000)
loan = st.sidebar.number_input("Loan Repayment (₹)", 0, 100000, 5000, step=1000)
insurance = st.sidebar.number_input("Insurance (₹)", 0, 20000, 2000, step=500)
groceries = st.sidebar.number_input("Groceries (₹)", 0, 30000, 8000, step=500)
transport = st.sidebar.number_input("Transport (₹)", 0, 20000, 4000, step=500)
eating_out = st.sidebar.number_input("Eating Out (₹)", 0, 20000, 3000, step=500)
entertainment = st.sidebar.number_input("Entertainment (₹)", 0, 20000, 2000, step=500)
utilities = st.sidebar.number_input("Utilities (₹)", 0, 10000, 2000, step=500)
healthcare = st.sidebar.number_input("Healthcare (₹)", 0, 20000, 1500, step=500)
education = st.sidebar.number_input("Education (₹)", 0, 30000, 2000, step=500)
miscellaneous = st.sidebar.number_input("Miscellaneous (₹)", 0, 20000, 2000, step=500)

# Predict button
predict_button = st.sidebar.button("🔮 Predict My Savings", type="primary")

# Main area
if predict_button:
    
    # Prepare data
    user_data = {
        'Age': age,
        'Income': income,
        'Dependents': dependents,
        'Occupation': occupation,
        'City_Tier': city_tier,
        'Rent': rent,
        'Loan_Repayment': loan,
        'Insurance': insurance,
        'Groceries': groceries,
        'Transport': transport,
        'Eating_Out': eating_out,
        'Entertainment': entertainment,
        'Utilities': utilities,
        'Healthcare': healthcare,
        'Education': education,
        'Miscellaneous': miscellaneous
    }
    
    total_expenses = sum([rent, loan, insurance, groceries, transport, 
                         eating_out, entertainment, utilities, healthcare, 
                         education, miscellaneous])
    
    # Make prediction
    with st.spinner("🤖 AI is analyzing your profile..."):
        try:
            input_df = preprocess_input(user_data, scaler, label_encoders, feature_config)
            pred_scaled = model.predict(input_df)
            predicted_savings = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            predicted_savings = max(0, predicted_savings)
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.stop()
    
    # Display results
    st.markdown("## 📊 Your Financial Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly Income", f"₹{income:,}")
    
    with col2:
        st.metric("Total Expenses", f"₹{total_expenses:,}")
    
    with col3:
        st.metric("Predicted Savings", f"₹{predicted_savings:,.0f}")
    
    savings_ratio = (predicted_savings / income) * 100
    with col4:
        st.metric("Savings Rate", f"{savings_ratio:.1f}%")
    
    # Summary
    st.divider()
    st.markdown("### 🎯 Summary")
    
    if savings_ratio >= 20:
        st.success(f"🎉 Excellent! You're saving {savings_ratio:.1f}% of your income (₹{predicted_savings:,.0f}/month)")
    elif savings_ratio >= 10:
        st.info(f"👍 Good! You're saving {savings_ratio:.1f}% of your income (₹{predicted_savings:,.0f}/month)")
    else:
        st.warning(f"⚠️ Your savings rate is {savings_ratio:.1f}% (₹{predicted_savings:,.0f}/month). Consider reducing expenses.")

else:
    # Landing page
    st.markdown("### 👋 Welcome!")
    st.markdown("""
    This AI tool helps you:
    - 💰 Predict monthly savings
    - 📊 Analyze spending patterns  
    - 💡 Get personalized recommendations
    
    **Fill in your details in the sidebar and click "Predict My Savings"!**
    """)

# Footer
st.divider()
st.markdown("*Built with TabTransformer AI • Trained on 20,000+ Indian households*")