import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Financial Inclusion Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left: 5px solid #28a745;
    }
    .warning-metric {
        border-left: 5px solid #ffc107;
    }
    .danger-metric {
        border-left: 5px solid #dc3545;
    }
    .feature-importance {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        margin: 0.1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏦 Financial Inclusion Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Predict account ownership probability using Global Findex 2024 data**")

# Load sample data (you would replace this with your actual data loading)
@st.cache_data
def load_sample_data():
    """Load and prepare sample data - replace with your actual data loading logic"""
    np.random.seed(42)
    
    regions = ['Sub-Saharan Africa', 'East Asia & Pacific', 'Europe & Central Asia', 
               'Latin America & Caribbean', 'Middle East & North Africa', 'South Asia', 'North America']
    
    countries_by_region = {
        'Sub-Saharan Africa': ['Kenya', 'Nigeria', 'South Africa', 'Ghana', 'Tanzania', 'Uganda'],
        'East Asia & Pacific': ['China', 'Indonesia', 'Thailand', 'Philippines', 'Vietnam', 'Malaysia'],
        'Europe & Central Asia': ['Germany', 'France', 'United Kingdom', 'Poland', 'Russia', 'Turkey'],
        'Latin America & Caribbean': ['Brazil', 'Mexico', 'Argentina', 'Colombia', 'Chile', 'Peru'],
        'Middle East & North Africa': ['Egypt', 'Saudi Arabia', 'Morocco', 'UAE', 'Jordan', 'Tunisia'],
        'South Asia': ['India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal', 'Afghanistan'],
        'North America': ['United States', 'Canada']
    }
    
    # Generate sample data
    n_samples = 1000
    data = {
        'biz_loan_source': np.random.uniform(0, 1, n_samples),
        'biz_loan': np.random.uniform(0, 1, n_samples),
        'emergency_funds': np.random.uniform(0, 1, n_samples),
        'digital_pay': np.random.uniform(0, 1, n_samples),
        'digital_pay_acc': np.random.uniform(0, 1, n_samples),
        'loan_purpose_group': np.random.uniform(0, 1, n_samples),
        'mobile_pay_s_r': np.random.uniform(0, 1, n_samples),
        'prefer_digital_fin': np.random.uniform(0, 1, n_samples),
        'digital_payment_other': np.random.uniform(0, 1, n_samples),
        'govt_payment_recv': np.random.uniform(0, 1, n_samples),
        'saved_any': np.random.uniform(0, 1, n_samples),
        'mobile_payment_bill': np.random.uniform(0, 1, n_samples),
        'borrowed_any': np.random.uniform(0, 1, n_samples),
        'saved_for_purchase': np.random.uniform(0, 1, n_samples),
        'loan_purpose': np.random.uniform(0, 1, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'income_level': np.random.choice([1, 2, 3, 4], n_samples),
        'education_level': np.random.choice([1, 2, 3, 4], n_samples),
        'region': np.random.choice(regions, n_samples),
        'has_account': np.random.choice([0, 1], n_samples, p=[0.38, 0.62])
    }
    
    df = pd.DataFrame(data)
    
    # Add country based on region
    df['country'] = df['region'].apply(lambda x: np.random.choice(countries_by_region[x]))
    
    return df, countries_by_region

# Train model function
@st.cache_data
def train_model(df):
    """Train the Random Forest model"""
    feature_cols = ['biz_loan_source', 'biz_loan', 'emergency_funds', 'digital_pay', 
                   'digital_pay_acc', 'loan_purpose_group', 'mobile_pay_s_r', 
                   'prefer_digital_fin', 'digital_payment_other', 'govt_payment_recv',
                   'saved_any', 'mobile_payment_bill', 'borrowed_any', 'saved_for_purchase',
                   'loan_purpose', 'age', 'income_level', 'education_level']
    
    X = df[feature_cols]
    y = df['has_account']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_imputed, y)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rf_model, imputer, importance_df, feature_cols

# Load data and train model
df, countries_by_region = load_sample_data()
model, imputer, feature_importance, feature_cols = train_model(df)

# Sidebar for user inputs
st.sidebar.header("🔧 Model Configuration")

# Region and Country selection
selected_region = st.sidebar.selectbox(
    "Select Region:",
    options=list(countries_by_region.keys()),
    index=0
)

selected_country = st.sidebar.selectbox(
    "Select Country:",
    options=countries_by_region[selected_region],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("👤 Individual Prediction")

# Create input form
with st.sidebar.form("prediction_form"):
    st.subheader("Financial Profile")
    
    # Business & Lending
    biz_loan_source = st.slider("Business Loan Access", 0.0, 1.0, 0.5, 0.1)
    biz_loan = st.slider("Business Loan Usage", 0.0, 1.0, 0.3, 0.1)
    emergency_funds = st.slider("Emergency Fund Access", 0.0, 1.0, 0.4, 0.1)
    
    # Digital Financial Services
    digital_pay = st.slider("Digital Payment Usage", 0.0, 1.0, 0.6, 0.1)
    digital_pay_acc = st.slider("Digital Payment Account", 0.0, 1.0, 0.5, 0.1)
    mobile_pay_s_r = st.slider("Mobile Payment Send/Receive", 0.0, 1.0, 0.4, 0.1)
    
    # Demographics
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    income_level = st.selectbox("Income Level", [1, 2, 3, 4], index=1)
    education_level = st.selectbox("Education Level", [1, 2, 3, 4], index=1)
    
    # Additional features (simplified for UI)
    loan_purpose_group = st.slider("Loan Purpose Clarity", 0.0, 1.0, 0.3, 0.1)
    prefer_digital_fin = st.slider("Digital Finance Preference", 0.0, 1.0, 0.5, 0.1)
    digital_payment_other = st.slider("Other Digital Payments", 0.0, 1.0, 0.3, 0.1)
    govt_payment_recv = st.slider("Government Payment Receipt", 0.0, 1.0, 0.2, 0.1)
    saved_any = st.slider("Any Savings", 0.0, 1.0, 0.4, 0.1)
    mobile_payment_bill = st.slider("Mobile Bill Payment", 0.0, 1.0, 0.3, 0.1)
    borrowed_any = st.slider("Any Borrowing", 0.0, 1.0, 0.3, 0.1)
    saved_for_purchase = st.slider("Saved for Purchase", 0.0, 1.0, 0.3, 0.1)
    loan_purpose = st.slider("Loan Purpose Score", 0.0, 1.0, 0.2, 0.1)
    
    predict_button = st.form_submit_button("🔮 Predict Account Ownership", use_container_width=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Regional Analysis
    st.subheader(f"📊 Regional Analysis: {selected_region}")
    
    # Filter data for selected region and country
    region_data = df[df['region'] == selected_region]
    country_data = df[df['country'] == selected_country]
    
    # Regional metrics
    region_inclusion_rate = region_data['has_account'].mean()
    country_inclusion_rate = country_data['has_account'].mean()
    global_inclusion_rate = df['has_account'].mean()
    
    # Display metrics
    met_col1, met_col2, met_col3 = st.columns(3)
    
    with met_col1:
        st.metric(
            "Global Rate", 
            f"{global_inclusion_rate:.1%}",
            delta=None
        )
    
    with met_col2:
        delta_region = region_inclusion_rate - global_inclusion_rate
        st.metric(
            f"{selected_region}", 
            f"{region_inclusion_rate:.1%}",
            delta=f"{delta_region:+.1%}"
        )
    
    with met_col3:
        delta_country = country_inclusion_rate - region_inclusion_rate
        st.metric(
            f"{selected_country}", 
            f"{country_inclusion_rate:.1%}",
            delta=f"{delta_country:+.1%}"
        )
    
    # Regional comparison chart
    regional_stats = df.groupby('region')['has_account'].agg(['mean', 'count']).reset_index()
    regional_stats.columns = ['Region', 'Inclusion_Rate', 'Sample_Size']
    
    fig_regional = px.bar(
        regional_stats, 
        x='Region', 
        y='Inclusion_Rate',
        title='Financial Inclusion Rates by Region',
        color='Inclusion_Rate',
        color_continuous_scale='RdYlGn',
        text=regional_stats['Inclusion_Rate'].apply(lambda x: f'{x:.1%}')
    )
    fig_regional.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    fig_regional.update_traces(textposition='outside')
    st.plotly_chart(fig_regional, use_container_width=True)

with col2:
    st.subheader("🎯 Model Performance")
    
    # Model metrics (from your results)
    metrics_data = {
        'Metric': ['Accuracy', 'AUC-ROC', 'AUC-PR', 'F1-Score', 'Precision', 'Recall'],
        'Score': [0.8962, 0.9607, 0.9743, 0.9163, 0.9103, 0.9225],
        'Category': ['Good', 'Excellent', 'Excellent', 'Excellent', 'Excellent', 'Excellent']
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Color coding
    colors = {'Excellent': '#28a745', 'Good': '#ffc107', 'Poor': '#dc3545'}
    
    fig_metrics = px.bar(
        metrics_df, 
        y='Metric', 
        x='Score',
        orientation='h',
        color='Category',
        color_discrete_map=colors,
        text=metrics_df['Score'].apply(lambda x: f'{x:.3f}')
    )
    fig_metrics.update_layout(
        title="Model Performance Metrics",
        height=400,
        showlegend=False
    )
    fig_metrics.update_traces(textposition='auto')
    st.plotly_chart(fig_metrics, use_container_width=True)

# Feature Importance Section
st.subheader("🔍 Feature Importance Analysis")

col1, col2 = st.columns([3, 2])

with col1:
    # Top 10 features chart
    top_features = feature_importance.head(10)
    
    fig_importance = px.bar(
        top_features, 
        y='feature', 
        x='importance',
        orientation='h',
        title='Top 10 Most Important Features',
        color='importance',
        color_continuous_scale='viridis'
    )
    fig_importance.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        height=400
    )
    st.plotly_chart(fig_importance, use_container_width=True)

with col2:
    st.markdown("**Feature Categories:**")
    
    # Group features by category
    business_features = ['biz_loan_source', 'biz_loan', 'loan_purpose_group', 'loan_purpose']
    digital_features = ['digital_pay', 'digital_pay_acc', 'mobile_pay_s_r', 'prefer_digital_fin', 
                       'digital_payment_other', 'mobile_payment_bill']
    saving_features = ['emergency_funds', 'saved_any', 'saved_for_purchase']
    
    business_importance = feature_importance[feature_importance['feature'].isin(business_features)]['importance'].sum()
    digital_importance = feature_importance[feature_importance['feature'].isin(digital_features)]['importance'].sum()
    saving_importance = feature_importance[feature_importance['feature'].isin(saving_features)]['importance'].sum()
    
    category_data = {
        'Category': ['Business & Credit', 'Digital Services', 'Savings Behavior', 'Other'],
        'Importance': [business_importance, digital_importance, saving_importance, 
                      1 - business_importance - digital_importance - saving_importance]
    }
    
    fig_pie = px.pie(
        values=category_data['Importance'], 
        names=category_data['Category'],
        title='Feature Importance by Category'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Individual Prediction Results
if predict_button:
    st.subheader("🔮 Individual Prediction Results")
    
    # Prepare input data
    input_data = pd.DataFrame({
        'biz_loan_source': [biz_loan_source],
        'biz_loan': [biz_loan],
        'emergency_funds': [emergency_funds],
        'digital_pay': [digital_pay],
        'digital_pay_acc': [digital_pay_acc],
        'loan_purpose_group': [loan_purpose_group],
        'mobile_pay_s_r': [mobile_pay_s_r],
        'prefer_digital_fin': [prefer_digital_fin],
        'digital_payment_other': [digital_payment_other],
        'govt_payment_recv': [govt_payment_recv],
        'saved_any': [saved_any],
        'mobile_payment_bill': [mobile_payment_bill],
        'borrowed_any': [borrowed_any],
        'saved_for_purchase': [saved_for_purchase],
        'loan_purpose': [loan_purpose],
        'age': [age],
        'income_level': [income_level],
        'education_level': [education_level]
    })
    
    # Make prediction
    input_imputed = imputer.transform(input_data)
    probability = model.predict_proba(input_imputed)[0][1]
    prediction = model.predict(input_imputed)[0]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Account Ownership Probability"},
            delta = {'reference': 62},  # Global average
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Interpretation
    if probability >= 0.8:
        st.success(f"🎯 **High Probability ({probability:.1%})**: Strong likelihood of having a bank account")
        recommendation = "**Retention Focus**: Ensure continued satisfaction with banking services"
    elif probability >= 0.5:
        st.warning(f"⚠️ **Moderate Probability ({probability:.1%})**: Uncertain account ownership status")
        recommendation = "**Targeted Outreach**: Focus on digital services and emergency fund products"
    else:
        st.error(f"🚨 **Low Probability ({probability:.1%})**: Likely unbanked individual")
        recommendation = "**Priority Intervention**: Immediate outreach with business loan and mobile payment focus"
    
    st.markdown(f"**🎯 Recommendation**: {recommendation}")
    
    # Top contributing factors for this prediction
    feature_contributions = []
    for i, feature in enumerate(feature_cols):
        contribution = input_imputed[0][i] * feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
        feature_contributions.append({'feature': feature, 'contribution': contribution})
    
    contrib_df = pd.DataFrame(feature_contributions).sort_values('contribution', ascending=False).head(5)
    
    st.markdown("**🔍 Top Contributing Factors:**")
    for _, row in contrib_df.iterrows():
        st.markdown(f"• **{row['feature']}**: {row['contribution']:.3f}")

# Country Deep Dive
st.subheader(f"🌍 Country Deep Dive: {selected_country}")

col1, col2 = st.columns(2)

with col1:
    # Country-specific metrics
    country_stats = country_data.groupby('has_account').agg({
        'age': 'mean',
        'income_level': 'mean',
        'education_level': 'mean'
    }).round(2)
    
    st.markdown("**Demographic Breakdown by Account Status:**")
    st.dataframe(country_stats)

with col2:
    # Feature distribution for country
    if len(country_data) > 0:
        fig_dist = px.box(
            country_data, 
            x='has_account', 
            y='age',
            title=f'Age Distribution by Account Status - {selected_country}',
            labels={'has_account': 'Has Account', 'age': 'Age'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# Policy Recommendations
st.subheader("📋 Policy Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card success-metric">
    <h4>🏢 Business-First Strategy</h4>
    <p>Focus on business loan access and entrepreneurial financial services as primary inclusion drivers (29.13% model importance)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card warning-metric">
    <h4>📱 Digital Infrastructure</h4>
    <p>Invest in mobile payment systems and digital literacy programs (12.33% model importance)</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card danger-metric">
    <h4>🛡️ Emergency Preparedness</h4>
    <p>Link disaster preparedness with financial inclusion through emergency fund products (9.80% importance)</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p>📊 Data: Global Findex Database (World Bank, 2024) | 🤖 Model: Random Forest (96.07% AUC) | 
🎯 Confidence: High-accuracy predictions for targeted interventions</p>
</div>
""", unsafe_allow_html=True)