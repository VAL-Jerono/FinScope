import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Financial Inclusion Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .mission-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
        text-align: center;
    }
    .impact-metric {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-medium {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #27ae60, #229954);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .deployment-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e6ed;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Mock model for demonstration (replace with your actual trained model)
class MockRandomForestModel:
    def __init__(self):
        self.feature_importances_ = {
            'biz_loan_source': 0.108,
            'biz_loan': 0.086,
            'digital_pay_acc': 0.085,
            'digital_engagement_score': 0.071,
            'prefer_digital_acc': 0.066,
            'credit_card': 0.056,
            'mobile_pay_s_r': 0.049,
            'loan_purpose_group': 0.045,
            'govt_services_score': 0.041,
            'mobile_payment': 0.028
        }
    
    def predict_proba(self, X):
        # Mock prediction based on feature values
        # In reality, load your actual model: joblib.load('financial_inclusion_model.pkl')
        
        # Simple logic for demonstration
        risk_score = 0.6  # Base risk
        
        # Business loan factors (most important)
        if hasattr(X, 'iloc') and len(X) > 0:
            row = X.iloc[0] if hasattr(X, 'iloc') else X
            
            # Digital engagement reduces risk significantly
            if 'digital_engagement_score' in row:
                risk_score -= row['digital_engagement_score'] * 0.3
            
            # Business loan source impacts risk
            if 'biz_loan_source' in row and row['biz_loan_source'] == 'Traditional Bank':
                risk_score -= 0.2
            
            # Credit card ownership reduces risk
            if 'credit_card' in row and row['credit_card'] == 1:
                risk_score -= 0.15
            
            # Mobile payment usage reduces risk
            if 'mobile_payment_usage' in row and row['mobile_payment_usage'] == 'High':
                risk_score -= 0.1
        
        # Ensure probability is between 0 and 1
        risk_score = max(0.05, min(0.95, risk_score))
        return np.array([[1-risk_score, risk_score]])

# Initialize mock model
@st.cache_resource
def load_model():
    return MockRandomForestModel()

model = load_model()

# Header
st.markdown('<h1 class="main-header">🏦 Financial Inclusion Risk Predictor</h1>', unsafe_allow_html=True)

# Mission Statement
st.markdown("""
<div class="mission-box">
    <h2>Our Mission: Banking the Unbanked</h2>
    <p>Helping 1.4 billion adults worldwide gain access to financial services through AI-powered risk prediction and targeted outreach.</p>
    <p><strong>Model Performance: 98.33% AUC | Predicting Financial Exclusion Risk</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("🎯 Individual Risk Assessment")
st.sidebar.markdown("Enter details to predict financial inclusion risk and get targeted recommendations.")

# Input fields based on your top features
with st.sidebar:
    st.subheader("📊 Business & Loan Profile")
    
    biz_loan_source = st.selectbox(
        "Business Loan Source",
        ["None", "Traditional Bank", "Microfinance Institution", "Digital Lender", "Government Program"],
        help="Source of business financing affects default risk significantly"
    )
    
    has_biz_loan = st.radio(
        "Has Business Loan?",
        ["No", "Yes"],
        help="Business loan status is the 2nd most important feature"
    )
    
    loan_purpose = st.selectbox(
        "Loan Purpose Group",
        ["Personal", "Business", "Education", "Housing", "Agriculture", "Emergency"],
        help="Purpose of loan affects repayment patterns"
    )
    
    st.subheader("📱 Digital Financial Engagement")
    
    digital_engagement_score = st.slider(
        "Digital Engagement Score",
        0.0, 1.0, 0.5, 0.1,
        help="Higher scores indicate more digital financial activity"
    )
    
    digital_pay_acc = st.slider(
        "Digital Payment Account Score",
        0.0, 1.0, 0.5, 0.1,
        help="Usage level of digital payment accounts"
    )
    
    prefer_digital = st.radio(
        "Prefers Digital Accounts?",
        ["No", "Yes"],
        help="Preference for digital vs traditional banking"
    )
    
    st.subheader("💳 Traditional Financial Services")
    
    has_credit_card = st.radio(
        "Has Credit Card?",
        ["No", "Yes"],
        help="Credit card ownership indicates financial sophistication"
    )
    
    mobile_payment_usage = st.selectbox(
        "Mobile Payment Usage Level",
        ["None", "Low", "Medium", "High"],
        help="Frequency of mobile payment transactions"
    )
    
    govt_services_score = st.slider(
        "Government Services Digital Score",
        0.0, 1.0, 0.3, 0.1,
        help="Usage of government digital services"
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Prediction section
    st.header("🎯 Risk Prediction Results")
    
    if st.button("🔮 Predict Financial Inclusion Risk", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame([{
            'biz_loan_source': biz_loan_source,
            'biz_loan': 1 if has_biz_loan == "Yes" else 0,
            'digital_engagement_score': digital_engagement_score,
            'digital_pay_acc': digital_pay_acc,
            'prefer_digital_acc': 1 if prefer_digital == "Yes" else 0,
            'credit_card': 1 if has_credit_card == "Yes" else 0,
            'mobile_payment_usage': mobile_payment_usage,
            'loan_purpose_group': loan_purpose,
            'govt_services_score': govt_services_score
        }])
        
        # Make prediction
        risk_prob = model.predict_proba(input_data)[0][1]
        
        # Determine risk category and recommendations
        if risk_prob > 0.6:
            risk_category = "HIGH RISK"
            risk_class = "risk-high"
            priority = "🚨 PRIORITY 1"
            intervention = "Immediate Outreach Required"
            strategy = [
                "🎯 **Simplified Products**: Offer basic savings accounts with minimal requirements",
                "📱 **Mobile-First Approach**: Use SMS and voice-based services instead of apps",
                "🤝 **Community Outreach**: Deploy field agents for face-to-face enrollment",
                "💰 **Micro-incentives**: Small cash bonuses for account opening and usage",
                "🏫 **Financial Literacy**: Basic education programs before product introduction"
            ]
        elif risk_prob > 0.35:
            risk_category = "MEDIUM RISK"
            risk_class = "risk-medium"
            priority = "⚠️ PRIORITY 2"
            intervention = "Digital Engagement Programs"
            strategy = [
                "📚 **Digital Training**: Workshops on mobile payments and digital banking",
                "🎮 **Gamification**: Reward-based engagement programs",
                "👥 **Peer Networks**: Connect with digitally engaged community members",
                "📊 **Progressive Products**: Start with simple, gradually introduce complex services",
                "🎁 **Incentive Programs**: Cashback for digital transactions"
            ]
        else:
            risk_category = "LOW RISK"
            risk_class = "risk-low"
            priority = "✅ PRIORITY 3"
            intervention = "Premium Service Candidate"
            strategy = [
                "💎 **Premium Products**: Offer credit cards, investment products, insurance",
                "🏦 **Relationship Banking**: Assign dedicated relationship managers",
                "📈 **Advanced Services**: Loans, mortgages, business banking",
                "🌟 **VIP Programs**: Exclusive benefits and priority customer service",
                "📊 **Cross-selling**: Multiple product offerings based on usage patterns"
            ]
        
        # Display results
        st.markdown(f"""
        <div class="{risk_class}">
            <h2>{priority}: {risk_category}</h2>
            <h3>Financial Exclusion Probability: {risk_prob:.1%}</h3>
            <p><strong>Recommended Intervention:</strong> {intervention}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Strategic recommendations
        st.subheader("📋 Targeted Strategy Recommendations")
        for strategy_item in strategy:
            st.markdown(strategy_item)
        
        # Feature importance for this prediction
        st.subheader("🔍 Key Risk Factors Analysis")
        
        # Create feature importance chart
        features = list(model.feature_importances_.keys())[:8]
        importances = list(model.feature_importances_.values())[:8]
        
        fig = px.horizontal_bar(
            x=importances, 
            y=features,
            orientation='h',
            title="Most Important Features in Your Risk Assessment",
            labels={'x': 'Feature Importance', 'y': 'Features'},
            color=importances,
            color_continuous_scale="RdYlBu_r"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Impact metrics and information
    st.header("📊 Global Impact")
    
    st.markdown("""
    <div class="impact-metric">
        <h3>1.4B</h3>
        <p>Adults without bank accounts worldwide</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="impact-metric">
        <h3>98.33%</h3>
        <p>Model accuracy (AUC score)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="impact-metric">
        <h3>47.6%</h3>
        <p>Gender gap in financial inclusion</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="impact-metric">
        <h3>$50B+</h3>
        <p>Potential economic impact</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Target segments
    st.subheader("🎯 Priority Segments")
    
    target_segments = [
        {"segment": "Women in MENA & South Asia", "population": "300M+", "gap": "47.6%"},
        {"segment": "Youth (15-24) with Primary Education", "population": "200M+", "gap": "46.5%"},
        {"segment": "Rural Small Business Owners", "population": "150M+", "gap": "31.8%"},
        {"segment": "Out-of-Labor Force Adults", "population": "250M+", "gap": "46.8%"}
    ]
    
    for segment in target_segments:
        st.markdown(f"""
        <div class="deployment-card">
            <strong>{segment['segment']}</strong><br>
            Population: {segment['population']}<br>
            Inclusion Gap: {segment['gap']}
        </div>
        """, unsafe_allow_html=True)

# Bottom section - Deployment impact
st.header("🚀 Deployment Impact Projection")

col3, col4, col5 = st.columns(3)

with col3:
    st.subheader("📈 6-Month Targets")
    st.markdown("""
    - **New Accounts**: 5M+
    - **Partner Integration**: 15 organizations
    - **Geographic Reach**: 25 countries
    - **Cost Reduction**: 40% in outreach
    """)

with col4:
    st.subheader("📊 18-Month Goals")
    st.markdown("""
    - **New Accounts**: 25M+
    - **Policy Partnerships**: 50 governments
    - **Mobile Integration**: 100M+ users
    - **Economic Impact**: $10B+ activity
    """)

with col5:
    st.subheader("🌍 Long-term Vision")
    st.markdown("""
    - **New Accounts**: 100M+
    - **Global Coverage**: 100+ countries
    - **Inclusion Gap Closure**: 30%
    - **Systemic Change**: Policy integration
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Financial Inclusion Risk Predictor</strong> | Powered by Random Forest ML Model (98.33% AUC)</p>
    <p>Transforming 1.4 billion lives through data-driven financial inclusion</p>
</div>
""", unsafe_allow_html=True)

# How to use instructions
with st.expander("📖 How to Use This Tool"):
    st.markdown("""
    ### For Policy Makers & Program Managers:
    1. **Individual Assessment**: Use the sidebar to input beneficiary profiles
    2. **Risk Prediction**: Get instant risk scores and intervention recommendations
    3. **Resource Allocation**: Prioritize outreach based on risk categories
    4. **Strategy Planning**: Use targeted recommendations for program design
    
    ### For Field Teams:
    1. **Pre-Screening**: Assess prospects before expensive outreach
    2. **Product Matching**: Recommend appropriate financial products
    3. **Success Tracking**: Monitor intervention effectiveness
    4. **Training Tool**: Understand key risk factors for better field decisions
    
    ### For Financial Institutions:
    1. **Credit Risk**: Assess loan default probability
    2. **Product Development**: Design products for specific risk segments
    3. **Marketing Optimization**: Target marketing spend effectively
    4. **Portfolio Management**: Monitor and manage existing customer risk
    """)

# Deployment instructions
with st.expander("🚀 Deployment Instructions"):
    st.markdown("""
    ### Local Deployment:
    ```bash
    pip install streamlit pandas plotly scikit-learn
    streamlit run financial_inclusion_app.py
    ```
    
    ### Cloud Deployment Options:
    
    **1. Streamlit Cloud (Recommended)**
    - Fork this code to GitHub
    - Connect to Streamlit Cloud
    - Deploy with one click
    - Free for public repos
    
    **2. Heroku**
    ```bash
    heroku create financial-inclusion-app
    git push heroku main
    ```
    
    **3. Google Cloud Run**
    ```bash
    gcloud run deploy --source .
    ```
    
    ### Required Files:
    - `financial_inclusion_app.py` (this file)
    - `requirements.txt` (streamlit, pandas, plotly, scikit-learn)
    - `financial_inclusion_model.pkl` (your trained model)
    """)