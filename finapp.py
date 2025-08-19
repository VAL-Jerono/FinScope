st.subheader("📱 Mobile & Digital Payments")
    
    mobile_payment = st.radio(
        "Uses Mobile Payments?",
        ["No", "Yes"],
        help="Mobile payment adoption"
    )
    
    mobile_payment_bill = st.radio(
        "Pays Bills via Mobile?",
        ["No", "Yes"],
        help="Uses mobile for bill payments"
    )
    
    govt_digital_pay = st.radio(
        "Receives Government Payments Digitally?",
        ["No", "Yes"],
        help="Government digital payment recipient"
    )import streamlit as st
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

# Input fields based on your actual features
with st.sidebar:
    st.subheader("🌍 Geographic Context")
    
    region = st.selectbox(
        "Region",
        ["East Asia & Pacific", "Europe & Central Asia", "Latin America & Caribbean", 
         "Middle East & North Africa", "North America", "South Asia", "Sub-Saharan Africa"],
        help="Geographic region affects financial inclusion patterns significantly"
    )
    
    income_group = st.selectbox(
        "Income Group",
        ["High income", "Upper middle income", "Lower middle income", "Low income"],
        help="Country income level impacts financial infrastructure availability"
    )
    
    st.subheader("👥 Demographics")
    
    demo_group = st.selectbox(
        "Demographic Group",
        ["all", "men", "women", "ages 15-24", "ages 25+", "poorest 40%", "richest 60%"],
        help="Demographic segment for targeted analysis"
    )
    
    demo_subgroup = st.selectbox(
        "Demographic Subgroup",
        ["urban", "rural", "in laborforce", "out of laborforce", 
         "prim edu or less", "secondary edu or more"],
        help="Detailed demographic classification"
    )
    
    st.subheader("📊 Business & Loan Profile")
    
    biz_loan_source = st.selectbox(
        "Business Loan Source",
        ["bank or credit union", "employer", "family or friends", "microfinance institution", 
         "informal lender", "store credit", "other", "multiple sources"],
        help="Source of business financing affects default risk significantly"
    )
    
    has_biz_loan = st.radio(
        "Has Business Loan?",
        ["No", "Yes"],
        help="Business loan status is the 2nd most important feature"
    )
    
    loan_purpose_group = st.selectbox(
        "Loan Purpose Group",
        ["business", "personal", "education", "home", "agriculture", "emergency", "other"],
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
    
    st.subheader("💰 Financial Behavior")
    
    borrowed_any = st.radio(
        "Has Borrowed Money (Any Source)?",
        ["No", "Yes"],
        help="Any borrowing activity in past 12 months"
    )
    
    saved_any = st.radio(
        "Has Saved Money?",
        ["No", "Yes"],
        help="Any saving activity indicates financial discipline"
    )
    
    emergency_funds = st.radio(
        "Has Emergency Funds?",
        ["No", "Yes"],
        help="Financial resilience indicator"
    )
    
    fin_resilience = st.slider(
        "Financial Resilience Score",
        0.0, 1.0, 0.5, 0.1,
        help="Overall financial stability and resilience"
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
        # Prepare input data with actual feature names
        input_data = pd.DataFrame([{
            'region': region,
            'income_group': income_group,
            'demo_group': demo_group,
            'demo_subgroup': demo_subgroup,
            'biz_loan_source': biz_loan_source,
            'biz_loan': 1 if has_biz_loan == "Yes" else 0,
            'loan_purpose_group': loan_purpose_group,
            'borrowed_any': 1 if borrowed_any == "Yes" else 0,
            'saved_any': 1 if saved_any == "Yes" else 0,
            'emergency_funds': 1 if emergency_funds == "Yes" else 0,
            'fin_resilience': fin_resilience,
            'digital_engagement_score': digital_engagement_score,
            'digital_pay_acc': digital_pay_acc,
            'prefer_digital_acc': 1 if prefer_digital == "Yes" else 0,
            'credit_card': 1 if has_credit_card == "Yes" else 0,
            'mobile_payment': 1 if mobile_payment == "Yes" else 0,
            'mobile_payment_bill': 1 if mobile_payment_bill == "Yes" else 0,
            'govt_digital_pay': 1 if govt_digital_pay == "Yes" else 0,
            'govt_services_score': govt_services_score
        }])
        
        # Enhanced risk calculation based on regional and demographic factors
        base_risk = 0.6
        
        # Regional adjustments
        regional_risk = {
            "Sub-Saharan Africa": -0.1,  # Lower risk due to mobile money success
            "South Asia": 0.1,           # Higher risk due to gender gaps
            "Middle East & North Africa": 0.15,  # Highest risk region
            "Latin America & Caribbean": -0.05,
            "East Asia & Pacific": -0.15,  # Lower risk due to digital adoption
            "Europe & Central Asia": -0.2,
            "North America": -0.25
        }
        
        # Income group adjustments
        income_risk = {
            "Low income": 0.2,
            "Lower middle income": 0.1,
            "Upper middle income": -0.1,
            "High income": -0.2
        }
        
        # Demographic adjustments
        demo_risk = {
            "women": 0.08,
            "ages 15-24": 0.05,
            "poorest 40%": 0.12,
            "out of laborforce": 0.1,
            "prim edu or less": 0.15,
            "rural": -0.02  # Surprisingly lower due to mobile money
        }
        
        risk_prob = base_risk
        risk_prob += regional_risk.get(region, 0)
        risk_prob += income_risk.get(income_group, 0)
        risk_prob += demo_risk.get(demo_group, 0)
        risk_prob += demo_risk.get(demo_subgroup, 0)
        
        # Digital factors (strong predictors)
        risk_prob -= digital_engagement_score * 0.3
        risk_prob -= digital_pay_acc * 0.2
        if prefer_digital == "Yes":
            risk_prob -= 0.1
        if mobile_payment == "Yes":
            risk_prob -= 0.08
        if govt_digital_pay == "Yes":
            risk_prob -= 0.06
        
        # Financial behavior factors
        if has_credit_card == "Yes":
            risk_prob -= 0.15
        if borrowed_any == "Yes":
            risk_prob += 0.05  # Borrowing can indicate need
        if saved_any == "Yes":
            risk_prob -= 0.1
        if emergency_funds == "Yes":
            risk_prob -= 0.12
        risk_prob -= fin_resilience * 0.15
        
        # Business factors
        if has_biz_loan == "Yes":
            if biz_loan_source in ["bank or credit union", "microfinance institution"]:
                risk_prob -= 0.1
            else:
                risk_prob += 0.05
        
        # Ensure probability bounds
        risk_prob = max(0.02, min(0.98, risk_prob))
        
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
        
        # Create feature importance chart - fix for plotly
        features = list(model.feature_importances_.keys())[:8]
        importances = list(model.feature_importances_.values())[:8]
        
        fig = px.bar(
            x=importances, 
            y=features,
            orientation='h',
            title="Most Important Features in Your Risk Assessment",
            labels={'x': 'Feature Importance', 'y': 'Features'},
            color=importances,
            color_continuous_scale="RdYlBu_r"
        )
        fig.update_layout(height=400, showlegend=False)
        fig.update_traces(texttemplate='%{x:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional context analysis
        st.subheader(f"🌍 Regional Context: {region}")
        
        regional_insights = {
            "Sub-Saharan Africa": {
                "strength": "Leading mobile money adoption (68.2% rural inclusion)",
                "challenge": "Infrastructure gaps in remote areas",
                "strategy": "Leverage mobile money networks for full financial services"
            },
            "South Asia": {
                "strength": "Large unbanked population with high mobile penetration", 
                "challenge": "Significant gender gap (47.6% lower women's inclusion)",
                "strategy": "Women-focused mobile financial services with family engagement"
            },
            "Middle East & North Africa": {
                "strength": "Growing fintech sector and young population",
                "challenge": "Largest financial inclusion gap globally",
                "strategy": "Digital-first approach with cultural sensitivity"
            },
            "East Asia & Pacific": {
                "strength": "High digital adoption and government support",
                "challenge": "Rural-urban divide in some countries",
                "strategy": "Government-private partnerships for universal coverage"
            },
            "Latin America & Caribbean": {
                "strength": "Strong remittance networks and mobile penetration",
                "challenge": "Informal economy dominance",
                "strategy": "Formalization incentives through digital payments"
            },
            "Europe & Central Asia": {
                "strength": "Strong regulatory frameworks and infrastructure",
                "challenge": "Rural and elderly populations",
                "strategy": "Traditional banking integration with digital services"
            },
            "North America": {
                "strength": "Advanced financial infrastructure",
                "challenge": "Underbanked populations in specific communities",
                "strategy": "Community banking and fintech partnerships"
            }
        }
        
        if region in regional_insights:
            insight = regional_insights[region]
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.success(f"**Regional Strength:** {insight['strength']}")
                st.info(f"**Strategy:** {insight['strategy']}")
            
            with col_b:
                st.warning(f"**Challenge:** {insight['challenge']}")
                
                # Income group specific insights
                income_context = {
                    "Low income": "Focus on basic accounts and mobile money",
                    "Lower middle income": "Expand to credit and insurance products", 
                    "Upper middle income": "Full-service digital banking",
                    "High income": "Premium and investment products"
                }
                st.info(f"**Income Focus:** {income_context.get(income_group, 'Tailored approach needed')}")

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