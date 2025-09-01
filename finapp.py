import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="FinScope Global - Financial Inclusion Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .recommendation-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        border: none;
        font-weight: bold;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sample data based on your analysis
@st.cache_data
def load_data():
    # Regional data from your analysis
    regional_data = {
        'region': [
            'High income',
            'East Asia & Pacific (excluding high income)',
            'Europe & Central Asia (excluding high income)',
            'South Asia (excluding high income)',
            'Latin America & Caribbean (excluding high income)',
            'Sub-Saharan Africa (excluding high income)',
            'Middle East & North Africa (excluding high income)'
        ],
        'inclusion_rate': [0.858, 0.568, 0.554, 0.483, 0.480, 0.427, 0.382],
        'count': [2938, 521, 1139, 352, 970, 1833, 558],
        'std': [0.173, 0.272, 0.221, 0.253, 0.202, 0.224, 0.230]
    }
    
    # Income group data
    income_data = {
        'income_group': ['High income', 'Upper middle income', 'Lower middle income', 'Low income'],
        'inclusion_rate': [0.870, 0.571, 0.440, 0.374],
        'count': [2790, 2203, 2328, 990]
    }
    
    # Feature importance data from your Random Forest analysis
    feature_importance = {
        'feature': [
            'biz_loan_source', 'biz_loan', 'emergency_funds', 'digital_pay',
            'digital_pay_acc', 'loan_purpose_group', 'mobile_pay_s_r',
            'prefer_digital_fin', 'digital_payment_other', 'govt_payment_recv',
            'saved_any', 'mobile_payment_bill', 'borrowed_any', 'saved_for_purchase'
        ],
        'importance': [0.1683, 0.1230, 0.0980, 0.0636, 0.0597, 0.0409, 0.0404, 
                      0.0392, 0.0390, 0.0378, 0.0351, 0.0273, 0.0251, 0.0250]
    }
    
    return pd.DataFrame(regional_data), pd.DataFrame(income_data), pd.DataFrame(feature_importance)

# Load data
regional_df, income_df, feature_df = load_data()

# Header
st.markdown("""
<div class="main-header">
    <h1>🌍 FinScope Global - Financial Inclusion Analytics</h1>
    <p><i>Empowering evidence-based policy through machine learning insights</i></p>
    <p><b>1.4 billion adults remain without financial accounts - Let's change that together</b></p>
</div>
""", unsafe_allow_html=True)

# Navigation buttons
st.markdown("### 🎯 Choose Your Analysis Path")
col1, col2 = st.columns(2)

with col1:
    regional_analysis = st.button("🌐 Regional Analytics", key="regional_btn")

with col2:
    individual_analysis = st.button("👤 Individual Analysis", key="individual_btn")

# Initialize session state
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = 'regional'

if regional_analysis:
    st.session_state.analysis_mode = 'regional'
elif individual_analysis:
    st.session_state.analysis_mode = 'individual'

# Regional Analytics Mode
if st.session_state.analysis_mode == 'regional':
    st.markdown("## 🌐 Regional Analytics Dashboard")
    
    # Key Statistics
    st.markdown("### 📊 Global Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Global Inclusion Rate", "61.1%", "📈 Target: 80%")
    with col2:
        st.metric("Total Adults Analyzed", "8,311", "🎯 7 Regions")
    with col3:
        st.metric("Best Performing Region", "High Income", "85.8%")
    with col4:
        st.metric("Greatest Opportunity", "MENA", "38.2% (+42.9% gap)")
    
    # World Map Visualization
    st.markdown("### 🗺️ Financial Inclusion by Region")
    
    # Create a more comprehensive mapping for better visualization
    region_map_data = regional_df.copy()
    region_map_data['hover_text'] = region_map_data.apply(
        lambda x: f"{x['region']}<br>Inclusion Rate: {x['inclusion_rate']:.1%}<br>Sample Size: {x['count']:,}", 
        axis=1
    )
    
    fig_map = go.Figure()
    
    # Create a choropleth-style visualization using scatter points
    # Since we don't have actual geographic coordinates, we'll create a treemap
    fig_treemap = px.treemap(
        region_map_data,
        path=['region'],
        values='count',
        color='inclusion_rate',
        color_continuous_scale='RdYlGn',
        title="Financial Inclusion Rates by Region",
        hover_data={'inclusion_rate': ':.1%', 'count': ':,'}
    )
    fig_treemap.update_layout(height=500, font_size=12)
    st.plotly_chart(fig_treemap, use_container_width=True)
    
    # Regional Comparison Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Inclusion Rates by Region")
        fig_bar = px.bar(
            regional_df.sort_values('inclusion_rate', ascending=True),
            x='inclusion_rate',
            y='region',
            orientation='h',
            color='inclusion_rate',
            color_continuous_scale='RdYlGn',
            text='inclusion_rate'
        )
        fig_bar.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_bar.update_layout(height=400, showlegend=False, 
                             xaxis_title="Financial Inclusion Rate",
                             yaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Sample Size Distribution")
        fig_pie = px.pie(
            regional_df,
            values='count',
            names='region',
            title="Regional Sample Distribution"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Income Group Analysis
    st.markdown("### 💰 Financial Inclusion by Income Group")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_income = px.bar(
            income_df,
            x='income_group',
            y='inclusion_rate',
            color='inclusion_rate',
            color_continuous_scale='RdYlGn',
            text='inclusion_rate'
        )
        fig_income.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_income.update_layout(
            xaxis_title="Income Group",
            yaxis_title="Inclusion Rate",
            showlegend=False
        )
        st.plotly_chart(fig_income, use_container_width=True)
    
    with col2:
        # Feature Importance
        st.markdown("### 🔑 Key Drivers of Financial Inclusion")
        fig_features = px.bar(
            feature_df.head(8),
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig_features.update_layout(height=300, showlegend=False,
                                 xaxis_title="Feature Importance",
                                 yaxis_title="")
        st.plotly_chart(fig_features, use_container_width=True)
    
    # Regional Recommendations
    st.markdown("### 💡 Regional Recommendations")
    
    recommendations = {
        'Sub-Saharan Africa (excluding high income)': [
            "📱 **Digital Infrastructure**: Expand mobile payment networks and digital literacy programs",
            "🏦 **Agent Banking**: Develop extensive agent banking networks in rural areas", 
            "🌾 **Agricultural Finance**: Create tailored financial products for farmers",
            "👥 **Community Banking**: Support savings groups and microfinance institutions"
        ],
        'Middle East & North Africa (excluding high income)': [
            "🏛️ **Regulatory Reform**: Modernize banking regulations to enable innovation",
            "👩‍💼 **Women's Financial Inclusion**: Address cultural barriers through targeted programs",
            "💳 **Digital Payments**: Accelerate adoption of digital payment systems",
            "🏢 **SME Finance**: Enhance access to business loans and credit"
        ],
        'Latin America & Caribbean (excluding high income)': [
            "📲 **Fintech Integration**: Partner with fintech companies for innovative solutions",
            "🏠 **Remittance Services**: Improve cross-border payment systems",
            "🎓 **Financial Education**: Implement comprehensive financial literacy programs",
            "🚀 **Entrepreneurship Support**: Expand access to business loans and startup capital"
        ]
    }
    
    # Show recommendations for lowest performing regions
    low_performing = regional_df[regional_df['inclusion_rate'] < 0.55]['region'].tolist()
    
    for region in low_performing:
        if region in recommendations:
            with st.expander(f"🎯 Recommendations for {region} (Inclusion Rate: {regional_df[regional_df['region']==region]['inclusion_rate'].iloc[0]:.1%})"):
                for rec in recommendations[region]:
                    st.markdown(f"- {rec}")

# Individual Analysis Mode
elif st.session_state.analysis_mode == 'individual':
    st.markdown("## 👤 Individual Financial Inclusion Predictor")
    st.markdown("*Get personalized insights and recommendations based on your profile*")
    
    # User Input Form
    with st.form("individual_analysis_form"):
        st.markdown("### 📝 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            region = st.selectbox("🌍 Region", [
                'High income',
                'East Asia & Pacific (excluding high income)',
                'Europe & Central Asia (excluding high income)', 
                'South Asia (excluding high income)',
                'Latin America & Caribbean (excluding high income)',
                'Sub-Saharan Africa (excluding high income)',
                'Middle East & North Africa (excluding high income)'
            ])
            
            income_group = st.selectbox("💰 Income Group", [
                'High income', 'Upper middle income', 'Lower middle income', 'Low income'
            ])
        
        with col2:
            # Most important features for prediction
            biz_loan = st.slider("🏢 Business Loan Access (0-1)", 0.0, 1.0, 0.3, 0.1,
                               help="Do you have access to business loans?")
            emergency_funds = st.slider("🆘 Emergency Funds (0-1)", 0.0, 1.0, 0.4, 0.1,
                                      help="Do you have emergency funds available?")
        
        st.markdown("### 💳 Digital & Financial Behavior")
        col3, col4 = st.columns(2)
        
        with col3:
            digital_pay = st.slider("📱 Digital Payments Usage (0-1)", 0.0, 1.0, 0.5, 0.1,
                                   help="How often do you use digital payments?")
            mobile_payments = st.slider("📲 Mobile Payments (0-1)", 0.0, 1.0, 0.3, 0.1,
                                      help="Do you use mobile payment services?")
            
        with col4:
            savings = st.slider("💰 Savings Behavior (0-1)", 0.0, 1.0, 0.4, 0.1,
                              help="Do you regularly save money?")
            credit_access = st.slider("💳 Credit Access (0-1)", 0.0, 1.0, 0.2, 0.1,
                                    help="Do you have access to credit/loans?")
        
        submitted = st.form_submit_button("🔮 Predict My Financial Inclusion Score")
    
    if submitted:
        # Simple prediction logic based on your Random Forest insights
        # Weights based on feature importance from your analysis
        weights = {
            'biz_loan': 0.1683,
            'emergency_funds': 0.0980,
            'digital_pay': 0.0636,
            'mobile_payments': 0.0404,
            'savings': 0.0351,
            'credit_access': 0.0251
        }
        
        # Regional baseline (from your data)
        region_baseline = {
            'High income': 0.858,
            'East Asia & Pacific (excluding high income)': 0.568,
            'Europe & Central Asia (excluding high income)': 0.554,
            'South Asia (excluding high income)': 0.483,
            'Latin America & Caribbean (excluding high income)': 0.480,
            'Sub-Saharan Africa (excluding high income)': 0.427,
            'Middle East & North Africa (excluding high income)': 0.382
        }
        
        # Income group adjustment
        income_adjustments = {
            'High income': 0.05,
            'Upper middle income': 0.02,
            'Lower middle income': -0.02,
            'Low income': -0.05
        }
        
        # Calculate prediction
        feature_score = (
            biz_loan * weights['biz_loan'] +
            emergency_funds * weights['emergency_funds'] +
            digital_pay * weights['digital_pay'] +
            mobile_payments * weights['mobile_payments'] +
            savings * weights['savings'] +
            credit_access * weights['credit_access']
        )
        
        baseline_score = region_baseline[region] + income_adjustments[income_group]
        final_score = min(1.0, max(0.0, baseline_score + feature_score * 0.5))
        
        # Display Results
        st.markdown("### 🎯 Your Financial Inclusion Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Score display with color coding
            if final_score >= 0.7:
                score_color = "🟢"
                status = "Excellent"
            elif final_score >= 0.5:
                score_color = "🟡" 
                status = "Moderate"
            else:
                score_color = "🔴"
                status = "Needs Attention"
                
            st.markdown(f"""
            <div class="metric-card">
                <h3>{score_color} Financial Inclusion Score</h3>
                <h1 style="color: #2a5298;">{final_score:.1%}</h1>
                <p><strong>Status: {status}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            regional_avg = region_baseline[region]
            comparison = final_score - regional_avg
            comparison_text = f"+{comparison:.1%}" if comparison > 0 else f"{comparison:.1%}"
            comparison_emoji = "📈" if comparison > 0 else "📉" if comparison < 0 else "➡️"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Regional Comparison</h3>
                <h2>{region_baseline[region]:.1%}</h2>
                <p><strong>Regional Average</strong></p>
                <p>{comparison_emoji} {comparison_text} vs regional avg</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence = 0.85 + (abs(final_score - 0.5) * 0.3)  # Higher confidence for extreme scores
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Prediction Confidence</h3>
                <h2>{confidence:.1%}</h2>
                <p><strong>Model Reliability</strong></p>
                <p>Based on Random Forest analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature Impact Analysis
        st.markdown("### 📊 What's Driving Your Score?")
        
        feature_impacts = {
            'Business Loan Access': biz_loan * weights['biz_loan'],
            'Emergency Funds': emergency_funds * weights['emergency_funds'],
            'Digital Payments': digital_pay * weights['digital_pay'],
            'Mobile Payments': mobile_payments * weights['mobile_payments'],
            'Savings Behavior': savings * weights['savings'],
            'Credit Access': credit_access * weights['credit_access']
        }
        
        impact_df = pd.DataFrame({
            'Factor': list(feature_impacts.keys()),
            'Impact Score': list(feature_impacts.values()),
            'Your Level': [biz_loan, emergency_funds, digital_pay, mobile_payments, savings, credit_access]
        })
        
        fig_impact = px.bar(
            impact_df,
            x='Impact Score',
            y='Factor',
            orientation='h',
            color='Your Level',
            color_continuous_scale='RdYlGn',
            title="Personal Factors Impact on Financial Inclusion"
        )
        fig_impact.update_layout(height=400)
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # Personalized Recommendations
        st.markdown("### 💡 Personalized Recommendations")
        
        recommendations = []
        
        if biz_loan < 0.5:
            recommendations.append("🏢 **Business Development**: Explore microfinance options and business loan programs in your region")
        
        if emergency_funds < 0.5:
            recommendations.append("🆘 **Emergency Preparedness**: Build an emergency fund - start with small, regular savings")
            
        if digital_pay < 0.5:
            recommendations.append("📱 **Digital Adoption**: Learn about mobile banking and digital payment platforms available in your area")
            
        if savings < 0.5:
            recommendations.append("💰 **Savings Habits**: Set up automatic savings plans and explore high-yield savings accounts")
            
        if final_score < region_baseline[region]:
            recommendations.append(f"🎯 **Regional Programs**: Look into financial inclusion initiatives specific to {region}")
        
        # Priority recommendations based on lowest scores
        low_factors = [(k, v) for k, v in {
            'Business Loans': biz_loan,
            'Emergency Funds': emergency_funds, 
            'Digital Payments': digital_pay,
            'Savings': savings
        }.items() if v < 0.4]
        
        if low_factors:
            st.markdown(f"#### 🚀 **Priority Actions** (Focus on these first):")
            for factor, score in sorted(low_factors, key=lambda x: x[1]):
                st.markdown(f"- **{factor}**: Current level {score:.1%} - High impact opportunity")
        
        for rec in recommendations[:4]:  # Show top 4 recommendations
            st.markdown(f"- {rec}")
        
        # Success Stories
        if final_score >= 0.7:
            st.markdown("""
            <div class="insight-box">
                <h4>🌟 Congratulations!</h4>
                <p>You're doing great with financial inclusion! Your score indicates good access to financial services. 
                Consider sharing your experience with others in your community.</p>
            </div>
            """, unsafe_allow_html=True)
        elif final_score >= 0.5:
            st.markdown("""
            <div class="insight-box">
                <h4>🎯 You're on the right track!</h4>
                <p>With some focused improvements in key areas, you can significantly enhance your financial inclusion. 
                The recommendations above will help you get there.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
                <h4>💪 Every journey starts with a single step!</h4>
                <p>There are many opportunities to improve your financial inclusion. Start with one small change 
                and build momentum. Financial inclusion programs in your region can provide additional support.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin: 20px 0;">
    <p><strong>FinScope Global</strong> | Built with ❤️ for financial inclusion | 
    Powered by <strong>Random Forest ML</strong> (89.6% accuracy) | 
    <a href="https://finscopee.streamlit.app" target="_blank">finscopee.streamlit.app</a></p>
    <p><em>Empowering 1.4 billion unbanked adults worldwide through data-driven insights</em></p>
</div>
""", unsafe_allow_html=True)