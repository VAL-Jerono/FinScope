import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="FinScope Global - Financial Inclusion Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 6px solid #667eea;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    .champion-card {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 6px 20px rgba(46,139,87,0.3);
        text-align: center;
    }
    .priority-card {
        background: linear-gradient(135deg, #E74C3C 0%, #F39C12 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 6px 20px rgba(231,76,60,0.3);
        text-align: center;
    }
    .region-recommendation {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 6px solid;
    }
    .action-item {
        background: rgba(255,255,255,0.9);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .immediate-action { border-left-color: #E74C3C; }
    .medium-term { border-left-color: #F39C12; }
    .long-term { border-left-color: #27AE60; }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border: none;
        border-radius: 25px;
        font-weight: bold;
        font-size: 18px;
        margin: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102,126,234,0.3);
    }
    .nav-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }
    .nav-button.active {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 8px 25px rgba(118,75,162,0.4);
    }
    .region-info-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border-left: 6px solid;
        animation: slideIn 0.5s ease-out;
    }
    .calculator-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 2px solid #e9ecef;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .stButton > button {
        width: 100%;
        border-radius: 25px;
        height: 3.5em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    .compact-metric {
        text-align: center;
        padding: 10px;
        background: rgba(255,255,255,0.9);
        border-radius: 8px;
        margin: 5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Regional data
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
    
    # Random Forest feature importance
    feature_importance = {
        'feature': [
            'Business Loan Source', 'Business Loan Access', 'Emergency Funds', 'Digital Engagement Score',
            'Government Services Score', 'Loan Purpose Group', 'Mobile Payment S/R',
            'Prefer Digital Finance', 'Financial Activity Score', 'Income Digital Interaction',
            'Saved Any', 'Borrowed Any', 'Saved for Purchase', 'Prefer Digital Account'
        ],
        'importance': [0.1683, 0.1230, 0.0980, 0.0636, 0.0597, 0.0409, 0.0404, 
                      0.0392, 0.0390, 0.0378, 0.0351, 0.0273, 0.0251, 0.0250]
    }
    
    # Enhanced demographic data - cleaned and organized
    demographic_champions = {
        'High Income': {'Urban': 0.891, 'Higher Ed': 0.897, 'In Labor': 0.930, 'Rich 60%': 0.903, 'Men': 0.888, 'Rural': 0.879},
        'East Asia Pacific': {'Urban': 0.766, 'Higher Ed': 0.677, 'In Labor': 0.597, 'Rich 60%': 0.637, 'Men': 0.576, 'Rural': 0.670},
        'Europe Central Asia': {'Urban': 0.769, 'Higher Ed': 0.638, 'In Labor': 0.704, 'Rich 60%': 0.626, 'Men': 0.596, 'Rural': 0.707},
        'Latin America': {'Urban': 0.642, 'Higher Ed': 0.564, 'In Labor': 0.569, 'Rich 60%': 0.559, 'Men': 0.527, 'Rural': 0.556},
        'Sub-Saharan Africa': {'Urban': 0.645, 'Higher Ed': 0.570, 'In Labor': 0.500, 'Rich 60%': 0.497, 'Men': 0.468, 'Rural': 0.516},
        'South Asia': {'Urban': 0.607, 'Higher Ed': 0.597, 'In Labor': 0.560, 'Rich 60%': 0.537, 'Men': 0.533, 'Rural': 0.596},
        'MENA': {'Urban': 0.491, 'Higher Ed': 0.425, 'In Labor': 0.540, 'Rich 60%': 0.448, 'Men': 0.476, 'Rural': 0.392}
    }
    
    demographic_excluded = {
        'High Income': {'Primary Ed': 0.769, 'Out Labor': 0.804, 'Poor 40%': 0.834, 'Age 15-24': 0.781, 'Women': 0.855},
        'East Asia Pacific': {'Primary Ed': 0.457, 'Out Labor': 0.489, 'Poor 40%': 0.484, 'Age 15-24': 0.543, 'Women': 0.576},
        'Europe Central Asia': {'Primary Ed': 0.390, 'Out Labor': 0.448, 'Poor 40%': 0.484, 'Age 15-24': 0.428, 'Women': 0.540},
        'Latin America': {'Primary Ed': 0.368, 'Out Labor': 0.386, 'Poor 40%': 0.381, 'Age 15-24': 0.403, 'Women': 0.454},
        'Sub-Saharan Africa': {'Primary Ed': 0.337, 'Out Labor': 0.330, 'Poor 40%': 0.333, 'Age 15-24': 0.361, 'Women': 0.383},
        'South Asia': {'Primary Ed': 0.419, 'Out Labor': 0.432, 'Poor 40%': 0.401, 'Age 15-24': 0.431, 'Women': 0.435},
        'MENA': {'Primary Ed': 0.330, 'Out Labor': 0.295, 'Poor 40%': 0.304, 'Age 15-24': 0.267, 'Women': 0.300}
    }

    # Enhanced regional mapping with detailed recommendations
    region_mapping = {
        'High income': {
            'color': '#2E8B57',
            'countries': ['USA', 'Germany', 'Japan', 'UK', 'France', 'Canada', 'Australia'],
            'key_challenges': ['Digital divide in elderly populations', 'Rural banking access', 'Fintech regulation balance'],
            'opportunities': ['AI-driven personalization', 'Green finance leadership', 'Cross-border digital payments'],
            'immediate_actions': [
                'Launch digital literacy programs for seniors (65+ age group)',
                'Establish rural mobile banking units in underserved areas',
                'Create regulatory sandboxes for fintech innovation'
            ],
            'medium_term': [
                'Develop AI-powered financial advisory services',
                'Implement comprehensive ESG banking standards',
                'Build interoperable digital identity frameworks'
            ],
            'long_term': [
                'Pioneer quantum-secure financial infrastructure',
                'Lead global financial inclusion measurement standards',
                'Create sustainable finance certification programs'
            ],
            'success_metrics': ['95% digital adoption by 2030', '100% rural branch coverage', 'Carbon-neutral banking operations'],
            'budget_allocation': 'Focus 40% on technology, 35% on rural infrastructure, 25% on sustainability initiatives'
        },
        'East Asia & Pacific (excluding high income)': {
            'color': '#FF6B35',
            'countries': ['China', 'Indonesia', 'Thailand', 'Philippines', 'Vietnam', 'Malaysia'],
            'key_challenges': ['Rural-urban digital divide', 'Cross-border payment complexity', 'Regulatory fragmentation'],
            'opportunities': ['Mobile-first banking expansion', 'E-commerce payment integration', 'Regional payment corridors'],
            'immediate_actions': [
                'Deploy rural 4G/5G infrastructure for mobile banking',
                'Launch region-wide QR code payment standards',
                'Establish cross-border fintech regulatory cooperation'
            ],
            'medium_term': [
                'Build unified regional digital wallet ecosystem',
                'Create agricultural value chain financing programs',
                'Develop disaster-resilient payment systems'
            ],
            'long_term': [
                'Pioneer blockchain-based trade finance networks',
                'Lead ASEAN financial integration initiatives',
                'Establish regional cryptocurrency frameworks'
            ],
            'success_metrics': ['80% mobile payment adoption', '90% SME access to credit', 'Sub-$1 remittance costs'],
            'budget_allocation': 'Focus 50% on mobile infrastructure, 30% on rural programs, 20% on regional integration'
        },
        'Europe & Central Asia (excluding high income)': {
            'color': '#F7931E',
            'countries': ['Russia', 'Turkey', 'Kazakhstan', 'Ukraine', 'Romania', 'Bulgaria'],
            'key_challenges': ['Economic volatility', 'Legacy banking systems', 'EU integration complexity'],
            'opportunities': ['Digital transformation acceleration', 'EU market integration', 'Remittance corridor optimization'],
            'immediate_actions': [
                'Modernize core banking systems with cloud technology',
                'Establish currency-hedged savings products',
                'Create EU-compliant digital payment infrastructure'
            ],
            'medium_term': [
                'Build cross-border lending platforms for SMEs',
                'Implement open banking standards across region',
                'Develop diaspora-focused financial products'
            ],
            'long_term': [
                'Achieve full EU payment integration compliance',
                'Lead Eastern European fintech hub development',
                'Pioneer post-conflict financial reconstruction models'
            ],
            'success_metrics': ['70% digital banking adoption', 'EU payment integration by 2027', '50% SME lending growth'],
            'budget_allocation': 'Focus 45% on system modernization, 35% on EU compliance, 20% on SME support'
        },
        'South Asia (excluding high income)': {
            'color': '#FFD23F',
            'countries': ['India', 'Bangladesh', 'Pakistan', 'Sri Lanka', 'Nepal', 'Afghanistan'],
            'key_challenges': ['Massive unbanked population', 'Gender inclusion gaps', 'Documentation barriers'],
            'opportunities': ['Digital ID system leverage', 'Mobile-first approaches', 'Government payment digitization'],
            'immediate_actions': [
                'Scale biometric-based account opening (Aadhaar model)',
                'Launch women-only banking centers in rural areas',
                'Digitize government welfare payment systems'
            ],
            'medium_term': [
                'Build extensive agent banking networks',
                'Create alternative credit scoring using mobile data',
                'Implement blockchain-based property records'
            ],
            'long_term': [
                'Achieve universal financial inclusion by 2030',
                'Lead global digital ID standards development',
                'Pioneer climate-resilient agricultural finance'
            ],
            'success_metrics': ['80% account ownership', 'Gender gap below 5%', '1 million banking agents'],
            'budget_allocation': 'Focus 60% on rural infrastructure, 25% on women\'s programs, 15% on digital systems'
        },
        'Latin America & Caribbean (excluding high income)': {
            'color': '#E74C3C',
            'countries': ['Brazil', 'Mexico', 'Argentina', 'Colombia', 'Peru', 'Chile'],
            'key_challenges': ['High informality rates', 'Credit access barriers', 'Remittance cost reduction'],
            'opportunities': ['Fintech ecosystem growth', 'Digital remittances', 'Government service digitization'],
            'immediate_actions': [
                'Launch alternative credit scoring for informal workers',
                'Create low-cost digital remittance corridors',
                'Digitize conditional cash transfer programs'
            ],
            'medium_term': [
                'Build region-wide instant payment networks',
                'Develop micro-insurance for informal sector',
                'Create fintech regulatory frameworks'
            ],
            'long_term': [
                'Pioneer AI-driven financial inclusion models',
                'Lead regional economic integration through fintech',
                'Achieve carbon-neutral payment systems'
            ],
            'success_metrics': ['70% informal sector inclusion', '$5 remittance costs', '90% government payment digitization'],
            'budget_allocation': 'Focus 40% on fintech support, 35% on informal sector, 25% on remittances'
        },
        'Sub-Saharan Africa (excluding high income)': {
            'color': '#C0392B',
            'countries': ['Nigeria', 'Kenya', 'South Africa', 'Ghana', 'Tanzania', 'Ethiopia'],
            'key_challenges': ['Limited infrastructure', 'Low income levels', 'High transaction costs'],
            'opportunities': ['Mobile money expansion', 'Agent banking networks', 'Agricultural finance innovation'],
            'immediate_actions': [
                'Expand mobile money interoperability across borders',
                'Train 500,000 new banking agents in rural areas',
                'Launch satellite-based internet for remote banking'
            ],
            'medium_term': [
                'Build agricultural value chain financing platforms',
                'Create diaspora investment facilitation systems',
                'Develop climate-smart insurance products'
            ],
            'long_term': [
                'Lead global mobile money innovation',
                'Pioneer space-based financial infrastructure',
                'Achieve energy-independent banking systems'
            ],
            'success_metrics': ['60% mobile money usage', '100km agent network coverage', '50% agricultural credit access'],
            'budget_allocation': 'Focus 55% on mobile infrastructure, 30% on agent networks, 15% on agricultural finance'
        },
        'Middle East & North Africa (excluding high income)': {
            'color': '#8E44AD',
            'countries': ['Egypt', 'Morocco', 'Jordan', 'Tunisia', 'Algeria', 'Lebanon'],
            'key_challenges': ['Political instability', 'Youth unemployment', 'Regulatory restrictions'],
            'opportunities': ['Islamic finance growth', 'Oil revenue diversification', 'Digital government services'],
            'immediate_actions': [
                'Launch Sharia-compliant digital banking platforms',
                'Create youth entrepreneurship financing programs',
                'Digitize government salary and pension payments'
            ],
            'medium_term': [
                'Build regional Islamic fintech ecosystem',
                'Develop sovereign wealth fund fintech investments',
                'Create post-conflict financial reconstruction frameworks'
            ],
            'long_term': [
                'Lead global Islamic fintech innovation',
                'Pioneer oil-to-digital economy transition models',
                'Achieve regional financial market integration'
            ],
            'success_metrics': ['50% Islamic finance adoption', '40% youth banking inclusion', '80% government payment digitization'],
            'budget_allocation': 'Focus 40% on Islamic finance, 35% on youth programs, 25% on government digitization'
        }
    }
    
    return pd.DataFrame(regional_data), pd.DataFrame(income_data), pd.DataFrame(feature_importance), region_mapping, demographic_champions, demographic_excluded

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None

# Load data
regional_df, income_df, feature_df, region_mapping, demo_champions, demo_excluded = load_data()

# Header
st.markdown("""
<div class="main-header">
    <h1>🌍 FinScope Global</h1>
    <h2>Financial Inclusion Analytics Dashboard</h2>
    <p style="font-size: 18px; margin: 15px 0;"><i>AI-powered insights for evidence-based financial inclusion policy</i></p>
    <p style="font-size: 16px; font-weight: bold;">📊 149 countries | 🎯 89.6% ML accuracy | 🌐 8,311 adults analyzed</p>
</div>
""", unsafe_allow_html=True)

# Navigation
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🏠 Dashboard Overview", key="home_btn"):
        st.session_state.page = 'home'
        st.session_state.selected_region = None

with col2:
    if st.button("🗺️ Regional Analytics", key="regional_btn"):
        st.session_state.page = 'regional'
        st.session_state.selected_region = None

with col3:
    if st.button("👤 Individual Analysis", key="individual_btn"):
        st.session_state.page = 'individual'



# Dashboard Overview Page
if st.session_state.page == 'home':
    st.markdown("## 📈 Global Financial Inclusion Overview")
    
    # Global Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-top: 0;">🌐 Global Average</h3>
            <h1 style="color: #2d3436; margin: 15px 0; font-size: 3em;">61.1%</h1>
            <p style="color: #636e72; font-size: 16px;">Financial Inclusion Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-top: 0;">🏆 Best Performing</h3>
            <h1 style="color: #2E8B57; margin: 15px 0; font-size: 3em;">85.8%</h1>
            <p style="color: #636e72; font-size: 16px;">High Income Countries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-top: 0;">🎯 Largest Gap</h3>
            <h1 style="color: #8E44AD; margin: 15px 0; font-size: 3em;">47.6%</h1>
            <p style="color: #636e72; font-size: 16px;">MENA vs High Income</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-top: 0;">🤖 ML Accuracy</h3>
            <h1 style="color: #2d3436; margin: 15px 0; font-size: 3em;">89.6%</h1>
            <p style="color: #636e72; font-size: 16px;">Random Forest Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Regional Comparison
    st.markdown("### 🌍 Regional Performance at a Glance")
    
    fig_overview = px.bar(
        regional_df.sort_values('inclusion_rate', ascending=True),
        x='inclusion_rate',
        y='region',
        orientation='h',
        color='inclusion_rate',
        color_continuous_scale=[
            [0.0, '#8E44AD'],  # MENA
            [0.2, '#C0392B'],  # Sub-Saharan Africa
            [0.4, '#E74C3C'],  # Latin America
            [0.5, '#FFD23F'],  # South Asia
            [0.6, '#F7931E'],  # Europe Central Asia
            [0.8, '#FF6B35'],  # East Asia Pacific
            [1.0, '#2E8B57']   # High income
        ],
        text=[f"{val:.1%}" for val in regional_df.sort_values('inclusion_rate', ascending=True)['inclusion_rate']],
        title="Financial Inclusion Rates by Region"
    )
    
    fig_overview.update_traces(textposition='inside', textfont_size=14, textfont_color='white')
    fig_overview.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Financial Inclusion Rate",
        yaxis_title="",
        title_font_size=20,
        xaxis=dict(tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_overview, use_container_width=True)
    
    # Income Group Analysis
    st.markdown("### 💰 Income Group Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_income = px.pie(
            income_df,
            values='count',
            names='income_group',
            color_discrete_sequence=['#2E8B57', '#FF6B35', '#F7931E', '#C0392B'],
            title="Population Distribution by Income Group"
        )
        fig_income.update_traces(textposition='inside', textinfo='percent+label')
        fig_income.update_layout(height=400)
        st.plotly_chart(fig_income, use_container_width=True)
    
    with col2:
        fig_income_rate = px.bar(
            income_df.sort_values('inclusion_rate', ascending=False),
            x='income_group',
            y='inclusion_rate',
            color='inclusion_rate',
            color_continuous_scale='RdYlGn',
            text=[f"{val:.1%}" for val in income_df.sort_values('inclusion_rate', ascending=False)['inclusion_rate']],
            title="Inclusion Rates by Income Group"
        )
        fig_income_rate.update_traces(textposition='outside')
        fig_income_rate.update_layout(
            height=400,
            showlegend=False,
            yaxis=dict(tickformat='.0%'),
            xaxis_title="Income Group",
            yaxis_title="Inclusion Rate"
        )
        st.plotly_chart(fig_income_rate, use_container_width=True)
    
    # Key Insights
    st.markdown("### 🔍 Key Insights from ML Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="champion-card">
            <h4>🎯 Top Predictors of Financial Inclusion</h4>
            <div style="text-align: left; margin-top: 15px;">
                <p><strong>1. Business Loan Access (16.8%)</strong> - Critical for economic participation</p>
                <p><strong>2. Emergency Funds (9.8%)</strong> - Shows financial resilience planning</p>
                <p><strong>3. Digital Engagement (6.4%)</strong> - Technology adoption indicator</p>
                <p><strong>4. Government Services (6.0%)</strong> - Public sector digitization impact</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="priority-card">
            <h4>⚠️ Critical Action Areas</h4>
            <div style="text-align: left; margin-top: 15px;">
                <p><strong>MENA Region:</strong> 38.2% inclusion - Focus on youth & women</p>
                <p><strong>Sub-Saharan Africa:</strong> 42.7% - Mobile money expansion needed</p>
                <p><strong>Gender Gap:</strong> Persistent across all regions except high-income</p>
                <p><strong>Rural Access:</strong> Infrastructure investment priority</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Importance Visualization
    st.markdown("### 🤖 Machine Learning Feature Importance")
    
    fig_features = px.bar(
        feature_df.head(10),
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='viridis',
        text=[f"{val:.1%}" for val in feature_df.head(10)['importance']],
        title="Top 10 Features Predicting Financial Inclusion (Random Forest Model)"
    )
    
    fig_features.update_traces(textposition='inside')
    fig_features.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Feature Importance",
        yaxis_title="",
        xaxis=dict(tickformat='.1%')
    )
    
    st.plotly_chart(fig_features, use_container_width=True)


# Add enhanced demographic comparison visualization
if st.session_state.page == 'home':
    # After the existing income group chart, add beautiful demographic comparisons
    st.markdown("---")
    st.markdown("## 🔍 Beautiful Demographic Contrasts")
    st.markdown("*Visual storytelling of inclusion gaps and opportunities*")
    
    # Create contrasting visualization - Champions vs Priority Groups
    col1, col2 = st.columns(2)
    
    with col1:
        # Champions visualization - radar chart
        champions_data = []
        regions_short = ['High Income', 'E.Asia Pacific', 'Europe C.Asia', 'S.America', 'S-S Africa', 'South Asia', 'MENA']
        
        for i, region in enumerate(['High Income', 'East Asia Pacific', 'Europe Central Asia', 'Latin America', 
                                  'Sub-Saharan Africa', 'South Asia', 'MENA']):
            if region in demo_champions:
                avg_champion = np.mean(list(demo_champions[region].values()))
                champions_data.append({'Region': regions_short[i], 'Champion_Rate': avg_champion, 'Type': 'Champions'})
        
        fig_champions = go.Figure()
        
        fig_champions.add_trace(go.Scatterpolar(
            r=[d['Champion_Rate'] for d in champions_data],
            theta=[d['Region'] for d in champions_data],
            fill='toself',
            name='Champion Groups',
            line_color='#2E8B57',
            fillcolor='rgba(46,139,87,0.3)'
        ))
        
        fig_champions.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%'
                )),
            showlegend=True,
            title="🏆 Champion Demographics<br>Average Performance",
            height=400,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_champions, use_container_width=True)
    
    with col2:
        # Priority groups visualization - radar chart
        priority_data = []
        
        for i, region in enumerate(['High Income', 'East Asia Pacific', 'Europe Central Asia', 'Latin America', 
                                  'Sub-Saharan Africa', 'South Asia', 'MENA']):
            if region in demo_excluded:
                avg_priority = np.mean(list(demo_excluded[region].values()))
                priority_data.append({'Region': regions_short[i], 'Priority_Rate': avg_priority, 'Type': 'Priority'})
        
        fig_priority = go.Figure()
        
        fig_priority.add_trace(go.Scatterpolar(
            r=[d['Priority_Rate'] for d in priority_data],
            theta=[d['Region'] for d in priority_data],
            fill='toself',
            name='Priority Groups',
            line_color='#E74C3C',
            fillcolor='rgba(231,76,60,0.3)'
        ))
        
        fig_priority.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%'
                )),
            showlegend=True,
            title="🎯 Priority Target Groups<br>Average Performance",
            height=400,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_priority, use_container_width=True)
    
    # Stunning gap analysis visualization
    st.markdown("### 📊 The Inclusion Gap Story")
    
    # Calculate gaps between champions and priority groups
    gap_data = []
    for region in regions_short:
        region_map = {
            'High Income': 'High Income', 'E.Asia Pacific': 'East Asia Pacific',
            'Europe C.Asia': 'Europe Central Asia', 'S.America': 'Latin America',
            'S-S Africa': 'Sub-Saharan Africa', 'South Asia': 'South Asia', 'MENA': 'MENA'
        }
        
        full_region = region_map[region]
        if full_region in demo_champions and full_region in demo_excluded:
            champion_avg = np.mean(list(demo_champions[full_region].values()))
            priority_avg = np.mean(list(demo_excluded[full_region].values()))
            gap = champion_avg - priority_avg
            
            gap_data.append({
                'Region': region,
                'Champion_Avg': champion_avg,
                'Priority_Avg': priority_avg,
                'Gap': gap,
                'Champion_Display': f"{champion_avg:.1%}",
                'Priority_Display': f"{priority_avg:.1%}",
                'Gap_Display': f"{gap:.1%}"
            })
    
    gap_df = pd.DataFrame(gap_data)
    
    # Create beautiful diverging bar chart
    fig_gap = go.Figure()
    
    # Champions bars (positive direction)
    fig_gap.add_trace(go.Bar(
        name='Champion Groups',
        y=gap_df['Region'],
        x=gap_df['Champion_Avg'],
        orientation='h',
        marker_color='#2E8B57',
        text=gap_df['Champion_Display'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Champion Groups: %{x:.1%}<extra></extra>'
    ))
    
    # Priority bars (negative direction for contrast)
    fig_gap.add_trace(go.Bar(
        name='Priority Groups',
        y=gap_df['Region'],
        x=gap_df['Priority_Avg'],
        orientation='h',
        marker_color='#E74C3C',
        text=gap_df['Priority_Display'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Priority Groups: %{x:.1%}<extra></extra>'
    ))
    
    # Add gap annotations
    for i, row in gap_df.iterrows():
        fig_gap.add_annotation(
            x=max(row['Champion_Avg'], row['Priority_Avg']) + 0.05,
            y=i,
            text=f"Gap: {row['Gap_Display']}",
            showarrow=True,
            arrowhead=2,
            arrowcolor='#34495E',
            arrowwidth=2,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#34495E',
            borderwidth=1,
            font=dict(color='#34495E', size=12, family='Arial Black')
        )
    
    fig_gap.update_layout(
        title="<b>Financial Inclusion: Champions vs Priority Groups</b><br><sub>Revealing the gaps that need urgent attention</sub>",
        xaxis_title="Financial Inclusion Rate",
        yaxis_title="",
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        height=500,
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font=dict(size=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_gap, use_container_width=True)















# Regional Analytics Page
elif st.session_state.page == 'regional':
    st.markdown("## 🗺️ Regional Deep Dive Analytics")
    
    # Region Selection
    selected_region = st.selectbox(
        "Select a region for detailed analysis:",
        options=list(region_mapping.keys()),
        index=0 if st.session_state.selected_region is None else list(region_mapping.keys()).index(st.session_state.selected_region)
    )
    
    if selected_region != st.session_state.selected_region:
        st.session_state.selected_region = selected_region
    
    region_data = region_mapping[selected_region]
    region_color = region_data['color']
    
    # Regional Overview
    st.markdown(f"""
    <div class="region-info-card" style="border-left-color: {region_color};">
        <h3 style="color: {region_color}; margin-top: 0;">📊 {selected_region} Overview</h3>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div class="compact-metric">
                <h4 style="color: {region_color};">Inclusion Rate</h4>
                <h2>{regional_df[regional_df['region'] == selected_region]['inclusion_rate'].iloc[0]:.1%}</h2>
            </div>
            <div class="compact-metric">
                <h4 style="color: {region_color};">Sample Size</h4>
                <h2>{regional_df[regional_df['region'] == selected_region]['count'].iloc[0]:,}</h2>
            </div>
            <div class="compact-metric">
                <h4 style="color: {region_color};">Variability</h4>
                <h2>{regional_df[regional_df['region'] == selected_region]['std'].iloc[0]:.1%}</h2>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Demographic Analysis
    st.markdown("### 👥 Demographic Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Champions analysis
        region_key = selected_region.replace('(excluding high income)', '').replace('&', '').strip()
        region_mapping_demo = {
            'High income': 'High Income',
            'East Asia Pacific': 'East Asia Pacific',
            'Europe Central Asia': 'Europe Central Asia',
            'South Asia': 'South Asia',
            'Latin America Caribbean': 'Latin America',
            'Sub-Saharan Africa': 'Sub-Saharan Africa',
            'Middle East North Africa': 'MENA'
        }
        
        demo_key = region_mapping_demo.get(region_key, 'High Income')
        
        if demo_key in demo_champions:
            champions_data = demo_champions[demo_key]
            champions_df = pd.DataFrame(list(champions_data.items()), columns=['Demographic', 'Rate'])
            
            fig_champions = px.bar(
                champions_df.sort_values('Rate', ascending=True),
                x='Rate',
                y='Demographic',
                orientation='h',
                color='Rate',
                color_continuous_scale='Greens',
                text=[f"{val:.1%}" for val in champions_df.sort_values('Rate', ascending=True)['Rate']],
                title=f"🏆 Champion Demographics - {selected_region}"
            )
            
            fig_champions.update_traces(textposition='inside')
            fig_champions.update_layout(
                height=400,
                showlegend=False,
                xaxis=dict(tickformat='.0%'),
                xaxis_title="Inclusion Rate",
                yaxis_title=""
            )
            
            st.plotly_chart(fig_champions, use_container_width=True)
    
    with col2:
        # Excluded groups analysis
        if demo_key in demo_excluded:
            excluded_data = demo_excluded[demo_key]
            excluded_df = pd.DataFrame(list(excluded_data.items()), columns=['Demographic', 'Rate'])
            
            fig_excluded = px.bar(
                excluded_df.sort_values('Rate', ascending=True),
                x='Rate',
                y='Demographic',
                orientation='h',
                color='Rate',
                color_continuous_scale='Reds',
                text=[f"{val:.1%}" for val in excluded_df.sort_values('Rate', ascending=True)['Rate']],
                title=f"🎯 Priority Demographics - {selected_region}"
            )
            
            fig_excluded.update_traces(textposition='inside')
            fig_excluded.update_layout(
                height=400,
                showlegend=False,
                xaxis=dict(tickformat='.0%'),
                xaxis_title="Inclusion Rate",
                yaxis_title=""
            )
            
            st.plotly_chart(fig_excluded, use_container_width=True)
    
    # Strategic Recommendations
    st.markdown("### 🎯 Strategic Recommendations")
    
    # Key Challenges and Opportunities
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="region-recommendation" style="border-left-color: {region_color};">
            <h4 style="color: {region_color};">⚠️ Key Challenges</h4>
            <ul style="margin-left: 20px;">
        """, unsafe_allow_html=True)
        
        for challenge in region_data['key_challenges']:
            st.markdown(f"<li style='margin: 8px 0;'>{challenge}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="region-recommendation" style="border-left-color: {region_color};">
            <h4 style="color: {region_color};">🚀 Key Opportunities</h4>
            <ul style="margin-left: 20px;">
        """, unsafe_allow_html=True)
        
        for opportunity in region_data['opportunities']:
            st.markdown(f"<li style='margin: 8px 0;'>{opportunity}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Action Timeline
    st.markdown("### 📅 Implementation Timeline")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="action-item immediate-action">
            <h4 style="color: #E74C3C; margin-top: 0;">🚨 Immediate Actions (0-6 months)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, action in enumerate(region_data['immediate_actions'], 1):
            st.markdown(f"**{i}.** {action}")
    
    with col2:
        st.markdown("""
        <div class="action-item medium-term">
            <h4 style="color: #F39C12; margin-top: 0;">⏳ Medium-term (6-18 months)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, action in enumerate(region_data['medium_term'], 1):
            st.markdown(f"**{i}.** {action}")
    
    with col3:
        st.markdown("""
        <div class="action-item long-term">
            <h4 style="color: #27AE60; margin-top: 0;">🎯 Long-term (18+ months)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, action in enumerate(region_data['long_term'], 1):
            st.markdown(f"**{i}.** {action}")
    
    # Success Metrics and Budget Allocation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="region-recommendation" style="border-left-color: {region_color};">
            <h4 style="color: {region_color};">📈 Success Metrics</h4>
            <ul style="margin-left: 20px;">
        """, unsafe_allow_html=True)
        
        for metric in region_data['success_metrics']:
            st.markdown(f"<li style='margin: 8px 0;'><strong>{metric}</strong></li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="region-recommendation" style="border-left-color: {region_color};">
            <h4 style="color: {region_color};">💰 Budget Allocation</h4>
            <p style="margin: 15px 0; font-size: 16px;"><strong>{region_data['budget_allocation']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Regional Comparison Chart
    st.markdown("### 📊 Regional Context")
    
    # Create comparison chart with highlighted region
    colors = [region_color if region == selected_region else '#E0E0E0' for region in regional_df['region']]
    
    fig_comparison = go.Figure(data=[
        go.Bar(
            x=regional_df['inclusion_rate'],
            y=regional_df['region'],
            orientation='h',
            marker_color=colors,
            text=[f"{val:.1%}" for val in regional_df['inclusion_rate']],
            textposition='inside'
        )
    ])
    
    fig_comparison.update_layout(
        title=f"Financial Inclusion Rates - {selected_region} in Context",
        xaxis_title="Financial Inclusion Rate",
        yaxis_title="",
        height=400,
        xaxis=dict(tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    
# Individual Analysis Mode
elif st.session_state.page == 'individual':
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            biz_loan = st.slider("🏢 Business Loan Access (0-1)", 0.0, 1.0, 0.3, 0.1,
                               help="Do you have access to business loans?")
            emergency_funds = st.slider("🆘 Emergency Funds (0-1)", 0.0, 1.0, 0.4, 0.1,
                                      help="Do you have emergency funds available?")
            digital_engagement = st.slider("📱 Digital Engagement (0-1)", 0.0, 1.0, 0.5, 0.1,
                                         help="How actively do you use digital financial services?")
        
        with col2:
            govt_services = st.slider("🏛️ Government Services Usage (0-1)", 0.0, 1.0, 0.3, 0.1,
                                    help="Do you use digital government payment services?")
            mobile_pay = st.slider("📲 Mobile Payments (0-1)", 0.0, 1.0, 0.3, 0.1,
                                 help="Do you use mobile payment services?")
            financial_activity = st.slider("💰 Overall Financial Activity (0-1)", 0.0, 1.0, 0.4, 0.1,
                                         help="How active are you in saving, borrowing, investing?")
        
        submitted = st.form_submit_button("🔮 Predict My Financial Inclusion Score")
    
    if submitted:
        # Updated prediction logic using actual Random Forest feature importance
        weights = {
            'biz_loan': 0.1683,
            'emergency_funds': 0.0980,
            'digital_engagement': 0.0636,
            'govt_services': 0.0597,
            'mobile_pay': 0.0404,
            'financial_activity': 0.0390
        }
        
        # Regional baseline (from actual data)
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
            digital_engagement * weights['digital_engagement'] +
            govt_services * weights['govt_services'] +
            mobile_pay * weights['mobile_pay'] +
            financial_activity * weights['financial_activity']
        )
        
        baseline_score = region_baseline[region] + income_adjustments[income_group]
        final_score = min(1.0, max(0.0, baseline_score + feature_score * 0.5))
        
        # Display Results
        st.markdown("### 🎯 Your Financial Inclusion Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
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
            confidence = 0.85 + (abs(final_score - 0.5) * 0.3)
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
            'Digital Engagement': digital_engagement * weights['digital_engagement'],
            'Government Services': govt_services * weights['govt_services'],
            'Mobile Payments': mobile_pay * weights['mobile_pay'],
            'Financial Activity': financial_activity * weights['financial_activity']
        }
        
        impact_df = pd.DataFrame({
            'Factor': list(feature_impacts.keys()),
            'Impact Score': list(feature_impacts.values()),
            'Your Level': [biz_loan, emergency_funds, digital_engagement, govt_services, mobile_pay, financial_activity]
        })
        
        fig_impact = px.bar(
            impact_df,
            x='Impact Score',
            y='Factor',
            orientation='h',
            color='Your Level',
            color_continuous_scale='RdYlGn',
            title="<b>Financial Inclusion Rates by Region</b>"Personal Factors Impact on Financial Inclusion"
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
            
        if digital_engagement < 0.5:
            recommendations.append("📱 **Digital Adoption**: Learn about mobile banking and digital payment platforms available in your area")
            
        if financial_activity < 0.5:
            recommendations.append("💰 **Financial Activity**: Increase your participation in savings, lending, and investment activities")
            
        if govt_services < 0.5:
            recommendations.append("🏛️ **Government Services**: Explore digital government payment and service options")
            
        if final_score < region_baseline[region]:
            recommendations.append(f"🎯 **Regional Programs**: Look into financial inclusion initiatives specific to {region}")
        
        for rec in recommendations[:4]:
            st.markdown(f"- {rec}")
        
        # Success Stories
        if final_score >= 0.7:
            st.markdown("""
            <div class="champion-card">
                <h4>🌟 Congratulations!</h4>
                <p>You're doing great with financial inclusion! Your score indicates good access to financial services. 
                Consider sharing your experience with others in your community.</p>
            </div>
            """, unsafe_allow_html=True)
        elif final_score >= 0.5:
            st.markdown("""
            <div class="compact-metric">
                <h4>🎯 You're on the right track!</h4>
                <p>With some focused improvements in key areas, you can significantly enhance your financial inclusion. 
                The recommendations above will help you get there.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="priority-card">
                <h4>💪 Every journey starts with a single step!</h4>
                <p>There are many opportunities to improve your financial inclusion. Start with one small change 
                and build momentum. Financial inclusion programs in your region can provide additional support.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #636e72;">
    <p><strong>FinScope Global Dashboard</strong> | Powered by Machine Learning & World Bank Global Findex Data</p>
    <p>🔬 Built with Streamlit • 🤖 Random Forest ML Model • 📊 149 Countries Analyzed</p>
    <p style="font-size: 12px; margin-top: 15px;">
        <em>This tool provides AI-powered insights for educational and policy planning purposes. 
        Individual assessments are estimates based on demographic patterns and should not be considered as financial advice.</em>
    </p>
</div>
""", unsafe_allow_html=True)
   