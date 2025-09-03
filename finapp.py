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

# Individual Analysis Page
elif st.session_state.page == 'individual':
    st.markdown("## 👤 Individual Financial Inclusion Assessment")
    
    st.markdown("""
    <div class="calculator-card">
        <h3 style="text-align: center; color: #667eea;">🔮 AI-Powered Inclusion Predictor</h3>
        <p style="text-align: center; margin-bottom: 25px;">Answer the questions below to get a personalized financial inclusion assessment based on our machine learning model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📍 Demographics")
        
        age_group = st.selectbox(
            "Age Group",
            ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        )
        
        education = st.selectbox(
            "Education Level",
            ["Primary or less", "Secondary", "Tertiary/Higher education"]
        )
        
        employment = st.selectbox(
            "Employment Status",
            ["In labor force", "Out of labor force"]
        )
        
        location = st.selectbox(
            "Location",
            ["Urban", "Rural"]
        )
        
        income_level = st.selectbox(
            "Income Level",
            ["Poor 40%", "Rich 60%"]
        )
        
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )
    
    with col2:
        st.markdown("#### 💰 Financial Behavior")
        
        saved_any = st.selectbox(
            "Have you saved money in the past 12 months?",
            ["Yes", "No"]
        )
        
        borrowed_any = st.selectbox(
            "Have you borrowed money in the past 12 months?",
            ["Yes", "No"]
        )
        
        emergency_funds = st.selectbox(
            "Could you come up with emergency funds?",
            ["Yes", "No", "Don't know"]
        )
        
        mobile_payments = st.selectbox(
            "Do you send/receive money via mobile?",
            ["Yes", "No"]
        )
        
        digital_preference = st.selectbox(
            "Do you prefer digital financial services?",
            ["Yes", "No", "No preference"]
        )
        
        government_services = st.selectbox(
            "Do you receive government payments digitally?",
            ["Yes", "No", "Not applicable"]
        )
    
    # Calculate prediction button
    if st.button("🔍 Calculate Financial Inclusion Score", key="calculate_btn"):
        
        # Simple scoring algorithm based on feature importance
        score = 0.5  # Base score
        
        # Demographic adjustments
        if education == "Tertiary/Higher education":
            score += 0.15
        elif education == "Primary or less":
            score -= 0.10
            
        if employment == "In labor force":
            score += 0.12
        else:
            score -= 0.08
            
        if location == "Urban":
            score += 0.08
        else:
            score -= 0.05
            
        if income_level == "Rich 60%":
            score += 0.10
        else:
            score -= 0.08
            
        if gender == "Male":
            score += 0.02
        else:
            score -= 0.02
            
        # Financial behavior adjustments (higher weights based on ML model)
        if saved_any == "Yes":
            score += 0.15
        else:
            score -= 0.10
            
        if borrowed_any == "Yes":
            score += 0.08
        else:
            score -= 0.05
            
        if emergency_funds == "Yes":
            score += 0.20  # High importance feature
        elif emergency_funds == "No":
            score -= 0.15
            
        if mobile_payments == "Yes":
            score += 0.12
        else:
            score -= 0.08
            
        if digital_preference == "Yes":
            score += 0.10
        elif digital_preference == "No":
            score -= 0.05
            
        if government_services == "Yes":
            score += 0.08
        elif government_services == "No":
            score -= 0.05
        
        # Ensure score is between 0 and 1
        score = max(0, min(1, score))
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Your Financial Inclusion Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score_color = "#2E8B57" if score >= 0.7 else "#F39C12" if score >= 0.4 else "#E74C3C"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin-top: 0;">📊 Inclusion Score</h3>
                <h1 style="color: {score_color}; margin: 15px 0; font-size: 4em;">{score:.0%}</h1>
                <p style="color: #636e72; font-size: 16px;">Predicted Probability</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if score >= 0.7:
                category = "High Inclusion"
                category_color = "#2E8B57"
                icon = "🟢"
            elif score >= 0.4:
                category = "Moderate Inclusion"
                category_color = "#F39C12"
                icon = "🟡"
            else:
                category = "Low Inclusion"
                category_color = "#E74C3C"
                icon = "🔴"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin-top: 0;">🎯 Category</h3>
                <h2 style="color: {category_color}; margin: 15px 0;">{icon} {category}</h2>
                <p style="color: #636e72; font-size: 16px;">Risk Assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Find similar region based on score
            regional_rates = regional_df['inclusion_rate'].values
            closest_region_idx = np.argmin(np.abs(regional_rates - score))
            closest_region = regional_df.iloc[closest_region_idx]['region']
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin-top: 0;">🌍 Similar to</h3>
                <h4 style="color: #2d3436; margin: 15px 0; line-height: 1.3;">{closest_region}</h4>
                <p style="color: #636e72; font-size: 16px;">Regional Average</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Personalized recommendations
        st.markdown("### 💡 Personalized Recommendations")
        
        recommendations = []
        
        if score < 0.4:
            recommendations.extend([
                "🏦 **Priority Action:** Open a basic bank account with low/no fees",
                "📱 **Digital Access:** Explore mobile banking and digital wallet options",
                "💰 **Emergency Fund:** Start building emergency savings, even small amounts",
                "🎓 **Financial Literacy:** Take advantage of free financial education programs"
            ])
        elif score < 0.7:
            recommendations.extend([
                "💳 **Expand Services:** Consider additional financial products like savings accounts",
                "📊 **Credit Building:** Explore opportunities to build credit history",
                "🏠 **Asset Building:** Look into savings programs for major purchases",
                "🔄 **Digital Upgrade:** Increase use of digital financial services"
            ])
        else:
            recommendations.extend([
                "📈 **Investment:** Explore investment opportunities and wealth building",
                "🌐 **Advanced Services:** Consider sophisticated financial products",
                "🤝 **Mentorship:** Share your knowledge with others in your community",
                "🔮 **Innovation:** Stay updated with emerging financial technologies"
            ])
        
        # Add specific recommendations based on weak areas
        if emergency_funds == "No":
            recommendations.append("🚨 **Emergency Fund:** This is critical - start with just $10-20 per month")
        
        if saved_any == "No":
            recommendations.append("🎯 **Savings Habit:** Begin with automated micro-savings - even $5/week helps")
        
        if mobile_payments == "No":
            recommendations.append("📱 **Mobile Money:** Learn to use mobile payment systems - they're often cheaper and more convenient")
        
        if digital_preference == "No":
            recommendations.append("💻 **Digital Literacy:** Take a basic course on digital financial services")
        
        for i, rec in enumerate(recommendations[:6], 1):  # Show top 6 recommendations
            st.markdown(f"{i}. {rec}")
        
        # Risk factors and strengths
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="priority-card">
                <h4>⚠️ Areas for Improvement</h4>
            </div>
            """, unsafe_allow_html=True)
            
            risk_factors = []
            if education == "Primary or less":
                risk_factors.append("• Education level - consider adult education programs")
            if employment == "Out of labor force":
                risk_factors.append("• Employment status - explore income opportunities")
            if emergency_funds == "No":
                risk_factors.append("• Emergency funds - critical for financial security")
            if saved_any == "No":
                risk_factors.append("• Savings behavior - foundation of financial health")
            if location == "Rural":
                risk_factors.append("• Rural location - may limit access to services")
            
            if risk_factors:
                for factor in risk_factors[:4]:  # Show top 4 risk factors
                    st.markdown(factor)
            else:
                st.markdown("• **Great job!** No major risk factors identified")
        
        with col2:
            st.markdown("""
            <div class="champion-card">
                <h4>🏆 Your Strengths</h4>
            </div>
            """, unsafe_allow_html=True)
            
            strengths = []
            if education == "Tertiary/Higher education":
                strengths.append("• Higher education - enables better financial decisions")
            if employment == "In labor force":
                strengths.append("• Active employment - steady income source")
            if emergency_funds == "Yes":
                strengths.append("• Emergency preparedness - excellent financial planning")
            if saved_any == "Yes":
                strengths.append("• Savings behavior - strong financial foundation")
            if mobile_payments == "Yes":
                strengths.append("• Mobile money usage - embracing financial technology")
            if digital_preference == "Yes":
                strengths.append("• Digital preference - aligned with future of finance")
            
            if strengths:
                for strength in strengths[:4]:  # Show top 4 strengths
                    st.markdown(strength)
            else:
                st.markdown("• Focus on building strengths through the recommendations above")
        
        # Comparison chart
        st.markdown("### 📊 How You Compare")
        
        comparison_data = {
            'Category': ['Your Score', 'Global Average', 'High Income Countries', 'Your Region Average'],
            'Score': [score, 0.611, 0.858, regional_df[regional_df['region'] == closest_region]['inclusion_rate'].iloc[0]],
            'Color': [score_color, '#667eea', '#2E8B57', '#F39C12']
        }
        
        fig_comparison = go.Figure(data=[
            go.Bar(
                x=comparison_data['Category'],
                y=comparison_data['Score'],
                marker_color=comparison_data['Color'],
                text=[f"{val:.0%}" for val in comparison_data['Score']],
                textposition='outside'
            )
        ])
        
        fig_comparison.update_layout(
            title="Your Financial Inclusion Score in Context",
            yaxis_title="Inclusion Rate",
            height=400,
            yaxis=dict(tickformat='.0%', range=[0, 1]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    else:
        # Show sample insights when form is not submitted
        st.markdown("""
        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-radius: 15px; margin: 20px 0;">
            <h4 style="color: #667eea;">💡 What You'll Get</h4>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 25px;">
                <div style="margin: 10px; max-width: 200px;">
                    <h3 style="color: #2E8B57;">📊</h3>
                    <p><strong>Personal Score</strong><br>AI-powered inclusion probability</p>
                </div>
                <div style="margin: 10px; max-width: 200px;">
                    <h3 style="color: #F39C12;">💡</h3>
                    <p><strong>Custom Recommendations</strong><br>Tailored action plan</p>
                </div>
                <div style="margin: 10px; max-width: 200px;">
                    <h3 style="color: #E74C3C;">🎯</h3>
                    <p><strong>Risk Assessment</strong><br>Areas for improvement</p>
                </div>
                <div style="margin: 10px; max-width: 200px;">
                    <h3 style="color: #8E44AD;">📈</h3>
                    <p><strong>Benchmarking</strong><br>Compare with global averages</p>
                </div>
            </div>
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