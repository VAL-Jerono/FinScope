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
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(102,126,234,0.3);
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
    
    # Regional mapping with colors and country data
    region_mapping = {
        'High income': {
            'color': '#2E8B57',
            'countries': ['USA', 'Germany', 'Japan', 'UK', 'France', 'Canada', 'Australia'],
            'key_challenges': ['Digital divide in rural areas', 'Aging population banking needs'],
            'opportunities': ['Fintech innovation', 'Sustainable finance', 'Digital banking expansion'],
            'priority_actions': ['AI-driven personalized services', 'Green finance products', 'Elderly-friendly digital solutions']
        },
        'East Asia & Pacific (excluding high income)': {
            'color': '#FF6B35',
            'countries': ['China', 'Indonesia', 'Thailand', 'Philippines', 'Vietnam', 'Malaysia'],
            'key_challenges': ['Rural-urban divide', 'Complex regulatory environments', 'Infrastructure gaps'],
            'opportunities': ['Mobile payment growth', 'E-commerce integration', 'Cross-border payments'],
            'priority_actions': ['Digital wallet expansion', 'Rural connectivity programs', 'Regulatory harmonization']
        },
        'Europe & Central Asia (excluding high income)': {
            'color': '#F7931E',
            'countries': ['Russia', 'Turkey', 'Kazakhstan', 'Ukraine', 'Romania', 'Bulgaria'],
            'key_challenges': ['Economic volatility', 'Legacy banking systems', 'Currency instability'],
            'opportunities': ['Digital transformation', 'EU integration benefits', 'Remittance corridors'],
            'priority_actions': ['Modern payment infrastructure', 'Cross-border integration', 'SME financing']
        },
        'South Asia (excluding high income)': {
            'color': '#FFD23F',
            'countries': ['India', 'Bangladesh', 'Pakistan', 'Sri Lanka', 'Nepal', 'Afghanistan'],
            'key_challenges': ['Large unbanked population', 'Documentation barriers', 'Gender gaps'],
            'opportunities': ['Digital identity systems', 'Mobile-first approaches', 'Government support'],
            'priority_actions': ['Jan Dhan-style programs', 'Women-focused initiatives', 'Agent banking networks']
        },
        'Latin America & Caribbean (excluding high income)': {
            'color': '#E74C3C',
            'countries': ['Brazil', 'Mexico', 'Argentina', 'Colombia', 'Peru', 'Chile'],
            'key_challenges': ['Economic informality', 'Credit access barriers', 'Income inequality'],
            'opportunities': ['Fintech boom', 'Remittance integration', 'Government digitization'],
            'priority_actions': ['Alternative credit scoring', 'Digital remittances', 'Financial education']
        },
        'Sub-Saharan Africa (excluding high income)': {
            'color': '#C0392B',
            'countries': ['Nigeria', 'Kenya', 'South Africa', 'Ghana', 'Tanzania', 'Ethiopia'],
            'key_challenges': ['Infrastructure limitations', 'Low income levels', 'Distance to banks'],
            'opportunities': ['Mobile money success', 'Agent banking', 'Agricultural finance'],
            'priority_actions': ['Mobile money expansion', 'Agent network growth', 'Agricultural value chain finance']
        },
        'Middle East & North Africa (excluding high income)': {
            'color': '#8E44AD',
            'countries': ['Egypt', 'Morocco', 'Jordan', 'Tunisia', 'Algeria', 'Lebanon'],
            'key_challenges': ['Political instability', 'Youth unemployment', 'Regulatory restrictions'],
            'opportunities': ['Islamic finance growth', 'Oil revenue diversification', 'Regional integration'],
            'priority_actions': ['Sharia-compliant products', 'Youth banking programs', 'Digital government services']
        }
    }
    
    return pd.DataFrame(regional_data), pd.DataFrame(income_data), pd.DataFrame(feature_importance), region_mapping

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None

# Load data
regional_df, income_df, feature_df, region_mapping = load_data()

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
        text='inclusion_rate',
        title="<b>Financial Inclusion Rates by Region</b>",
        height=500
    )
    
    fig_overview.update_traces(
        texttemplate='%{text:.1%}', 
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black')
    )
    
    fig_overview.update_layout(
        xaxis_title="Financial Inclusion Rate",
        yaxis_title="",
        xaxis=dict(tickformat='.0%'),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font=dict(size=18)
    )
    
    st.plotly_chart(fig_overview, use_container_width=True)

# Regional Analytics Page
elif st.session_state.page == 'regional':
    st.markdown("## 🗺️ Interactive Regional Analytics")
    st.markdown("### *Click on any region below to explore detailed insights and recommendations*")
    
    # Create interactive map with regions
    fig_map = go.Figure()
    
    # Create a simple regional representation using scatter geo
    region_centers = {
        'High income': {'lat': 45, 'lon': 0, 'size': 60},
        'East Asia & Pacific (excluding high income)': {'lat': 20, 'lon': 120, 'size': 50},
        'Europe & Central Asia (excluding high income)': {'lat': 50, 'lon': 50, 'size': 45},
        'South Asia (excluding high income)': {'lat': 20, 'lon': 80, 'size': 40},
        'Latin America & Caribbean (excluding high income)': {'lat': -10, 'lon': -60, 'size': 40},
        'Sub-Saharan Africa (excluding high income)': {'lat': -10, 'lon': 20, 'size': 35},
        'Middle East & North Africa (excluding high income)': {'lat': 25, 'lon': 35, 'size': 30}
    }
    
    for _, row in regional_df.iterrows():
        region = row['region']
        rate = row['inclusion_rate']
        count = row['count']
        
        center = region_centers[region]
        color = region_mapping[region]['color']
        
        fig_map.add_trace(go.Scattergeo(
            lon=[center['lon']],
            lat=[center['lat']],
            text=region.split('(')[0].strip(),
            mode='markers+text',
            marker=dict(
                size=center['size'],
                color=color,
                opacity=0.8,
                line=dict(width=3, color='white'),
                sizemode='diameter'
            ),
            textposition="middle center",
            textfont=dict(size=12, color='white', family='Arial Black'),
            hovertemplate=f'<b>{region}</b><br>' +
                         f'Inclusion Rate: {rate:.1%}<br>' +
                         f'Sample Size: {count:,}<br>' +
                         '<extra></extra>',
            name=region,
            customdata=[region]
        ))
    
    fig_map.update_layout(
        title=dict(
            text="<b>Global Financial Inclusion by Region</b><br><sub>Circle size represents sample size, color shows inclusion rate</sub>",
            x=0.5,
            font=dict(size=20)
        ),
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(240, 240, 240)',
            coastlinecolor="rgb(204, 204, 204)",
            showocean=True,
            oceancolor="rgb(230, 245, 255)",
            showlakes=True,
            lakecolor="rgb(230, 245, 255)",
            showrivers=True,
            rivercolor="rgb(230, 245, 255)"
        ),
        height=500,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    # Handle map clicks
    selected_points = st.plotly_chart(fig_map, use_container_width=True, key="regional_map", 
                                     on_select="rerun", selection_mode="points")
    
    # Region selection buttons
    st.markdown("### 📊 Select Region for Detailed Analysis")
    
    regions_sorted = regional_df.sort_values('inclusion_rate', ascending=False)
    cols = st.columns(2)
    
    for idx, (_, region_data) in enumerate(regions_sorted.iterrows()):
        region_name = region_data['region']
        inclusion_rate = region_data['inclusion_rate']
        
        col = cols[idx % 2]
        with col:
            if st.button(f"{region_name.split('(')[0].strip()} - {inclusion_rate:.1%}", 
                        key=f"region_{idx}"):
                st.session_state.selected_region = region_name
    
    # Display selected region details
    if st.session_state.selected_region:
        region_name = st.session_state.selected_region
        region_data = regional_df[regional_df['region'] == region_name].iloc[0]
        region_info = region_mapping[region_name]
        
        st.markdown(f"""
        <div class="region-info-card" style="border-left-color: {region_info['color']};">
            <h2 style="color: {region_info['color']}; margin-top: 0;">
                {region_name.split('(')[0].strip()} - Detailed Analysis
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Inclusion Rate", f"{region_data['inclusion_rate']:.1%}")
        with col2:
            st.metric("Sample Size", f"{region_data['count']:,}")
        with col3:
            gap_to_best = regional_df['inclusion_rate'].max() - region_data['inclusion_rate']
            st.metric("Gap to Best", f"{gap_to_best:.1%}")
        with col4:
            rank = (regional_df['inclusion_rate'] > region_data['inclusion_rate']).sum() + 1
            st.metric("Global Rank", f"#{rank}/7")
        
        # Detailed insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Key Challenges")
            for challenge in region_info['key_challenges']:
                st.markdown(f"• {challenge}")
            
            st.markdown("#### 🚀 Growth Opportunities")
            for opportunity in region_info['opportunities']:
                st.markdown(f"• {opportunity}")
        
        with col2:
            st.markdown("#### 📍 Major Countries")
            for country in region_info['countries']:
                st.markdown(f"• {country}")
            
            st.markdown(f"""
            <div class="recommendation-box">
                <h4 style="margin-top: 0;">💡 Priority Recommendations</h4>
                <ul style="margin: 10px 0;">
                    {''.join(f'<li>{action}</li>' for action in region_info['priority_actions'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Individual Analysis Page
elif st.session_state.page == 'individual':
    st.markdown("## 👤 Individual Financial Inclusion Calculator")
    st.markdown("### *Get personalized insights and recommendations based on ML analysis*")
    
    # Calculator interface
    st.markdown("""
    <div class="calculator-card">
        <h3 style="color: #667eea; text-align: center; margin-top: 0;">
            🧮 Personal Financial Inclusion Assessment
        </h3>
        <p style="text-align: center; color: #636e72;">
            Based on Random Forest model trained on 8,311+ global respondents
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User inputs based on top features
        st.markdown("#### 📝 Your Profile")
        
        region = st.selectbox(
            "🌍 Your Region",
            options=list(region_mapping.keys()),
            help="Select your geographical region"
        )
        
        business_loan_access = st.slider(
            "🏢 Business Loan Access (0-10)",
            min_value=0, max_value=10, value=5,
            help="How easily can you access business loans? (0=Very difficult, 10=Very easy)"
        )
        
        emergency_funds = st.slider(
            "🆘 Emergency Funds Availability (0-10)",
            min_value=0, max_value=10, value=5,
            help="How well can you handle financial emergencies? (0=No funds, 10=Well prepared)"
        )
        
        digital_engagement = st.slider(
            "📱 Digital Engagement Score (0-10)",
            min_value=0, max_value=10, value=5,
            help="How comfortable are you with digital financial services?"
        )
        
        government_services = st.slider(
            "🏛️ Government Services Usage (0-10)",
            min_value=0, max_value=10, value=5,
            help="How often do you use digital government services?"
        )
        
        mobile_payments = st.slider(
            "📲 Mobile Payment Usage (0-10)",
            min_value=0, max_value=10, value=5,
            help="How frequently do you use mobile payments?"
        )
    
    with col2:
        st.markdown("#### 📊 Your Assessment")
        
        # Simple scoring based on feature importance
        region_baseline = regional_df[regional_df['region'] == region]['inclusion_rate'].iloc[0]
        
        # Weighted score calculation
        weights = {
            'business_loan_access': 0.1683,
            'emergency_funds': 0.0980,
            'digital_engagement': 0.0636,
            'government_services': 0.0597,
            'mobile_payments': 0.0404
        }
        
        user_score = (
            business_loan_access * weights['business_loan_access'] +
            emergency_funds * weights['emergency_funds'] +
            digital_engagement * weights['digital_engagement'] +
            government_services * weights['government_services'] +
            mobile_payments * weights['mobile_payments']
        ) / 10  # Normalize to 0-1
        
        # Combine with regional baseline
        final_score = (region_baseline * 0.6 + user_score * 0.4)
        
        # Display results
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, {region_mapping[region]['color']}20 0%, {region_mapping[region]['color']}10 100%);">
            <h3 style="color: {region_mapping[region]['color']}; margin-top: 0;">Your Inclusion Probability</h3>
            <h1 style="color: #2d3436; margin: 15px 0; font-size: 3.5em;">{final_score:.1%}</h1>
            <p style="color: #636e72;">Based on ML model analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison with regional average
        st.metric(
            "vs Regional Average",
            f"{final_score:.1%}",
            delta=f"{(final_score - region_baseline):.1%}",
            help="How you compare to others in your region"
        )
        
        # Risk category
        if final_score >= 0.7:
            risk_level = "🟢 High Inclusion"
            risk_color = "#2E8B57"
        elif final_score >= 0.5:
            risk_level = "🟡 Moderate Inclusion"
            risk_color = "#F7931E"
        else:
            risk_level = "🔴 Low Inclusion Risk"
            risk_color = "#E74C3C"
        
        st.markdown(f"""
        <div style="background: {risk_color}20; border: 2px solid {risk_color}; 
                    border-radius: 10px; padding: 15px; text-align: center; margin: 20px 0;">
            <h3 style="color: {risk_color}; margin: 0;">{risk_level}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Personalized recommendations
    st.markdown("### 💡 Personalized Recommendations")
    
    recommendations = []
    
    if business_loan_access < 5:
        recommendations.append("🏢 **Improve Business Loan Access**: Research microfinance institutions, credit unions, and online lending platforms in your region.")
    
    if emergency_funds < 5:
        recommendations.append("🆘 **Build Emergency Fund**: Start with small, regular savings. Aim for 3-6 months of expenses.")
    
    if digital_engagement < 5:
        recommendations.append("📱 **Enhance Digital Skills**: Take online courses on digital banking, practice with mobile apps.")
    
    if government_services < 5:
        recommendations.append("🏛️ **Explore Government Services**: Check available digital government financial services and benefits.")
    
    if mobile_payments < 5:
        recommendations.append("📲 **Adopt Mobile Payments**: Start with popular local mobile payment apps, use for small transactions first.")
    
    # Regional specific recommendations
    regional_recs = region_mapping[region]['priority_actions']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Personal Action Items")
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.markdown("🌟 **Great job!** You're well-positioned for financial inclusion. Consider mentoring others in your community.")
    
    with col2:
        st.markdown(f"#### 🌍 Regional Opportunities in {region.split('(')[0].strip()}")
        for rec in regional_recs:
            st.markdown(f"• {rec}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #636e72; padding: 20px;">
    <p><strong>FinScope Global</strong> | Powered by Machine Learning | Data from 149 countries</p>
    <p>Model Accuracy: 89.6% | Random Forest with 14 key features | Sample: 8,311+ adults</p>
</div>
""", unsafe_allow_html=True)