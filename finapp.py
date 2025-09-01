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
    
    # Country mapping with ISO codes for proper map visualization
    # ===============================================================
# Final Corrected Country Mapping (Only Given Countries Included)
# ===============================================================

    country_mapping = {
        'High income': {
            'countries': [
                'Australia','Austria','Bahrain','Belgium','Canada','Chile','Croatia','Cyprus','Czechia',
                'Denmark','Estonia','Finland','France','Germany','Greece','Hong Kong SAR, China','Hungary',
                'Iceland','Ireland','Israel','Italy','Japan','Korea, Rep.','Kuwait','Latvia','Lithuania',
                'Luxembourg','Malta','Netherlands','New Zealand','Norway','Oman','Panama','Poland',
                'Portugal','Puerto Rico','Qatar','Romania','Saudi Arabia','Singapore','Slovak Republic',
                'Slovenia','Spain','Sweden','Switzerland','Taiwan, China','Trinidad and Tobago',
                'United Arab Emirates','United Kingdom','United States','Uruguay'
            ],
            'iso_codes': [
                'AUS','AUT','BHR','BEL','CAN','CHL','HRV','CYP','CZE','DNK',
                'EST','FIN','FRA','DEU','GRC','HKG','HUN','ISL','IRL','ISR',
                'ITA','JPN','KOR','KWT','LVA','LTU','LUX','MLT','NLD','NZL',
                'NOR','OMN','PAN','POL','PRT','PRI','QAT','ROU','SAU','SGP',
                'SVK','SVN','ESP','SWE','CHE','TWN','TTO','ARE','GBR','USA','URY'
            ],
            'color': '#2E8B57'
        },

        'East Asia & Pacific (excluding high income)': {
            'countries': [
                'Cambodia','China','Indonesia','Lao PDR','Malaysia','Mongolia',
                'Myanmar','Philippines','Thailand','Viet Nam'
            ],
            'iso_codes': ['KHM','CHN','IDN','LAO','MYS','MNG','MMR','PHL','THA','VNM'],
            'color': '#FF6B35'
        },

        'Europe & Central Asia (excluding high income)': {
            'countries': [
                'Albania','Armenia','Azerbaijan','Belarus','Bosnia and Herzegovina','Bulgaria',
                'Georgia','Kazakhstan','Kosovo','Kyrgyz Republic','Moldova','Montenegro',
                'North Macedonia','Russian Federation','Serbia','Tajikistan','Turkiye',
                'Turkmenistan','Ukraine','Uzbekistan'
            ],
            'iso_codes': [
                'ALB','ARM','AZE','BLR','BIH','BGR','GEO','KAZ','XKX','KGZ',
                'MDA','MNE','MKD','RUS','SRB','TJK','TUR','TKM','UKR','UZB'
            ],
            'color': '#F7931E'
        },

        'South Asia (excluding high income)': {
            'countries': [
                'Afghanistan','Bangladesh','Bhutan','India','Maldives','Nepal','Pakistan','Sri Lanka'
            ],
            'iso_codes': ['AFG','BGD','BTN','IND','MDV','NPL','PAK','LKA'],
            'color': '#FFD23F'
        },

        'Latin America & Caribbean (excluding high income)': {
            'countries': [
                'Argentina','Belize','Bolivia','Brazil','Colombia','Costa Rica','Dominican Republic',
                'Ecuador','El Salvador','Guatemala','Haiti','Honduras','Jamaica','Mexico',
                'Nicaragua','Paraguay','Peru','Venezuela, RB'
            ],
            'iso_codes': [
                'ARG','BLZ','BOL','BRA','COL','CRI','DOM','ECU','SLV','GTM',
                'HTI','HND','JAM','MEX','NIC','PRY','PER','VEN'
            ],
            'color': '#E74C3C'
        },

        'Sub-Saharan Africa (excluding high income)': {
            'countries': [
                'Angola','Benin','Botswana','Burkina Faso','Burundi','Cameroon','Central African Republic',
                'Chad','Comoros','Congo, Dem. Rep.','Congo, Rep.','Cote d\'Ivoire','Eswatini','Ethiopia',
                'Gabon','Gambia, The','Ghana','Guinea','Kenya','Lesotho','Liberia','Madagascar','Malawi',
                'Mali','Mauritania','Mauritius','Mozambique','Namibia','Niger','Nigeria','Rwanda',
                'Senegal','Sierra Leone','Somalia','South Africa','South Sudan','Sudan','Tanzania',
                'Togo','Uganda','Zambia','Zimbabwe'
            ],
            'iso_codes': [
                'AGO','BEN','BWA','BFA','BDI','CMR','CAF','TCD','COM','COD','COG','CIV','SWZ','ETH',
                'GAB','GMB','GHA','GIN','KEN','LSO','LBR','MDG','MWI','MLI','MRT','MUS','MOZ','NAM',
                'NER','NGA','RWA','SEN','SLE','SOM','ZAF','SSD','SDN','TZA','TGO','UGA','ZMB','ZWE'
            ],
            'color': '#C0392B'
        },

        'Middle East & North Africa (excluding high income)': {
            'countries': [
                'Algeria','Djibouti','Egypt, Arab Rep.','Iran, Islamic Rep.','Iraq','Jordan','Lebanon',
                'Libya','Morocco','Syrian Arab Republic','Tunisia','West Bank and Gaza','Yemen, Rep.'
            ],
            'iso_codes': ['DZA','DJI','EGY','IRN','IRQ','JOR','LBN','LBY','MAR','SYR','TUN','PSE','YEM'],
            'color': '#8E44AD'
        }
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
    
    return pd.DataFrame(regional_data), pd.DataFrame(income_data), pd.DataFrame(feature_importance), region_mapping, country_mapping

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None

# Load data
regional_df, income_df, feature_df, region_mapping, country_mapping = load_data()

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
    st.markdown("### *Click on any region to explore detailed insights and recommendations*")
    
    # Create choropleth map with actual countries
    country_data = []
    for region, region_info in country_mapping.items():
        region_rate = regional_df[regional_df['region'] == region]['inclusion_rate'].iloc[0]
        region_count = regional_df[regional_df['region'] == region]['count'].iloc[0]
        
        for i, (country, iso_code) in enumerate(zip(region_info['countries'], region_info['iso_codes'])):
            country_data.append({
                'country': country,
                'iso_code': iso_code,
                'region': region,
                'inclusion_rate': region_rate,
                'color': region_info['color'],
                'sample_size': region_count
            })

    country_df = pd.DataFrame(country_data)

    # Create the choropleth map
    fig_choropleth = go.Figure(data=go.Choropleth(
        locations=country_df['iso_code'],
        z=country_df['inclusion_rate'],
        locationmode='ISO-3',
        colorscale=[
            [0.0, '#8E44AD'],    # MENA
            [0.15, '#C0392B'],   # Sub-Saharan Africa  
            [0.30, '#E74C3C'],   # Latin America
            [0.45, '#FFD23F'],   # South Asia
            [0.60, '#F7931E'],   # Europe & Central Asia
            [0.75, '#FF6B35'],   # East Asia Pacific
            [1.0, '#2E8B57']     # High income
        ],
        text=country_df['country'],
        hovertemplate='<b>%{text}</b><br>' +
                      'Region: %{customdata[0]}<br>' +
                      'Inclusion Rate: %{z:.1%}<br>' +
                      'Sample Size: %{customdata[1]:,}<br>' +
                      '<extra></extra>',
        customdata=country_df[['region', 'sample_size']].values,
        colorbar=dict(
            title="Financial<br>Inclusion Rate",
            titlefont=dict(size=14),
            tickformat='.0%',
            len=0.8
        ),
        showscale=True
    ))

    fig_choropleth.update_layout(
        title={
            'text': '<b>Global Financial Inclusion Rates by Country</b><br><sub>Countries color-coded by regional inclusion rates - Click to explore regions</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=18)
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgb(180,180,180)",
            projection_type='natural earth',
            bgcolor='rgba(240,240,240,0.1)',
            showland=True,
            landcolor='rgb(250,250,250)',
            showocean=True,
            oceancolor='rgb(230,245,255)',
            showlakes=True,
            lakecolor='rgb(230,245,255)'
        ),
        height=600,
        margin=dict(l=0, r=0, t=80, b=0)
    )

    # Display the interactive map
    st.plotly_chart(fig_choropleth, use_container_width=True, key="country_choropleth")
    
    # Region selection buttons
    st.markdown("### 📊 Select Region for Detailed Analysis")
    
    regions_sorted = regional_df.sort_values('inclusion_rate', ascending=False)
    cols = st.columns(2)
    
    for idx, (_, region_data) in enumerate(regions_sorted.iterrows()):
        region_name = region_data['region']
        inclusion_rate = region_data['inclusion_rate']
        region_color = country_mapping[region_name]['color']
        
        col = cols[idx % 2]
        with col:
            # Create custom button styling based on region color
            button_style = f"""
            <div style="margin: 10px 0;">
                <button onclick="this.style.transform='scale(0.95)'" 
                        style="width: 100%; padding: 15px; border: none; border-radius: 12px;
                               background: linear-gradient(135deg, {region_color} 0%, {region_color}CC 100%);
                               color: white; font-weight: bold; font-size: 16px;
                               box-shadow: 0 4px 15px {region_color}40;
                               transition: all 0.2s ease; cursor: pointer;">
                    {region_name.split('(')[0].strip()} - {inclusion_rate:.1%}
                </button>
            </div>
            """
            
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


# Individual Prediction Page
elif st.session_state.page == 'individual':
    st.markdown("## 🎯 Individual Financial Inclusion Predictor")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 👤 Personal Profile")
        
        with st.form("prediction_form"):
            # Demographics
            st.markdown("**📋 Demographics**")
            age = st.slider("Age", 18, 80, 35, help="Individual's age in years")
            income_level = st.selectbox("Income Level", [1, 2, 3, 4], index=1, 
                                      help="1=Lowest, 4=Highest income quartile")
            education_level = st.selectbox("Education Level", [1, 2, 3, 4], index=1,
                                         help="1=Primary, 2=Secondary, 3=Tertiary, 4=Advanced")
            
            st.markdown("**💼 Business & Credit**")
            biz_loan_source = st.slider("Business Loan Access", 0.0, 1.0, 0.5, 0.1,
                                      help="Access to business loans (0=None, 1=Full access)")
            biz_loan = st.slider("Business Loan Usage", 0.0, 1.0, 0.3, 0.1,
                               help="Current business loan usage")
            emergency_funds = st.slider("Emergency Fund Access", 0.0, 1.0, 0.4, 0.1,
                                      help="Ability to access emergency funds")
            
            st.markdown("**📱 Digital Financial Services**")
            digital_pay = st.slider("Digital Payment Usage", 0.0, 1.0, 0.6, 0.1,
                                   help="Regular use of digital payments")
            digital_pay_acc = st.slider("Digital Payment Account", 0.0, 1.0, 0.5, 0.1,
                                      help="Has digital payment account")
            mobile_pay_s_r = st.slider("Mobile Send/Receive", 0.0, 1.0, 0.4, 0.1,
                                     help="Mobile money send/receive frequency")
            prefer_digital_fin = st.slider("Digital Finance Preference", 0.0, 1.0, 0.5, 0.1,
                                         help="Preference for digital financial services")
            
            st.markdown("**💰 Savings & Other**")
            saved_any = st.slider("Savings Behavior", 0.0, 1.0, 0.4, 0.1,
                                help="Any form of savings")
            borrowed_any = st.slider("Borrowing Behavior", 0.0, 1.0, 0.3, 0.1,
                                   help="Any form of borrowing")
            
            # Simplified additional features for better UX
            loan_purpose_group = st.slider("Loan Clarity", 0.0, 1.0, 0.3, 0.1,
                                         help="Clear purpose for loans")
            digital_payment_other = st.slider("Other Digital Services", 0.0, 1.0, 0.3, 0.1)
            govt_payment_recv = st.slider("Government Payments", 0.0, 1.0, 0.2, 0.1)
            mobile_payment_bill = st.slider("Mobile Bill Payments", 0.0, 1.0, 0.3, 0.1)
            saved_for_purchase = st.slider("Targeted Savings", 0.0, 1.0, 0.3, 0.1)
            loan_purpose = st.slider("Loan Planning", 0.0, 1.0, 0.2, 0.1)
            
            predict_button = st.form_submit_button("🚀 Predict Financial Inclusion", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Model Performance Metrics")
        
        # Enhanced model metrics display
        metrics_data = {
            'Metric': ['Accuracy', 'AUC-ROC', 'AUC-PR', 'F1-Score', 'Precision', 'Recall'],
            'Score': [0.8962, 0.9607, 0.9743, 0.9163, 0.9103, 0.9225],
            'Category': ['Excellent', 'Excellent', 'Excellent', 'Excellent', 'Excellent', 'Excellent']
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig_metrics = px.bar(
            metrics_df, 
            x='Metric', 
            y='Score',
            color='Score',
            color_continuous_scale='viridis',
            title="AI Model Performance Dashboard",
            text=metrics_df['Score'].apply(lambda x: f'{x:.3f}')
        )
        fig_metrics.update_layout(
            height=300,
            showlegend=False
        )
        fig_metrics.update_traces(textposition='outside')
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Model confidence indicator
        st.markdown("""
        <div class="metric-card success-metric">
            <h3>🎯 Model Confidence</h3>
            <h2>96.07% AUC-ROC</h2>
            <p>High-accuracy predictions you can trust</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction Results
    if predict_button:
        st.markdown("---")
        st.markdown("## 🔮 Prediction Results")
        
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
            # Enhanced probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Financial Inclusion Probability", 'font': {'size': 24}},
                delta = {'reference': 68, 'valueformat': '.1f'},  # Global average reference
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue", 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#ff6b6b'},
                        {'range': [30, 50], 'color': '#ffa726'},
                        {'range': [50, 70], 'color': '#ffeb3b'},
                        {'range': [70, 85], 'color': '#66bb6a'},
                        {'range': [85, 100], 'color': '#4caf50'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Detailed interpretation with modern styling
        if probability >= 0.8:
            st.markdown(f"""
            <div class="prediction-result">
                <h2>🎯 HIGH PROBABILITY ({probability:.1%})</h2>
                <p><strong>Strong likelihood of having a bank account</strong></p>
                <p>This individual shows excellent financial inclusion indicators</p>
            </div>
            """, unsafe_allow_html=True)
            recommendation = "**🎯 Strategy**: Focus on premium services and investment products"
            risk_level = "Low Risk"
            action = "Retention & Upselling"
        elif probability >= 0.5:
            st.markdown(f"""
            <div class="prediction-result" style="background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);">
                <h2>⚠️ MODERATE PROBABILITY ({probability:.1%})</h2>
                <p><strong>Uncertain account ownership status</strong></p>
                <p>Mixed financial inclusion signals detected</p>
            </div>
            """, unsafe_allow_html=True)
            recommendation = "**📱 Strategy**: Targeted digital services and emergency fund products"
            risk_level = "Medium Risk"
            action = "Targeted Outreach"
        else:
            st.markdown(f"""
            <div class="prediction-result" style="background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);">
                <h2>🚨 LOW PROBABILITY ({probability:.1%})</h2>
                <p><strong>Likely unbanked individual</strong></p>
                <p>Immediate intervention recommended</p>
            </div>
            """, unsafe_allow_html=True)
            recommendation = "**🚀 Strategy**: Priority intervention with mobile payment focus"
            risk_level = "High Risk"
            action = "Immediate Outreach"
        
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Recommendation Analysis")
            st.markdown(f"""
            **Risk Assessment**: {risk_level}  
            **Recommended Action**: {action}  
            {recommendation}
            
            **🎯 Priority Areas:**
            - Business loan accessibility
            - Digital payment adoption
            - Emergency financial planning
            """)
        
        with col2:
            st.markdown("### 🔍 Top Contributing Factors")
            
            # Calculate feature contributions for this prediction
            feature_contributions = []
            for i, feature in enumerate(feature_cols):
                contribution = input_imputed[0][i] * feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
                feature_contributions.append({
                    'feature': feature.replace('_', ' ').title(), 
                    'contribution': contribution,
                    'value': input_imputed[0][i]
                })
            
            contrib_df = pd.DataFrame(feature_contributions).sort_values('contribution', ascending=False).head(5)
            
            for _, row in contrib_df.iterrows():
                contribution_pct = (row['contribution'] / contrib_df['contribution'].sum()) * 100
                st.markdown(f"""
                <div class="feature-chip">
                    {row['feature']}: {contribution_pct:.1f}% influence
                </div>
                """, unsafe_allow_html=True)
        
        # Visual breakdown of prediction factors
        st.markdown("### 📊 Prediction Factor Analysis")
        
        # Create a comprehensive analysis chart
        fig_factors = px.bar(
            contrib_df.head(8), 
            y='feature', 
            x='contribution',
            orientation='h',
            title='Individual Prediction Drivers',
            color='contribution',
            color_continuous_scale='viridis',
            text=contrib_df.head(8)['contribution'].apply(lambda x: f'{x:.3f}')
        )
        fig_factors.update_layout(
            yaxis=dict(categoryorder='total ascending'),
            height=400,
            showlegend=False
        )
        fig_factors.update_traces(textposition='auto')
        st.plotly_chart(fig_factors, use_container_width=True)

# Policy Recommendations Section (shown on both pages)
st.markdown("---")
st.markdown("## 📋 Strategic Policy Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card">
    <h3>🏢 Business-First Inclusion</h3>
    <p><strong>29.13%</strong> model importance</p>
    <p>Prioritize business loan accessibility and entrepreneurial financial services as the primary pathway to financial inclusion.</p>
    <br>
    <p><strong>Action Items:</strong></p>
    <ul>
    <li>Micro-enterprise lending programs</li>
    <li>Simplified business account opening</li>
    <li>SME-focused digital platforms</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
    <h3>📱 Digital Infrastructure</h3>
    <p><strong>12.33%</strong> model importance</p>
    <p>Accelerate mobile payment systems and digital financial literacy to bridge the inclusion gap.</p>
    <br>
    <p><strong>Action Items:</strong></p>
    <ul>
    <li>National digital ID systems</li>
    <li>Mobile money interoperability</li>
    <li>Digital literacy campaigns</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card">
    <h3>🛡️ Emergency Preparedness</h3>
    <p><strong>9.80%</strong> model importance</p>
    <p>Link disaster resilience with financial inclusion through emergency savings products and crisis support systems.</p>
    <br>
    <p><strong>Action Items:</strong></p>
    <ul>
    <li>Emergency savings incentives</li>
    <li>Crisis support funds</li>
    <li>Insurance product integration</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #636e72; padding: 20px;">
    <p><strong>FinScope Global</strong> | Powered by Machine Learning | Data from 149 countries</p>
    <p>Model Accuracy: 89.6% | Random Forest with 14 key features | Sample: 8,311+ adults</p>
</div>
""", unsafe_allow_html=True)