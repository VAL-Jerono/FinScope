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
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        margin-bottom: 15px;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
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
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 6px 25px rgba(102,126,234,0.3);
    }
    .region-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 6px solid;
        transition: all 0.3s ease;
    }
    .region-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .stats-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Your actual regional data
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
    
    # Income group data from your analysis
    income_data = {
        'income_group': ['High income', 'Upper middle income', 'Lower middle income', 'Low income'],
        'inclusion_rate': [0.870, 0.571, 0.440, 0.374],
        'count': [2790, 2203, 2328, 990]
    }
    
    # Your actual Random Forest feature importance
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
    country_mapping = {
        'High income': {
            'countries': ['Australia', 'Austria', 'Bahrain', 'Belgium', 'Canada', 'Chile', 'Croatia', 
                         'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 
                         'Greece', 'Hong Kong', 'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 
                         'Japan', 'South Korea', 'Kuwait', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 
                         'Netherlands', 'New Zealand', 'Norway', 'Oman', 'Panama', 'Poland', 'Portugal', 
                         'Qatar', 'Romania', 'Saudi Arabia', 'Singapore', 'Slovakia', 'Slovenia', 
                         'Spain', 'Sweden', 'Switzerland', 'Taiwan', 'Trinidad and Tobago', 
                         'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay'],
            'iso_codes': ['AUS', 'AUT', 'BHR', 'BEL', 'CAN', 'CHL', 'HRV', 'CYP', 'CZE', 'DNK', 
                         'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'HKG', 'HUN', 'ISL', 'IRL', 'ISR', 
                         'ITA', 'JPN', 'KOR', 'KWT', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'NZL', 
                         'NOR', 'OMN', 'PAN', 'POL', 'PRT', 'QAT', 'ROU', 'SAU', 'SGP', 'SVK', 
                         'SVN', 'ESP', 'SWE', 'CHE', 'TWN', 'TTO', 'ARE', 'GBR', 'USA', 'URY'],
            'color': '#2E8B57'
        },
        'East Asia & Pacific (excluding high income)': {
            'countries': ['Cambodia', 'China', 'Indonesia', 'Laos', 'Malaysia', 'Mongolia', 
                         'Myanmar', 'Philippines', 'Thailand', 'Vietnam'],
            'iso_codes': ['KHM', 'CHN', 'IDN', 'LAO', 'MYS', 'MNG', 'MMR', 'PHL', 'THA', 'VNM'],
            'color': '#FF6B35'
        },
        'Europe & Central Asia (excluding high income)': {
            'countries': ['Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herzegovina', 
                         'Bulgaria', 'Georgia', 'Kazakhstan', 'Kosovo', 'Kyrgyzstan', 'Moldova', 
                         'Montenegro', 'North Macedonia', 'Russia', 'Serbia', 'Tajikistan', 
                         'Turkey', 'Turkmenistan', 'Ukraine', 'Uzbekistan'],
            'iso_codes': ['ALB', 'ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'GEO', 'KAZ', 'XKX', 'KGZ', 
                         'MDA', 'MNE', 'MKD', 'RUS', 'SRB', 'TJK', 'TUR', 'TKM', 'UKR', 'UZB'],
            'color': '#F7931E'
        },
        'South Asia (excluding high income)': {
            'countries': ['Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 
                         'Pakistan', 'Sri Lanka'],
            'iso_codes': ['AFG', 'BGD', 'BTN', 'IND', 'MDV', 'NPL', 'PAK', 'LKA'],
            'color': '#FFD23F'
        },
        'Latin America & Caribbean (excluding high income)': {
            'countries': ['Argentina', 'Belize', 'Bolivia', 'Brazil', 'Colombia', 'Costa Rica', 
                         'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Haiti', 
                         'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Paraguay', 'Peru', 'Venezuela'],
            'iso_codes': ['ARG', 'BLZ', 'BOL', 'BRA', 'COL', 'CRI', 'DOM', 'ECU', 'SLV', 'GTM', 
                         'HTI', 'HND', 'JAM', 'MEX', 'NIC', 'PRY', 'PER', 'VEN'],
            'color': '#FF6B35'
        },
        'Sub-Saharan Africa (excluding high income)': {
            'countries': ['Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 
                         'Central African Republic', 'Chad', 'Comoros', 'Democratic Republic of the Congo', 
                         'Republic of the Congo', 'Ivory Coast', 'Eswatini', 'Ethiopia', 'Gabon', 
                         'Gambia', 'Ghana', 'Guinea', 'Kenya', 'Lesotho', 'Liberia', 'Madagascar', 
                         'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Mozambique', 'Namibia', 
                         'Niger', 'Nigeria', 'Rwanda', 'Senegal', 'Sierra Leone', 'Somalia', 
                         'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Uganda', 
                         'Zambia', 'Zimbabwe'],
            'iso_codes': ['AGO', 'BEN', 'BWA', 'BFA', 'BDI', 'CMR', 'CAF', 'TCD', 'COM', 'COD', 
                         'COG', 'CIV', 'SWZ', 'ETH', 'GAB', 'GMB', 'GHA', 'GIN', 'KEN', 'LSO', 
                         'LBR', 'MDG', 'MWI', 'MLI', 'MRT', 'MUS', 'MOZ', 'NAM', 'NER', 'NGA', 
                         'RWA', 'SEN', 'SLE', 'SOM', 'ZAF', 'SSD', 'SDN', 'TZA', 'TGO', 'UGA', 
                         'ZMB', 'ZWE'],
            'color': '#E74C3C'
        },
        'Middle East & North Africa (excluding high income)': {
            'countries': ['Algeria', 'Djibouti', 'Egypt', 'Iran', 'Iraq', 'Jordan', 'Lebanon', 
                         'Libya', 'Morocco', 'Syria', 'Tunisia', 'Palestine', 'Yemen'],
            'iso_codes': ['DZA', 'DJI', 'EGY', 'IRN', 'IRQ', 'JOR', 'LBN', 'LBY', 'MAR', 'SYR', 
                         'TUN', 'PSE', 'YEM'],
            'color': '#C0392B'
        }
    }
    
    return pd.DataFrame(regional_data), pd.DataFrame(income_data), pd.DataFrame(feature_importance), country_mapping

# Load data
regional_df, income_df, feature_df, country_mapping = load_data()

# Session state for selected region
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None

# Enhanced Header
st.markdown("""
<div class="main-header">
    <h1>🌍 FinScope Global</h1>
    <h2>Financial Inclusion Analytics Dashboard</h2>
    <p style="font-size: 18px; margin: 15px 0;"><i>Empowering evidence-based policy through machine learning insights</i></p>
    <p style="font-size: 16px; font-weight: bold;">📊 Covering 149 countries across 7 regions | 🎯 1.4 billion adults without financial accounts</p>
</div>
""", unsafe_allow_html=True)

# Global Statistics Dashboard
st.markdown("## 📈 Global Financial Inclusion Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea;">🌐 Global Average</h3>
        <h2 style="color: #2d3436; margin: 10px 0;">61.1%</h2>
        <p style="color: #636e72;">Inclusion Rate</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea;">📊 Total Sample</h3>
        <h2 style="color: #2d3436; margin: 10px 0;">8,311</h2>
        <p style="color: #636e72;">Adults Surveyed</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea;">🏆 Best Region</h3>
        <h2 style="color: #2d3436; margin: 10px 0;">85.8%</h2>
        <p style="color: #636e72;">High Income</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea;">⚡ ML Accuracy</h3>
        <h2 style="color: #2d3436; margin: 10px 0;">89.6%</h2>
        <p style="color: #636e72;">Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea;">🎯 Countries</h3>
        <h2 style="color: #2d3436; margin: 10px 0;">149</h2>
        <p style="color: #636e72;">Analyzed</p>
    </div>
    """, unsafe_allow_html=True)

# Interactive World Map
st.markdown("## 🗺️ Interactive World Financial Inclusion Map")
st.markdown("### *Click on any country to explore regional insights and recommendations*")

# Create country-level data for the map
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
fig_world = go.Figure(data=go.Choropleth(
    locations=country_df['iso_code'],
    z=country_df['inclusion_rate'],
    locationmode='ISO-3',
    colorscale=[
        [0.0, '#C0392B'],    # Lowest (MENA color)
        [0.2, '#E74C3C'],    # Sub-Saharan Africa
        [0.4, '#F7931E'],    # Europe & Central Asia
        [0.5, '#FFD23F'],    # South Asia
        [0.6, '#FF6B35'],    # Latin America & East Asia Pacific
        [1.0, '#2E8B57']     # Highest (High income)
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
        tickformat='.0%'
    )
))

fig_world.update_layout(
    title={
        'text': '<b>Global Financial Inclusion Rates by Country</b><br><sub>Based on Regional Analysis of 149 Countries</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': dict(size=20)
    },
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth',
        bgcolor='rgba(240,240,240,0.1)'
    ),
    height=600,
    margin=dict(l=0, r=0, t=80, b=0)
)

# Display the map with click handling
map_click = st.plotly_chart(fig_world, use_container_width=True, key="world_map")

# Regional Analysis Cards
st.markdown("## 🌍 Regional Performance Analysis")

# Create regional cards with enhanced styling
regions_sorted = regional_df.sort_values('inclusion_rate', ascending=False)

for idx, (_, region_data) in enumerate(regions_sorted.iterrows()):
    region_name = region_data['region']
    inclusion_rate = region_data['inclusion_rate']
    sample_size = region_data['count']
    
    # Get color from mapping
    region_color = country_mapping[region_name]['color']
    
    # Determine rank and performance category
    rank = idx + 1
    if inclusion_rate >= 0.7:
        performance = "🟢 Excellent"
    elif inclusion_rate >= 0.5:
        performance = "🟡 Moderate"
    else:
        performance = "🔴 Needs Focus"
    
    # Create expandable region card
    with st.expander(f"#{rank} {region_name.split('(')[0].strip()} - {inclusion_rate:.1%} Financial Inclusion", expanded=(rank <= 3)):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div style="border-left: 6px solid {region_color}; padding-left: 15px;">
                <h4 style="color: {region_color}; margin: 0;">{region_name}</h4>
                <p style="margin: 5px 0; color: #666;">
                    <strong>Countries:</strong> {len(country_mapping[region_name]['countries'])} nations<br>
                    <strong>Sample Size:</strong> {sample_size:,} adults<br>
                    <strong>Performance:</strong> {performance}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Countries in this region
            countries = country_mapping[region_name]['countries']
            st.markdown("**Key Countries:**")
            for country in countries[:5]:  # Show first 5 countries
                st.markdown(f"• {country}")
            if len(countries) > 5:
                st.markdown(f"*...and {len(countries)-5} more*")
        
        with col3:
            # Quick stats
            gap_to_best = regions_sorted.iloc[0]['inclusion_rate'] - inclusion_rate
            st.metric(
                "Gap to Best", 
                f"{gap_to_best:.1%}" if gap_to_best > 0 else "Leading Region",
                delta=f"Rank #{rank}"
            )
            
        # Quick recommendations based on performance
        if inclusion_rate < 0.5:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%); 
                        border-left: 4px solid #e53e3e; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>🎯 Priority Actions:</strong> Basic financial infrastructure, mobile banking expansion, 
                regulatory frameworks, and financial literacy programs.
            </div>
            """, unsafe_allow_html=True)
        elif inclusion_rate < 0.7:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fffbeb 0%, #fef5e7 100%); 
                        border-left: 4px solid #d69e2e; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>🚀 Growth Opportunities:</strong> Digital payment systems, fintech partnerships, 
                SME financing, and youth financial inclusion programs.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); 
                        border-left: 4px solid #38a169; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>🌟 Excellence Areas:</strong> Innovation leadership, underbanked support, 
                global best practice sharing, and fintech innovation hubs.
            </div>
            """, unsafe_allow_html=True)

# Enhanced Visualizations
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Regional Inclusion Rates Comparison")
    
    # Enhanced bar chart
    fig_regions = px.bar(
        regions_sorted,
        x='inclusion_rate',
        y='region',
        orientation='h',
        color='inclusion_rate',
        color_continuous_scale=['#C0392B', '#E74C3C', '#F7931E', '#FFD23F', '#FF6B35', '#2E8B57'],
        text='inclusion_rate',
        title="Financial Inclusion by Region"
    )
    
    fig_regions.update_traces(
        texttemplate='%{text:.1%}', 
        textposition='outside',
        textfont=dict(size=12, color='black', family='Arial Black')
    )
    
    fig_regions.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Financial Inclusion Rate",
        yaxis_title="",
        xaxis=dict(tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_regions, use_container_width=True)

with col2:
    st.markdown("### 🎯 Income Group Analysis")
    
    fig_income = px.bar(
        income_df,
        x='income_group',
        y='inclusion_rate',
        color='inclusion_rate',
        color_continuous_scale='Viridis',
        text='inclusion_rate',
        title="Inclusion Rate by Income Level"
    )
    
    fig_income.update_traces(
        texttemplate='%{text:.1%}',
        textposition='outside',
        textfont=dict(size=12, color='black', family='Arial Black')
    )
    
    fig_income.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Income Group",
        yaxis_title="Inclusion Rate",
        yaxis=dict(tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_income, use_container_width=True)

# Machine Learning Insights
st.markdown("## 🤖 Machine Learning Model Insights")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🔑 Top Predictive Features (Random Forest Analysis)")
    
    # Enhanced feature importance chart
    top_features = feature_df.head(10)
    
    fig_features = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='Plasma',
        text='importance',
        title="Most Important Factors for Financial Inclusion"
    )
    
    fig_features.update_traces(
        texttemplate='%{text:.3f}',
        textposition='outside',
        textfont=dict(size=11)
    )
    
    fig_features.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Feature Importance Score",
        yaxis_title="",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_features, use_container_width=True)

with col2:
    st.markdown("### 🎯 Model Performance")
    
    st.markdown("""
    <div class="stats-container">
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #667eea;">Random Forest Classifier</h3>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin: 15px 0;">
            <div style="text-align: center;">
                <h2 style="color: #2d3436; margin: 5px 0;">89.6%</h2>
                <p style="color: #636e72;">Accuracy</p>
            </div>
            <div style="text-align: center;">
                <h2 style="color: #2d3436; margin: 5px 0;">14</h2>
                <p style="color: #636e72;">Features</p>
            </div>
        </div>
        
        <div style="margin: 20px 0;">
            <h4 style="color: #667eea;">Key Insights:</h4>
            <ul style="color: #636e72; line-height: 1.6;">
                <li><strong>Business loan access</strong> is the strongest predictor</li>
                <li><strong>Emergency funds</strong> availability is crucial</li>
                <li><strong>Digital engagement</strong> drives inclusion</li>
                <li><strong>Government services</strong> usage correlates highly</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Action Dashboard
st.markdown("## 🎯 Strategic Action Dashboard")

# Priority regions for intervention
low_performing = regional_df[regional_df['inclusion_rate'] < 0.55].sort_values('inclusion_rate')

st.markdown("### 🚨 Priority Regions for Intervention")

for _, region_data in low_performing.iterrows():
    region_name = region_data['region']
    inclusion_rate = region_data['inclusion_rate']
    sample_size = region_data['count']
    
    potential_impact = (0.61 - inclusion_rate) * sample_size  # Assuming global average as target
    
    st.markdown(f"""
    <div class="region-card" style="border-left-color: #E74C3C;">
        <h4 style="color: #E74C3C; margin-top: 0;">{region_name.split('(')[0].strip()}</h4>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="margin: 5px 0;"><strong>Current Rate:</strong> {inclusion_rate:.1%}</p>
                <p style="margin: 5px 0;"><strong>Gap to Global Avg:</strong