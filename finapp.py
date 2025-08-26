import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="FinScope - Global Financial Inclusion Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Regional cards */
    .region-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
        transition: all 0.3s ease;
    }
    
    .region-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Recommendations */
    .policy-recommendation {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #2c3e50;
    }
    
    .individual-recommendation {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #2c3e50;
    }
    
    /* Alert styles */
    .success-alert {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .warning-alert {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .danger-alert {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Economy to region mapping based on World Bank classification
ECONOMY_TO_REGION = {
    # East Asia & Pacific (excluding high income)
    'China': 'East Asia & Pacific (excluding high income)',
    'Indonesia': 'East Asia & Pacific (excluding high income)',
    'Philippines': 'East Asia & Pacific (excluding high income)',
    'Vietnam': 'East Asia & Pacific (excluding high income)',
    'Thailand': 'East Asia & Pacific (excluding high income)',
    'Malaysia': 'East Asia & Pacific (excluding high income)',
    'Myanmar': 'East Asia & Pacific (excluding high income)',
    'Cambodia': 'East Asia & Pacific (excluding high income)',
    'Laos': 'East Asia & Pacific (excluding high income)',
    'Mongolia': 'East Asia & Pacific (excluding high income)',
    'Papua New Guinea': 'East Asia & Pacific (excluding high income)',
    'Fiji': 'East Asia & Pacific (excluding high income)',
    
    # Europe & Central Asia (excluding high income)
    'Russia': 'Europe & Central Asia (excluding high income)',
    'Turkey': 'Europe & Central Asia (excluding high income)',
    'Kazakhstan': 'Europe & Central Asia (excluding high income)',
    'Ukraine': 'Europe & Central Asia (excluding high income)',
    'Uzbekistan': 'Europe & Central Asia (excluding high income)',
    'Belarus': 'Europe & Central Asia (excluding high income)',
    'Azerbaijan': 'Europe & Central Asia (excluding high income)',
    'Georgia': 'Europe & Central Asia (excluding high income)',
    'Armenia': 'Europe & Central Asia (excluding high income)',
    'Albania': 'Europe & Central Asia (excluding high income)',
    'Bosnia and Herzegovina': 'Europe & Central Asia (excluding high income)',
    'Serbia': 'Europe & Central Asia (excluding high income)',
    'North Macedonia': 'Europe & Central Asia (excluding high income)',
    'Moldova': 'Europe & Central Asia (excluding high income)',
    'Kosovo': 'Europe & Central Asia (excluding high income)',
    'Montenegro': 'Europe & Central Asia (excluding high income)',
    'Kyrgyzstan': 'Europe & Central Asia (excluding high income)',
    'Tajikistan': 'Europe & Central Asia (excluding high income)',
    'Turkmenistan': 'Europe & Central Asia (excluding high income)',
    
    # High income
    'United States': 'High income',
    'Germany': 'High income',
    'Japan': 'High income',
    'United Kingdom': 'High income',
    'France': 'High income',
    'Canada': 'High income',
    'Australia': 'High income',
    'South Korea': 'High income',
    'Spain': 'High income',
    'Italy': 'High income',
    'Netherlands': 'High income',
    'Belgium': 'High income',
    'Switzerland': 'High income',
    'Austria': 'High income',
    'Sweden': 'High income',
    'Norway': 'High income',
    'Denmark': 'High income',
    'Finland': 'High income',
    'Ireland': 'High income',
    'New Zealand': 'High income',
    'Singapore': 'High income',
    'Hong Kong': 'High income',
    'Taiwan': 'High income',
    'Israel': 'High income',
    'United Arab Emirates': 'High income',
    'Saudi Arabia': 'High income',
    'Kuwait': 'High income',
    'Qatar': 'High income',
    'Bahrain': 'High income',
    'Oman': 'High income',
    'Chile': 'High income',
    'Uruguay': 'High income',
    'Poland': 'High income',
    'Czech Republic': 'High income',
    'Slovakia': 'High income',
    'Slovenia': 'High income',
    'Estonia': 'High income',
    'Latvia': 'High income',
    'Lithuania': 'High income',
    'Croatia': 'High income',
    'Hungary': 'High income',
    'Portugal': 'High income',
    'Greece': 'High income',
    'Cyprus': 'High income',
    'Malta': 'High income',
    
    # Latin America & Caribbean (excluding high income)
    'Brazil': 'Latin America & Caribbean (excluding high income)',
    'Mexico': 'Latin America & Caribbean (excluding high income)',
    'Argentina': 'Latin America & Caribbean (excluding high income)',
    'Colombia': 'Latin America & Caribbean (excluding high income)',
    'Peru': 'Latin America & Caribbean (excluding high income)',
    'Venezuela': 'Latin America & Caribbean (excluding high income)',
    'Ecuador': 'Latin America & Caribbean (excluding high income)',
    'Guatemala': 'Latin America & Caribbean (excluding high income)',
    'Cuba': 'Latin America & Caribbean (excluding high income)',
    'Bolivia': 'Latin America & Caribbean (excluding high income)',
    'Dominican Republic': 'Latin America & Caribbean (excluding high income)',
    'Honduras': 'Latin America & Caribbean (excluding high income)',
    'Paraguay': 'Latin America & Caribbean (excluding high income)',
    'Nicaragua': 'Latin America & Caribbean (excluding high income)',
    'El Salvador': 'Latin America & Caribbean (excluding high income)',
    'Costa Rica': 'Latin America & Caribbean (excluding high income)',
    'Panama': 'Latin America & Caribbean (excluding high income)',
    'Jamaica': 'Latin America & Caribbean (excluding high income)',
    'Trinidad and Tobago': 'Latin America & Caribbean (excluding high income)',
    'Guyana': 'Latin America & Caribbean (excluding high income)',
    'Suriname': 'Latin America & Caribbean (excluding high income)',
    'Haiti': 'Latin America & Caribbean (excluding high income)',
    'Belize': 'Latin America & Caribbean (excluding high income)',
    
    # Middle East & North Africa (excluding high income)
    'Egypt': 'Middle East & North Africa (excluding high income)',
    'Iran': 'Middle East & North Africa (excluding high income)',
    'Iraq': 'Middle East & North Africa (excluding high income)',
    'Morocco': 'Middle East & North Africa (excluding high income)',
    'Algeria': 'Middle East & North Africa (excluding high income)',
    'Tunisia': 'Middle East & North Africa (excluding high income)',
    'Jordan': 'Middle East & North Africa (excluding high income)',
    'Lebanon': 'Middle East & North Africa (excluding high income)',
    'Libya': 'Middle East & North Africa (excluding high income)',
    'Yemen': 'Middle East & North Africa (excluding high income)',
    'Syria': 'Middle East & North Africa (excluding high income)',
    'Palestine': 'Middle East & North Africa (excluding high income)',
    'Djibouti': 'Middle East & North Africa (excluding high income)',
    
    # South Asia (excluding high income)
    'India': 'South Asia (excluding high income)',
    'Pakistan': 'South Asia (excluding high income)',
    'Bangladesh': 'South Asia (excluding high income)',
    'Sri Lanka': 'South Asia (excluding high income)',
    'Nepal': 'South Asia (excluding high income)',
    'Afghanistan': 'South Asia (excluding high income)',
    'Bhutan': 'South Asia (excluding high income)',
    'Maldives': 'South Asia (excluding high income)',
    
    # Sub-Saharan Africa (excluding high income)
    'Nigeria': 'Sub-Saharan Africa (excluding high income)',
    'Ethiopia': 'Sub-Saharan Africa (excluding high income)',
    'South Africa': 'Sub-Saharan Africa (excluding high income)',
    'Kenya': 'Sub-Saharan Africa (excluding high income)',
    'Uganda': 'Sub-Saharan Africa (excluding high income)',
    'Tanzania': 'Sub-Saharan Africa (excluding high income)',
    'Ghana': 'Sub-Saharan Africa (excluding high income)',
    'Madagascar': 'Sub-Saharan Africa (excluding high income)',
    'Cameroon': 'Sub-Saharan Africa (excluding high income)',
    'Angola': 'Sub-Saharan Africa (excluding high income)',
    'Burkina Faso': 'Sub-Saharan Africa (excluding high income)',
    'Niger': 'Sub-Saharan Africa (excluding high income)',
    'Malawi': 'Sub-Saharan Africa (excluding high income)',
    'Mali': 'Sub-Saharan Africa (excluding high income)',
    'Zambia': 'Sub-Saharan Africa (excluding high income)',
    'Senegal': 'Sub-Saharan Africa (excluding high income)',
    'Somalia': 'Sub-Saharan Africa (excluding high income)',
    'Chad': 'Sub-Saharan Africa (excluding high income)',
    'Zimbabwe': 'Sub-Saharan Africa (excluding high income)',
    'Guinea': 'Sub-Saharan Africa (excluding high income)',
    'Rwanda': 'Sub-Saharan Africa (excluding high income)',
    'Benin': 'Sub-Saharan Africa (excluding high income)',
    'Burundi': 'Sub-Saharan Africa (excluding high income)',
    'Tunisia': 'Sub-Saharan Africa (excluding high income)',
    'South Sudan': 'Sub-Saharan Africa (excluding high income)',
    'Togo': 'Sub-Saharan Africa (excluding high income)',
    'Sierra Leone': 'Sub-Saharan Africa (excluding high income)',
    'Laos': 'Sub-Saharan Africa (excluding high income)',
    'Libya': 'Sub-Saharan Africa (excluding high income)',
    'Central African Republic': 'Sub-Saharan Africa (excluding high income)',
    'Mauritania': 'Sub-Saharan Africa (excluding high income)',
    'Eritrea': 'Sub-Saharan Africa (excluding high income)',
    'Gambia': 'Sub-Saharan Africa (excluding high income)',
    'Botswana': 'Sub-Saharan Africa (excluding high income)',
    'Namibia': 'Sub-Saharan Africa (excluding high income)',
    'Gabon': 'Sub-Saharan Africa (excluding high income)',
    'Lesotho': 'Sub-Saharan Africa (excluding high income)',
    'Guinea-Bissau': 'Sub-Saharan Africa (excluding high income)',
    'Equatorial Guinea': 'Sub-Saharan Africa (excluding high income)',
    'Mauritius': 'Sub-Saharan Africa (excluding high income)',
    'Eswatini': 'Sub-Saharan Africa (excluding high income)',
    'Djibouti': 'Sub-Saharan Africa (excluding high income)',
    'Comoros': 'Sub-Saharan Africa (excluding high income)',
    'Cape Verde': 'Sub-Saharan Africa (excluding high income)',
    'Sao Tome and Principe': 'Sub-Saharan Africa (excluding high income)',
    'Seychelles': 'Sub-Saharan Africa (excluding high income)',
}

# Load and process data functions
@st.cache_data
def load_actual_data():
    """Load sample data matching your actual dataset structure"""
    np.random.seed(42)
    
    # Regional statistics from your actual data
    regions_data = {
        'East Asia & Pacific (excluding high income)': {
            'inclusion_rate': 0.568, 'std': 0.272, 'count': 521, 'coords': [20, 120]
        },
        'Europe & Central Asia (excluding high income)': {
            'inclusion_rate': 0.554, 'std': 0.221, 'count': 1139, 'coords': [50, 30]
        },
        'High income': {
            'inclusion_rate': 0.858, 'std': 0.173, 'count': 2938, 'coords': [45, 0]
        },
        'Latin America & Caribbean (excluding high income)': {
            'inclusion_rate': 0.480, 'std': 0.202, 'count': 970, 'coords': [-10, -60]
        },
        'Middle East & North Africa (excluding high income)': {
            'inclusion_rate': 0.382, 'std': 0.230, 'count': 558, 'coords': [25, 35]
        },
        'South Asia (excluding high income)': {
            'inclusion_rate': 0.483, 'std': 0.253, 'count': 352, 'coords': [20, 77]
        },
        'Sub-Saharan Africa (excluding high income)': {
            'inclusion_rate': 0.427, 'std': 0.224, 'count': 1833, 'coords': [0, 20]
        }
    }
    
    # Income group statistics from your actual data  
    income_groups_data = {
        'High income': {'inclusion_rate': 0.870, 'std': 0.165, 'count': 2790},
        'Upper middle income': {'inclusion_rate': 0.571, 'std': 0.221, 'count': 2203},
        'Lower middle income': {'inclusion_rate': 0.440, 'std': 0.229, 'count': 2328},
        'Low income': {'inclusion_rate': 0.374, 'std': 0.211, 'count': 990}
    }
    
    # Generate sample data matching your structure
    n_samples = 8311
    regions = list(regions_data.keys())
    income_groups = list(income_groups_data.keys())
    
    # Generate proportional samples per region
    data = []
    for region, region_info in regions_data.items():
        n_region_samples = region_info['count']
        
        for _ in range(n_region_samples):
            # Select income group based on regional characteristics
            if region == 'High income':
                income_group = 'High income'
            elif region_info['inclusion_rate'] > 0.55:
                income_group = np.random.choice(income_groups, p=[0.1, 0.4, 0.4, 0.1])
            else:
                income_group = np.random.choice(income_groups, p=[0.05, 0.15, 0.4, 0.4])
            
            # Generate features based on regional and income characteristics
            base_rate = region_info['inclusion_rate']
            income_multiplier = income_groups_data[income_group]['inclusion_rate'] / 0.611  # Global average
            
            record = {
                'region': region,
                'income_group': income_group,
                'demo_group': np.random.choice(['All adults', 'Female', 'Male', 'Young adults']),
                'demo_subgroup': np.random.choice(['All', 'Primary education or less', 'Secondary education or more']),
                
                # Core features for individual prediction
                'biz_loan_source': np.random.beta(2, 5) * base_rate * income_multiplier,
                'emergency_funds': np.random.beta(3, 4) * base_rate * income_multiplier,
                'digital_pay': np.random.beta(4, 3) * base_rate * income_multiplier,
                'mobile_pay_s_r': np.random.beta(3, 4) * base_rate * income_multiplier,
                'saved_any': np.random.beta(3, 3) * base_rate * income_multiplier,
                
                # Additional features from your dataset
                'borrowed_any': np.random.beta(2, 6) * base_rate,
                'credit_card': np.random.beta(2, 8) * income_multiplier,
                'biz_loan': np.random.beta(2, 7) * base_rate,
                'loan_purpose_group': np.random.beta(2, 6),
                'loan_purpose': np.random.beta(2, 7),
                'saved_old_age': np.random.beta(2, 8) * income_multiplier,
                'saved_for_purchase': np.random.beta(3, 5) * base_rate,
                'saved_no_purpose': np.random.beta(2, 6),
                'digital_payment_other': np.random.beta(2, 5) * base_rate,
                'mobile_payment': np.random.beta(3, 4) * base_rate,
                'mobile_payment_bill': np.random.beta(3, 6) * base_rate,
                'govt_digital_pay': np.random.beta(1, 8) * base_rate,
                'govt_digital_pay_acc': np.random.beta(1, 9) * base_rate,
                'digital_pay_acc': np.random.beta(4, 3) * base_rate * income_multiplier,
                'govt_payment_recv': np.random.beta(1, 7),
                'fin_resilience': np.random.beta(3, 4) * base_rate * income_multiplier,
                'prefer_digital_acc': np.random.beta(3, 4) * base_rate,
                'prefer_digital_fin': np.random.beta(3, 4) * base_rate,
                'prefer_digital': np.random.beta(3, 4) * base_rate,
            }
            
            # Generate account ownership based on regional and income characteristics
            feature_effect = (record['digital_pay'] + record['emergency_funds'] + 
                            record['biz_loan_source'] + record['saved_any']) / 4
            final_prob = base_rate * 0.6 + feature_effect * 0.4
            
            record['has_account'] = max(0.004049, min(1.0, np.random.normal(final_prob, region_info['std']/3)))
            data.append(record)
    
    df = pd.DataFrame(data)
    return df, regions_data, income_groups_data

@st.cache_data
def get_simplified_model(df):
    """Get model using core features for individual prediction"""
    
    # Core features based on your model's top importance
    core_features = [
        'biz_loan_source',    # Most important 
        'emergency_funds',    # High importance
        'digital_pay',        # High importance  
        'mobile_pay_s_r',     # Important
        'saved_any'           # Important
    ]
    
    X = df[core_features]
    y = df['has_account']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    rf_model.fit(X_imputed, y)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': core_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rf_model, imputer, importance_df, core_features

def create_regional_map(regions_data):
    """Create interactive regional map"""
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
    
    for region, info in regions_data.items():
        # Color based on inclusion rate
        if info['inclusion_rate'] >= 0.8:
            color = '#2E8B57'  # SeaGreen
        elif info['inclusion_rate'] >= 0.6:
            color = '#FFD700'  # Gold  
        elif info['inclusion_rate'] >= 0.45:
            color = '#FFA500'  # Orange
        else:
            color = '#DC143C'  # Crimson
        
        # Simplify region names for display
        display_name = region.replace(' (excluding high income)', '')
        
        folium.CircleMarker(
            location=info['coords'],
            radius=max(8, min(25, info['count'] / 150)),
            popup=f"""
            <div style='font-family: Arial; min-width: 220px;'>
                <h4 style='margin: 0; color: #1e3c72;'>{display_name}</h4>
                <hr style='margin: 5px 0;'>
                <p><strong>Inclusion Rate:</strong> {info['inclusion_rate']:.1%}</p>
                <p><strong>Sample Size:</strong> {info['count']:,}</p>
                <p><strong>Std Deviation:</strong> {info['std']:.3f}</p>
            </div>
            """,
            tooltip=f"{display_name}: {info['inclusion_rate']:.1%}",
            color=color,
            fillColor=color,
            fillOpacity=0.7,
            weight=3
        ).add_to(m)
    
    return m

def get_regional_recommendations(region, inclusion_rate):
    """Generate practical regional recommendations"""
    recommendations = []
    
    if inclusion_rate < 0.45:
        recommendations.extend([
            "🏦 **Basic Infrastructure**: Establish mobile money services and banking agents",
            "🎓 **Financial Education**: Launch mass media financial literacy campaigns", 
            "📋 **Simplified Access**: Reduce documentation requirements for basic accounts",
            "💰 **Government Programs**: Link social payments to bank accounts"
        ])
    elif inclusion_rate < 0.65:
        recommendations.extend([
            "💳 **Digital Ecosystem**: Expand merchant acceptance of digital payments",
            "🏪 **Agent Networks**: Scale banking agent presence in rural areas",
            "💼 **SME Support**: Create small business financing programs", 
            "🤝 **Partnerships**: Foster fintech-bank collaborations"
        ])
    else:
        recommendations.extend([
            "🎯 **Advanced Services**: Develop sophisticated digital financial products",
            "📊 **Data Analytics**: Use AI for personalized financial services",
            "🌐 **Integration**: Build cross-border payment capabilities",
            "💎 **Premium Tiers**: Create differentiated service levels"
        ])
    
    return recommendations

def get_individual_recommendations(probability, economy_region, income_group):
    """Generate practical individual recommendations"""
    recommendations = []
    
    if probability < 0.4:
        recommendations.extend([
            "📱 **Start Simple**: Register for mobile money (M-Pesa, Airtel Money, etc.)",
            "🏦 **Community Banking**: Visit local microfinance institutions or savings groups", 
            "💰 **Small Steps**: Begin saving $1-2 weekly to build the habit",
            "🎓 **Learn Basics**: Attend free financial literacy sessions in your area"
        ])
    elif probability < 0.7:
        recommendations.extend([
            "💳 **Open Account**: Visit bank with required documents for basic account",
            "📱 **Go Digital**: Use mobile banking apps for bill payments",
            "🛡️ **Emergency Fund**: Aim to save 1 month's expenses for emergencies",
            "💼 **Business Banking**: Consider business account if self-employed"
        ])
    else:
        recommendations.extend([
            "✅ **Stay Active**: Use account regularly to maintain good standing",
            "📈 **Grow Wealth**: Explore savings products and investment options", 
            "🎯 **Set Goals**: Create specific financial targets and track progress",
            "🤝 **Share Knowledge**: Help others in your community access banking"
        ])
    
    # Add region-specific recommendations
    if 'Sub-Saharan Africa' in economy_region:
        recommendations.append("📲 **Mobile First**: Prioritize mobile money and digital payments")
    elif 'South Asia' in economy_region:
        recommendations.append("🏪 **Agent Banking**: Use banking correspondents in your area")
    elif 'Middle East' in economy_region:
        recommendations.append("🏛️ **Islamic Banking**: Consider Sharia-compliant financial products")
    
    # Add income-specific recommendations  
    if income_group == 'Low income':
        recommendations.append("💡 **Microfinance**: Look into small loans from MFIs")
    elif income_group in ['Upper middle income', 'High income']:
        recommendations.append("💼 **Business Loans**: Explore formal business financing options")
    
    return recommendations[:4]

# Load data and train model
df, regions_data, income_groups_data = load_actual_data()
model, imputer, feature_importance, core_features = get_simplified_model(df)

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 FinScope Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Global Financial Inclusion Analytics & Decision Support System</p>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col2:
        if st.button("📊 General Statistics", use_container_width=True):
            st.session_state.view = 'general'
    with col3:
        if st.button("🗺️ Regional Analytics", use_container_width=True):
            st.session_state.view = 'regional'
    with col4:
        if st.button("👤 Individual Predictor", use_container_width=True):
            st.session_state.view = 'individual'
    
    # Initialize session state
    if 'view' not in st.session_state:
        st.session_state.view = 'general'
    
    st.markdown("---")
    
    # Route to different views
    if st.session_state.view == 'general':
        show_general_statistics()
    elif st.session_state.view == 'regional':
        show_regional_analytics() 
    elif st.session_state.view == 'individual':
        show_individual_predictor()

def show_general_statistics():
    st.header("📊 Global Financial Inclusion Overview")
    
    # Global metrics from actual data
    global_inclusion_rate = 0.611  # From your dataset
    total_samples = 8311
    banked_samples = int(total_samples * global_inclusion_rate)
    unbanked_samples = total_samples - banked_samples
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">61.1%</div>
            <div class="metric-label">Global Inclusion Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{banked_samples:,}</div>
            <div class="metric-label">Banked Adults (Sample)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{unbanked_samples:,}</div>
            <div class="metric-label">Unbanked Adults (Sample)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">7</div>
            <div class="metric-label">Regions Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Regional and income analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍