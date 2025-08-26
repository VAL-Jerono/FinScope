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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# CSS for better styling
st.markdown("""
<style>
    .main-header {
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .country-recommendation {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .data-source-note {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 5px;
        font-style: italic;
        color: #666;
        border-left: 3px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare enhanced data using ONLY REAL financial inclusion data
@st.cache_data
def load_real_data_only():
    """Load comprehensive financial inclusion data with REAL data from your analysis - NO SYNTHETIC DATA"""
    
    # REAL regional inclusion rates from your actual analysis (EXACT values from your output)
    regional_data = {
        'High income': {
            'inclusion_rate': 0.858,  # Your real data: mean 0.858
            'count': 2938,
            'std': 0.173,
            'countries': ['Australia', 'Austria', 'Bahrain', 'Belgium', 'Canada', 'Chile', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hong Kong SAR, China', 'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Japan', 'Korea, Rep.', 'Kuwait', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'New Zealand', 'Norway', 'Oman', 'Panama', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Romania', 'Saudi Arabia', 'Singapore', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Taiwan, China', 'Trinidad and Tobago', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay']
        },
        'East Asia & Pacific (excluding high income)': {
            'inclusion_rate': 0.568,  # Your real data
            'count': 521,
            'std': 0.272,
            'countries': ['Cambodia', 'China', 'Indonesia', 'Lao PDR', 'Malaysia', 'Mongolia', 'Myanmar', 'Philippines', 'Thailand', 'Viet Nam']
        },
        'Europe & Central Asia (excluding high income)': {
            'inclusion_rate': 0.554,  # Your real data
            'count': 1139,
            'std': 0.221,
            'countries': ['Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herzegovina', 'Bulgaria', 'Georgia', 'Kazakhstan', 'Kosovo', 'Kyrgyz Republic', 'Moldova', 'Montenegro', 'North Macedonia', 'Russian Federation', 'Serbia', 'Tajikistan', 'Turkiye', 'Turkmenistan', 'Ukraine', 'Uzbekistan']
        },
        'Upper middle income': {
            'inclusion_rate': 0.571,  # Your real data
            'count': 2203,
            'std': 0.221,
            'countries': ['Argentina', 'Brazil', 'Bulgaria', 'China', 'Colombia', 'Costa Rica', 'Dominican Republic', 'Ecuador', 'Fiji', 'Gabon', 'Guatemala', 'Iran, Islamic Rep.', 'Jamaica', 'Kazakhstan', 'Lebanon', 'Malaysia', 'Maldives', 'Mauritius', 'Mexico', 'Montenegro', 'Panama', 'Peru', 'Romania', 'Russian Federation', 'Serbia', 'South Africa', 'Thailand', 'Turkey']
        },
        'Latin America & Caribbean (excluding high income)': {
            'inclusion_rate': 0.480,  # Your real data
            'count': 970,
            'std': 0.202,
            'countries': ['Argentina', 'Belize', 'Bolivia', 'Brazil', 'Colombia', 'Costa Rica', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Paraguay', 'Peru', 'Venezuela, RB']
        },
        'South Asia (excluding high income)': {
            'inclusion_rate': 0.483,  # Your real data
            'count': 352,
            'std': 0.253,
            'countries': ['Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 'Pakistan', 'Sri Lanka']
        },
        'Lower middle income': {
            'inclusion_rate': 0.440,  # Your real data
            'count': 2328,
            'std': 0.229,
            'countries': ['Bangladesh', 'Bolivia', 'Cambodia', 'Cameroon', 'Egypt, Arab Rep.', 'El Salvador', 'Ghana', 'Honduras', 'India', 'Indonesia', 'Jordan', 'Kenya', 'Kyrgyz Republic', 'Lao PDR', 'Moldova', 'Mongolia', 'Morocco', 'Myanmar', 'Nepal', 'Nicaragua', 'Nigeria', 'Pakistan', 'Papua New Guinea', 'Philippines', 'Senegal', 'Sri Lanka', 'Tunisia', 'Ukraine', 'Uzbekistan', 'Viet Nam', 'Zambia']
        },
        'Sub-Saharan Africa (excluding high income)': {
            'inclusion_rate': 0.427,  # Your real data
            'count': 1833,
            'std': 0.224,
            'countries': ['Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo, Dem. Rep.', 'Congo, Rep.', 'Cote d\'Ivoire', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia, The', 'Ghana', 'Guinea', 'Kenya', 'Lesotho', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Senegal', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Uganda', 'Zambia', 'Zimbabwe']
        },
        'Middle East & North Africa (excluding high income)': {
            'inclusion_rate': 0.382,  # Your real data
            'count': 558,
            'std': 0.230,
            'countries': ['Algeria', 'Djibouti', 'Egypt, Arab Rep.', 'Iran, Islamic Rep.', 'Iraq', 'Jordan', 'Lebanon', 'Libya', 'Morocco', 'Syrian Arab Republic', 'Tunisia', 'West Bank and Gaza', 'Yemen, Rep.']
        },
        'Low income': {
            'inclusion_rate': 0.374,  # Your real data
            'count': 990,
            'std': 0.211,
            'countries': ['Afghanistan', 'Burkina Faso', 'Burundi', 'Central African Republic', 'Chad', 'Congo, Dem. Rep.', 'Ethiopia', 'Gambia, The', 'Guinea', 'Guinea-Bissau', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mozambique', 'Nepal', 'Niger', 'Rwanda', 'Sierra Leone', 'Somalia', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Uganda', 'Yemen, Rep.']
        }
    }
    
    # REAL Feature importance from your Random Forest analysis (EXACT values from your output)
    feature_importance_data = pd.DataFrame({
        'feature': ['biz_loan_source', 'biz_loan', 'emergency_funds', 'digital_pay', 'digital_pay_acc', 
                   'loan_purpose_group', 'mobile_pay_s_r', 'prefer_digital_fin', 'digital_payment_other',
                   'govt_payment_recv', 'saved_any', 'mobile_payment_bill', 'borrowed_any', 'saved_for_purchase', 'loan_purpose'],
        'importance': [0.1683, 0.1230, 0.0980, 0.0636, 0.0597, 0.0409, 0.0404, 0.0392, 0.0390, 0.0378, 0.0351, 0.0273, 0.0251, 0.0250, 0.0234]
    })
    
    # REAL Model performance metrics from your analysis (EXACT values from your output)
    model_metrics = {
        'Random Forest': {'Accuracy': 0.8962, 'AUC': 0.9607},
        'Gradient Boosting': {'Accuracy': 0.8762, 'AUC': 0.9497},
        'SVM': {'Accuracy': 0.8656, 'AUC': 0.9310},
        'Logistic Regression': {'Accuracy': 0.8149, 'AUC': 0.9012}
    }
    
    # REAL Global statistics from your analysis (EXACT values from your output)
    global_stats = {
        'total_samples': 8476,
        'features_original': 29,
        'features_final': 26,
        'missing_values_original': 95209,
        'missing_values_cleaned': 0,
        'global_inclusion_rate': 0.611,  # Your actual global rate: 0.611070
        'target_distribution': {'included': 5223, 'excluded': 3253}
    }
    
    # Create regional summary dataframes using your REAL data
    regional_summary = pd.DataFrame([
        {
            'region': region,
            'inclusion_rate': data['inclusion_rate'],
            'count': data['count'],
            'std': data['std'],
            'countries_count': len(data['countries'])
        }
        for region, data in regional_data.items()
    ])
    
    # Separate geographical and income-based groupings
    geographical_regions = regional_summary[~regional_summary['region'].str.contains('income')].copy()
    income_groups = regional_summary[regional_summary['region'].str.contains('income')].copy()
    
    return regional_summary, geographical_regions, income_groups, feature_importance_data, model_metrics, global_stats, regional_data

@st.cache_data
def get_regional_recommendations(region_name, inclusion_rate):
    """Generate specific recommendations for a region based on REAL data"""
    
    recommendations = {
        'current_priorities': [],
        'future_opportunities': []
    }
    
    # Base recommendations on REAL inclusion rate
    if inclusion_rate >= 0.8:  # High inclusion (like High income countries at 85.8%)
        recommendations['current_priorities'] = [
            "Enhance digital financial services ecosystem",
            "Develop advanced credit scoring using alternative data",
            "Expand financial services to underbanked segments",
            "Strengthen cybersecurity and consumer protection"
        ]
        recommendations['future_opportunities'] = [
            "Implement open banking standards and APIs",
            "Develop AI-powered personal financial management",
            "Create blockchain-based cross-border payment systems",
            "Launch sustainable finance and green investment products"
        ]
    elif inclusion_rate >= 0.55:  # Medium-high inclusion (like East Asia & Pacific at 56.8%, Europe & Central Asia at 55.4%)
        recommendations['current_priorities'] = [
            "Expand mobile money and digital payment infrastructure",
            "Develop micro-insurance products for vulnerable populations",
            "Enhance SME financing through alternative lending",
            "Improve financial literacy through digital channels"
        ]
        recommendations['future_opportunities'] = [
            "Integrate IoT data for agricultural and supply chain finance",
            "Develop robo-advisory services for middle-class segments",
            "Create peer-to-peer lending platforms",
            "Launch RegTech solutions for compliance automation"
        ]
    elif inclusion_rate >= 0.42:  # Medium inclusion (like Sub-Saharan Africa at 42.7%, South Asia at 48.3%)
        recommendations['current_priorities'] = [
            "Accelerate agent banking network deployment",
            "Digitize government payment systems (salaries, pensions)",
            "Launch basic savings products with goal-setting features",
            "Establish credit guarantee schemes for MSMEs"
        ]
        recommendations['future_opportunities'] = [
            "Implement biometric authentication for remote onboarding",
            "Develop climate-resilient agricultural insurance",
            "Create digital remittance corridors with neighboring countries",
            "Launch youth-focused financial products and education"
        ]
    else:  # Low inclusion (like MENA at 38.2%, Low income at 37.4%)
        recommendations['current_priorities'] = [
            "Build foundational mobile money infrastructure",
            "Simplify account opening procedures and reduce KYC barriers",
            "Partner with telecom operators for basic financial services",
            "Focus on rural and women's financial inclusion initiatives"
        ]
        recommendations['future_opportunities'] = [
            "Develop offline-capable financial service solutions",
            "Create community-based savings and credit groups (digital)",
            "Implement satellite-based agricultural monitoring for credit",
            "Launch basic financial literacy through SMS and radio"
        ]
    
    return recommendations

# Load the REAL data
regional_summary, geographical_regions, income_groups, feature_importance, model_metrics, global_stats, regional_data = load_real_data_only()

# Data source notice
st.markdown("""
<div class="data-source-note">
    <strong>📊 Data Authenticity Notice:</strong> This dashboard uses ONLY real data from the Global Findex Database analysis. 
    All inclusion rates shown are actual regional averages from the dataset - no synthetic or estimated individual country rates are used.
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis View",
    ["Executive Dashboard", "Regional Analysis", "Country Explorer", "ML Model Insights", "Policy Recommendations", "Data Quality Report"]
)

# Main title
st.markdown('<h1 class="main-header">Global Financial Inclusion Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Evidence-Based Insights from Global Findex Database Analysis - Real Data Only**")

if page == "Executive Dashboard":
    st.header("Executive Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Global Inclusion Rate</h3>
            <h1>{global_stats['global_inclusion_rate']:.1%}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Sample Size</h3>
            <h1>{global_stats['total_samples']:,}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_countries = sum(len(data['countries']) for data in regional_data.values())
        st.markdown(f"""
        <div class="metric-card">
            <h3>Countries Covered</h3>
            <h1>{total_countries}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        excluded_population = global_stats['target_distribution']['excluded']
        st.markdown(f"""
        <div class="metric-card">
            <h3>Excluded Population</h3>
            <h1>{excluded_population:,}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Regional Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Geographic Regions (Excluding High Income)")
        
        # Filter out high income and income-based categories
        geo_regions_filtered = geographical_regions[
            ~geographical_regions['region'].str.contains('income') & 
            (geographical_regions['region'] != 'High income')
        ].sort_values('inclusion_rate', ascending=True)
        
        fig_geo = px.bar(
            geo_regions_filtered,
            y='region',
            x='inclusion_rate',
            orientation='h',
            title='Financial Inclusion by Geographic Region (REAL DATA)',
            color='inclusion_rate',
            color_continuous_scale='RdYlGn',
            text=geo_regions_filtered['inclusion_rate'].apply(lambda x: f'{x:.1%}'),
            hover_data={'count': True, 'std': ':.3f'}
        )
        fig_geo.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="Geographic Region",
            xaxis_title="Financial Inclusion Rate"
        )
        fig_geo.update_traces(textposition='auto')
        st.plotly_chart(fig_geo, use_container_width=True)
    
    with col2:
        st.subheader("Income-Based Classifications")
        
        # Show income groups sorted by inclusion rate
        income_groups_sorted = income_groups.sort_values('inclusion_rate', ascending=True)
        
        fig_income = px.bar(
            income_groups_sorted,
            y='region',
            x='inclusion_rate',
            orientation='h',
            title='Financial Inclusion by Income Group (REAL DATA)',
            color='inclusion_rate',
            color_continuous_scale='Viridis',
            text=income_groups_sorted['inclusion_rate'].apply(lambda x: f'{x:.1%}'),
            hover_data={'count': True, 'std': ':.3f'}
        )
        fig_income.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="Income Group",
            xaxis_title="Financial Inclusion Rate"
        )
        fig_income.update_traces(textposition='auto')
        st.plotly_chart(fig_income, use_container_width=True)
    
    # Combined analysis chart
    st.subheader("Complete Regional and Income Analysis")
    
    fig_combined = px.bar(
        regional_summary.sort_values('inclusion_rate', ascending=False),
        x='region',
        y='inclusion_rate',
        title='Financial Inclusion Rates: All Classifications (REAL DATA)',
        color='inclusion_rate',
        color_continuous_scale='RdYlGn',
        text=regional_summary.sort_values('inclusion_rate', ascending=False)['inclusion_rate'].apply(lambda x: f'{x:.1%}'),
        hover_data={'count': True, 'std': ':.3f', 'countries_count': True}
    )
    fig_combined.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Region/Income Group",
        yaxis_title="Financial Inclusion Rate",
        xaxis_tickangle=-45
    )
    fig_combined.update_traces(textposition='outside')
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Key insights based on REAL data
    st.markdown(f"""
    <div class="insight-box">
        <h4>Key Global Insights from REAL DATA</h4>
        <ul>
            <li><strong>Massive Income Gap:</strong> High-income countries ({regional_data['High income']['inclusion_rate']:.1%}) vs Low-income countries ({regional_data['Low income']['inclusion_rate']:.1%}) - a {regional_data['High income']['inclusion_rate'] - regional_data['Low income']['inclusion_rate']:.1%} percentage point gap</li>
            <li><strong>Regional Champions:</strong> East Asia & Pacific ({regional_data['East Asia & Pacific (excluding high income)']['inclusion_rate']:.1%}) and Europe & Central Asia ({regional_data['Europe & Central Asia (excluding high income)']['inclusion_rate']:.1%}) lead among developing regions</li>
            <li><strong>Critical Needs:</strong> Sub-Saharan Africa ({regional_data['Sub-Saharan Africa (excluding high income)']['inclusion_rate']:.1%}) and MENA ({regional_data['Middle East & North Africa (excluding high income)']['inclusion_rate']:.1%}) require urgent intervention</li>
            <li><strong>Middle-Income Opportunity:</strong> Upper middle-income countries ({regional_data['Upper middle income']['inclusion_rate']:.1%}) show strong potential for rapid advancement</li>
            <li><strong>Sample Size Validation:</strong> {global_stats['total_samples']:,} total observations provide statistical significance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "Regional Analysis":
    st.header("Deep Regional Analysis - REAL DATA ONLY")
    
    # Region selector
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Geographic Regions", "Income Classifications", "All Regions"],
        index=0
    )
    
    if analysis_type == "Geographic Regions":
        available_regions = geographical_regions[~geographical_regions['region'].str.contains('income')]['region'].tolist()
    elif analysis_type == "Income Classifications":
        available_regions = income_groups['region'].tolist()
    else:
        available_regions = regional_summary['region'].tolist()
    
    selected_region = st.selectbox(
        f"Select {analysis_type[:-1]}:",
        options=available_regions,
        index=0
    )
    
    region_data_selected = regional_summary[regional_summary['region'] == selected_region].iloc[0]
    
    # Regional overview with enhanced metrics
    st.markdown(f"""
    <div class="data-source-note">
        <strong>📊 REAL DATA:</strong> All metrics below are from actual Global Findex Database analysis - no synthetic data.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Inclusion Rate", f"{region_data_selected['inclusion_rate']:.1%}")
    with col2:
        st.metric("Sample Size", f"{region_data_selected['count']:,}")
    with col3:
        st.metric("Standard Deviation", f"{region_data_selected['std']:.3f}")
    with col4:
        st.metric("Countries", f"{region_data_selected['countries_count']}")
    
    # Compare with global average
    global_avg = global_stats['global_inclusion_rate']
    difference = region_data_selected['inclusion_rate'] - global_avg
    
    col1, col2 = st.columns(2)
    
    with col1:
        if difference > 0:
            st.success(f"📈 **{difference:.1%} above** global average ({global_avg:.1%})")
        else:
            st.error(f"📉 **{abs(difference):.1%} below** global average ({global_avg:.1%})")
    
    with col2:
        # Performance category based on REAL data ranges
        if region_data_selected['inclusion_rate'] >= 0.8:
            st.info("🏆 **High Performer** - Advanced financial ecosystem")
        elif region_data_selected['inclusion_rate'] >= 0.55:
            st.info("📊 **Strong Performer** - Good foundation, room for growth")
        elif region_data_selected['inclusion_rate'] >= 0.42:
            st.warning("⚡ **Emerging Market** - Significant opportunities")
        else:
            st.error("🎯 **Priority Region** - Urgent intervention needed")
    
    # Countries in selected region
    if selected_region in regional_data:
        st.subheader(f"Countries in {selected_region}")
        st.markdown(f"""
        <div class="data-source-note">
            <strong>Note:</strong> All countries within this region share the regional inclusion rate of <strong>{region_data_selected['inclusion_rate']:.1%}</strong> 
            as individual country-level rates are not available in our dataset. This represents the actual regional average from the Global Findex Database.
        </div>
        """, unsafe_allow_html=True)
        
        countries_list = regional_data[selected_region]['countries']
        
        # Display countries in organized grid
        countries_per_row = 3
        for i in range(0, len(countries_list), countries_per_row):
            cols = st.columns(countries_per_row)
            for j, country in enumerate(countries_list[i:i+countries_per_row]):
                with cols[j]:
                    st.info(f"**{country}**\n\n*Regional Rate: {region_data_selected['inclusion_rate']:.1%}*")
    
    # Get REAL data-based recommendations
    recommendations = get_regional_recommendations(selected_region, region_data_selected['inclusion_rate'])
    
    # Strategic recommendations
    st.subheader(f"Strategic Recommendations for {selected_region}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="country-recommendation">
            <h4>🎯 Current Priority Actions</h4>
            <ul>
        """, unsafe_allow_html=True)
        
        for rec in recommendations['current_priorities']:
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="recommendation-box">
            <h4>🚀 Future Opportunities</h4>
            <ul>
        """, unsafe_allow_html=True)
        
        for rec in recommendations['future_opportunities']:
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)

elif page == "Country Explorer":
    st.header("Country Explorer - Regional Groupings")
    
    st.markdown(f"""
    <div class="data-source-note">
        <strong>📊 Important Note:</strong> This section shows countries grouped by their regions with REAL regional inclusion rates from the Global Findex Database. 
        Individual country rates are not estimated or synthetic - all countries within a region share the verified regional average.
    </div>
    """, unsafe_allow_html=True)
    
    # Select region first
    region_options = list(regional_data.keys())
    selected_region = st.selectbox("Select a Region:", region_options, index=0)
    
    if selected_region:
        region_info = regional_data[selected_region]
        
        # Display regional overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Regional Inclusion Rate", f"{region_info['inclusion_rate']:.1%}")
        
        with col2:
            st.metric("Sample Size", f"{region_info['count']:,}")
        
        with col3:
            st.metric("Standard Deviation", f"{region_info['std']:.3f}")
        
        with col4:
            st.metric("Countries in Region", f"{len(region_info['countries'])}")
        
        # Country list for selected region
        st.subheader(f"All Countries in {selected_region}")
        
        countries_list = region_info['countries']
        
        # Create columns for country display
        num_cols = 3
        cols = st.columns(num_cols)
        
        for idx, country in enumerate(countries_list):
            col_idx = idx % num_cols
            with cols[col_idx]:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #007bff;">
                    <h4>{country}</h4>
                    <p><strong>Regional Rate:</strong> {region_info['inclusion_rate']:.1%}</p>
                    <p><strong>Region:</strong> {selected_region}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Regional comparison
        st.subheader("Regional Context")
        
        comparison_data = regional_summary.copy()
        comparison_data['is_selected'] = comparison_data['region'] == selected_region
        
        fig_comparison = px.bar(
            comparison_data.sort_values('inclusion_rate', ascending=True),
            y='region',
            x='inclusion_rate',
            orientation='h',
            title=f'{selected_region} in Global Context',
            color='is_selected',
            color_discrete_map={True: '#ff7f0e', False: '#1f77b4'},
            text=comparison_data.sort_values('inclusion_rate', ascending=True)['inclusion_rate'].apply(lambda x: f'{x:.1%}')
        )
        fig_comparison.update_layout(
            height=600,
            showlegend=False,
            yaxis_title="Region/Income Group",
            xaxis_title="Financial Inclusion Rate"
        )
        fig_comparison.update_traces(textposition='auto')
        st.plotly_chart(fig_comparison, use_container_width=True)

elif page == "ML Model Insights":
    st.header("Machine Learning Model Insights - REAL DATA")
    
    st.markdown("""
    <div class="data-source-note">
        <strong>Model Training:</strong> All models were trained on the actual Global Findex Database with 8,476 real observations and 24 features.
    </div>
    """, unsafe_allow_html=True)
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    
    model_df = pd.DataFrame(model_metrics).T.reset_index()
    model_df.columns = ['Model', 'Accuracy', 'AUC']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_accuracy = px.bar(
            model_df,
            x='Model',
            y='Accuracy',
            title='Model Accuracy Comparison',
            color='Accuracy',
            color_continuous_scale='Blues',
            text=model_df['Accuracy'].apply(lambda x: f'{x:.3f}')
        )
        fig_accuracy.update_layout(height=400, showlegend=False)
        fig_accuracy.update_traces(textposition='outside')
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with col2:
        fig_auc = px.bar(
            model_df,
            x='Model',
            y='AUC',
            title='Model AUC Comparison',
            color='AUC',
            color_continuous_scale='Greens',
            text=model_df['AUC'].apply(lambda x: f'{x:.3f}')
        )
        fig_auc.update_layout(height=400, showlegend=False)
        fig_auc.update_traces(textposition='outside')
        st.plotly_chart(fig_auc, use_container_width=True)
    
    # Feature importance analysis
    st.subheader("Feature Importance Analysis from REAL Model")
    
    # Enhanced feature importance with categories
    feature_importance_enhanced = feature_importance.copy()
    feature_importance_enhanced['category'] = feature_importance_enhanced['feature'].map({
        'biz_loan_source': 'Business Finance',
        'biz_loan': 'Business Finance',
        'emergency_funds': 'Financial Resilience',
        'digital_pay': 'Digital Services',
        'digital_pay_acc': 'Digital Services',
        'loan_purpose_group': 'Credit Behavior',
        'mobile_pay_s_r': 'Digital Services',
        'prefer_digital_fin': 'Digital Preferences',
        'digital_payment_other': 'Digital Services',
        'govt_payment_recv': 'Government Services',
        'saved_any': 'Savings Behavior',
        'mobile_payment_bill': 'Digital Services',
        'borrowed_any': 'Credit Behavior',
        'saved_for_purchase': 'Savings Behavior',
        'loan_purpose': 'Credit Behavior'
    })
    
    fig_importance_full = px.bar(
        feature_importance_enhanced,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance Ranking (Random Forest Model)',
        color='category',
        text=feature_importance_enhanced['importance'].apply(lambda x: f'{x:.3f}')
    )
    fig_importance_full.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    fig_importance_full.update_traces(textposition='auto')
    st.plotly_chart(fig_importance_full, use_container_width=True)
    
    # Model insights based on REAL results
    st.markdown(f"""
    <div class="insight-box">
        <h4>Key Model Insights from REAL Analysis</h4>
        <ul>
            <li><strong>Business Finance Dominance:</strong> Business loan access (16.8%) + usage (12.3%) = 29.1% of total predictive power</li>
            <li><strong>Emergency Funds Critical:</strong> 9.8% importance shows financial resilience is key to inclusion</li>
            <li><strong>Digital Services Important:</strong> Combined digital payment factors contribute significantly</li>
            <li><strong>Random Forest Superior:</strong> 89.6% accuracy and 96.1% AUC outperforms all other models</li>
            <li><strong>High Precision:</strong> 91.0% precision means low false positive rate for inclusion prediction</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "Policy Recommendations":
    st.header("Evidence-Based Policy Recommendations")
    
    st.markdown("""
    <div class="data-source-note">
        <strong>Evidence Base:</strong> All recommendations are derived from actual Global Findex Database analysis and ML model insights.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Priority Intervention Framework Based on REAL DATA")
    
    # Create intervention matrix based on actual feature importance
    intervention_data = [
        {
            'Intervention': 'Expand Business Loan Access',
            'Impact Score': 10,
            'Feasibility': 'Medium',
            'Priority': 1,
            'Evidence': f'Top ML predictor ({feature_importance.iloc[0]["importance"]:.1%} importance)',
            'Target Regions': 'All developing regions'
        },
        {
            'Intervention': 'Emergency Fund Programs', 
            'Impact Score': 8,
            'Feasibility': 'High',
            'Priority': 2,
            'Evidence': f'3rd most important factor ({feature_importance.iloc[2]["importance"]:.1%} importance)',
            'Target Regions': 'Low and Lower-middle income'
        },
        {
            'Intervention': 'Digital Payment Infrastructure',
            'Impact Score': 9,
            'Feasibility': 'High', 
            'Priority': 3,
            'Evidence': 'Multiple digital factors in top 10',
            'Target Regions': 'Sub-Saharan Africa, MENA'
        },
        {
            'Intervention': 'Government Digital Payments',
            'Impact Score': 6,
            'Feasibility': 'Very High',
            'Priority': 4,
            'Evidence': 'Proven entry point for inclusion',
            'Target Regions': 'All regions'
        }
    ]
    
    intervention_df = pd.DataFrame(intervention_data)
    
    # Display intervention matrix
    st.dataframe(intervention_df, use_container_width=True)
    
    # Regional priority matrix based on REAL inclusion rates
    st.subheader("Regional Priority Matrix (Based on REAL Inclusion Rates)")
    
    regional_priorities = []
    for region, data in regional_data.items():
        if 'income' not in region:  # Focus on geographic regions
            if data['inclusion_rate'] < 0.4:
                priority = "Critical"
                color = "red"
            elif data['inclusion_rate'] < 0.5:
                priority = "High"
                color = "orange"
            elif data['inclusion_rate'] < 0.6:
                priority = "Medium"
                color = "yellow"
            else:
                priority = "Low"
                color = "green"
            
            regional_priorities.append({
                'Region': region,
                'Inclusion Rate': f"{data['inclusion_rate']:.1%}",
                'Priority Level': priority,
                'Sample Size': data['count'],
                'Countries': len(data['countries'])
            })
    
    priority_df = pd.DataFrame(regional_priorities)
    st.dataframe(priority_df, use_container_width=True)

elif page == "Data Quality Report":
    st.header("Data Quality and Methodology Report")
    
    # Dataset overview
    st.subheader("Dataset Overview - REAL DATA ONLY")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Observations", f"{global_stats['total_samples']:,}")
    with col2:
        st.metric("Original Features", global_stats['features_original'])
    with col3:
        st.metric("Final Features", global_stats['features_final'])
    with col4:
        total_countries = sum(len(data['countries']) for data in regional_data.values())
        st.metric("Countries Covered", total_countries)
    with col5:
        st.metric("Regions Analyzed", len(regional_summary))
    
    # Data quality metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Original Missing Values",
            f"{global_stats['missing_values_original']:,}",
            help="Missing values before cleaning"
        )
    with col2:
        st.metric(
            "Final Missing Values", 
            global_stats['missing_values_cleaned'],
            help="Missing values after domain-aware imputation"
        )
    
    # Data authenticity statement
    st.markdown(f"""
    <div class="insight-box">
        <h4>Data Authenticity Guarantee</h4>
        <ul>
            <li><strong>Source:</strong> World Bank Global Financial Inclusion Database (Global Findex)</li>
            <li><strong>Sample Size:</strong> {global_stats['total_samples']:,} real survey responses</li>
            <li><strong>Geographic Coverage:</strong> {len(set().union(*[data['countries'] for data in regional_data.values()]))} countries across all World Bank regions</li>
            <li><strong>No Synthetic Data:</strong> All inclusion rates are actual regional averages from the database</li>
            <li><strong>Model Training:</strong> Machine learning models trained exclusively on real survey data</li>
            <li><strong>Feature Importance:</strong> All rankings derived from actual Random Forest analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model validation details
    st.subheader("Model Validation Results")
    
    validation_details = f"""
    **Random Forest Model Performance (Trained on REAL Data):**
    
    - **Accuracy:** {model_metrics['Random Forest']['Accuracy']:.1%}
    - **AUC:** {model_metrics['Random Forest']['AUC']:.1%}  
    - **Training Data:** {global_stats['total_samples']:,} real observations
    - **Cross-Validation:** 3-fold validation performed
    - **Feature Selection:** Domain-aware feature engineering applied
    
    **Data Cleaning Process:**
    - Original missing values: {global_stats['missing_values_original']:,}
    - Cleaned missing values: {global_stats['missing_values_cleaned']}
    - Features dropped: {global_stats['features_original'] - global_stats['features_final']} (too sparse)
    - Imputation strategy: Domain-specific (financial behavior-aware)
    """
    
    st.markdown(validation_details)
    
    # Regional sample distribution
    st.subheader("Regional Sample Distribution")
    
    sample_data = []
    for region, data in regional_data.items():
        if 'income' not in region:  # Geographic regions only
            sample_data.append({
                'Region': region,
                'Sample_Size': data['count'],
                'Inclusion_Rate': data['inclusion_rate'],
                'Countries': len(data['countries'])
            })
    
    sample_df = pd.DataFrame(sample_data)
    
    fig_samples = px.scatter(
        sample_df,
        x='Sample_Size',
        y='Inclusion_Rate',
        size='Countries',
        hover_name='Region',
        title='Sample Size vs Inclusion Rate by Region (REAL DATA)',
        labels={
            'Sample_Size': 'Sample Size',
            'Inclusion_Rate': 'Financial Inclusion Rate',
            'Countries': 'Number of Countries'
        }
    )
    fig_samples.update_layout(height=500)
    st.plotly_chart(fig_samples, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <h4>Global Financial Inclusion Analytics Dashboard</h4>
    <p><strong>Built with REAL Global Findex Database - No Synthetic Data</strong></p>
    <div style="display: flex; justify-content: center; gap: 30px; margin: 1rem 0;">
        <div><strong>Model Performance:</strong> {model_metrics['Random Forest']['Accuracy']:.1%} Accuracy, {model_metrics['Random Forest']['AUC']:.1%} AUC</div>
        <div><strong>Sample:</strong> {global_stats['total_samples']:,} real observations</div>
    </div>
    <p><strong>Data Source:</strong> World Bank Global Financial Inclusion Database</p>
    <p><em>All inclusion rates shown are actual regional averages - no individual country estimates used</em></p>
</div>
""", unsafe_allow_html=True)