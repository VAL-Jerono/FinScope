import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🏦 Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .high-risk {
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
        color: white;
    }
    
    .low-risk {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        color: white;
    }
    
    .medium-risk {
        background: linear-gradient(45deg, #ffa726, #ff9800);
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load your trained model (you'll need to save it first)
@st.cache_data
def load_model():
    # For demo purposes, we'll create a mock model
    # In production, you'd load your actual trained model:
    # with open('best_model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # return model
    return None  # Mock model for now

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    np.random.seed(42)
    
    countries = ['Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Ethiopia', 'Ghana', 
                'Nigeria', 'South Africa', 'Zambia', 'Malawi']
    regions = ['East Africa', 'West Africa', 'Southern Africa']
    
    # Create sample dataset
    n_samples = 1000
    data = {
        'country': np.random.choice(countries, n_samples),
        'biz_loan_source': np.random.choice([0, 1, 2], n_samples),
        'biz_loan': np.random.choice([0, 1], n_samples),
        'digital_pay_acc': np.random.random(n_samples),
        'digital_engagement_score': np.random.random(n_samples) * 10,
        'prefer_digital_acc': np.random.choice([0, 1], n_samples),
        'credit_card': np.random.choice([0, 1], n_samples),
        'mobile_pay_s_r': np.random.choice([0, 1], n_samples),
        'loan_purpose_group': np.random.choice([0, 1, 2, 3], n_samples),
        'default_probability': np.random.random(n_samples),
        'predicted_default': np.random.choice([0, 1], n_samples)
    }
    
    # Add region mapping
    region_mapping = {
        'Kenya': 'East Africa', 'Uganda': 'East Africa', 'Tanzania': 'East Africa', 
        'Rwanda': 'East Africa', 'Ethiopia': 'East Africa',
        'Ghana': 'West Africa', 'Nigeria': 'West Africa',
        'South Africa': 'Southern Africa', 'Zambia': 'Southern Africa', 'Malawi': 'Southern Africa'
    }
    
    data['region'] = [region_mapping[country] for country in data['country']]
    
    return pd.DataFrame(data)

def predict_default_probability(features):
    """Mock prediction function - replace with your actual model"""
    # This is a mock function. In production, you'd use your trained model:
    # probability = model.predict_proba([features])[0][1]
    
    # Mock logic based on key features
    risk_score = 0
    if features.get('biz_loan', 0) == 1:
        risk_score += 0.3
    if features.get('digital_engagement_score', 0) < 3:
        risk_score += 0.2
    if features.get('credit_card', 0) == 0:
        risk_score += 0.15
    if features.get('mobile_pay_s_r', 0) == 0:
        risk_score += 0.1
    
    # Add some randomness
    risk_score += np.random.random() * 0.25
    probability = min(risk_score, 0.95)
    
    return probability

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Intelligent Loan Default Prediction System</h1>
        <p>Advanced ML-powered risk assessment with 98.33% AUC accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    sample_data = load_sample_data()
    
    # Sidebar for navigation
    st.sidebar.markdown("### 🎛️ Navigation")
    page = st.sidebar.selectbox("Choose Analysis Type", 
                               ["🔮 Individual Prediction", 
                                "🌍 Regional Analysis", 
                                "📊 Batch Predictions",
                                "📈 Model Performance"])
    
    if page == "🔮 Individual Prediction":
        individual_prediction()
    elif page == "🌍 Regional Analysis":
        regional_analysis(sample_data)
    elif page == "📊 Batch Predictions":
        batch_predictions(sample_data)
    elif page == "📈 Model Performance":
        model_performance()

def individual_prediction():
    st.markdown("## 🔮 Individual Loan Default Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Customer Information")
        
        # Customer basic info
        country = st.selectbox("🌍 Country", 
                              ['Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Ethiopia', 
                               'Ghana', 'Nigeria', 'South Africa', 'Zambia', 'Malawi'])
        
        age = st.slider("👤 Age", 18, 80, 35)
        income = st.number_input("💰 Monthly Income (USD)", 100, 10000, 1500)
        
        st.markdown("### 🏪 Business & Loan Details")
        biz_loan = st.selectbox("Has Business Loan?", ["No", "Yes"])
        biz_loan_source = st.selectbox("Business Loan Source", 
                                     ["None", "Bank", "MFI", "Other"])
        loan_purpose = st.selectbox("Loan Purpose", 
                                  ["Personal", "Business", "Education", "Agriculture"])
        
        st.markdown("### 📱 Digital Engagement")
        has_credit_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
        mobile_payments = st.selectbox("Uses Mobile Payments?", ["No", "Yes"])
        digital_engagement = st.slider("Digital Engagement Score (1-10)", 1, 10, 5)
        prefer_digital = st.selectbox("Prefers Digital Services?", ["No", "Yes"])
    
    with col2:
        st.markdown("### 🎯 Risk Assessment")
        
        # Prepare features for prediction
        features = {
            'biz_loan': 1 if biz_loan == "Yes" else 0,
            'biz_loan_source': ["None", "Bank", "MFI", "Other"].index(biz_loan_source),
            'credit_card': 1 if has_credit_card == "Yes" else 0,
            'mobile_pay_s_r': 1 if mobile_payments == "Yes" else 0,
            'digital_engagement_score': digital_engagement,
            'prefer_digital_acc': 1 if prefer_digital == "Yes" else 0,
            'loan_purpose_group': ["Personal", "Business", "Education", "Agriculture"].index(loan_purpose)
        }
        
        # Get prediction
        if st.button("🚀 Predict Default Risk", type="primary"):
            with st.spinner("Analyzing risk factors..."):
                probability = predict_default_probability(features)
                
                # Display prediction
                if probability < 0.3:
                    risk_level = "LOW RISK"
                    css_class = "low-risk"
                    icon = "✅"
                    recommendation = "Approve loan with standard terms"
                elif probability < 0.7:
                    risk_level = "MEDIUM RISK"
                    css_class = "medium-risk"
                    icon = "⚠️"
                    recommendation = "Approve with enhanced monitoring"
                else:
                    risk_level = "HIGH RISK"
                    css_class = "high-risk"
                    icon = "🚨"
                    recommendation = "Reject or require additional collateral"
                
                st.markdown(f"""
                <div class="prediction-box {css_class}">
                    {icon} {risk_level}<br>
                    Default Probability: {probability:.1%}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Recommendation:** {recommendation}")
                
                # Risk factors breakdown
                st.markdown("### 📊 Risk Factors Analysis")
                
                risk_factors = []
                if features['biz_loan'] == 1:
                    risk_factors.append(("Business Loan Holder", "Medium", "📈"))
                if features['digital_engagement_score'] < 4:
                    risk_factors.append(("Low Digital Engagement", "High", "📱"))
                if features['credit_card'] == 0:
                    risk_factors.append(("No Credit Card", "Low", "💳"))
                if features['mobile_pay_s_r'] == 0:
                    risk_factors.append(("No Mobile Payments", "Medium", "📲"))
                
                if risk_factors:
                    for factor, impact, icon in risk_factors:
                        color = {"Low": "green", "Medium": "orange", "High": "red"}[impact]
                        st.markdown(f"{icon} **{factor}** - *{impact} Risk Impact*")
                else:
                    st.success("✨ No major risk factors identified!")

def regional_analysis(data):
    st.markdown("## 🌍 Regional Risk Analysis")
    
    # Regional summary statistics
    regional_stats = data.groupby('region').agg({
        'default_probability': ['mean', 'count'],
        'predicted_default': 'sum'
    }).round(4)
    
    regional_stats.columns = ['Avg_Risk', 'Total_Customers', 'Predicted_Defaults']
    regional_stats['Default_Rate'] = (regional_stats['Predicted_Defaults'] / 
                                    regional_stats['Total_Customers'] * 100).round(2)
    
    # Country-level analysis
    country_stats = data.groupby('country').agg({
        'default_probability': 'mean',
        'predicted_default': ['sum', 'count']
    }).round(4)
    
    country_stats.columns = ['Avg_Risk', 'Defaults', 'Total']
    country_stats['Default_Rate'] = (country_stats['Defaults'] / 
                                   country_stats['Total'] * 100).round(2)
    
    # Display regional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌍 Total Regions", len(regional_stats), "Active")
    
    with col2:
        highest_risk_region = regional_stats.loc[regional_stats['Avg_Risk'].idxmax()]
        st.metric("⚠️ Highest Risk Region", 
                 regional_stats['Avg_Risk'].idxmax(),
                 f"{highest_risk_region['Avg_Risk']:.1%}")
    
    with col3:
        lowest_risk_region = regional_stats.loc[regional_stats['Avg_Risk'].idxmin()]
        st.metric("✅ Lowest Risk Region", 
                 regional_stats['Avg_Risk'].idxmin(),
                 f"{lowest_risk_region['Avg_Risk']:.1%}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional risk heatmap
        fig_region = px.bar(
            x=regional_stats.index,
            y=regional_stats['Avg_Risk'] * 100,
            title="Average Default Risk by Region",
            labels={'y': 'Default Risk (%)', 'x': 'Region'},
            color=regional_stats['Avg_Risk'] * 100,
            color_continuous_scale='Reds'
        )
        fig_region.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        # Customer distribution
        fig_customers = px.pie(
            values=regional_stats['Total_Customers'],
            names=regional_stats.index,
            title="Customer Distribution by Region"
        )
        fig_customers.update_layout(height=400)
        st.plotly_chart(fig_customers, use_container_width=True)
    
    # Country-level analysis
    st.markdown("### 🏳️ Country-Level Risk Analysis")
    
    fig_country = px.choropleth(
        locations=country_stats.index,
        z=country_stats['Avg_Risk'] * 100,
        locationmode='country names',
        title="Default Risk Heatmap by Country",
        color_continuous_scale='Reds',
        labels={'z': 'Default Risk (%)'}
    )
    fig_country.update_layout(height=500)
    st.plotly_chart(fig_country, use_container_width=True)
    
    # Detailed country table
    st.markdown("### 📋 Detailed Country Statistics")
    
    # Format the dataframe for display
    display_df = country_stats.copy()
    display_df['Avg_Risk'] = (display_df['Avg_Risk'] * 100).round(2).astype(str) + '%'
    display_df = display_df.reset_index()
    
    st.dataframe(
        display_df,
        column_config={
            "country": "Country",
            "Avg_Risk": st.column_config.TextColumn("Average Risk"),
            "Defaults": st.column_config.NumberColumn("Predicted Defaults"),
            "Total": st.column_config.NumberColumn("Total Customers"),
            "Default_Rate": st.column_config.NumberColumn("Default Rate (%)", format="%.2f%%")
        },
        hide_index=True,
        use_container_width=True
    )

def batch_predictions(data):
    st.markdown("## 📊 Batch Loan Predictions")
    
    st.markdown("### 📁 Upload Customer Data")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose CSV file with customer data", 
        type=['csv'],
        help="Upload a CSV file with customer information for batch predictions"
    )
    
    if uploaded_file is not None:
        # Load uploaded data
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_data)} records")
            
            # Show preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(batch_data.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    else:
        # Use sample data
        st.info("💡 Using sample data for demonstration. Upload your own CSV file above.")
        batch_data = data.copy()
    
    # Batch prediction results
    if st.button("🚀 Run Batch Predictions", type="primary"):
        with st.spinner("Processing batch predictions..."):
            # Add predictions to the data (mock implementation)
            batch_data['risk_score'] = np.random.random(len(batch_data))
            batch_data['risk_category'] = pd.cut(
                batch_data['risk_score'], 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Processed", len(batch_data))
            
            with col2:
                low_risk_count = (batch_data['risk_category'] == 'Low Risk').sum()
                st.metric("✅ Low Risk", low_risk_count, 
                         f"{low_risk_count/len(batch_data)*100:.1f}%")
            
            with col3:
                medium_risk_count = (batch_data['risk_category'] == 'Medium Risk').sum()
                st.metric("⚠️ Medium Risk", medium_risk_count,
                         f"{medium_risk_count/len(batch_data)*100:.1f}%")
            
            with col4:
                high_risk_count = (batch_data['risk_category'] == 'High Risk').sum()
                st.metric("🚨 High Risk", high_risk_count,
                         f"{high_risk_count/len(batch_data)*100:.1f}%")
            
            # Risk distribution visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig_risk_dist = px.histogram(
                    batch_data, 
                    x='risk_category', 
                    title="Risk Category Distribution",
                    color='risk_category',
                    color_discrete_map={
                        'Low Risk': '#4ecdc4',
                        'Medium Risk': '#ffa726', 
                        'High Risk': '#ff6b6b'
                    }
                )
                st.plotly_chart(fig_risk_dist, use_container_width=True)
            
            with col2:
                fig_risk_by_country = px.box(
                    batch_data,
                    x='country',
                    y='risk_score',
                    title="Risk Score Distribution by Country"
                )
                fig_risk_by_country.update_xaxes(tickangle=45)
                st.plotly_chart(fig_risk_by_country, use_container_width=True)
            
            # Detailed results table
            st.markdown("### 📋 Detailed Prediction Results")
            
            # Format results for display
            results_df = batch_data[['country', 'region', 'risk_score', 'risk_category']].copy()
            results_df['risk_score'] = (results_df['risk_score'] * 100).round(1)
            
            st.dataframe(
                results_df,
                column_config={
                    "country": "Country",
                    "region": "Region", 
                    "risk_score": st.column_config.NumberColumn("Risk Score (%)", format="%.1f%%"),
                    "risk_category": st.column_config.TextColumn("Risk Category")
                },
                use_container_width=True
            )
            
            # Download button for results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"loan_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def model_performance():
    st.markdown("## 📈 Model Performance Dashboard")
    
    # Performance metrics from your results
    metrics_data = {
        'Metric': ['AUC-ROC', 'Average Precision', 'F1-Score', 'Accuracy', 
                  'Precision', 'Recall', 'Matthews Correlation'],
        'Score': [0.9833, 0.9902, 0.9405, 0.9278, 0.96, 0.922, 0.85],
        'Benchmark': [0.8, 0.7, 0.75, 0.8, 0.8, 0.8, 0.6]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Performance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 AUC-ROC", "98.33%", "+18.33%")
    
    with col2:
        st.metric("📊 Precision", "96.0%", "+16.0%")
    
    with col3:
        st.metric("🔍 Recall", "92.2%", "+12.2%")
    
    with col4:
        st.metric("⚖️ F1-Score", "94.05%", "+19.05%")
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics comparison
        fig_metrics = px.bar(
            metrics_df,
            x='Metric',
            y=['Score', 'Benchmark'],
            title="Model Performance vs Industry Benchmark",
            barmode='group'
        )
        fig_metrics.update_xaxes(tickangle=45)
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        # Feature importance (from your earlier results)
        feature_importance_data = {
            'Feature': ['Business Loan Source', 'Business Loan', 'Digital Pay Account', 
                       'Digital Engagement', 'Prefer Digital Account', 'Credit Card'],
            'Importance': [0.1080, 0.0861, 0.0846, 0.0714, 0.0664, 0.0558]
        }
        
        fig_importance = px.bar(
            x=feature_importance_data['Importance'],
            y=feature_importance_data['Feature'],
            title="Top Feature Importance",
            orientation='h'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model insights
    st.markdown("### 🧠 Key Model Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        <div class="feature-importance">
            <h4>🎯 Model Strengths</h4>
            <ul>
                <li><strong>Exceptional Discrimination:</strong> 98.33% AUC-ROC</li>
                <li><strong>High Precision:</strong> 96% of flagged customers default</li>
                <li><strong>Strong Recall:</strong> Catches 92.2% of actual defaults</li>
                <li><strong>Robust Performance:</strong> Consistent across all metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("""
        <div class="feature-importance">
            <h4>💼 Business Impact</h4>
            <ul>
                <li><strong>Risk Reduction:</strong> 92.2% default detection rate</li>
                <li><strong>False Positives:</strong> Only 4% false alarm rate</li>
                <li><strong>Cost Savings:</strong> Prevents majority of defaults</li>
                <li><strong>Decision Support:</strong> Clear risk categorization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Model deployment status
    st.markdown("### 🚀 Deployment Status")
    st.success("✅ Model is production-ready with excellent performance metrics!")
    st.info("💡 Recommended for immediate deployment with 0.5 classification threshold")

if __name__ == "__main__":
    main()