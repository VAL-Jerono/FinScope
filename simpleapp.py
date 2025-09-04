import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FinScope AI - Financial Inclusion Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-high {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .prediction-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .prediction-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .stats-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Actual feature data from your results
FEATURES_DATA = {
    'biz_loan_source': 0.1683,
    'biz_loan': 0.1230,
    'emergency_funds': 0.0980,
    'digital_pay': 0.0636,
    'digital_pay_acc': 0.0597,
    'loan_purpose_group': 0.0409,
    'mobile_pay_s_r': 0.0404,
    'prefer_digital_fin': 0.0392,
    'digital_payment_other': 0.0390,
    'govt_payment_recv': 0.0378,
    'saved_any': 0.0351,
    'mobile_payment_bill': 0.0273,
    'borrowed_any': 0.0251,
    'saved_for_purchase': 0.0250,
    'loan_purpose': 0.0234
}

# Actual model performance metrics from your results
MODEL_METRICS = {
    'accuracy': 0.8962,
    'balanced_accuracy': 0.8883,
    'precision': 0.9103,
    'recall': 0.9225,
    'f1_score': 0.9163,
    'auc_roc': 0.9607,
    'auc_pr': 0.9743,
    'matthews_corr': 0.7798,
    'log_loss': 0.2887,
    'specificity': 0.8541
}

# Actual confusion matrix data
CONFUSION_MATRIX = {
    'true_negatives': 556,
    'false_positives': 95,
    'false_negatives': 81,
    'true_positives': 964
}

# Actual confidence analysis data
CONFIDENCE_DATA = [
    {'threshold': 0.5, 'accuracy': 0.8962, 'coverage': 100.0},
    {'threshold': 0.6, 'accuracy': 0.9168, 'coverage': 94.2},
    {'threshold': 0.7, 'accuracy': 0.9705, 'coverage': 77.9},
    {'threshold': 0.8, 'accuracy': 0.9770, 'coverage': 69.1},
    {'threshold': 0.9, 'accuracy': 0.9844, 'coverage': 56.8},
    {'threshold': 1.0, 'accuracy': 0.9958, 'coverage': 28.4}
]

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'biz_loan_source': 'Has access to business loan sources',
    'biz_loan': 'Currently has or had business loan',
    'emergency_funds': 'Has emergency funds available',
    'digital_pay': 'Uses digital payment methods',
    'digital_pay_acc': 'Has digital payment account',
    'loan_purpose_group': 'Loan purpose category',
    'mobile_pay_s_r': 'Mobile payment send/receive capability',
    'prefer_digital_fin': 'Prefers digital financial services',
    'digital_payment_other': 'Uses other digital payment methods',
    'govt_payment_recv': 'Receives government payments',
    'saved_any': 'Has any form of savings',
    'mobile_payment_bill': 'Pays bills via mobile payment',
    'borrowed_any': 'Has borrowed money from any source',
    'saved_for_purchase': 'Saves money for specific purchases',
    'loan_purpose': 'Specific loan purpose identified'
}

def load_model():
    """Load the actual trained Random Forest model"""
    try:
        with open('finance_app.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, True
    except FileNotFoundError:
        st.error("⚠️ Model file 'finance_app.pkl' not found. Please upload the trained model.")
        return None, False

def create_feature_importance_chart():
    """Create feature importance visualization using actual data"""
    df = pd.DataFrame([
        {'Feature': feature, 'Importance': importance, 'Description': FEATURE_DESCRIPTIONS[feature]}
        for feature, importance in FEATURES_DATA.items()
    ]).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Top 15 Most Important Features for Financial Inclusion Prediction',
        hover_data=['Description'],
        color='Importance',
        color_continuous_scale='viridis',
        text='Importance'
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        height=600,
        xaxis_title="Feature Importance",
        yaxis_title="Features",
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig

def create_model_performance_chart():
    """Create model comparison chart using actual results"""
    models_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM'],
        'Accuracy': [0.8149, 0.8962, 0.8762, 0.8656],
        'AUC': [0.9012, 0.9607, 0.9497, 0.9310]
    }
    
    df = pd.DataFrame(models_data)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Accuracy Comparison', 'Model AUC Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy chart
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['Accuracy'], name='Accuracy', 
               marker_color=['lightblue', 'darkblue', 'lightgreen', 'orange']),
        row=1, col=1
    )
    
    # AUC chart
    fig.add_trace(
        go.Bar(x=df['Model'], y=df['AUC'], name='AUC',
               marker_color=['lightblue', 'darkblue', 'lightgreen', 'orange']),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Model Performance Comparison")
    return fig

def create_confidence_analysis_chart():
    """Create confidence analysis chart using actual data"""
    df = pd.DataFrame(CONFIDENCE_DATA)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Accuracy vs Confidence Threshold', 'Coverage vs Confidence Threshold'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=df['threshold'], y=df['accuracy'], mode='lines+markers', 
                  name='Accuracy', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['threshold'], y=df['coverage']/100, mode='lines+markers', 
                  name='Coverage', line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Model Confidence Analysis")
    fig.update_xaxes(title_text="Confidence Threshold", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Threshold", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Coverage", row=1, col=2)
    
    return fig

def create_confusion_matrix_chart():
    """Create confusion matrix visualization"""
    cm_data = [
        [CONFUSION_MATRIX['true_negatives'], CONFUSION_MATRIX['false_positives']],
        [CONFUSION_MATRIX['false_negatives'], CONFUSION_MATRIX['true_positives']]
    ]
    
    fig = px.imshow(
        cm_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        title='Confusion Matrix - Random Forest Model',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['No Account (0)', 'Has Account (1)'],
        y=['No Account (0)', 'Has Account (1)']
    )
    
    fig.update_layout(height=400, width=500)
    return fig

def get_prediction_interpretation(probability):
    """Interpret prediction results based on actual confidence thresholds"""
    if probability >= 0.9:
        return "prediction-high", "🟢 VERY HIGH INCLUSION LIKELIHOOD", f"98.4% model accuracy at this confidence level. This individual has a {probability:.1%} chance of having a financial account."
    elif probability >= 0.8:
        return "prediction-high", "🟢 HIGH INCLUSION LIKELIHOOD", f"97.7% model accuracy at this confidence level. This individual has a {probability:.1%} chance of having a financial account."
    elif probability >= 0.7:
        return "prediction-medium", "🟡 MODERATE INCLUSION LIKELIHOOD", f"97.1% model accuracy at this confidence level. This individual has a {probability:.1%} chance of having a financial account."
    elif probability >= 0.6:
        return "prediction-medium", "🟡 MODERATE EXCLUSION RISK", f"91.7% model accuracy at this confidence level. This individual has a {probability:.1%} chance of having a financial account."
    else:
        return "prediction-low", "🔴 HIGH EXCLUSION RISK", f"89.6% model accuracy at this confidence level. This individual has a {probability:.1%} chance of having a financial account."

def main():
    # Header
    st.markdown('<p class="main-header">🏦 FinScope AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predicting Financial Inclusion with Machine Learning</p>', unsafe_allow_html=True)
    
    # Model loading status
    model, model_loaded = load_model()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction Tool", "📊 Model Performance", "🔍 Feature Analysis", "📈 Model Insights"])
    
    with tab1:
        st.header("Financial Inclusion Prediction")
        
        if not model_loaded:
            st.warning("⚠️ Upload your trained model file (finance_app.pkl) to use the prediction functionality.")
            uploaded_file = st.file_uploader("Upload Model File", type=['pkl'])
            if uploaded_file is not None:
                try:
                    model = pickle.load(uploaded_file)
                    model_loaded = True
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Individual Profile Input")
            
            # Create input form based on actual features
            with st.form("prediction_form"):
                inputs = {}
                
                # Business & Credit features
                st.markdown("**💼 Business & Credit**")
                inputs['biz_loan_source'] = st.selectbox("Has access to business loan sources", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['biz_loan'] = st.selectbox("Currently has business loan", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['credit_card'] = st.selectbox("Owns credit card", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['borrowed_any'] = st.selectbox("Has borrowed money", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                
                # Digital Payments features
                st.markdown("**📱 Digital Payments**")
                inputs['digital_pay'] = st.selectbox("Uses digital payment methods", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['digital_pay_acc'] = st.selectbox("Has digital payment account", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['mobile_payment'] = st.selectbox("Uses mobile payment", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['mobile_pay_s_r'] = st.selectbox("Mobile payment send/receive", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['mobile_payment_bill'] = st.selectbox("Pays bills via mobile", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                
                # Savings features
                st.markdown("**💰 Savings Behavior**")
                inputs['saved_any'] = st.selectbox("Has any form of savings", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['emergency_funds'] = st.selectbox("Has emergency funds", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['saved_for_purchase'] = st.selectbox("Saves for purchases", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['saved_old_age'] = st.selectbox("Saves for old age", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                
                # Government & Preferences
                st.markdown("**🏛️ Government & Preferences**")
                inputs['govt_payment_recv'] = st.selectbox("Receives government payments", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                inputs['prefer_digital_fin'] = st.selectbox("Prefers digital financial services", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                
                submit_button = st.form_submit_button("🔮 Predict Financial Inclusion")
        
        with col2:
            if submit_button and model_loaded:
                # Create prediction
                input_array = np.array([[inputs[feature] if feature in inputs else 0 for feature in FEATURES_DATA.keys()]])
                
                try:
                    prediction_proba = model.predict_proba(input_array)[0]
                    inclusion_probability = prediction_proba[1]  # Probability of having account
                    
                    # Display prediction
                    st.subheader("🎯 Prediction Results")
                    
                    class_name, title, description = get_prediction_interpretation(inclusion_probability)
                    
                    st.markdown(f'<div class="{class_name}"><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)
                    
                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = inclusion_probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Financial Inclusion Probability"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Feature importance for this prediction
                    st.subheader("🔍 Key Factors")
                    feature_contrib = []
                    for feature, importance in FEATURES_DATA.items():
                        if feature in inputs:
                            contrib = inputs[feature] * importance
                            feature_contrib.append((feature, contrib, FEATURE_DESCRIPTIONS.get(feature, feature)))
                    
                    feature_contrib.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (feature, contrib, desc) in enumerate(feature_contrib[:5]):
                        if contrib > 0:
                            st.write(f"**{i+1}. {desc}:** Contributing +{contrib:.3f} to inclusion likelihood")
                
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
    
    with tab2:
        st.header("🎯 Random Forest Model Performance")
        
        # Model metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{MODEL_METRICS['accuracy']:.2%}")
            st.metric("Precision", f"{MODEL_METRICS['precision']:.2%}")
        
        with col2:
            st.metric("AUC-ROC", f"{MODEL_METRICS['auc_roc']:.4f}")
            st.metric("Recall", f"{MODEL_METRICS['recall']:.2%}")
        
        with col3:
            st.metric("F1-Score", f"{MODEL_METRICS['f1_score']:.4f}")
            st.metric("AUC-PR", f"{MODEL_METRICS['auc_pr']:.4f}")
        
        with col4:
            st.metric("Specificity", f"{MODEL_METRICS['specificity']:.2%}")
            st.metric("Matthews Corr", f"{MODEL_METRICS['matthews_corr']:.4f}")
        
        # Model comparison
        st.plotly_chart(create_model_performance_chart(), use_container_width=True)
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_confusion_matrix_chart(), use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="stats-box">
            <h4>📊 Dataset Statistics</h4>
            <ul>
            <li><strong>Total Samples:</strong> 8,476</li>
            <li><strong>Features:</strong> 24</li>
            <li><strong>Training Set:</strong> 6,780 samples</li>
            <li><strong>Test Set:</strong> 1,696 samples</li>
            <li><strong>Target Distribution:</strong> 5,223 included, 3,253 excluded</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.header("🔍 Feature Importance Analysis")
        
        st.plotly_chart(create_feature_importance_chart(), use_container_width=True)
        
        # Feature importance table
        st.subheader("📋 Feature Importance Details")
        df_features = pd.DataFrame([
            {'Rank': i+1, 'Feature': feature, 'Importance': f"{importance:.4f}", 
             'Description': FEATURE_DESCRIPTIONS[feature]}
            for i, (feature, importance) in enumerate(FEATURES_DATA.items())
        ])
        st.dataframe(df_features, use_container_width=True)
        
        # Key insights
        st.markdown("""
        <div class="stats-box">
        <h4>🔑 Key Insights</h4>
        <ul>
        <li><strong>Business Access:</strong> Having access to business loan sources is the strongest predictor (16.83%)</li>
        <li><strong>Credit History:</strong> Current business loans significantly impact inclusion (12.30%)</li>
        <li><strong>Financial Resilience:</strong> Emergency funds availability is crucial (9.80%)</li>
        <li><strong>Digital Infrastructure:</strong> Digital payment usage shows strong correlation with inclusion</li>
        <li><strong>Entrepreneurship:</strong> Business-related features dominate the top predictors</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.header("📈 Model Confidence & Insights")
        
        # Confidence analysis
        st.plotly_chart(create_confidence_analysis_chart(), use_container_width=True)
        
        # Confidence interpretation
        st.subheader("🎯 Confidence Levels Guide")
        
        confidence_df = pd.DataFrame(CONFIDENCE_DATA)
        confidence_df['Accuracy %'] = (confidence_df['accuracy'] * 100).round(2)
        confidence_df['Coverage %'] = confidence_df['coverage'].round(1)
        confidence_df['Threshold'] = confidence_df['threshold']
        
        st.dataframe(
            confidence_df[['Threshold', 'Accuracy %', 'Coverage %']], 
            use_container_width=True
        )
        
        # Cross-validation results
        st.markdown("""
        <div class="stats-box">
        <h4>🔄 Cross-Validation Performance</h4>
        <ul>
        <li><strong>CV AUC Scores:</strong> 0.9494, 0.9403, 0.9542</li>
        <li><strong>CV AUC Mean:</strong> 0.9479</li>
        <li><strong>CV AUC Std:</strong> 0.0058</li>
        <li><strong>Model Stability:</strong> Excellent consistency across folds</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Deployment info
        st.markdown("""
        <div class="stats-box">
        <h4>🚀 Deployment Information</h4>
        <p>This model was trained on the <strong>Global Findex 2024 dataset</strong> to address financial exclusion. 
        The Random Forest classifier achieved superior performance compared to Logistic Regression, Gradient Boosting, 
        and SVM models, making it ideal for:</p>
        <ul>
        <li><strong>Policy targeting:</strong> Identify high-risk populations for financial exclusion</li>
        <li><strong>Resource allocation:</strong> Optimize intervention strategies</li>
        <li><strong>Impact assessment:</strong> Monitor program effectiveness</li>
        <li><strong>Risk assessment:</strong> Evaluate individual inclusion likelihood</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Model Information")
        
        st.markdown(f"""
        **Model Type:** Random Forest  
        **Accuracy:** {MODEL_METRICS['accuracy']:.2%}  
        **AUC-ROC:** {MODEL_METRICS['auc_roc']:.4f}  
        **Training Samples:** 6,780  
        **Test Samples:** 1,696  
        """)
        
        st.header("📊 Quick Stats")
        st.markdown(f"""
        **True Positives:** {CONFUSION_MATRIX['true_positives']}  
        **True Negatives:** {CONFUSION_MATRIX['true_negatives']}  
        **False Positives:** {CONFUSION_MATRIX['false_positives']}  
        **False Negatives:** {CONFUSION_MATRIX['false_negatives']}  
        """)
        
        st.header("ℹ️ About")
        st.markdown("""
        This tool predicts financial inclusion likelihood using machine learning trained on Global Findex 2024 data.
        
        **Usage:**
        1. Input individual characteristics
        2. Get inclusion probability
        3. Review key contributing factors
        4. Plan targeted interventions
        
        **Target:** 1.4 billion unbanked adults globally
        """)

if __name__ == "__main__":
    main()