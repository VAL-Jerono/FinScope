# train_and_save_model.py
"""
Script to train and save the optimized top 10 features model
Run this script to generate the optimized_financial_inclusion_model.pkl file
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class OptimizedModelTrainer:
    """
    Train and save optimized model using top 10 features from your analysis
    """
    
    def __init__(self):
        # Top 10 features based on your pipeline results
        self.top_features = [
            'biz_loan_source',           # Weight: 321
            'saved_any',                 # Weight: 313
            'borrowed_any',              # Weight: 285
            'income_digital_interaction', # Weight: 260
            'biz_loan',                  # Weight: 254
            'demo_subgroup',             # Weight: 214
            'saved_for_purchase',        # Weight: 198
            'region_cleaned',            # Weight: 182
            'financial_activity_score',  # Weight: 167
            'saved_no_purpose'           # Weight: 138
        ]
        
        self.feature_weights = [321, 313, 285, 260, 254, 214, 198, 182, 167, 138]
        
        # Regional baselines from your analysis
        self.regional_baselines = {
            0: 0.882,  # High income
            1: 0.564,  # East Asia & Pacific
            2: 0.557,  # Europe & Central Asia
            3: 0.440,  # South Asia
            4: 0.496,  # Latin America & Caribbean
            5: 0.394,  # Sub-Saharan Africa
            6: 0.378   # Middle East & North Africa
        }
        
        # Demo subgroup mappings
        self.demo_mappings = {
            'all': 0, 'male': 1, 'female': 2, 'urban': 3, 'rural': 4,
            'young': 5, 'adult': 6, 'ages 15-24': 7, 'men': 8, 'women': 9,
            'in laborforce': 10, 'out of laborforce': 11, 'richest 60%': 12,
            'poorest 40%': 13, 'secondary edu or more': 14, 'prim edu or less': 15
        }
    
    def prepare_data(self, df, target_col='Has_Account'):
        """
        Prepare data with only top 10 features
        """
        print("Preparing Data with Top 10 Features...")
        print("=" * 50)
        
        # Check if all required features exist
        missing_features = [f for f in self.top_features if f not in df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            # Create dummy features if needed (for demonstration)
            for feature in missing_features:
                if 'score' in feature:
                    # Create composite score from available features
                    df[feature] = (df.get('saved_any', 0) + df.get('borrowed_any', 0)) / 2
                elif 'interaction' in feature:
                    # Create interaction feature
                    df[feature] = df.get('saved_any', 0) * 0.5
                else:
                    df[feature] = 0.3  # Default value
        
        # Select features and target
        X = df[self.top_features].copy()
        y = df[target_col].copy()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target range: {y.min():.3f} to {y.max():.3f}")
        print(f"Target mean: {y.mean():.3f}")
        
        return X, y
    
    def encode_categorical_features(self, X_train, X_test):
        """
        Encode categorical features and store encoders
        """
        encoders = {}
        
        # Handle demo_subgroup
        if 'demo_subgroup' in X_train.columns:
            le_demo = LabelEncoder()
            X_train['demo_subgroup'] = le_demo.fit_transform(X_train['demo_subgroup'].astype(str))
            X_test['demo_subgroup'] = le_demo.transform(X_test['demo_subgroup'].astype(str))
            encoders['demo_subgroup'] = le_demo
            print(f"Encoded demo_subgroup: {len(le_demo.classes_)} categories")
        
        # Handle region_cleaned (should already be numeric, but ensure consistency)
        if 'region_cleaned' in X_train.columns:
            X_train['region_cleaned'] = X_train['region_cleaned'].astype(int)
            X_test['region_cleaned'] = X_test['region_cleaned'].astype(int)
        
        return X_train, X_test, encoders
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple models and select the best one
        """
        print("\nTraining Models...")
        print("=" * 30)
        
        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                random_state=42,
                verbose=-1,
                objective='regression',
                num_leaves=31,
                learning_rate=0.1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                eval_metric='rmse',
                learning_rate=0.1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
        }
        
        results = []
        trained_models = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name == 'LinearRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Clip predictions
            y_pred = np.clip(y_pred, 0, 1)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # For classification metrics, convert to binary
            y_test_bin = (y_test >= 0.5).astype(int)
            y_pred_bin = (y_pred >= 0.5).astype(int)
            accuracy = accuracy_score(y_test_bin, y_pred_bin)
            
            results.append({
                'model': name,
                'r2_score': r2,
                'rmse': rmse,
                'accuracy': accuracy
            })
            
            trained_models[name] = model
            print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, Accuracy = {accuracy:.4f}")
        
        # Select best model (by R²)
        best_result = max(results, key=lambda x: x['r2_score'])
        best_model = trained_models[best_result['model']]
        
        print(f"\nBest Model: {best_result['model']} (R² = {best_result['r2_score']:.4f})")
        
        return best_model, best_result['model'], results, scaler
    
    def save_production_model(self, model, model_name, encoders, scaler, results):
        """
        Save complete production-ready model package
        """
        print(f"\nSaving Production Model...")
        print("=" * 30)
        
        # Create comprehensive model package
        production_package = {
            'model': model,
            'model_name': model_name,
            'scalers': {'standard': scaler},
            'encoders': encoders,
            'top_features': self.top_features,
            'feature_weights': dict(zip(self.top_features, self.feature_weights)),
            'regional_baselines': self.regional_baselines,
            'demo_mappings': self.demo_mappings,
            'metadata': {
                'training_date': datetime.now().isoformat(),
                'best_model': model_name,
                'model_performance': results,
                'features_count': len(self.top_features),
                'version': '1.0_optimized'
            }
        }
        
        # Save to file
        filepath = 'optimized_financial_inclusion_model.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(production_package, f)
        
        file_size = os.path.getsize(filepath) / 1024  # KB
        
        print(f"✅ Model saved successfully!")
        print(f"   File: {filepath}")
        print(f"   Size: {file_size:.1f} KB")
        print(f"   Model: {model_name}")
        print(f"   Features: {len(self.top_features)}")
        
        return filepath
    
    def run_complete_training(self, df, target_col='Has_Account'):
        """
        Complete training pipeline
        """
        print("OPTIMIZED FINANCIAL INCLUSION MODEL TRAINING")
        print("=" * 60)
        
        # Step 1: Prepare data
        X, y = self.prepare_data(df, target_col)
        
        # Step 2: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=(y >= 0.5).astype(int)
        )
        
        print(f"\nTrain-test split:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Testing: {X_test.shape[0]} samples")
        
        # Step 3: Encode categorical features
        X_train, X_test, encoders = self.encode_categorical_features(X_train, X_test)
        
        # Step 4: Train models
        best_model, best_model_name, results, scaler = self.train_models(
            X_train, y_train, X_test, y_test
        )
        
        # Step 5: Save production model
        filepath = self.save_production_model(
            best_model, best_model_name, encoders, scaler, results
        )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Production model ready at: {filepath}")
        
        return filepath, best_model, results

def test_saved_model(model_path='optimized_financial_inclusion_model.pkl'):
    """
    Test the saved model with sample data
    """
    print(f"\nTesting saved model...")
    
    # Load model
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    # Sample test data
    test_individual = {
        'biz_loan_source': 0.3,
        'saved_any': 0.6,
        'borrowed_any': 0.4,
        'income_digital_interaction': 0.5,
        'biz_loan': 0.1,
        'demo_subgroup': 'adult',
        'saved_for_purchase': 0.2,
        'region_cleaned': 1,  # East Asia & Pacific
        'financial_activity_score': 0.4,
        'saved_no_purpose': 0.1
    }
    
    # Make prediction
    prediction = make_test_prediction(model_package, test_individual)
    
    print(f"✅ Test successful!")
    print(f"   Sample prediction: {prediction:.1%}")
    print(f"   Model: {model_package['model_name']}")
    
    return prediction

def make_test_prediction(model_package, input_data):
    """
    Make a test prediction using the model package
    """
    # Extract components
    model = model_package['model']
    model_name = model_package['model_name']
    encoders = model_package.get('encoders', {})
    scalers = model_package.get('scalers', {})
    demo_mappings = model_package.get('demo_mappings', {})
    top_features = model_package['top_features']
    
    # Prepare features
    feature_values = []
    for feature in top_features:
        value = input_data[feature]
        
        # Handle categorical encoding
        if feature == 'demo_subgroup' and isinstance(value, str):
            value = demo_mappings.get(value, 0)
        elif feature in encoders:
            try:
                value = encoders[feature].transform([str(value)])[0]
            except ValueError:
                value = 0
        
        feature_values.append(value)
    
    feature_array = np.array([feature_values])
    
    # Apply scaling if needed
    if model_name == 'LinearRegression' and 'standard' in scalers:
        feature_array = scalers['standard'].transform(feature_array)
    
    # Predict
    prediction = model.predict(feature_array)[0]
    return np.clip(prediction, 0, 1)

if __name__ == "__main__":
    # Example usage
    
    # You would load your actual dataset here
    # For demonstration, create sample data structure
    print("Creating sample dataset for demonstration...")
    
    # Create synthetic sample data that matches your pipeline structure
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'biz_loan_source': np.random.beta(2, 5, n_samples),
        'saved_any': np.random.beta(3, 2, n_samples),
        'borrowed_any': np.random.beta(2, 3, n_samples),
        'income_digital_interaction': np.random.beta(2, 3, n_samples),
        'biz_loan': np.random.beta(1, 4, n_samples),
        'demo_subgroup': np.random.choice(['adult', 'young', 'urban', 'rural', 'richest 60%', 'poorest 40%'], n_samples),
        'saved_for_purchase': np.random.beta(2, 4, n_samples),
        'region_cleaned': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples),
        'financial_activity_score': np.random.beta(3, 3, n_samples),
        'saved_no_purpose': np.random.beta(2, 4, n_samples)
    }
    
    # Create target variable based on weighted combination (realistic relationship)
    weights = [0.321, 0.313, 0.285, 0.260, 0.254, 0.214, 0.198, 0.182, 0.167, 0.138]
    features_array = np.column_stack([
        sample_data['biz_loan_source'], sample_data['saved_any'], sample_data['borrowed_any'],
        sample_data['income_digital_interaction'], sample_data['biz_loan'],
        np.random.random(n_samples),  # demo placeholder
        sample_data['saved_for_purchase'], sample_data['region_cleaned'] / 6,  # normalize region
        sample_data['financial_activity_score'], sample_data['saved_no_purpose']
    ])
    
    target = np.clip(np.dot(features_array, weights) / sum(weights) + np.random.normal(0, 0.1, n_samples), 0, 1)
    sample_data['Has_Account'] = target
    
    df_sample = pd.DataFrame(sample_data)
    
    print(f"Sample dataset created: {df_sample.shape}")
    
    # Train model
    trainer = OptimizedModelTrainer()
    model_path, best_model, results = trainer.run_complete_training(df_sample)
    
    # Test model
    test_prediction = test_saved_model(model_path)
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Model trained and saved: ✅")
    print(f"   File location: {model_path}")
    print(f"   Ready for Streamlit deployment: ✅")
    print(f"   Test prediction: {test_prediction:.1%}")
    
    # Instructions for deployment
    print(f"\n📋 DEPLOYMENT INSTRUCTIONS:")
    print(f"1. Copy '{model_path}' to your Streamlit app directory")
    print(f"2. Use the Streamlit deployment code provided")
    print(f"3. The model will automatically load the top 10 features")
    print(f"4. Users input only the most important predictors")
    print(f"5. Fast, accurate predictions with minimal data collection")
    
    
    
    