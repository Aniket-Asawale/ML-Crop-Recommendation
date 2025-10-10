"""
Retrain overfitted models with better regularization
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def retrain_overfitted_models():
    print("🔧 RETRAINING OVERFITTED MODELS WITH REGULARIZATION")
    print("=" * 60)
    
    # Load data
    train_data = pd.read_csv('data/processed/train.csv')
    val_data = pd.read_csv('data/processed/validation.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    scaler = joblib.load('data/processed/scaler.pkl')
    label_encoder = joblib.load('data/processed/label_encoder.pkl')
    
    # Combine train and validation for better training
    combined_data = pd.concat([train_data, val_data], ignore_index=True)
    
    # Prepare features
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X_combined = combined_data[feature_cols]
    y_combined = combined_data['label']
    X_test = test_data[feature_cols]
    y_test = test_data['label']
    
    # Scale features
    X_combined_scaled = scaler.transform(X_combined)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data loaded: {len(X_combined)} training samples, {len(X_test)} test samples")
    
    # 1. Retrain Gradient Boosting with regularization
    print(f"\n🌳 RETRAINING GRADIENT BOOSTING...")
    gb_regularized = GradientBoostingClassifier(
        n_estimators=50,  # Reduced from default 100
        max_depth=3,      # Reduced from default 3 (keep conservative)
        min_samples_split=20,  # Increased from default 2
        min_samples_leaf=10,   # Increased from default 1
        learning_rate=0.05,    # Reduced from default 0.1
        subsample=0.8,         # Add subsampling
        random_state=42
    )
    
    gb_regularized.fit(X_combined_scaled, y_combined)
    
    # Evaluate
    cv_scores = cross_val_score(gb_regularized, X_combined_scaled, y_combined, cv=5, scoring='accuracy')
    test_pred = gb_regularized.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Overfitting Score: {test_accuracy - cv_scores.mean():.4f}")
    
    # Save if improved
    if test_accuracy < 0.99:  # Less overfitted
        joblib.dump(gb_regularized, 'models/saved_models/gradient_boosting_regularized.pkl')
        print(f"  ✅ Saved regularized model")
    else:
        print(f"  ⚠️  Still overfitted, not saving")
    
    # 2. Retrain Random Forest with regularization
    print(f"\n🌲 RETRAINING RANDOM FOREST...")
    rf_regularized = RandomForestClassifier(
        n_estimators=50,       # Reduced from default 100
        max_depth=10,          # Limited depth
        min_samples_split=20,  # Increased from default 2
        min_samples_leaf=5,    # Increased from default 1
        max_features='sqrt',   # Use sqrt instead of all features
        bootstrap=True,
        oob_score=True,        # Out-of-bag scoring
        random_state=42
    )
    
    rf_regularized.fit(X_combined_scaled, y_combined)
    
    # Evaluate
    cv_scores = cross_val_score(rf_regularized, X_combined_scaled, y_combined, cv=5, scoring='accuracy')
    test_pred = rf_regularized.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  OOB Score: {rf_regularized.oob_score_:.4f}")
    print(f"  Overfitting Score: {test_accuracy - cv_scores.mean():.4f}")
    
    # Save if improved
    if test_accuracy < 0.99:  # Less overfitted
        joblib.dump(rf_regularized, 'models/saved_models/random_forest_regularized.pkl')
        print(f"  ✅ Saved regularized model")
    else:
        print(f"  ⚠️  Still overfitted, not saving")
    
    # 3. Retrain XGBoost with regularization
    print(f"\n🚀 RETRAINING XGBOOST...")
    xgb_regularized = xgb.XGBClassifier(
        n_estimators=50,       # Reduced
        max_depth=4,           # Limited depth
        min_child_weight=5,    # Increased from default 1
        gamma=0.1,             # Add regularization
        subsample=0.8,         # Subsampling
        colsample_bytree=0.8,  # Feature subsampling
        learning_rate=0.05,    # Reduced learning rate
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=1.0,        # L2 regularization
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_regularized.fit(X_combined_scaled, y_combined)
    
    # Evaluate
    cv_scores = cross_val_score(xgb_regularized, X_combined_scaled, y_combined, cv=5, scoring='accuracy')
    test_pred = xgb_regularized.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Overfitting Score: {test_accuracy - cv_scores.mean():.4f}")
    
    # Save if improved
    if test_accuracy < 0.99:  # Less overfitted
        joblib.dump(xgb_regularized, 'models/saved_models/xgboost_regularized.pkl')
        print(f"  ✅ Saved regularized model")
    else:
        print(f"  ⚠️  Still overfitted, not saving")
    
    # Test grape prediction consistency
    print(f"\n🍇 TESTING GRAPE PREDICTION CONSISTENCY...")
    grape_input = pd.DataFrame({
        'N': [20], 'P': [130], 'K': [200],
        'temperature': [20], 'humidity': [85],
        'ph': [6.2], 'rainfall': [70]
    })
    grape_scaled = scaler.transform(grape_input)
    
    models_to_test = [
        ('GB Regularized', gb_regularized),
        ('RF Regularized', rf_regularized),
        ('XGB Regularized', xgb_regularized)
    ]
    
    for name, model in models_to_test:
        pred = model.predict(grape_scaled)
        crop = label_encoder.inverse_transform(pred)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(grape_scaled)[0]
            confidence = np.max(proba) * 100
            print(f"  {name}: {crop} ({confidence:.1f}%)")
        else:
            print(f"  {name}: {crop}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"  - Retrained models with regularization to reduce overfitting")
    print(f"  - Models with test accuracy < 99% were saved as regularized versions")
    print(f"  - Use these models if you want less overfitted predictions")
    print(f"  - Original models are still available for comparison")

if __name__ == "__main__":
    retrain_overfitted_models()
