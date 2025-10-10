"""
Create deployment-compatible models for Streamlit Cloud
Uses conservative approaches that work across scikit-learn versions
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def create_deployment_models():
    print("🚀 CREATING DEPLOYMENT-COMPATIBLE MODELS")
    print("=" * 60)
    
    # Load data
    train_data = pd.read_csv('data/processed/train.csv')
    val_data = pd.read_csv('data/processed/validation.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    scaler = joblib.load('data/processed/scaler.pkl')
    label_encoder = joblib.load('data/processed/label_encoder.pkl')
    
    # Combine train and validation
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
    
    # 1. Logistic Regression (Most Compatible)
    print(f"\n📊 TRAINING LOGISTIC REGRESSION...")
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
        multi_class='ovr'
    )
    
    lr_model.fit(X_combined_scaled, y_combined)
    
    # Evaluate
    cv_scores = cross_val_score(lr_model, X_combined_scaled, y_combined, cv=5)
    test_pred = lr_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    # Save
    joblib.dump(lr_model, 'models/saved_models/logistic_regression_deployment.pkl')
    print(f"  ✅ Saved deployment-compatible model")
    
    # 2. Random Forest (Conservative Settings)
    print(f"\n🌲 TRAINING RANDOM FOREST...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1  # Single thread for compatibility
    )
    
    rf_model.fit(X_combined_scaled, y_combined)
    
    # Evaluate
    cv_scores = cross_val_score(rf_model, X_combined_scaled, y_combined, cv=5)
    test_pred = rf_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    # Save
    joblib.dump(rf_model, 'models/saved_models/random_forest_deployment.pkl')
    print(f"  ✅ Saved deployment-compatible model")
    
    # 3. SVM (Linear - Most Stable)
    print(f"\n🎯 TRAINING SVM LINEAR...")
    svm_model = SVC(
        kernel='linear',
        probability=True,
        random_state=42,
        C=1.0
    )
    
    svm_model.fit(X_combined_scaled, y_combined)
    
    # Evaluate
    cv_scores = cross_val_score(svm_model, X_combined_scaled, y_combined, cv=5)
    test_pred = svm_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    # Save
    joblib.dump(svm_model, 'models/saved_models/svm_linear_deployment.pkl')
    print(f"  ✅ Saved deployment-compatible model")
    
    # 4. SVM RBF
    print(f"\n🎯 TRAINING SVM RBF...")
    svm_rbf_model = SVC(
        kernel='rbf',
        probability=True,
        random_state=42,
        C=1.0,
        gamma='scale'
    )
    
    svm_rbf_model.fit(X_combined_scaled, y_combined)
    
    # Evaluate
    cv_scores = cross_val_score(svm_rbf_model, X_combined_scaled, y_combined, cv=5)
    test_pred = svm_rbf_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    # Save
    joblib.dump(svm_rbf_model, 'models/saved_models/svm_rbf_deployment.pkl')
    print(f"  ✅ Saved deployment-compatible model")
    
    # Test grape prediction consistency
    print(f"\n🍇 TESTING GRAPE PREDICTION CONSISTENCY...")
    grape_input = pd.DataFrame({
        'N': [20], 'P': [130], 'K': [200],
        'temperature': [20], 'humidity': [85],
        'ph': [6.2], 'rainfall': [70]
    })
    grape_scaled = scaler.transform(grape_input)
    
    models_to_test = [
        ('Logistic Regression', lr_model),
        ('Random Forest', rf_model),
        ('SVM Linear', svm_model),
        ('SVM RBF', svm_rbf_model)
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
    
    # Create a simple ensemble
    print(f"\n🤝 CREATING SIMPLE ENSEMBLE...")
    
    # Simple voting ensemble
    ensemble_predictions = []
    for name, model in models_to_test:
        pred = model.predict(X_test_scaled)
        ensemble_predictions.append(pred)
    
    # Majority vote
    ensemble_predictions = np.array(ensemble_predictions)
    final_predictions = []
    
    for i in range(len(X_test_scaled)):
        votes = ensemble_predictions[:, i]
        unique, counts = np.unique(votes, return_counts=True)
        majority_vote = unique[np.argmax(counts)]
        final_predictions.append(majority_vote)
    
    ensemble_accuracy = accuracy_score(y_test, final_predictions)
    print(f"  Ensemble Test Accuracy: {ensemble_accuracy:.4f}")
    
    # Save ensemble info
    ensemble_info = {
        'models': ['logistic_regression_deployment.pkl', 'random_forest_deployment.pkl', 
                  'svm_linear_deployment.pkl', 'svm_rbf_deployment.pkl'],
        'accuracy': ensemble_accuracy
    }
    joblib.dump(ensemble_info, 'models/saved_models/ensemble_info.pkl')
    print(f"  ✅ Saved ensemble information")
    
    print(f"\n🎯 DEPLOYMENT SUMMARY:")
    print(f"  - Created 4 deployment-compatible models")
    print(f"  - All models use conservative settings")
    print(f"  - Compatible with scikit-learn 1.0+ versions")
    print(f"  - Simple ensemble approach available")
    print(f"  - Models saved with '_deployment' suffix")

if __name__ == "__main__":
    create_deployment_models()
