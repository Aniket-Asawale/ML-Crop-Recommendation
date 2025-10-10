"""
Comprehensive Model Performance Validation Script
Checks for overfitting and validates model performance using proper cross-validation
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def validate_model_performance():
    print("🔍 COMPREHENSIVE MODEL PERFORMANCE VALIDATION")
    print("=" * 60)
    
    # Load data
    train_data = pd.read_csv('data/processed/train.csv')
    val_data = pd.read_csv('data/processed/validation.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    scaler = joblib.load('data/processed/scaler.pkl')
    label_encoder = joblib.load('data/processed/label_encoder.pkl')
    
    # Combine train and validation for proper cross-validation
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
    
    print(f"✅ Data loaded: {len(X_combined)} combined samples, {len(X_test)} test samples")
    
    # Models to validate
    models_to_check = [
        ('Logistic Regression', 'models/saved_models/logistic_regression.pkl'),
        ('Random Forest', 'models/saved_models/random_forest_model.pkl'),
        ('Gradient Boosting', 'models/saved_models/gradient_boosting_model.pkl'),
        ('XGBoost', 'models/saved_models/xgboost_model.pkl'),
        ('SVM Best', 'models/saved_models/svm_best_model.pkl'),
        ('Voting Classifier', 'models/saved_models/voting_soft.pkl'),
        ('Stacking Classifier', 'models/saved_models/stacking_classifier.pkl'),
    ]
    
    results = []
    
    for model_name, model_path in models_to_check:
        try:
            print(f"\n📊 Validating {model_name}...")
            
            # Load model
            model = joblib.load(model_path)
            
            # Cross-validation on combined data (proper validation)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_combined_scaled, y_combined, cv=cv, scoring='accuracy')
            
            # Test set performance
            y_test_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Calculate overfitting indicator
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            overfitting_score = test_accuracy - cv_mean
            
            # Determine if overfitting
            is_overfitting = overfitting_score > 0.05 or test_accuracy > 0.99
            
            results.append({
                'Model': model_name,
                'CV_Mean': cv_mean,
                'CV_Std': cv_std,
                'Test_Accuracy': test_accuracy,
                'Overfitting_Score': overfitting_score,
                'Is_Overfitting': is_overfitting
            })
            
            print(f"   CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"   Test Accuracy: {test_accuracy:.4f}")
            print(f"   Overfitting Score: {overfitting_score:.4f}")
            
            if is_overfitting:
                print(f"   ⚠️  POTENTIAL OVERFITTING DETECTED")
                if test_accuracy > 0.99:
                    print(f"      - Test accuracy too high (>99%)")
                if overfitting_score > 0.05:
                    print(f"      - Large gap between CV and test performance")
            else:
                print(f"   ✅ Performance looks good")
                
        except Exception as e:
            print(f"   ❌ Error validating {model_name}: {str(e)}")
            results.append({
                'Model': model_name,
                'CV_Mean': np.nan,
                'CV_Std': np.nan,
                'Test_Accuracy': np.nan,
                'Overfitting_Score': np.nan,
                'Is_Overfitting': True
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    
    # Models with potential overfitting
    overfitted_models = results_df[results_df['Is_Overfitting'] == True]['Model'].tolist()
    good_models = results_df[results_df['Is_Overfitting'] == False]['Model'].tolist()
    
    print(f"\n✅ Models with good performance ({len(good_models)}):")
    for model in good_models:
        row = results_df[results_df['Model'] == model].iloc[0]
        print(f"   - {model}: CV={row['CV_Mean']:.4f}, Test={row['Test_Accuracy']:.4f}")
    
    if overfitted_models:
        print(f"\n⚠️  Models with potential overfitting ({len(overfitted_models)}):")
        for model in overfitted_models:
            row = results_df[results_df['Model'] == model].iloc[0]
            if not pd.isna(row['CV_Mean']):
                print(f"   - {model}: CV={row['CV_Mean']:.4f}, Test={row['Test_Accuracy']:.4f}")
            else:
                print(f"   - {model}: Error during validation")
    
    print("\n📋 RECOMMENDATIONS:")
    
    if len(overfitted_models) > 0:
        print("   1. Models showing 100% test accuracy may be overfitted")
        print("   2. Consider using models with CV accuracy 95-98% for better generalization")
        print("   3. Logistic Regression and SVM often provide good balance")
        print("   4. Ensemble methods may be overfitting to the specific test set")
    else:
        print("   ✅ All models show reasonable performance without overfitting")
    
    print(f"\n💡 BEST MODELS FOR PRODUCTION:")
    # Recommend models with CV accuracy > 95% and test accuracy < 99%
    production_models = results_df[
        (results_df['CV_Mean'] > 0.95) & 
        (results_df['Test_Accuracy'] < 0.99) & 
        (results_df['Is_Overfitting'] == False)
    ]['Model'].tolist()
    
    if production_models:
        for model in production_models:
            row = results_df[results_df['Model'] == model].iloc[0]
            print(f"   ✅ {model}: Reliable and generalizable")
    else:
        print("   ⚠️  Consider retraining models with regularization")
    
    return results_df

if __name__ == "__main__":
    results = validate_model_performance()
    
    # Save results
    results.to_csv('data/processed/model_validation_results.csv', index=False)
    print(f"\n💾 Results saved to 'data/processed/model_validation_results.csv'")
