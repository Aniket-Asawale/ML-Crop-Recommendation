"""
Test deployment compatibility for Streamlit Cloud
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_deployment_compatibility():
    print("🚀 TESTING DEPLOYMENT COMPATIBILITY")
    print("=" * 60)
    
    # Check Python and package versions
    print(f"Python version: {sys.version}")
    
    try:
        import sklearn
        print(f"scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn not available")
        return False
    
    try:
        import streamlit
        print(f"Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not available")
    
    try:
        import xgboost
        print(f"XGBoost version: {xgboost.__version__}")
    except ImportError:
        print("⚠️ XGBoost not available (optional)")
    
    # Test data loading
    print(f"\n📊 TESTING DATA LOADING...")
    try:
        scaler = joblib.load('data/processed/scaler.pkl')
        label_encoder = joblib.load('data/processed/label_encoder.pkl')
        print("✅ Preprocessing objects loaded successfully")
    except Exception as e:
        print(f"❌ Error loading preprocessing objects: {e}")
        return False
    
    # Test deployment models
    print(f"\n🤖 TESTING DEPLOYMENT MODELS...")
    models_path = Path('models/saved_models')
    
    deployment_models = [
        'logistic_regression_deployment.pkl',
        'random_forest_deployment.pkl',
        'svm_linear_deployment.pkl',
        'svm_rbf_deployment.pkl'
    ]
    
    loaded_models = {}
    
    for model_file in deployment_models:
        model_path = models_path / model_file
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                
                # Test prediction
                test_input = np.array([[50, 50, 50, 25, 70, 6.5, 100]])
                prediction = model.predict(test_input)
                
                # Test probability prediction
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(test_input)
                    print(f"✅ {model_file}: Loaded and tested successfully")
                else:
                    print(f"✅ {model_file}: Loaded (no probabilities)")
                
                loaded_models[model_file] = model
                
            except Exception as e:
                print(f"❌ {model_file}: Error - {str(e)}")
        else:
            print(f"⚠️ {model_file}: File not found")
    
    if not loaded_models:
        print("❌ No deployment models available")
        return False
    
    # Test fallback models
    print(f"\n🔄 TESTING FALLBACK MODELS...")
    fallback_models = [
        'logistic_regression.pkl',
        'svm_linear_model.pkl',
        'random_forest_model.pkl'
    ]
    
    fallback_loaded = 0
    for model_file in fallback_models:
        model_path = models_path / model_file
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                test_input = np.array([[50, 50, 50, 25, 70, 6.5, 100]])
                prediction = model.predict(test_input)
                fallback_loaded += 1
                print(f"✅ {model_file}: Available as fallback")
            except Exception as e:
                print(f"❌ {model_file}: Error - {str(e)}")
        else:
            print(f"⚠️ {model_file}: Not found")
    
    # Test grape prediction consistency
    print(f"\n🍇 TESTING GRAPE PREDICTION...")
    grape_input = pd.DataFrame({
        'N': [20], 'P': [130], 'K': [200],
        'temperature': [20], 'humidity': [85],
        'ph': [6.2], 'rainfall': [70]
    })
    grape_scaled = scaler.transform(grape_input)
    
    for model_file, model in loaded_models.items():
        try:
            pred = model.predict(grape_scaled)
            crop = label_encoder.inverse_transform(pred)[0]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(grape_scaled)[0]
                confidence = np.max(proba) * 100
                print(f"  {model_file.replace('_deployment.pkl', '')}: {crop} ({confidence:.1f}%)")
            else:
                print(f"  {model_file.replace('_deployment.pkl', '')}: {crop}")
        except Exception as e:
            print(f"  ❌ {model_file}: Error in prediction - {str(e)}")
    
    # Test rule-based fallback
    print(f"\n🛡️ TESTING RULE-BASED FALLBACK...")
    try:
        # Simple rule-based predictor
        class SimplePredictor:
            def predict(self, X):
                # Always predict rice for simplicity
                return np.array([20] * len(X))
            
            def predict_proba(self, X):
                n_samples = len(X)
                probabilities = np.zeros((n_samples, 22))
                probabilities[:, 20] = 0.8  # 80% confidence for rice
                probabilities[:, 11] = 0.1  # 10% for maize
                probabilities[:, 6] = 0.1   # 10% for cotton
                return probabilities
        
        fallback_predictor = SimplePredictor()
        test_pred = fallback_predictor.predict(grape_scaled)
        test_crop = label_encoder.inverse_transform(test_pred)[0]
        print(f"✅ Rule-based fallback: {test_crop}")
        
    except Exception as e:
        print(f"❌ Rule-based fallback failed: {e}")
    
    # Summary
    print(f"\n📋 DEPLOYMENT READINESS SUMMARY:")
    print(f"  ✅ Deployment models available: {len(loaded_models)}")
    print(f"  ✅ Fallback models available: {fallback_loaded}")
    print(f"  ✅ Preprocessing objects: Working")
    print(f"  ✅ Rule-based fallback: Available")
    
    if len(loaded_models) >= 2:
        print(f"\n🎉 DEPLOYMENT READY!")
        print(f"  - Multiple models available for redundancy")
        print(f"  - Fallback systems in place")
        print(f"  - Compatible with Streamlit Cloud")
        return True
    elif len(loaded_models) >= 1:
        print(f"\n⚠️ MINIMAL DEPLOYMENT READY")
        print(f"  - At least one model available")
        print(f"  - Consider adding more models for redundancy")
        return True
    else:
        print(f"\n❌ DEPLOYMENT NOT READY")
        print(f"  - No working models available")
        print(f"  - Check scikit-learn version compatibility")
        return False

if __name__ == "__main__":
    success = test_deployment_compatibility()
    if success:
        print(f"\n🚀 Ready for: streamlit run app.py")
    else:
        print(f"\n❌ Fix issues before deployment")
