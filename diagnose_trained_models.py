"""
Comprehensive diagnostic to check if models were trained correctly
"""
import pandas as pd
import joblib
import numpy as np
import os
import warnings

print("=" * 80)
print("TRAINED MODELS DIAGNOSTIC - FEATURE NAMES CHECK")
print("=" * 80)

# Load preprocessors
scaler = joblib.load('data/processed/scaler.pkl')
label_encoder = joblib.load('data/processed/label_encoder.pkl')

# Test input (as DataFrame - like the app does)
test_input_df = pd.DataFrame({
    'N': [80],
    'P': [47],
    'K': [40],
    'temperature': [23.7],
    'humidity': [82.2],
    'ph': [6.4],
    'rainfall': [233.1]
})

# Scale it (returns numpy array)
test_input_scaled = scaler.transform(test_input_df)

print(f"\n📊 Test Input Info:")
print(f"   Input DataFrame shape: {test_input_df.shape}")
print(f"   Input DataFrame type: {type(test_input_df)}")
print(f"   Scaled input shape: {test_input_scaled.shape}")
print(f"   Scaled input type: {type(test_input_scaled)}")
print(f"   Scaled input is numpy array: {isinstance(test_input_scaled, np.ndarray)}")

# Find all model files
model_dir = 'models/saved_models'
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'model' in f.lower()]

print(f"\n📁 Found {len(model_files)} model files")
print("=" * 80)

# Test each model
for model_file in sorted(model_files):
    model_path = os.path.join(model_dir, model_file)
    model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
    
    try:
        model = joblib.load(model_path)
        
        # Check if model has feature_names_in_ attribute (sklearn models trained on DataFrames have this)
        has_feature_names = hasattr(model, 'feature_names_in_')
        
        print(f"\n🔍 {model_name}")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Has feature_names_in_: {has_feature_names}")
        
        if has_feature_names:
            print(f"   ⚠️  PROBLEM: Model was trained with DataFrame (has feature names)")
            print(f"   Feature names: {model.feature_names_in_}")
        else:
            print(f"   ✅ OK: Model was trained with numpy array (no feature names)")
        
        # Try prediction with warnings captured
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            prediction = model.predict(test_input_scaled)
            
            if w:
                print(f"   ⚠️  WARNINGS during prediction:")
                for warning in w:
                    print(f"      - {warning.message}")
            else:
                print(f"   ✅ No warnings during prediction")
        
        # Get prediction
        predicted_crop = label_encoder.inverse_transform(prediction)[0]
        print(f"   Prediction: {predicted_crop}")
        
    except Exception as e:
        print(f"\n❌ ERROR - {model_name}: {str(e)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n🎯 KEY FINDINGS:")
print("   If models have 'feature_names_in_' attribute, they were trained with DataFrames")
print("   This causes the 'X does not have valid feature names' warning")
print("   The warning itself doesn't break predictions, but indicates training issue")
print("\n💡 SOLUTION:")
print("   Models should be trained with numpy arrays (from scaler.transform())")
print("   This ensures no feature names are stored in the model")
print("   Notebooks are already fixed - models just need to be retrained correctly")

