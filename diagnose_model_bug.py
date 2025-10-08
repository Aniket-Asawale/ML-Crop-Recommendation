"""
Diagnostic script to identify why models are stuck predicting kidneybeans
"""
import pandas as pd
import joblib
import numpy as np
import os

print("=" * 80)
print("MODEL BUG DIAGNOSTIC REPORT")
print("=" * 80)

# Load preprocessors
scaler = joblib.load('data/processed/scaler.pkl')
label_encoder = joblib.load('data/processed/label_encoder.pkl')

print(f"\n✅ Loaded scaler and label encoder")
print(f"Available crops: {list(label_encoder.classes_)}")

# Test inputs - diverse conditions
test_cases = {
    'Rice-like': {'N': 80, 'P': 47, 'K': 40, 'temp': 23.7, 'humidity': 82.2, 'ph': 6.4, 'rainfall': 233.1},
    'Chickpea-like': {'N': 39, 'P': 68, 'K': 79, 'temp': 18.9, 'humidity': 16.7, 'ph': 7.4, 'rainfall': 79.7},
    'Cotton-like': {'N': 117, 'P': 46, 'K': 19, 'temp': 24.0, 'humidity': 80.0, 'ph': 6.8, 'rainfall': 80.2},
    'Maize-like': {'N': 76, 'P': 48, 'K': 20, 'temp': 22.8, 'humidity': 65.3, 'ph': 6.3, 'rainfall': 83.5},
    'High-N': {'N': 140, 'P': 50, 'K': 50, 'temp': 25.0, 'humidity': 70.0, 'ph': 6.5, 'rainfall': 100.0},
    'Low-N': {'N': 10, 'P': 50, 'K': 50, 'temp': 25.0, 'humidity': 70.0, 'ph': 6.5, 'rainfall': 100.0},
}

# Find all model files
model_dir = 'models/saved_models'
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'model' in f.lower()]

print(f"\n📁 Found {len(model_files)} model files")
print("=" * 80)

# Test each model
results = {}

for model_file in sorted(model_files):
    model_path = os.path.join(model_dir, model_file)
    model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
    
    try:
        model = joblib.load(model_path)
        
        # Test with all test cases
        predictions = []
        for case_name, params in test_cases.items():
            input_data = pd.DataFrame({
                'N': [params['N']],
                'P': [params['P']],
                'K': [params['K']],
                'temperature': [params['temp']],
                'humidity': [params['humidity']],
                'ph': [params['ph']],
                'rainfall': [params['rainfall']]
            })
            
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]
            predictions.append(predicted_crop)
        
        # Check if model is stuck (all predictions the same)
        unique_predictions = set(predictions)
        is_stuck = len(unique_predictions) == 1
        
        results[model_name] = {
            'predictions': predictions,
            'unique_count': len(unique_predictions),
            'is_stuck': is_stuck,
            'stuck_on': list(unique_predictions)[0] if is_stuck else None
        }
        
        # Print results
        status = "❌ STUCK" if is_stuck else "✅ WORKING"
        print(f"\n{status} - {model_name}")
        if is_stuck:
            print(f"   Always predicts: {results[model_name]['stuck_on']}")
        else:
            print(f"   Predicts {len(unique_predictions)} different crops: {sorted(unique_predictions)}")
        print(f"   Predictions: {predictions}")
        
    except Exception as e:
        print(f"\n⚠️  ERROR - {model_name}: {str(e)}")
        results[model_name] = {'error': str(e)}

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

working_models = [name for name, data in results.items() if not data.get('is_stuck', True) and 'error' not in data]
stuck_models = [name for name, data in results.items() if data.get('is_stuck', False)]
error_models = [name for name, data in results.items() if 'error' in data]

print(f"\n✅ Working models ({len(working_models)}):")
for name in working_models:
    print(f"   - {name}")

print(f"\n❌ Stuck models ({len(stuck_models)}):")
for name in stuck_models:
    stuck_on = results[name]['stuck_on']
    print(f"   - {name} (always predicts: {stuck_on})")

if error_models:
    print(f"\n⚠️  Models with errors ({len(error_models)}):")
    for name in error_models:
        print(f"   - {name}: {results[name]['error']}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if stuck_models:
    print("\n🔍 Models are stuck predicting the same crop for all inputs.")
    print("   This indicates a fundamental bug in the training process.")
    print("\n   Possible causes:")
    print("   1. Features not scaled properly during training")
    print("   2. Feature order mismatch between training and prediction")
    print("   3. Model trained on wrong data")
    print("   4. Feature names not preserved in pipeline")
    
    # Check if all stuck on same crop
    stuck_crops = [results[name]['stuck_on'] for name in stuck_models]
    if len(set(stuck_crops)) == 1:
        print(f"\n   ⚠️  ALL stuck models predict the same crop: {stuck_crops[0]}")
        print("      This suggests a systematic issue in the training pipeline.")
else:
    print("\n✅ All models are working correctly!")

print("\n" + "=" * 80)

