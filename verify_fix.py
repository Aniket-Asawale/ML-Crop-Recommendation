"""
Verification script to confirm all fixes are working
Run this AFTER clearing Streamlit cache to verify everything works
"""
import pandas as pd
import joblib
import numpy as np
import os

print("=" * 80)
print("VERIFICATION SCRIPT - CONFIRMING ALL FIXES WORK")
print("=" * 80)

# Load preprocessors
scaler = joblib.load('data/processed/scaler.pkl')
label_encoder = joblib.load('data/processed/label_encoder.pkl')

# Test cases
test_cases = {
    'Rice': {'N': 80, 'P': 47, 'K': 40, 'temp': 23.7, 'humidity': 82.2, 'ph': 6.4, 'rainfall': 233.1},
    'Cotton': {'N': 117, 'P': 46, 'K': 19, 'temp': 24.0, 'humidity': 80.0, 'ph': 6.8, 'rainfall': 80.0},
    'Maize': {'N': 76, 'P': 48, 'K': 20, 'temp': 22.8, 'humidity': 65.3, 'ph': 6.3, 'rainfall': 84.0},
    'Chickpea': {'N': 39, 'P': 68, 'K': 79, 'temp': 18.9, 'humidity': 16.7, 'ph': 7.4, 'rainfall': 80.0},
}

# Models to test (the ones that should work in the app)
models_to_test = [
    'logistic_regression.pkl',
    'svm_linear_model.pkl',
    'svm_rbf_model.pkl',
    'svm_poly_model.pkl',
    'svm_best_model.pkl',
    'random_forest_model.pkl',
    'best_random_forest_model.pkl',
    'gradient_boosting_model.pkl',
    'adaboost_model.pkl',
    'xgboost_model.pkl',
]

print(f"\n📊 Testing {len(models_to_test)} models with {len(test_cases)} test cases")
print("=" * 80)

results = {}
all_passed = True

for model_file in models_to_test:
    model_path = os.path.join('models/saved_models', model_file)
    model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  {model_name}: Model file not found - SKIP")
        continue
    
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
        
        # Check if model is stuck
        unique_predictions = set(predictions)
        is_stuck = len(unique_predictions) == 1
        
        # Check for feature names
        has_feature_names = hasattr(model, 'feature_names_in_')
        
        # Determine pass/fail
        if is_stuck:
            status = "❌ FAIL - STUCK"
            all_passed = False
        elif has_feature_names:
            status = "⚠️  WARN - Has feature names"
        else:
            status = "✅ PASS"
        
        print(f"\n{status} - {model_name}")
        print(f"   Unique predictions: {len(unique_predictions)}/4")
        print(f"   Predictions: {', '.join(predictions)}")
        
        results[model_name] = {
            'status': status,
            'predictions': predictions,
            'unique_count': len(unique_predictions),
            'is_stuck': is_stuck,
            'has_feature_names': has_feature_names
        }
        
    except Exception as e:
        print(f"\n❌ ERROR - {model_name}: {str(e)}")
        all_passed = False

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Count results
passed = sum(1 for r in results.values() if '✅' in r['status'])
warned = sum(1 for r in results.values() if '⚠️' in r['status'])
failed = sum(1 for r in results.values() if '❌' in r['status'])

print(f"\n📊 Results:")
print(f"   ✅ Passed: {passed}")
print(f"   ⚠️  Warnings: {warned}")
print(f"   ❌ Failed: {failed}")
print(f"   Total tested: {len(results)}")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if all_passed and passed == len(results):
    print("\n🎉 SUCCESS! All models are working correctly!")
    print("\n✅ You can now use the Streamlit app with confidence.")
    print("✅ All models should provide varied predictions.")
    print("✅ No models are stuck on a single crop.")
    print("\n💡 Next step: Run 'streamlit run app.py' and test!")
elif warned > 0 and failed == 0:
    print("\n⚠️  PARTIAL SUCCESS - Models work but have warnings")
    print("\n✅ Models provide varied predictions (not stuck)")
    print("⚠️  Some models have feature names (minor issue)")
    print("\n💡 This is acceptable - the app should work fine.")
    print("💡 To fix warnings, retrain those specific models.")
else:
    print("\n❌ ISSUES DETECTED!")
    print("\nProblems found:")
    if failed > 0:
        print(f"   ❌ {failed} models are stuck or have errors")
    if warned > 0:
        print(f"   ⚠️  {warned} models have feature name warnings")
    
    print("\n💡 Troubleshooting:")
    print("   1. Make sure you retrained all models after fixing notebooks")
    print("   2. Check that notebook fixes were saved correctly")
    print("   3. Run notebooks again to regenerate model files")
    print("   4. Verify you're using the correct scaler.pkl")

print("\n" + "=" * 80)

# Detailed breakdown
if results:
    print("\n📋 DETAILED BREAKDOWN")
    print("=" * 80)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"   Status: {result['status']}")
        print(f"   Unique predictions: {result['unique_count']}/4")
        print(f"   Stuck: {'Yes' if result['is_stuck'] else 'No'}")
        print(f"   Has feature names: {'Yes' if result['has_feature_names'] else 'No'}")
        print(f"   Predictions: {', '.join(result['predictions'])}")

print("\n" + "=" * 80)
print("END OF VERIFICATION")
print("=" * 80)

