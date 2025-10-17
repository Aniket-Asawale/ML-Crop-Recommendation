"""
Test grape prediction inconsistencies across models
"""

import joblib
import pandas as pd
import numpy as np

def test_grape_predictions():
    print('🍇 TESTING GRAPE PREDICTION INCONSISTENCY')
    print('=' * 50)

    # Load preprocessors
    scaler = joblib.load('data/processed/scaler.pkl')
    label_encoder = joblib.load('data/processed/label_encoder.pkl')

    # Ideal grape conditions (from the app)
    grape_conditions = {
        'N': 20, 'P': 130, 'K': 200, 
        'temp': 20, 'humidity': 85, 
        'ph': 6.2, 'rainfall': 70
    }

    print('Testing with ideal grape conditions:')
    print(f'N={grape_conditions["N"]}, P={grape_conditions["P"]}, K={grape_conditions["K"]}')
    print(f'Temp={grape_conditions["temp"]}, Humidity={grape_conditions["humidity"]}, pH={grape_conditions["ph"]}, Rainfall={grape_conditions["rainfall"]}')

    # Prepare input data
    input_data = pd.DataFrame({
        'N': [grape_conditions['N']],
        'P': [grape_conditions['P']],
        'K': [grape_conditions['K']],
        'temperature': [grape_conditions['temp']],
        'humidity': [grape_conditions['humidity']],
        'ph': [grape_conditions['ph']],
        'rainfall': [grape_conditions['rainfall']]
    })

    input_scaled = scaler.transform(input_data)

    # Test key models
    models_to_test = [
        ('Logistic Regression', 'models/saved_models/logistic_regression.pkl'),
        ('XGBoost', 'models/saved_models/xgboost_model.pkl'),
        ('Random Forest', 'models/saved_models/random_forest_model.pkl'),
        ('Gradient Boosting', 'models/saved_models/gradient_boosting_model.pkl'),
        ('SVM Best', 'models/saved_models/svm_best_model.pkl'),
        ('Stacking Classifier', 'models/saved_models/stacking_classifier.pkl'),
        ('Voting Classifier', 'models/saved_models/voting_soft.pkl'),
    ]

    print('\n📊 Model Predictions for Ideal Grape Conditions:')
    print('-' * 70)
    print(f'{"Model":<20} | {"Prediction":<12} | {"Confidence":<8} | Top 3 Predictions')
    print('-' * 70)

    results = []

    for model_name, model_path in models_to_test:
        try:
            model = joblib.load(model_path)
            
            # Get prediction and probabilities
            prediction = model.predict(input_scaled)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_scaled)[0]
                max_prob = np.max(probabilities) * 100
                
                # Get top 3 predictions
                top_3_idx = np.argsort(probabilities)[-3:][::-1]
                top_3_crops = label_encoder.inverse_transform(top_3_idx)
                top_3_probs = probabilities[top_3_idx] * 100
                
                results.append({
                    'model': model_name,
                    'prediction': predicted_crop,
                    'confidence': max_prob,
                    'top_3': list(zip(top_3_crops, top_3_probs))
                })
                
                top_3_str = f"{top_3_crops[0]}({top_3_probs[0]:.1f}%), {top_3_crops[1]}({top_3_probs[1]:.1f}%), {top_3_crops[2]}({top_3_probs[2]:.1f}%)"
                print(f'{model_name:<20} | {predicted_crop:<12} | {max_prob:6.1f}% | {top_3_str}')
            else:
                results.append({
                    'model': model_name,
                    'prediction': predicted_crop,
                    'confidence': 'N/A',
                    'top_3': []
                })
                print(f'{model_name:<20} | {predicted_crop:<12} | {"N/A":<8} | No probabilities available')
                
        except Exception as e:
            print(f'{model_name:<20} | ERROR: {str(e)}')

    # Analysis
    print(f'\n🔍 ANALYSIS:')
    print('-' * 30)

    predictions = [r['prediction'] for r in results if r['prediction']]
    unique_predictions = set(predictions)
    print(f'Unique predictions: {len(unique_predictions)} - {sorted(unique_predictions)}')

    # Check for grapes predictions
    grape_predictions = [r for r in results if r['prediction'] == 'grapes']
    non_grape_predictions = [r for r in results if r['prediction'] != 'grapes']

    print(f'\nModels predicting GRAPES: {len(grape_predictions)}')
    for r in grape_predictions:
        if isinstance(r['confidence'], (int, float)):
            print(f'  - {r["model"]}: {r["confidence"]:.1f}%')
        else:
            print(f'  - {r["model"]}: {r["confidence"]}')

    print(f'\nModels predicting OTHER crops: {len(non_grape_predictions)}')
    for r in non_grape_predictions:
        if isinstance(r['confidence'], (int, float)):
            print(f'  - {r["model"]}: {r["prediction"]} ({r["confidence"]:.1f}%)')
        else:
            print(f'  - {r["model"]}: {r["prediction"]} ({r["confidence"]})')

    # Check confidence score variations
    confidences = [r['confidence'] for r in results if isinstance(r['confidence'], (int, float))]
    if confidences:
        print(f'\nConfidence score statistics:')
        print(f'  Range: {min(confidences):.1f}% - {max(confidences):.1f}%')
        print(f'  Average: {np.mean(confidences):.1f}%')
        print(f'  Standard deviation: {np.std(confidences):.1f}%')
        
        if max(confidences) - min(confidences) > 30:
            print(f'  ⚠️  HIGH VARIATION: Confidence scores vary by {max(confidences) - min(confidences):.1f}%')
        
    # Check if any model predicts apple with high confidence
    apple_predictions = [r for r in results if r['prediction'] == 'apple']
    if apple_predictions:
        print(f'\n🍎 Models predicting APPLE (potential issue):')
        for r in apple_predictions:
            if isinstance(r['confidence'], (int, float)):
                print(f'  - {r["model"]}: {r["confidence"]:.1f}% confidence')
                if r['confidence'] > 90:
                    print(f'    ⚠️  VERY HIGH CONFIDENCE for wrong crop!')

if __name__ == "__main__":
    test_grape_predictions()
