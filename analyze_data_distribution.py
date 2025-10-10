"""
Analyze training data distribution to understand model inconsistencies
"""

import pandas as pd
import numpy as np

def analyze_data_distribution():
    print('🔍 ANALYZING TRAINING DATA DISTRIBUTION')
    print('=' * 50)

    # Load data
    train_data = pd.read_csv('data/processed/train.csv')
    val_data = pd.read_csv('data/processed/validation.csv')
    test_data = pd.read_csv('data/processed/test.csv')

    # Combine all data to see full distribution
    all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

    # Check crop distribution
    crop_counts = all_data['label'].value_counts()
    print('Crop distribution in dataset:')
    for crop, count in crop_counts.items():
        print(f'  {crop}: {count} samples ({count/len(all_data)*100:.1f}%)')

    print(f'\nTotal samples: {len(all_data)}')

    # Check grapes vs apple data
    grapes_data = all_data[all_data['label'] == 'grapes']
    apple_data = all_data[all_data['label'] == 'apple']

    print(f'\n🍇 GRAPES data ({len(grapes_data)} samples):')
    if len(grapes_data) > 0:
        print('  Average values:')
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            print(f'    {col}: {grapes_data[col].mean():.1f} (std: {grapes_data[col].std():.1f})')

    print(f'\n🍎 APPLE data ({len(apple_data)} samples):')
    if len(apple_data) > 0:
        print('  Average values:')
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            print(f'    {apple_data[col].mean():.1f} (std: {apple_data[col].std():.1f})')

    # Test input vs actual data
    test_input = {
        'N': 20, 'P': 130, 'K': 200, 
        'temp': 20, 'humidity': 85, 
        'ph': 6.2, 'rainfall': 70
    }

    print(f'\n📊 TEST INPUT vs ACTUAL DATA:')
    print('Test input:')
    print(f'  N={test_input["N"]}, P={test_input["P"]}, K={test_input["K"]}')
    print(f'  Temp={test_input["temp"]}, Humidity={test_input["humidity"]}, pH={test_input["ph"]}, Rainfall={test_input["rainfall"]}')

    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    test_values = np.array([
        test_input['N'], test_input['P'], test_input['K'], 
        test_input['temp'], test_input['humidity'], 
        test_input['ph'], test_input['rainfall']
    ])

    if len(grapes_data) > 0:
        print(f'\nClosest grapes samples to test input:')
        distances = []
        for idx, row in grapes_data.iterrows():
            sample_values = row[feature_cols].values
            distance = np.sqrt(np.sum((test_values - sample_values) ** 2))
            distances.append((distance, idx, row))
        
        distances.sort(key=lambda x: x[0])
        
        for i, (dist, idx, row) in enumerate(distances[:3]):
            print(f'  #{i+1} (distance: {dist:.1f}):')
            print(f'    N={row["N"]}, P={row["P"]}, K={row["K"]}, Temp={row["temperature"]}, Humidity={row["humidity"]}, pH={row["ph"]}, Rainfall={row["rainfall"]}')

    if len(apple_data) > 0:
        print(f'\nClosest apple samples to test input:')
        distances = []
        for idx, row in apple_data.iterrows():
            sample_values = row[feature_cols].values
            distance = np.sqrt(np.sum((test_values - sample_values) ** 2))
            distances.append((distance, idx, row))
        
        distances.sort(key=lambda x: x[0])
        
        for i, (dist, idx, row) in enumerate(distances[:3]):
            print(f'  #{i+1} (distance: {dist:.1f}):')
            print(f'    N={row["N"]}, P={row["P"]}, K={row["K"]}, Temp={row["temperature"]}, Humidity={row["humidity"]}, pH={row["ph"]}, Rainfall={row["rainfall"]}')

    # Check if the test input is actually closer to apple data
    if len(grapes_data) > 0 and len(apple_data) > 0:
        # Find closest sample overall
        all_distances = []
        
        for idx, row in grapes_data.iterrows():
            sample_values = row[feature_cols].values
            distance = np.sqrt(np.sum((test_values - sample_values) ** 2))
            all_distances.append((distance, 'grapes', row))
            
        for idx, row in apple_data.iterrows():
            sample_values = row[feature_cols].values
            distance = np.sqrt(np.sum((test_values - sample_values) ** 2))
            all_distances.append((distance, 'apple', row))
        
        all_distances.sort(key=lambda x: x[0])
        
        print(f'\n🎯 CLOSEST SAMPLES OVERALL:')
        for i, (dist, crop, row) in enumerate(all_distances[:5]):
            print(f'  #{i+1} {crop} (distance: {dist:.1f}):')
            print(f'    N={row["N"]}, P={row["P"]}, K={row["K"]}, Temp={row["temperature"]}, Humidity={row["humidity"]}, pH={row["ph"]}, Rainfall={row["rainfall"]}')

    # Check data quality issues
    print(f'\n⚠️  POTENTIAL DATA QUALITY ISSUES:')
    
    # Check for duplicate or very similar samples with different labels
    print('Checking for conflicting samples...')
    
    # Group by rounded values to find similar samples
    all_data_rounded = all_data.copy()
    for col in feature_cols:
        all_data_rounded[col + '_rounded'] = all_data_rounded[col].round(0)
    
    rounded_cols = [col + '_rounded' for col in feature_cols]
    grouped = all_data_rounded.groupby(rounded_cols)['label'].apply(list).reset_index()
    
    conflicts = 0
    for idx, row in grouped.iterrows():
        labels = row['label']
        if len(set(labels)) > 1:
            conflicts += 1
            if conflicts <= 3:  # Show first 3 conflicts
                print(f'  Conflict #{conflicts}: Same conditions -> {set(labels)}')
                condition_str = ', '.join([f'{col.replace("_rounded", "")}={row[col]}' for col in rounded_cols])
                print(f'    Conditions: {condition_str}')
    
    if conflicts > 3:
        print(f'  ... and {conflicts - 3} more conflicts')
    
    if conflicts == 0:
        print('  ✅ No obvious conflicts found')
    else:
        print(f'  ⚠️  Found {conflicts} potential conflicts in training data')

if __name__ == "__main__":
    analyze_data_distribution()
