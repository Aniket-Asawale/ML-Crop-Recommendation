# Model Training Bug - Root Cause Analysis & Fix Guide

**Date:** October 7, 2025  
**Status:** 🔴 CRITICAL BUG IDENTIFIED

---

## 🔍 Bug Summary

**11 out of 16 models are stuck predicting the same crop for ALL inputs**, regardless of parameter values.

### Affected Models:
- ❌ AdaBoost → Always predicts: **muskmelon**
- ❌ Best Random Forest → Always predicts: **muskmelon**
- ❌ Gradient Boosting → Always predicts: **kidneybeans**
- ❌ LDA Best Model → Always predicts: **kidneybeans**
- ❌ LDA Full Model → Always predicts: **kidneybeans**
- ❌ Random Forest → Always predicts: **kidneybeans**
- ❌ SVM Best → Always predicts: **kidneybeans**
- ❌ SVM Linear → Always predicts: **kidneybeans**
- ❌ SVM Poly → Always predicts: **mothbeans**
- ❌ SVM RBF → Always predicts: **mothbeans**
- ❌ XGBoost → Always predicts: **kidneybeans**

### Working Models:
- ✅ Logistic Regression (only this one works correctly)

---

## 🎯 Root Cause Identified

### The Critical Warning:
```
UserWarning: X does not have valid feature names, but [Model] was fitted with feature names
```

This warning reveals the exact problem:

1. **During Training**: Models were fitted with **DataFrames** (which have feature names)
   ```python
   model.fit(X_train, y_train)  # X_train is a DataFrame
   ```

2. **During Prediction**: Models receive **numpy arrays** (which have NO feature names)
   ```python
   input_scaled = scaler.transform(input_data)  # Returns numpy array
   prediction = model.predict(input_scaled)  # Numpy array, no feature names!
   ```

3. **The Problem**: When sklearn models are trained on DataFrames, they store the feature names internally. During prediction, if they receive a numpy array, they can't verify the feature order is correct, leading to **feature order mismatch**.

---

## 📊 Detailed Analysis

### What Logistic Regression Does RIGHT:

```python
# Load data
X_train = train_data[feature_cols]  # DataFrame with feature names

# Scale features
X_train_scaled = scaler.transform(X_train)  # Returns numpy array

# Fit model with SCALED numpy array
lr_model.fit(X_train_scaled, y_train)  # ✅ Numpy array, no feature names stored
```

**Result**: Model doesn't store feature names, so it works fine with numpy arrays during prediction.

### What Other Notebooks Do WRONG:

#### Example from SVM Notebook (04_SVM_Classification.ipynb):

```python
# Load data
X_train = train_data[feature_cols]  # DataFrame with feature names

# NO SCALING STEP!

# Fit model with UNSCALED DataFrame
svm_linear.fit(X_train, y_train)  # ❌ DataFrame with feature names stored
```

**Problems**:
1. Model is trained on **unscaled data** (SVM requires scaling!)
2. Model stores **feature names** from DataFrame
3. During prediction, receives **numpy array** without feature names
4. Feature order mismatch causes wrong predictions

#### Example from Ensemble Methods (07_Ensemble_Methods.ipynb):

```python
# Load data
X_train = train_data[feature_cols]  # DataFrame with feature names

# NO SCALING STEP!

# Fit models with UNSCALED DataFrame
rf_model.fit(X_train, y_train)  # ❌ DataFrame, no scaling
gb_model.fit(X_train, y_train)  # ❌ DataFrame, no scaling
ada_model.fit(X_train, y_train)  # ❌ DataFrame, no scaling
xgb_model.fit(X_train, y_train)  # ❌ DataFrame, no scaling
```

**Same problems**: Unscaled data + feature name mismatch

---

## 🔧 The Fix

### Solution: Always fit models with SCALED numpy arrays

**Pattern to follow** (from working Logistic Regression notebook):

```python
# 1. Load preprocessed data
train_data = pd.read_csv('../data/processed/train.csv')
val_data = pd.read_csv('../data/processed/validation.csv')
test_data = pd.read_csv('../data/processed/test.csv')

# 2. Load scaler and label encoder
scaler = joblib.load('../data/processed/scaler.pkl')
label_encoder = joblib.load('../data/processed/label_encoder.pkl')

# 3. Extract features as DataFrame
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X_train = train_data[feature_cols]
X_val = val_data[feature_cols]
X_test = test_data[feature_cols]
y_train = train_data['label']
y_val = val_data['label']
y_test = test_data['label']

# 4. Scale features (returns numpy arrays)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. Fit model with SCALED numpy arrays
model.fit(X_train_scaled, y_train)  # ✅ CORRECT!

# 6. Predict with SCALED numpy arrays
y_pred = model.predict(X_val_scaled)  # ✅ CORRECT!
```

---

## 📝 Notebooks to Fix

### High Priority (Stuck Models):

1. **04_SVM_Classification.ipynb**
   - Lines ~115, 164, 213: Change `model.fit(X_train, y_train)` to `model.fit(X_train_scaled, y_train)`
   - Add scaling step before training
   - Update all predictions to use scaled data

2. **07_Ensemble_Methods.ipynb**
   - Lines ~139, 209, 278, 349: Change all `model.fit(X_train, y_train)` to `model.fit(X_train_scaled, y_train)`
   - Add scaling step before training
   - Update all predictions to use scaled data

3. **06_LDA_Analysis.ipynb**
   - Add scaling step
   - Change model.fit() to use scaled data
   - Update predictions

4. **08_CART_Decision_Trees.ipynb**
   - Note: Decision Trees don't require scaling, but should still use numpy arrays for consistency
   - Change `model.fit(X_train, y_train)` to `model.fit(X_train.values, y_train)` OR use scaled data
   - This removes feature names while keeping unscaled data (trees are scale-invariant)

### Medium Priority (Error Models):

5. **02_Linear_Regression.ipynb**
   - Error: "X has 7 features, but LinearRegression is expecting 6 features"
   - Check if model was trained on wrong feature set
   - Retrain with correct 7 features

6. **05_PCA_Analysis.ipynb**
   - Error: "X has 7 features, but LogisticRegression is expecting 5 features"
   - PCA reduces dimensions, so prediction pipeline needs to apply PCA first
   - Save PCA transformer and apply before prediction

---

## ✅ Verification Steps

After fixing each notebook:

1. **Delete old model file**:
   ```bash
   rm models/saved_models/[model_name].pkl
   ```

2. **Re-run the notebook** to retrain the model

3. **Test with diagnostic script**:
   ```bash
   python diagnose_model_bug.py
   ```

4. **Verify**:
   - Model should predict different crops for different inputs
   - No "feature names" warnings
   - Reasonable accuracy on test data

---

## 🎯 Success Criteria

- [ ] All models predict varied crops based on input parameters
- [ ] No models stuck on single crop prediction
- [ ] No "feature names" warnings during prediction
- [ ] All models achieve >80% accuracy on test data
- [ ] Streamlit app works correctly with any selected model

---

## 📌 Key Takeaways

1. **Always use scaled data** for models that require it (SVM, Neural Networks, etc.)
2. **Always fit models with numpy arrays**, not DataFrames, to avoid feature name issues
3. **Pattern**: `model.fit(X_train_scaled, y_train)` where `X_train_scaled` is a numpy array
4. **Consistency**: Use the same data type (numpy array) for both training and prediction
5. **Test thoroughly**: Use diverse test cases to ensure models aren't stuck

---

**Next Steps:**
1. Fix notebooks 04, 06, 07, 08 (stuck models)
2. Fix notebooks 02, 05 (error models)
3. Delete all broken model files
4. Retrain all models
5. Verify with diagnostic script
6. Test in Streamlit app

