# 🌾 Intelligent Agriculture - Crop Recommendation System
## Machine Learning for Smart Farming

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Ready%20to%20Implement-brightgreen)

## 📋 Project Overview

This project implements a comprehensive machine learning analysis for **intelligent agriculture** based on the research paper "Leveraging Machine Learning for Intelligent Agriculture" (Springer, 2025). The project demonstrates **10 major machine learning algorithms** for crop recommendation and agricultural decision-making.

### 🎯 Objectives
- Implement crop recommendation system using ML
- Compare 10+ ML algorithms for agricultural applications
- Analyze soil and climate factors affecting crop selection
- Demonstrate model training from scratch (no APIs)
- Provide actionable insights for farmers

---

## 📊 Research Paper

**Title:** "Leveraging Machine Learning for Intelligent Agriculture"

**Authors:** B.J. Sowmya, A.K. Meeradevi, S. Supreeth, et al.

**Publication:**
- **Journal:** Discover Internet of Things (Springer)
- **Volume:** 5, Article 33
- **Date:** March 26, 2025
- **DOI:** 10.1007/s43926-025-00132-6
- **Link:** https://link.springer.com/article/10.1007/s43926-025-00132-6
- **Status:** ✅ Scopus Indexed, ✅ Open Access

---

## 📈 Dataset Information

### **Crop Recommendation Dataset**

- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- **Size:** 2,200 instances
- **Features:** 7 numerical + 1 target (22 crop classes)
- **Status:** ✅ Verified, ✅ No Missing Values, ✅ Balanced

### **Dataset Features:**

| Feature | Description | Unit |
|---------|-------------|------|
| N | Nitrogen content in soil | kg/ha |
| P | Phosphorous content | kg/ha |
| K | Potassium content | kg/ha |
| temperature | Average temperature | °C |
| humidity | Relative humidity | % |
| ph | Soil pH value | 0-14 |
| rainfall | Average rainfall | mm |
| **label** | **Crop recommendation** | **Target** |

### **Target Crops (22 classes):**
Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

## 🔬 ML Algorithms Implemented

### ✅ All 10 Required Topics Covered (100%)

| # | Algorithm | Category | Application |
|---|-----------|----------|-------------|
| 1 | **Linear Regression** | Regression | Predict optimal NPK levels |
| 2 | **Logistic Regression** | Classification | Binary crop suitability |
| 3 | **SVM** | Classification | Multi-class crop recommendation |
| 4 | **Multivariate Linear Regression** | Regression | Multiple feature prediction |
| 5 | **Random Forest** | Ensemble (Bagging) | Crop recommendation |
| 6 | **AdaBoost/XGBoost** | Ensemble (Boosting) | Enhanced accuracy |
| 7 | **Decision Tree (CART)** | Classification/Regression | Rule-based recommendations |
| 8 | **K-Means Clustering** | Clustering | Group similar crops |
| 9 | **DBSCAN** | Clustering | Density-based crop groups |
| 10 | **PCA/LDA** | Dimensionality Reduction | Feature reduction |

### 🎁 Bonus Algorithms
- LightGBM (Advanced Boosting)
- Hierarchical Clustering
- Naive Bayes
- K-Nearest Neighbors (KNN)

---

## 📁 Project Structure

```
ML/
│
├── data/
│   ├── raw/                           # Raw dataset
│   │   └── crop_recommendation.csv    # Download from Kaggle
│   └── processed/                     # Preprocessed data (auto-generated)
│       ├── train.csv
│       ├── test.csv
│       └── validation.csv
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Linear_Regression.ipynb
│   ├── 03_Logistic_Regression.ipynb
│   ├── 04_SVM_Classification.ipynb
│   ├── 05_PCA_LDA_Analysis.ipynb
│   ├── 06_Ensemble_Methods.ipynb
│   ├── 07_CART_Decision_Trees.ipynb
│   ├── 08_Clustering_Analysis.ipynb
│   ├── 09_DBSCAN_Analysis.ipynb
│   ├── 10_Advanced_Techniques.ipynb
│   └── 11_Model_Comparison.ipynb
│
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data preprocessing utilities
│   ├── feature_engineering.py         # Feature creation
│   ├── evaluation.py                  # Model evaluation
│   ├── visualization.py               # Plotting functions
│   └── models/                        # Model implementations
│       ├── __init__.py
│       ├── linear_models.py
│       ├── svm_models.py
│       ├── ensemble_models.py
│       ├── tree_models.py
│       └── clustering_models.py
│
├── models/                            # Saved trained models
│   └── saved_models/
│
├── results/                           # Results and outputs
│   ├── figures/                       # Generated plots
│   ├── metrics/                       # Performance metrics
│   └── comparisons/                   # Model comparisons
│
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── RESEARCH_PAPER_AND_DATASET.md     # Paper and dataset details
└── problem_statement.md              # Original requirements
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook
- 5 GB free disk space

### Installation Steps

#### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- pandas, numpy, scipy (data processing)
- matplotlib, seaborn, plotly (visualization)
- scikit-learn (ML algorithms)
- xgboost, lightgbm (advanced algorithms)
- jupyter (notebook environment)

#### 3. Download Dataset

**Option A: Manual Download (Recommended)**
1. Visit: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Sign in to Kaggle
3. Click "Download"
4. Save `Crop_recommendation.csv` to `data/raw/`

**Option B: Kaggle API**
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d atharvaingle/crop-recommendation-dataset

# Extract to data/raw/
unzip crop-recommendation-dataset.zip -d data/raw/
```

#### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Navigate to `notebooks/` and start with `01_EDA_and_Preprocessing.ipynb`

---

## 📊 Implementation Roadmap

### Phase 1: Data Preparation ✅
- [ ] Download dataset from Kaggle
- [ ] Exploratory Data Analysis (EDA)
- [ ] Data cleaning and preprocessing
- [ ] Feature engineering
- [ ] Train-test-validation split

### Phase 2: Linear Models 📋
- [ ] Simple Linear Regression
- [ ] Multivariate Linear Regression
- [ ] Logistic Regression (binary classification)
- [ ] Model evaluation and visualization

### Phase 3: Support Vector Machines 📋
- [ ] Linear SVM
- [ ] RBF Kernel SVM
- [ ] Polynomial Kernel SVM
- [ ] Hyperparameter tuning (GridSearchCV)

### Phase 4: Dimensionality Reduction 📋
- [ ] Principal Component Analysis (PCA)
- [ ] Linear Discriminant Analysis (LDA)
- [ ] Visualize reduced dimensions
- [ ] Compare model performance

### Phase 5: Ensemble Methods 📋
- [ ] Random Forest (Bagging)
- [ ] AdaBoost (Boosting)
- [ ] Gradient Boosting
- [ ] XGBoost
- [ ] LightGBM
- [ ] Feature importance analysis

### Phase 6: Decision Trees (CART) 📋
- [ ] Decision Tree Classifier
- [ ] Tree visualization
- [ ] Pruning techniques
- [ ] Rule extraction

### Phase 7: Clustering Analysis 📋
- [ ] K-Means Clustering
- [ ] DBSCAN
- [ ] Hierarchical Clustering
- [ ] Cluster validation metrics

### Phase 8: Model Comparison 📋
- [ ] Compare all 10+ models
- [ ] Performance metrics dashboard
- [ ] Statistical significance tests
- [ ] Final recommendations

---

## 🔍 Evaluation Metrics

### Classification Metrics
- **Accuracy:** Overall correctness
- **Precision:** Positive predictive value
- **Recall:** Sensitivity
- **F1-Score:** Harmonic mean
- **ROC-AUC:** Area under curve
- **Confusion Matrix:** Detailed classification

### Regression Metrics
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R² Score:** Coefficient of determination

### Clustering Metrics
- **Silhouette Score:** Cluster cohesion
- **Davies-Bouldin Index:** Cluster separation
- **Calinski-Harabasz Index:** Variance ratio

---

## 📈 Expected Results

Based on literature review and similar studies:

| Model | Expected Accuracy | F1-Score |
|-------|------------------|----------|
| Logistic Regression | 88-92% | 0.87-0.91 |
| SVM (RBF) | 90-94% | 0.89-0.93 |
| Random Forest | 95-98% | 0.94-0.97 |
| XGBoost | 96-99% | 0.95-0.98 |
| Decision Tree | 85-90% | 0.84-0.89 |

**Goal:** Achieve 95%+ accuracy for crop recommendation

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **NumPy & Pandas:** Data manipulation
- **Scikit-learn:** ML algorithms
- **XGBoost & LightGBM:** Advanced ensemble methods
- **Matplotlib, Seaborn, Plotly:** Visualization
- **Jupyter Notebook:** Interactive development
- **SHAP:** Model interpretability

---

## 📚 Key Features

### 1. Comprehensive Preprocessing
- Automated missing value handling
- Feature scaling and normalization
- Data split with stratification
- Feature engineering pipelines

### 2. Multiple ML Algorithms
- 10+ algorithms implemented
- Hyperparameter tuning
- Cross-validation
- Model comparison

### 3. Rich Visualizations
- Feature distributions
- Correlation heatmaps
- Confusion matrices
- ROC curves
- Feature importance plots
- Cluster visualizations
- PCA/LDA plots

### 4. Model Interpretability
- Feature importance rankings
- SHAP values (if time permits)
- Decision tree rules
- Clustering insights

### 5. Practical Application
- Crop recommendation system
- Soil nutrient analysis
- Climate suitability assessment
- Farming decision support

---

## 🎯 Project Applications

### 1. Crop Recommendation
- Input: Soil NPK, climate data
- Output: Best crop to grow
- Accuracy: 95%+

### 2. Soil Analysis
- Identify soil deficiencies
- Recommend fertilizers
- Optimize crop selection

### 3. Climate Suitability
- Assess temperature/humidity/rainfall
- Recommend suitable crops
- Risk analysis

### 4. Crop Clustering
- Group crops with similar requirements
- Crop rotation planning
- Diversification strategies

---

## 📖 Research Paper Implementation

The project implements techniques from the research paper:

1. **Computer Vision** → Feature visualization
2. **Deep Learning** → Optional neural network implementation
3. **Machine Learning** → All 10 algorithms
4. **Smart Agriculture** → Practical crop recommendations
5. **IoT Integration** → Data-driven decision making

---

## 🔬 Academic Requirements Met

- ✅ **2025 Research Paper:** Yes (March 2025)
- ✅ **Scopus Indexed:** Yes (Discover Internet of Things)
- ✅ **Open Access:** Yes (Free to read)
- ✅ **DOI Available:** Yes (10.1007/s43926-025-00132-6)
- ✅ **Agriculture Related:** Yes (Crop Recommendation)
- ✅ **Dataset Available:** Yes (Kaggle, 2,200 samples)
- ✅ **Dataset Verified:** Yes (No missing values, balanced)
- ✅ **Covers 7-8 ML Topics:** Yes (Covers all 10 - 100%)
- ✅ **No API Usage:** Yes (All models trained from scratch)
- ✅ **AIML Related:** Yes (ML for Agriculture)

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

- **[Aniket-Asawale](https://github.com/Aniket-Asawale)**
- **Project:** ML Mini-Project (SEM 7)
- **Focus:** Intelligent Agriculture
- **Year:** 2025

---

## 🙏 Acknowledgments

- Research paper authors (Sowmya, B.J., et al.)
- Kaggle community for the dataset
- Springer Nature for open access publication
- Scikit-learn and open-source ML libraries

---

## 📧 Support

For questions or issues:
1. Check `RESEARCH_PAPER_AND_DATASET.md` for details
2. Review code comments and docstrings
3. Follow notebook cell explanations

---

## 📌 Project Status

**Current Status:** ✅ Working Models but few models are bugged (due to invalid feature conversion to numpy)

**Last Updated:** October 6, 2025

**Next Steps:**
1. Download dataset from Kaggle
2. Place in `data/raw/` directory
3. Install dependencies: `pip install -r requirements.txt`
4. Launch Jupyter: `jupyter notebook`
5. Start with notebook 01

---

## 🎉 Success Criteria

Project is complete when:
- ✅ All 11 notebooks run without errors
- ✅ All 10+ ML algorithms implemented
- ✅ Models trained and evaluated
- ✅ Visualizations generated (20+ plots)
- ✅ Final comparison report created
- ✅ Crop recommendation system working
- ✅ Results documented and analyzed

---

**⭐ If you find this project useful, please star the repository!**

**📢 Ready to revolutionize agriculture with Machine Learning? Let's begin!** 🌾🚜🤖

