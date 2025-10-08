"""
Intelligent Agriculture - Crop Recommendation System
Streamlit Web Application

This application provides an interactive interface for crop recommendation
based on soil and environmental parameters using trained ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Load Models and Preprocessing Objects
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        # Define paths
        base_path = Path(__file__).parent
        models_path = base_path / 'models' / 'saved_models'
        data_path = base_path / 'data' / 'processed'

        # Load preprocessing objects
        scaler = joblib.load(data_path / 'scaler.pkl')
        label_encoder = joblib.load(data_path / 'label_encoder.pkl')

        # Load best performing models
        models = {}

        # Load all available trained models
        # All models have been retrained with scaled numpy arrays (no feature name issues)
        model_files = [
            ('Logistic Regression', models_path / 'logistic_regression.pkl'),
            ('SVM Linear', models_path / 'svm_linear_model.pkl'),
            ('SVM RBF', models_path / 'svm_rbf_model.pkl'),
            ('SVM Polynomial', models_path / 'svm_poly_model.pkl'),
            ('SVM Best', models_path / 'svm_best_model.pkl'),
            ('Random Forest', models_path / 'random_forest_model.pkl'),
            ('Random Forest Optimized', models_path / 'random_forest_optimized.pkl'),
            ('Best Random Forest', models_path / 'best_random_forest_model.pkl'),
            ('Gradient Boosting', models_path / 'gradient_boosting_model.pkl'),
            ('AdaBoost', models_path / 'adaboost_model.pkl'),
            ('XGBoost', models_path / 'xgboost_model.pkl'),
            ('XGBoost Optimized', models_path / 'xgboost_optimized.pkl'),
            ('Stacking Classifier', models_path / 'stacking_classifier.pkl'),
            ('Voting Classifier', models_path / 'voting_soft.pkl'),
        ]

        for name, path in model_files:
            if path.exists():
                models[name] = joblib.load(path)

        if not models:
            st.error("No trained models found! Please train models first.")
            return None, None, None, None, None

        # Load model performance data
        model_comparison = pd.read_csv(data_path / 'final_model_comparison.csv')
        feature_importance = pd.read_csv(data_path / 'feature_importance_comparison.csv')

        return models, scaler, label_encoder, model_comparison, feature_importance

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

# ============================================================================
# Helper Functions
# ============================================================================

def get_crop_info(crop_name):
    """Get information about recommended crop"""
    crop_info = {
        'rice': {
            'season': 'Kharif (June-November)',
            'water': 'High water requirement',
            'soil': 'Clayey loam soil',
            'temp': '20-35°C'
        },
        'chickpea': {
            'season': 'Rabi (October-March)',
            'water': 'Low water requirement',
            'soil': 'Well-drained loamy soil',
            'temp': '15-25°C'
        },
        'maize': {
            'season': 'Kharif & Rabi',
            'water': 'Moderate water requirement',
            'soil': 'Well-drained loamy soil',
            'temp': '18-27°C'
        },
        'cotton': {
            'season': 'Kharif (April-October)',
            'water': 'Moderate to high water',
            'soil': 'Black cotton soil',
            'temp': '21-30°C'
        },
        'sugarcane': {
            'season': 'Year-round',
            'water': 'Very high water requirement',
            'soil': 'Loamy soil',
            'temp': '20-35°C'
        },
    }

    return crop_info.get(crop_name.lower(), {
        'season': 'Varies by region',
        'water': 'Moderate',
        'soil': 'Well-drained soil',
        'temp': 'Moderate climate'
    })

def create_gauge_chart(value, title, min_val, max_val, optimal_range):
    """Create a gauge chart for parameter visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, optimal_range[0]], 'color': "lightgray"},
                {'range': optimal_range, 'color': "lightgreen"},
                {'range': [optimal_range[1], max_val], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ============================================================================
# Main Application
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🌾 Intelligent Agriculture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Crop Recommendation System using Machine Learning</div>', unsafe_allow_html=True)

    # Load models
    models, scaler, label_encoder, model_comparison, feature_importance = load_models()

    if models is None:
        st.stop()

    # Sidebar - Model Selection
    st.sidebar.header("⚙️ Configuration")
    selected_model = st.sidebar.selectbox(
        "Select ML Model",
        list(models.keys()),
        help="Choose the machine learning model for prediction"
    )

    # Warning about model compatibility
    if selected_model != 'Logistic Regression':
        st.sidebar.warning(
            "⚠️ Note: Some models may have limited prediction variety due to feature name compatibility issues. "
            "Logistic Regression is recommended for best results."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About")
    st.sidebar.info(
        "This system uses advanced machine learning algorithms to recommend "
        "the most suitable crop based on soil nutrients and environmental conditions."
    )

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🌱 Crop Recommendation", "📈 Parameter Analysis", "📊 ML Insights", "ℹ️ About"])

    # ========================================================================
    # Tab 1: Crop Recommendation
    # ========================================================================

    with tab1:
        st.header("Enter Soil and Environmental Parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💧 Soil Nutrients (kg/ha)")

            nitrogen = st.slider(
                "Nitrogen (N)",
                min_value=0,
                max_value=140,
                value=50,
                help="Nitrogen content in soil (kg/ha)"
            )

            phosphorus = st.slider(
                "Phosphorus (P)",
                min_value=5,
                max_value=145,
                value=50,
                help="Phosphorus content in soil (kg/ha)"
            )

            potassium = st.slider(
                "Potassium (K)",
                min_value=5,
                max_value=205,
                value=50,
                help="Potassium content in soil (kg/ha)"
            )

            ph = st.slider(
                "pH Level",
                min_value=3.5,
                max_value=9.5,
                value=6.5,
                step=0.1,
                help="Soil pH level (3.5-9.5)"
            )

        with col2:
            st.subheader("🌡️ Environmental Conditions")

            temperature = st.slider(
                "Temperature (°C)",
                min_value=8.0,
                max_value=44.0,
                value=25.0,
                step=0.1,
                help="Average temperature in Celsius"
            )

            humidity = st.slider(
                "Humidity (%)",
                min_value=14.0,
                max_value=100.0,
                value=70.0,
                step=0.1,
                help="Relative humidity percentage"
            )

            rainfall = st.slider(
                "Rainfall (mm)",
                min_value=20.0,
                max_value=300.0,
                value=100.0,
                step=1.0,
                help="Annual rainfall in millimeters"
            )

        st.markdown("---")

        # Predict button
        if st.button("🔍 Get Crop Recommendation", type="primary", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame({
                'N': [nitrogen],
                'P': [phosphorus],
                'K': [potassium],
                'temperature': [temperature],
                'humidity': [humidity],
                'ph': [ph],
                'rainfall': [rainfall]
            })

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Make prediction
            model = models[selected_model]
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled) if hasattr(model, 'predict_proba') else None

            # Decode prediction
            recommended_crop = label_encoder.inverse_transform(prediction)[0]

            # Display recommendation
            st.markdown(f'<div class="recommendation-box">🌾 Recommended Crop: {recommended_crop.upper()}</div>',
                       unsafe_allow_html=True)

            # Display confidence
            if prediction_proba is not None:
                confidence = np.max(prediction_proba) * 100
                st.metric("Prediction Confidence", f"{confidence:.2f}%")

            # Display crop information
            crop_info = get_crop_info(recommended_crop)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"**Season**<br>{crop_info['season']}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"**Water Need**<br>{crop_info['water']}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"**Soil Type**<br>{crop_info['soil']}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"**Temperature**<br>{crop_info['temp']}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Top 3 predictions
            if prediction_proba is not None:
                st.markdown("---")
                st.subheader("📊 Top 3 Crop Recommendations")

                top_3_idx = np.argsort(prediction_proba[0])[-3:][::-1]
                top_3_crops = label_encoder.inverse_transform(top_3_idx)
                top_3_proba = prediction_proba[0][top_3_idx] * 100

                chart_data = pd.DataFrame({
                    'Crop': top_3_crops,
                    'Confidence (%)': top_3_proba
                })

                fig = px.bar(chart_data, x='Crop', y='Confidence (%)',
                            color='Confidence (%)',
                            color_continuous_scale='Greens')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # Tab 2: Parameter Analysis
    # ========================================================================

    with tab2:
        st.header("📈 Input Parameter Analysis")
        st.write("Visual analysis of your input parameters against optimal ranges")

        # Create gauge charts
        col1, col2, col3 = st.columns(3)

        with col1:
            fig_n = create_gauge_chart(nitrogen, "Nitrogen (N)", 0, 140, [20, 80])
            st.plotly_chart(fig_n, use_container_width=True)

        with col2:
            fig_p = create_gauge_chart(phosphorus, "Phosphorus (P)", 5, 145, [20, 80])
            st.plotly_chart(fig_p, use_container_width=True)

        with col3:
            fig_k = create_gauge_chart(potassium, "Potassium (K)", 5, 205, [20, 100])
            st.plotly_chart(fig_k, use_container_width=True)

        col4, col5, col6 = st.columns(3)

        with col4:
            fig_temp = create_gauge_chart(temperature, "Temperature (°C)", 8, 44, [20, 30])
            st.plotly_chart(fig_temp, use_container_width=True)

        with col5:
            fig_hum = create_gauge_chart(humidity, "Humidity (%)", 14, 100, [60, 85])
            st.plotly_chart(fig_hum, use_container_width=True)

        with col6:
            fig_ph = create_gauge_chart(ph, "pH Level", 3.5, 9.5, [6.0, 7.5])
            st.plotly_chart(fig_ph, use_container_width=True)

        # Parameter summary
        st.markdown("---")
        st.subheader("📋 Parameter Summary")

        summary_data = pd.DataFrame({
            'Parameter': ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
            'Value': [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall],
            'Unit': ['kg/ha', 'kg/ha', 'kg/ha', '°C', '%', '', 'mm'],
            'Status': ['✅ Good' if 20 <= nitrogen <= 80 else '⚠️ Check',
                      '✅ Good' if 20 <= phosphorus <= 80 else '⚠️ Check',
                      '✅ Good' if 20 <= potassium <= 100 else '⚠️ Check',
                      '✅ Good' if 20 <= temperature <= 30 else '⚠️ Check',
                      '✅ Good' if 60 <= humidity <= 85 else '⚠️ Check',
                      '✅ Good' if 6.0 <= ph <= 7.5 else '⚠️ Check',
                      '✅ Good' if 50 <= rainfall <= 200 else '⚠️ Check']
        })

        st.dataframe(summary_data, use_container_width=True, hide_index=True)

    # ========================================================================
    # Tab 3: ML Insights
    # ========================================================================

    with tab3:
        st.header("📊 Machine Learning Insights")

        # Model Performance Comparison
        st.subheader("🏆 Model Performance Comparison")

        # Display performance table
        st.dataframe(
            model_comparison[['Model', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score', 'Train_Time']].style.format({
                'Test_Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1_Score': '{:.4f}',
                'Train_Time': '{:.2f}s'
            }).background_gradient(subset=['Test_Accuracy'], cmap='Greens'),
            use_container_width=True
        )

        # Accuracy comparison chart
        st.markdown("---")
        st.subheader("📈 Accuracy Comparison")

        fig_acc = px.bar(
            model_comparison,
            x='Model',
            y='Test_Accuracy',
            color='Test_Accuracy',
            color_continuous_scale='Greens',
            title='Model Test Accuracy Comparison',
            labels={'Test_Accuracy': 'Test Accuracy'}
        )
        fig_acc.update_layout(xaxis_tickangle=-45, height=500, showlegend=False)
        fig_acc.update_traces(text=model_comparison['Test_Accuracy'].apply(lambda x: f'{x:.2%}'), textposition='outside')
        st.plotly_chart(fig_acc, use_container_width=True)

        # Feature Importance
        st.markdown("---")
        st.subheader("🔍 Feature Importance Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_feat = px.bar(
                feature_importance,
                x='Feature',
                y='Average',
                color='Average',
                color_continuous_scale='Viridis',
                title='Average Feature Importance Across Models',
                labels={'Average': 'Importance Score'}
            )
            fig_feat.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            fig_feat.update_traces(text=feature_importance['Average'].apply(lambda x: f'{x:.3f}'), textposition='outside')
            st.plotly_chart(fig_feat, use_container_width=True)

        with col2:
            st.markdown("#### 📊 Importance Ranking")
            for idx, row in feature_importance.iterrows():
                percentage = row['Average'] * 100
                st.metric(
                    label=f"{idx+1}. {row['Feature'].upper()}",
                    value=f"{percentage:.1f}%"
                )

        # Detailed feature importance by model
        st.markdown("---")
        st.subheader("📊 Feature Importance by Model")

        fig_feat_detailed = px.bar(
            feature_importance.melt(id_vars='Feature', var_name='Model', value_name='Importance'),
            x='Feature',
            y='Importance',
            color='Model',
            barmode='group',
            title='Feature Importance Comparison Across Models'
        )
        fig_feat_detailed.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig_feat_detailed, use_container_width=True)

        # Training Time vs Accuracy
        st.markdown("---")
        st.subheader("⏱️ Training Time vs Accuracy Trade-off")

        fig_scatter = px.scatter(
            model_comparison,
            x='Train_Time',
            y='Test_Accuracy',
            size='Test_Accuracy',
            color='Model',
            hover_data=['Precision', 'Recall', 'F1_Score'],
            title='Model Efficiency: Training Time vs Accuracy',
            labels={'Train_Time': 'Training Time (seconds)', 'Test_Accuracy': 'Test Accuracy'}
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Model metrics radar chart
        st.markdown("---")
        st.subheader("🎯 Selected Model Performance Metrics")

        selected_model_data = model_comparison[model_comparison['Model'] == selected_model.replace(' Optimized', '').replace(' Classifier', '')]

        if len(selected_model_data) > 0:
            metrics_data = selected_model_data.iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Test Accuracy", f"{metrics_data['Test_Accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{metrics_data['Precision']:.2%}")
            with col3:
                st.metric("Recall", f"{metrics_data['Recall']:.2%}")
            with col4:
                st.metric("F1-Score", f"{metrics_data['F1_Score']:.2%}")

            # Additional metrics
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                st.metric("Cohen's Kappa", f"{metrics_data['Cohen_Kappa']:.4f}")
            with col6:
                st.metric("MCC", f"{metrics_data['MCC']:.4f}")
            with col7:
                st.metric("CV Mean", f"{metrics_data['CV_Mean']:.2%}")
            with col8:
                st.metric("Training Time", f"{metrics_data['Train_Time']:.2f}s")
        else:
            st.info(f"Detailed metrics for {selected_model} are not available in the comparison data.")

        # Key Insights
        st.markdown("---")
        st.subheader("💡 Key Insights")

        best_model = model_comparison.loc[model_comparison['Test_Accuracy'].idxmax()]
        fastest_model = model_comparison.loc[model_comparison['Train_Time'].idxmin()]
        most_important_feature = feature_importance.loc[feature_importance['Average'].idxmax()]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏆 Best Model</h4>
                <p><strong>{best_model['Model']}</strong></p>
                <p>Accuracy: {best_model['Test_Accuracy']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚡ Fastest Model</h4>
                <p><strong>{fastest_model['Model']}</strong></p>
                <p>Time: {fastest_model['Train_Time']:.2f}s</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔑 Most Important Feature</h4>
                <p><strong>{most_important_feature['Feature'].upper()}</strong></p>
                <p>Importance: {most_important_feature['Average']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================================
    # Tab 4: About
    # ========================================================================

    with tab4:
        st.header("ℹ️ About This System")

        st.markdown("""
        ### 🌾 Intelligent Agriculture - Crop Recommendation System

        This application uses **Machine Learning** to recommend the most suitable crop
        for cultivation based on soil nutrients and environmental conditions.

        #### 📊 Dataset Information
        - **Total Samples**: 2,200 agricultural data points
        - **Features**: 7 (N, P, K, Temperature, Humidity, pH, Rainfall)
        - **Crop Classes**: 22 different crops
        - **Data Split**: 70% Training, 15% Validation, 15% Testing

        #### 🤖 Machine Learning Models
        The system implements and compares multiple ML algorithms:

        1. **Logistic Regression** - Baseline linear model
        2. **Support Vector Machine (SVM)** - Kernel-based classification
        3. **Decision Trees (CART)** - Rule-based decision making
        4. **Random Forest** - Ensemble of decision trees
        5. **Gradient Boosting** - Sequential ensemble method
        6. **XGBoost** - Optimized gradient boosting
        7. **Voting Classifier** - Combines multiple models
        8. **Stacking Classifier** - Meta-learning ensemble

        #### 🎯 Model Performance
        - **Best Accuracy**: ~96% (Stacking Classifier)
        - **Average Accuracy**: ~94% across all models
        - **Precision**: >93% for most crop classes
        - **Recall**: >92% for most crop classes

        #### 🔬 Methodology
        1. **Data Preprocessing**: StandardScaler normalization, Label encoding
        2. **Feature Engineering**: PCA, LDA dimensionality reduction
        3. **Model Training**: Cross-validation, hyperparameter tuning
        4. **Ensemble Methods**: Voting, stacking, boosting
        5. **Model Evaluation**: Accuracy, precision, recall, F1-score

        #### 💡 Key Features
        - ✅ Real-time crop recommendations
        - ✅ Multiple ML model options
        - ✅ Confidence scores for predictions
        - ✅ Visual parameter analysis
        - ✅ Top 3 crop suggestions
        - ✅ Crop-specific cultivation information

        #### 🛠️ Technology Stack
        - **Frontend**: Streamlit
        - **ML Framework**: scikit-learn, XGBoost
        - **Data Processing**: pandas, NumPy
        - **Visualization**: Plotly, Matplotlib
        - **Deployment**: Streamlit Cloud

        #### 📚 Project Structure
        ```
        ML/
        ├── data/
        │   ├── raw/              # Original dataset
        │   └── processed/        # Preprocessed data
        ├── models/
        │   └── saved_models/     # Trained ML models
        ├── notebooks/            # 11 Jupyter notebooks
        ├── src/                  # Source code modules
        ├── app.py               # Streamlit application
        └── requirements.txt      # Dependencies
        ```

        #### 👨‍💻 Development
        This project was developed as part of an academic research initiative
        to apply machine learning techniques to precision agriculture.

        #### 📧 Contact
        For questions or feedback, please refer to the project repository.

        ---

        **Version**: 1.0.0
        **Last Updated**: 2025-10-07
        **License**: MIT
        """)

        st.markdown("---")
        st.info("💡 For detailed model performance metrics and visualizations, check the **📊 ML Insights** tab!")

# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    main()
