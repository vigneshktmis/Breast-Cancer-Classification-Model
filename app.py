import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Model Training Functions
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.feature_names

def clean_data(df):
    """Clean the dataset by removing null values and duplicates"""
    df_clean = df.dropna()
    df_clean = df_clean.drop_duplicates()
    return df_clean

def display_data_statistics(df):
    st.subheader("Data Statistics")
    st.write("Original Data Shape:", df.shape)
    
    null_count = df.isnull().sum().sum()
    st.write("Null Values Count:", null_count)
    
    duplicate_count = df.duplicated().sum()
    st.write("Duplicate Values Count:", duplicate_count)
    
    df_clean = clean_data(df)
    st.write("Data Shape After Cleaning:", df_clean.shape)
    
    st.subheader("Data Description")
    st.write(df_clean.describe())
    
    st.subheader("Data Visualizations")
    
    # Correlation heatmap
    st.write("Correlation Heatmap")
    fig = px.imshow(df_clean.corr(), 
                    color_continuous_scale='RdBu_r',
                    aspect='auto')
    st.plotly_chart(fig)
    
    return df_clean

def train_and_evaluate_models(X, y, test_size, random_state, scaler_type):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    if scaler_type == "Standard Scaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMax Scaler":
        scaler = MinMaxScaler()
    else:  # Robust Scaler
        scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate models using LazyPredict
    start_time = time.time()
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models_train, predictions_train = clf.fit(X_train_scaled, X_test_scaled, y_train, y_test)
    end_time = time.time()
    
    # Add time taken for each model
    models_train['Time Taken'] = end_time - start_time
    
    # Get the best model name and create a new instance
    best_model_name = models_train.index[0]
    
    # Create and train the best model based on its name
    if 'LogisticRegression' in best_model_name:
        best_model = LogisticRegression()
    elif 'RandomForestClassifier' in best_model_name:
        best_model = RandomForestClassifier()
    elif 'GradientBoostingClassifier' in best_model_name:
        best_model = GradientBoostingClassifier()
    elif 'SVC' in best_model_name:
        best_model = SVC(probability=True)
    elif 'BernoulliNB' in best_model_name:
        best_model = BernoulliNB()
    elif 'GaussianNB' in best_model_name:
        best_model = GaussianNB()
    elif 'MultinomialNB' in best_model_name:
        best_model = MultinomialNB()
    else:
        best_model = LogisticRegression()
    
    # Train the best model
    best_model.fit(X_train_scaled, y_train)
    
    # Save the best model and scaler
    joblib.dump(best_model, 'best_model.joblib')
    joblib.dump(scaler, 'best_scaler.joblib')
    
    # Save model name for reference
    with open('best_model_name.txt', 'w') as f:
        f.write(best_model_name)
    
    return models_train, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_current_model():
    # Check if best model exists
    if os.path.exists('best_model.joblib') and os.path.exists('best_scaler.joblib'):
        model = joblib.load('best_model.joblib')
        scaler = joblib.load('best_scaler.joblib')
        
        # Get model name if available
        model_name = "Best Model"
        if os.path.exists('best_model_name.txt'):
            with open('best_model_name.txt', 'r') as f:
                model_name = f.read().strip()
    else:
        # Use default LogisticRegression model
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train default model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save default model and scaler
        joblib.dump(model, 'best_model.joblib')
        joblib.dump(scaler, 'best_scaler.joblib')
        model_name = "LogisticRegression"
        
        # Save model name
        with open('best_model_name.txt', 'w') as f:
            f.write(model_name)
    
    return model, scaler

def main():
    st.set_page_config(page_title="Breast Cancer Classification", layout="wide")
    
    st.title("🔬 Breast Cancer Classification Model")
    st.markdown("""
    This application uses machine learning to classify breast cancer cases as malignant or benign.
    Use the sample data or enter values manually to get predictions.
    """)
    
    # Load the dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    feature_names = data.feature_names
    
    # Get current model and scaler
    model, scaler = get_current_model()
    
    # Sidebar
    st.sidebar.header("Options")
    option = st.sidebar.radio(
        "Choose option:",
        ["Data", "Train & Test", "Predict"]
    )
    
    if option == "Data":
        data_option = st.radio(
            "Select data view:",
            ["Raw Data", "Statistics"]
        )
        
        if data_option == "Raw Data":
            st.subheader("Raw Data (First 15 rows)")
            st.dataframe(df.head(15))
        else:  # Statistics
            df_clean = display_data_statistics(df)
            
    elif option == "Train & Test":
        st.subheader("Model Training and Testing")
        
        # Clean the data first
        df_clean = clean_data(df)
        st.info(f"Using cleaned dataset with shape: {df_clean.shape}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            # Display train and test shapes based on test size
            X = df_clean.drop('target', axis=1)
            y = df_clean['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.info(f"Train Shape: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            st.info(f"Test Shape: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            
            random_state = st.number_input("Random State", 0, 100, 42)
            scaler_type = st.selectbox(
                "Select Scaler",
                ["Standard Scaler", "MinMax Scaler", "Robust Scaler"]
            )
        
        if st.button("Train Models"):
            X = df_clean.drop('target', axis=1)
            y = df_clean['target']
            
            with st.spinner("Training models... This may take a few minutes."):
                models_df, X_train_scaled, X_test_scaled, y_train, y_test, new_scaler = train_and_evaluate_models(
                    X, y, test_size, random_state, scaler_type
                )
            
            st.success("Training completed! The best performing model has been saved and will be used for predictions.")
            
            # Display results
            st.subheader("Model Performance Comparison")
            
            # Calculate additional metrics for each model
            metrics_data = []
            for model_name in models_df.index:
                # Create and train the model
                if 'LogisticRegression' in model_name:
                    model = LogisticRegression()
                elif 'RandomForestClassifier' in model_name:
                    model = RandomForestClassifier()
                elif 'GradientBoostingClassifier' in model_name:
                    model = GradientBoostingClassifier()
                elif 'SVC' in model_name:
                    model = SVC(probability=True)
                elif 'BernoulliNB' in model_name:
                    model = BernoulliNB()
                elif 'GaussianNB' in model_name:
                    model = GaussianNB()
                elif 'MultinomialNB' in model_name:
                    model = MultinomialNB()
                else:
                    continue
                
                # Train the model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Calculate metrics
                metrics = {
                    'Model': model_name,
                    'Accuracy': balanced_accuracy_score(y_test, y_pred),
                    'F1 Score': f1_score(y_test, y_pred),
                    'ROC AUC': roc_auc_score(y_test, y_pred_proba[:, 1]),
                    'Time Taken': models_df.loc[model_name, 'Time Taken']
                }
                metrics_data.append(metrics)
            
            # Create DataFrame with all metrics
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df = metrics_df.set_index('Model')
            
            # Display metrics with formatting
            st.dataframe(
                metrics_df.style
                .background_gradient(cmap='RdYlGn', subset=['Accuracy', 'F1 Score', 'ROC AUC'])
                .format({
                    'Accuracy': '{:.4f}',
                    'F1 Score': '{:.4f}',
                    'ROC AUC': '{:.4f}',
                    'Time Taken': '{:.2f} sec'
                })
            )
            
            # Plot performance metrics
            fig = go.Figure()
            metrics = ['Accuracy', 'F1 Score', 'ROC AUC']
            colors = ['blue', 'green', 'red']
            
            for metric, color in zip(metrics, colors):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=metrics_df.index,
                    y=metrics_df[metric],
                    marker_color=color,
                    opacity=0.7
                ))
            
            fig.update_layout(
                title="Model Performance Metrics Comparison",
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig)
            
            # Display the best model name
            with open('best_model_name.txt', 'r') as f:
                best_model_name = f.read().strip()
            st.success(f"Best performing model: {best_model_name}")
            
            # Display top 3 models
            st.subheader("Top 3 Performing Models")
            top_3_models = metrics_df.nlargest(3, 'Accuracy')
            for i, (model_name, metrics) in enumerate(top_3_models.iterrows(), 1):
                st.info(f"{i}. {model_name}")
                st.write(f"   Accuracy: {metrics['Accuracy']:.4f}")
                st.write(f"   F1 Score: {metrics['F1 Score']:.4f}")
                st.write(f"   ROC AUC: {metrics['ROC AUC']:.4f}")
                st.write("---")
    
    else:  # Predict option
        st.subheader("Make Prediction")
        
        # Prediction method selection
        pred_method = st.radio(
            "Choose prediction method:",
            ["Manual Input", "Sample from Dataset", "Bulk Prediction (CSV)"]
        )
        
        try:
            model = joblib.load('best_model.joblib')
            scaler = joblib.load('best_scaler.joblib')
            with open('best_model_name.txt', 'r') as f:
                model_name = f.read().strip()
            st.info(f"Using model: {model_name}")
        except:
            st.error("Error: Model file not found. Please train a model first using the Train & Test option.")
            return
        
        if pred_method == "Manual Input":
            st.subheader("Enter Feature Values")
            
            # Get feature ranges
            feature_ranges = {}
            for feature in feature_names:
                feature_ranges[feature] = {
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'mean': df[feature].mean()
                }
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            input_data = {}
            
            for i, feature in enumerate(feature_names):
                with col1 if i < len(feature_names)/2 else col2:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=float(feature_ranges[feature]['min']),
                        max_value=float(feature_ranges[feature]['max']),
                        value=float(feature_ranges[feature]['min']),
                        help=f"Range: {feature_ranges[feature]['min']:.2f} to {feature_ranges[feature]['max']:.2f}"
                    )
            
            input_df = pd.DataFrame([input_data])
        
        elif pred_method == "Sample from Dataset":
            st.subheader("Select Sample from Dataset")
            
            # Allow both slider and direct input for row selection
            col1, col2 = st.columns(2)
            
            with col1:
                row_index = st.number_input(
                    "Enter row number",
                    min_value=0,
                    max_value=len(df)-1,
                    value=0
                )
            
            with col2:
                row_index = st.slider(
                    "Or use slider to select row",
                    0,
                    len(df)-1,
                    int(row_index)
                )
            
            # Display selected row data
            input_df = df.iloc[[row_index]].drop('target', axis=1)
            st.write("Selected data:")
            st.dataframe(input_df)
            
            # Display actual target value
            actual_target = "Malignant" if df.iloc[row_index]['target'] == 1 else "Benign"
            st.info(f"Actual diagnosis: {actual_target}")
        
        else:  # Bulk Prediction
            st.subheader("Bulk Prediction using CSV")
            st.info("Upload a CSV file containing breast cancer features. The file should have the same column names as the training data.")
            
            # Display sample format
            st.write("Sample format:")
            sample_df = df.drop('target', axis=1).head(1)
            st.dataframe(sample_df)
            
            # File uploader
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Read the CSV file
                    input_df = pd.read_csv(uploaded_file)
                    
                    # Verify columns
                    missing_cols = set(feature_names) - set(input_df.columns)
                    if missing_cols:
                        st.error(f"Missing columns in CSV: {missing_cols}")
                        return
                    
                    # Display the uploaded data
                    st.write("Uploaded data:")
                    st.dataframe(input_df)
                    
                    # Make predictions
                    if st.button("Predict"):
                        try:
                            # Scale the input data
                            input_scaled = scaler.transform(input_df)
                            
                            # Make predictions
                            predictions = model.predict(input_scaled)
                            probabilities = model.predict_proba(input_scaled)
                            
                            # Create results DataFrame
                            results_df = pd.DataFrame({
                                'Prediction': ['Malignant' if p == 1 else 'Benign' for p in predictions],
                                'Confidence': [prob[1] if p == 1 else prob[0] for p, prob in zip(predictions, probabilities)],
                                'Malignant Probability': probabilities[:, 1],
                                'Benign Probability': probabilities[:, 0]
                            })
                            
                            # Display results
                            st.subheader("Prediction Results")
                            
                            # Create a color-coded DataFrame for better visualization
                            def color_prediction(val):
                                if val == 'Malignant':
                                    return 'background-color: #ffcccc'
                                return 'background-color: #ccffcc'
                            
                            def color_confidence(val):
                                # Get the prediction value from the same row
                                prediction = results_df.loc[results_df['Confidence'] == val, 'Prediction'].iloc[0] if len(results_df.loc[results_df['Confidence'] == val, 'Prediction']) > 0 else None
                                
                                if val >= 0.9:
                                    return 'background-color: #ff9999' if prediction == 'Malignant' else 'background-color: #99ff99'
                                elif val >= 0.7:
                                    return 'background-color: #ffcccc' if prediction == 'Malignant' else 'background-color: #ccffcc'
                                return 'background-color: #ffeeee' if prediction == 'Malignant' else 'background-color: #eeffee'
                            
                            # Apply styling to the results DataFrame
                            styled_results = results_df.style.applymap(
                                color_prediction, subset=['Prediction']
                            ).applymap(
                                color_confidence, subset=['Confidence']
                            ).format({
                                'Confidence': '{:.2%}',
                                'Malignant Probability': '{:.2%}',
                                'Benign Probability': '{:.2%}'
                            })
                            
                            st.dataframe(styled_results)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="prediction_results.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            st.subheader("Summary Statistics")
                            total = len(predictions)
                            malignant = sum(predictions == 1)
                            benign = sum(predictions == 0)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Samples", total)
                            with col2:
                                st.metric("Malignant", malignant, f"{(malignant/total)*100:.1f}%")
                            with col3:
                                st.metric("Benign", benign, f"{(benign/total)*100:.1f}%")
                            
                            # Additional insights
                            st.subheader("Insights and Recommendations")
                            
                            # High confidence predictions
                            high_confidence = results_df[results_df['Confidence'] >= 0.9]
                            high_confidence_malignant = high_confidence[high_confidence['Prediction'] == 'Malignant']
                            high_confidence_benign = high_confidence[high_confidence['Prediction'] == 'Benign']
                            
                            # Medium confidence predictions
                            medium_confidence = results_df[(results_df['Confidence'] >= 0.7) & (results_df['Confidence'] < 0.9)]
                            medium_confidence_malignant = medium_confidence[medium_confidence['Prediction'] == 'Malignant']
                            medium_confidence_benign = medium_confidence[medium_confidence['Prediction'] == 'Benign']
                            
                            # Low confidence predictions
                            low_confidence = results_df[results_df['Confidence'] < 0.7]
                            low_confidence_malignant = low_confidence[low_confidence['Prediction'] == 'Malignant']
                            low_confidence_benign = low_confidence[low_confidence['Prediction'] == 'Benign']
                            
                            # Display insights
                            st.write(f"**High Confidence Predictions ({len(high_confidence)} samples):**")
                            if len(high_confidence_malignant) > 0:
                                st.warning(f"- {len(high_confidence_malignant)} samples predicted as Malignant with high confidence")
                                st.info("Recommendation: Immediate medical consultation is strongly recommended for these cases.")
                            if len(high_confidence_benign) > 0:
                                st.success(f"- {len(high_confidence_benign)} samples predicted as Benign with high confidence")
                                st.info("Recommendation: Regular monitoring as per healthcare provider's recommendations.")
                            
                            # Medium confidence insights
                            if len(medium_confidence) > 0:
                                st.warning(f"**Medium Confidence Predictions ({len(medium_confidence)} samples):**")
                                if len(medium_confidence_malignant) > 0:
                                    st.warning(f"- {len(medium_confidence_malignant)} samples predicted as Malignant with medium confidence")
                                    st.info("Recommendation: Medical consultation is recommended for further evaluation.")
                                if len(medium_confidence_benign) > 0:
                                    st.success(f"- {len(medium_confidence_benign)} samples predicted as Benign with medium confidence")
                                    st.info("Recommendation: Follow-up appointment in 6-12 months to monitor any changes.")
                            
                            # Low confidence insights
                            if len(low_confidence) > 0:
                                st.error(f"**Low Confidence Predictions ({len(low_confidence)} samples):**")
                                if len(low_confidence_malignant) > 0:
                                    st.warning(f"- {len(low_confidence_malignant)} samples predicted as Malignant with low confidence")
                                    st.info("Recommendation: Additional diagnostic tests or a second opinion is strongly advised.")
                                if len(low_confidence_benign) > 0:
                                    st.warning(f"- {len(low_confidence_benign)} samples predicted as Benign with low confidence")
                                    st.info("Recommendation: Further evaluation may be needed to confirm the findings.")
                            
                            # Overall recommendations
                            st.subheader("Overall Recommendations")
                            
                            if malignant > 0:
                                malignant_percentage = (malignant/total)*100
                                if malignant_percentage > 30:
                                    st.error(f"**High proportion of Malignant predictions ({malignant_percentage:.1f}%)**")
                                    st.info("This dataset shows a concerning pattern with a high percentage of malignant cases. Consider a comprehensive review of these cases by medical specialists.")
                                elif malignant_percentage > 10:
                                    st.warning(f"**Moderate proportion of Malignant predictions ({malignant_percentage:.1f}%)**")
                                    st.info("This dataset shows a moderate percentage of malignant cases. Prioritize these cases for medical review.")
                                else:
                                    st.success(f"**Low proportion of Malignant predictions ({malignant_percentage:.1f}%)**")
                                    st.info("This dataset shows a relatively low percentage of malignant cases, which is generally positive.")
                            
                            # Visualization of confidence distribution
                            st.subheader("Data Visualizations")
                            
                            # Create a simple bar chart for prediction distribution
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=['Benign', 'Malignant'],
                                    y=[benign, malignant],
                                    marker_color=['green', 'red'],
                                    text=[f"{benign} ({benign/total*100:.1f}%)", f"{malignant} ({malignant/total*100:.1f}%)"],
                                    textposition='auto',
                                )
                            ])
                            fig.update_layout(
                                title="Prediction Distribution",
                                xaxis_title="Diagnosis",
                                yaxis_title="Number of Cases",
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig)
                            
                            # Create a pie chart for Benign and Malignant percentages
                            fig = go.Figure(data=[go.Pie(
                                labels=['Benign', 'Malignant'],
                                values=[benign, malignant],
                                hole=.4,
                                marker_colors=['green', 'red'],
                                textinfo='label+percent',
                                textposition='inside'
                            )])
                            fig.update_layout(
                                title="Benign vs Malignant Distribution",
                                height=400,
                                annotations=[dict(text='Diagnosis<br>Distribution', x=0.5, y=0.5, font_size=20, showarrow=False)]
                            )
                            st.plotly_chart(fig)
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    return
        
        # Make prediction for Manual Input and Sample from Dataset
        if pred_method != "Bulk Prediction (CSV)" and st.button("Predict"):
            try:
                # Scale the input data
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)
                probabilities = model.predict_proba(input_scaled)
                
                st.header("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Classification")
                    result = "Malignant" if prediction[0] == 1 else "Benign"
                    color = "red" if result == "Malignant" else "green"
                    confidence = probabilities[0][1] if result == "Malignant" else probabilities[0][0]
                    confidence_percent = confidence * 100
                    
                    st.markdown(f"<h2 style='color: {color};'>{result}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h3>Confidence: {confidence_percent:.2f}%</h3>", unsafe_allow_html=True)
                    
                    # Add suggestions based on prediction confidence
                    st.subheader("Suggestions")
                    if result == "Malignant":
                        if confidence_percent >= 90:
                            st.warning("High confidence of malignancy. Immediate medical consultation is strongly recommended.")
                            st.info("Consider scheduling a biopsy and additional diagnostic tests as soon as possible.")
                        elif confidence_percent >= 70:
                            st.warning("Moderate confidence of malignancy. Medical consultation is recommended.")
                            st.info("Schedule a follow-up appointment with your healthcare provider for further evaluation.")
                        else:
                            st.warning("Low to moderate confidence of malignancy. Further evaluation is needed.")
                            st.info("Consider a second opinion or additional diagnostic tests to confirm the findings.")
                    else:  # Benign
                        if confidence_percent >= 90:
                            st.success("High confidence of benign condition. Regular monitoring is recommended.")
                            st.info("Continue with regular check-ups as per your healthcare provider's recommendations.")
                        elif confidence_percent >= 70:
                            st.success("Moderate confidence of benign condition. Follow-up may be beneficial.")
                            st.info("Schedule a follow-up appointment in 6-12 months to monitor any changes.")
                        else:
                            st.warning("Low to moderate confidence of benign condition. Further evaluation may be needed.")
                            st.info("Consider additional tests or a second opinion to confirm the findings.")
                
                with col2:
                    st.subheader("Probability Scores")
                    prob_df = pd.DataFrame({
                        'Class': ['Benign', 'Malignant'],
                        'Probability': probabilities[0]
                    })
                    fig = go.Figure(data=[
                        go.Bar(
                            x=prob_df['Class'],
                            y=prob_df['Probability'],
                            marker_color=['green', 'red']
                        )
                    ])
                    fig.update_layout(
                        title="Prediction Probabilities",
                        xaxis_title="Class",
                        yaxis_title="Probability",
                        yaxis_range=[0, 1]
                    )
                    st.plotly_chart(fig)
                    
                    # Add a bar chart for prediction count
                    prediction_count = pd.DataFrame({
                        'Prediction': [result],
                        'Count': [1]
                    })
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=prediction_count['Prediction'],
                            y=prediction_count['Count'],
                            marker_color=['green' if result == 'Benign' else 'red'],
                            text=['1 (100%)'],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Prediction Count",
                        xaxis_title="Diagnosis",
                        yaxis_title="Count",
                        yaxis_range=[0, 1.5],
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main() 