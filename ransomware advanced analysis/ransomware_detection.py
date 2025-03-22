import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyClassifier
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    cols_to_drop = ['FileName', 'md5Hash']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    columns = ["Machine", "DebugSize", "NumberOfSections", "SizeOfStackReserve", "MajorOSVersion", "BitcoinAddresses"]
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
    df = df.drop_duplicates(keep='last')
    return df

def normalize_data(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'Benign']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def train_svm_model(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='rbf', random_state=0, probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = svm.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    class_report = classification_report(y_test, y_pred)
    return svm, accuracy, cm, precision, recall, f1_score, class_report

def train_lazy_models(X_train, X_test, y_train, y_test):
    # Select a subset of classifiers to avoid memory issues and long computation times
    classifiers = [
        'RandomForestClassifier', 'ExtraTreesClassifier', 'BaggingClassifier',
        'XGBClassifier', 'LGBMClassifier', 'DecisionTreeClassifier',
        'KNeighborsClassifier', 'AdaBoostClassifier'
    ]
    clf = LazyClassifier(predictions=True, verbose=0, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    return models, predictions

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket", cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    return fig

def main():
    st.title("Advanced Ransomware Detection System")
    st.write("Upload your CSV file for comprehensive ransomware detection using multiple ML models")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = load_and_preprocess_data(uploaded_file)
            st.write("Preview of data (first 5 rows):")
            st.dataframe(df.head())
            
            if 'Benign' not in df.columns:
                st.error("Error: 'Benign' column not found in the dataset")
                return
                
            if st.button("Analyze Ransomware"):
                with st.spinner("Training models and analyzing..."):
                    cleaned_df = normalize_data(df)
                    
                    # Prepare data
                    X = cleaned_df.drop('Benign', axis=1).values
                    y = cleaned_df['Benign'].values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                    
                    # Train SVM model
                    svm_model, svm_accuracy, svm_cm, svm_precision, svm_recall, svm_f1, svm_report = train_svm_model(
                        X_train, X_test, y_train, y_test)
                    
                    # Train multiple models with LazyPredict
                    lazy_models, predictions = train_lazy_models(X_train, X_test, y_train, y_test)
                    
                    # Display results
                    st.subheader("Model Performance Comparison")
                    
                    # SVM Results
                    st.write("### SVM Results")
                    st.write(f"Accuracy: {svm_accuracy:.4f}")
                    st.pyplot(plot_confusion_matrix(svm_cm, "Confusion Matrix - SVM"))
                    st.write("Classification Report:")
                    st.text(svm_report)
                    st.write(f"Precision: {svm_precision:.4f}")
                    st.write(f"Recall: {svm_recall:.4f}")
                    st.write(f"F1 Score: {svm_f1:.4f}")
                    
                    # LazyPredict Results
                    st.write("### Ensemble Model Results")
                    st.dataframe(lazy_models.sort_values(by='Accuracy', ascending=False))
                    
                    # Top 3 models visualization
                    top_models = lazy_models.sort_values(by='Accuracy', ascending=False).head(3)
                    st.write("### Top 3 Performing Models")
                    for model_name in top_models.index:
                        cm = confusion_matrix(y_test, predictions[model_name])
                        st.pyplot(plot_confusion_matrix(cm, f"Confusion Matrix - {model_name}"))
                    
                    # Summary Metrics
                    total_samples = len(df)
                    benign_samples = len(df[df['Benign'] == 1])
                    ransomware_samples = len(df[df['Benign'] == 0])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", total_samples)
                        st.metric("Benign Samples", benign_samples)
                        st.metric("Ransomware Samples", ransomware_samples)
                    
                    with col2:
                        st.metric("SVM Accuracy", f"{svm_accuracy:.2%}")
                        st.metric("Best Model Accuracy", f"{top_models['Accuracy'].iloc[0]:.2%}")
                    
                    # Threat Assessment
                    ensemble_threat_score = np.mean([svm_accuracy, top_models['Accuracy'].iloc[0]])
                    with col3:
                        if ensemble_threat_score < 0.9:  # Adjusted threshold based on ensemble
                            st.error(f"⚠️ Threat Detected! ({(1-ensemble_threat_score):.2%} suspicious activity)")
                        else:
                            st.success(f"✅ Low Threat Level ({(1-ensemble_threat_score):.2%} suspicious activity)")
                    
                    # Model Agreement
                    agreement = np.mean([predictions[model_name].equals(predictions[top_models.index[0]]) 
                                       for model_name in top_models.index[1:]])
                    st.write(f"Top Model Prediction Agreement: {agreement:.2%}")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please check your CSV file format and try again.")

if __name__ == "__main__":
    main()