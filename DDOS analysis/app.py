import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        df[col] = df[col].astype(str)
    return df

def normalize_data(df):
    """Normalize the numerical columns in the dataset using StandardScaler"""
    df_normalized = df.copy()
    
    # Convert IP addresses to numerical values
    if 'ip.src' in df.columns:
        df_normalized['ip.src'] = df_normalized['ip.src'].apply(lambda x: int(x.split('.')[-1]))
    if 'ip.dst' in df.columns:
        df_normalized['ip.dst'] = df_normalized['ip.dst'].apply(lambda x: int(x.split('.')[-1]))
    
    # Select numerical columns
    numerical_cols = df_normalized.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'Label' and 'time' not in col.lower()]
    
    # Use StandardScaler for faster normalization
    scaler = StandardScaler()
    df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])
    
    return df_normalized

def labbeled_data(df):
    """Prepare labeled data for training"""
    # Make sure 'Label' column exists and contains the target values
    if 'Label' not in df.columns:
        raise ValueError("DataFrame must contain a 'Label' column")
    
    # Select only numerical columns for features
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in feature_cols if col != 'Label' and 'time' not in col.lower()]
    
    # Separate features and labels
    X = df[feature_cols]  # Features
    y = df['Label']       # Target variable
    
    return X, y

def unlabbeled_data(df):
    """Prepare unlabeled data for training"""
    # Select only numerical columns for features
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in feature_cols if col != 'Label' and 'time' not in col.lower()]
    
    # Separate features and labels
    X = df[feature_cols]  # Features
    y = df['Label']       # Target variable
    return X, y

def train_svm(X_train, X_test, y_train, y_test):
    """Train SVM model and make predictions"""
    svm_model = svm.SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)
    Y_pred = svm_model.predict(X_test)
    return svm_model, Y_pred

def train_isolation_forest(X_train, X_test):
    """Train Isolation Forest model and make predictions"""
    iforest = IsolationForest(contamination=0.1, random_state=42)
    iforest.fit(X_train)
    # Convert predictions from {1: normal, -1: anomaly} to {0: normal, 1: anomaly}
    Y_pred = [(1 if x == -1 else 0) for x in iforest.predict(X_test)]
    return iforest, Y_pred

def predict_and_combine(iforest, svm_model, X_test_svm, y_test_svm, X_test_iforest, y_test_iforest):
    """Combine predictions from both models and calculate accuracies"""
    # Get predictions
    y_pred_svm = svm_model.predict(X_test_svm)
    y_pred_iforest = [(1 if x == -1 else 0) for x in iforest.predict(X_test_iforest)]
    
    # Combine predictions (simple OR operation)
    y_pred_combined = np.logical_or(y_pred_svm, y_pred_iforest).astype(int)
    
    # Calculate accuracies
    accuracy_svm = np.mean(y_pred_svm == y_test_svm)
    accuracy_iforest = np.mean(y_pred_iforest == y_test_iforest)
    accuracy_combined = np.mean(y_pred_combined == y_test_svm)
    
    return y_pred_combined, accuracy_iforest, accuracy_svm, accuracy_combined

def main():
    st.title("DDoS Detection System")
    st.write("Upload your CSV file for DDoS detection")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data with caching
            df = load_and_preprocess_data(uploaded_file)
            
            # Show sample of data
            st.write("Preview of data (first 5 rows):")
            st.dataframe(df.head())

            # Convert labels to binary
            df['Label'] = df['Label'].apply(lambda x: 1 if 'DDoS' in str(x) else 0)
            
            # Display data distribution
            st.write("Data Distribution:")
            st.write(f"Normal traffic: {len(df[df['Label'] == 0])} samples")
            st.write(f"DDoS traffic: {len(df[df['Label'] == 1])} samples")

            if st.button("Analyze for DDoS"):
                with st.spinner("Training models and analyzing..."):
                    # Sample the data if it's too large
                    sample_size = 10000  # Adjust this number based on your needs
                    if len(df) > sample_size:
                        df_sampled = pd.concat([
                            df[df['Label'] == 0].sample(n=sample_size//2, random_state=42),
                            df[df['Label'] == 1].sample(n=sample_size//2, random_state=42)
                        ])
                    else:
                        df_sampled = df

                    # Normalize the sampled data
                    cleaned_df = normalize_data(df_sampled)

                    # Prepare features and labels
                    # First separate features and labels before splitting
                    X = cleaned_df.select_dtypes(include=['float64', 'int64'])
                    X = X.drop('Label', axis=1, errors='ignore')  # Drop Label column if it exists
                    y = cleaned_df['Label']  # Keep labels separate

                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    # Train SVM
                    svm_model = svm.SVC(kernel='rbf', random_state=42)
                    svm_model.fit(X_train, y_train)
                    y_pred_svm = svm_model.predict(X_test)

                    # Train Isolation Forest
                    iforest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
                    iforest.fit(X_train)
                    y_pred_iforest = [(1 if x == -1 else 0) for x in iforest.predict(X_test)]

                    # Combine predictions
                    y_pred_combined = np.logical_or(y_pred_svm, y_pred_iforest).astype(int)

                    # Calculate accuracies
                    accuracy_svm = np.mean(y_pred_svm == y_test)
                    accuracy_iforest = np.mean(y_pred_iforest == y_test)
                    accuracy_combined = np.mean(y_pred_combined == y_test)

                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Isolation Forest Accuracy", f"{accuracy_iforest:.2%}")
                    with col2:
                        st.metric("SVM Accuracy", f"{accuracy_svm:.2%}")
                    with col3:
                        st.metric("Combined Model Accuracy", f"{accuracy_combined:.2%}")

                    # Plot confusion matrix
                    st.write("Confusion Matrix:")
                    fig, ax = plt.subplots()
                    sns.heatmap(confusion_matrix(y_test, y_pred_combined), 
                               annot=True, fmt="d", cmap="rocket", cbar=False,
                               xticklabels=['Normal', 'DDoS'],
                               yticklabels=['Normal', 'DDoS'])
                    st.pyplot(fig)

                    # Final verdict
                    threat_percentage = np.mean(y_pred_combined)
                    st.write("---")
                    if threat_percentage > 0.2:
                        st.error(f"⚠️ Threat Detected! ({threat_percentage:.2%} of traffic flagged as suspicious)")
                    else:
                        st.success(f"✅ No Threat Detected ({threat_percentage:.2%} of traffic flagged as suspicious)")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please check your CSV file format and try again.")

if __name__ == "__main__":
    main()
