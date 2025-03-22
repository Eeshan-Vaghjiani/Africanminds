import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    cols_to_drop = ['FileName', 'md5Hash']
    df = df.drop(columns=cols_to_drop)
    columns = ["Machine", "DebugSize", "NumberOfSections", "SizeOfStackReserve", "MajorOSVersion", "BitcoinAddresses"]
    for col in columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    df.drop_duplicates(keep='last')
    return df

def normalize_data(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'Benign']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def train_evaluate_model(df):
    X = df.iloc[:, 1:-1].values
    Y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    svm = SVC(kernel='rbf', random_state=0)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = svm.score(X_test, y_test)
    cm_svm = confusion_matrix(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    class_report = classification_report(y_test, y_pred)
    return accuracy, cm_svm, precision, recall, f1_score, class_report

def main():
    st.title("Ransomware Detection System")
    st.write("Upload your CSV file for ransomware detection")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = load_and_preprocess_data(uploaded_file)
            st.write("Preview of data (first 5 rows):")
            st.dataframe(df.head())
            
            if st.button("Analyze Ransomware"):
                with st.spinner("Training model and analyzing..."):
                    cleaned_df = normalize_data(df)
                    accuracy, cm_svm, precision, recall, f1_score, class_report = train_evaluate_model(cleaned_df)
                    
                    st.write("Accuracy:", accuracy)
                    st.write("Confusion Matrix:")
                    fig, ax = plt.subplots()
                    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="rocket", cbar=False)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix - Support Vector Machine')
                    st.pyplot(fig)
                    
                    st.write("Classification Report:")
                    st.text(class_report)
                    
                    st.write("Precision:", precision)
                    st.write("Recall:", recall)
                    st.write("F1 Score:", f1_score)

                    # Display additional metrics and sample readings
                    total_samples = len(df)
                    benign_samples = len(df[df['Benign'] == 1])
                    ransomware_samples = len(df[df['Benign'] == 0])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", total_samples)
                        st.metric("Benign Samples", benign_samples)
                        st.metric("Ransomware Samples", ransomware_samples)
                    
                    with col2:
                        st.metric("SVM Accuracy", f"{accuracy:.2%}")
                    
                    threat_percentage = np.mean(cm_svm[1])
                    with col3:
                        if threat_percentage > 0.2:
                            st.error(f"⚠️ Threat Detected! ({threat_percentage:.2%} of traffic flagged as suspicious)")
                        else:
                            st.success(f"✅ No Threat Detected ({threat_percentage:.2%} of traffic flagged as suspicious)")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please check your CSV file format and try again.")

if __name__ == "__main__":
    main()