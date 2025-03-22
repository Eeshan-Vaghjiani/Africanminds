import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scapy.all import *
from scapy.utils import rdpcap

def load_pcap(file_path):
    # Read pcap file
    packets = rdpcap(file_path)
    
    # Extract features from packets
    data = []
    for packet in packets:
        if IP in packet and TCP in packet:
            features = {
                'ip_len': packet[IP].len,
                'ip_ttl': packet[IP].ttl,
                'tcp_sport': packet[TCP].sport,
                'tcp_dport': packet[TCP].dport,
                'tcp_flags': packet[TCP].flags,
                'tcp_window': packet[TCP].window
            }
            data.append(features)
    
    return pd.DataFrame(data)

def normalize_data(data):
    try:
        # Define the columns we want to use for prediction
        numeric_columns = [
            'frame.len', 'tcp.flags.syn', 'tcp.flags.rst', 
            'tcp.flags.push', 'tcp.flags.ack', 'ip.flags.mf', 
            'ip.flags.df', 'tcp.seq', 'tcp.ack', 
            'Packets', 'Bytes', 'TxPackets', 'TxBytes', 
            'RxPackets', 'RxBytes'
        ]
        
        # Select only numeric columns that exist in the data
        available_columns = [col for col in numeric_columns if col in data.columns]
        
        if len(available_columns) == 0:
            raise ValueError("No valid numeric columns found in the data")
        
        # Create a copy of the data with only numeric columns
        X = data[available_columns].copy()
        
        # Handle missing values if any
        X = X.fillna(0)
        
        # Normalize the data
        for column in X.columns:
            std = X[column].std()
            if std != 0:  # Avoid division by zero
                X[column] = (X[column] - X[column].mean()) / std
            else:
                X[column] = 0  # Set to 0 if standard deviation is 0
                
        return X
        
    except Exception as e:
        raise Exception(f"Error in normalize_data: {str(e)}")

def unlabbeled_data(data):
    # Return the features for prediction
    return data, None

def labbeled_data(data):
    # For training purposes, you might want to use the 'Label' column
    # But for prediction, we'll just return the features
    return data, None

def train_model():
    # Initialize models with appropriate parameters for DDoS detection
    iforest = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # Adjust based on expected percentage of anomalies
        random_state=42
    )
    
    svm_model = OneClassSVM(
        kernel='rbf',
        nu=0.1,  # Adjust based on expected percentage of anomalies
        gamma='scale'
    )
    
    return iforest, svm_model

def predict(iforest, svm_model, unlabelled_X, labelled_X):
    try:
        # Make predictions
        Y_pred_iforest = iforest.fit_predict(unlabelled_X)  # Fit and predict
        Y_pred_svm = svm_model.fit_predict(unlabelled_X)    # Fit and predict
        
        # Convert predictions to binary (1 for DDoS, 0 for normal)
        # IsolationForest returns -1 for anomalies and 1 for normal
        Y_pred_iforest = (Y_pred_iforest == -1).astype(int)
        # OneClassSVM returns -1 for anomalies and 1 for normal
        Y_pred_svm = (Y_pred_svm == -1).astype(int)
        
        # Combine predictions
        Y_pred_combined = np.logical_or(Y_pred_iforest == 1, Y_pred_svm == 1).astype(int)
        
        return Y_pred_iforest, Y_pred_svm, Y_pred_combined
    except Exception as e:
        raise Exception(f"Error in predict: {str(e)}")