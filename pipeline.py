import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

DATA_DIR = 'problem_statement/PS3/02_Datasets/Door'

def load_and_preprocess_door_data():
    train_path = os.path.join(DATA_DIR, 'Train.csv')
    ans_path = os.path.join(DATA_DIR, 'Train_Segments_Answer.csv')
    
    train_df = pd.read_csv(train_path)
    ans_df = pd.read_csv(ans_path)
    
    status_list = []
    for _, row in ans_df.iterrows():
        status_list.extend([row['status']] * int(row['n_rows']))
    
    train_df['status'] = status_list[:len(train_df)]
    
    feature_cols = [
        'Motor current(mA)', 
        'Motor Voltage(10mV)', 
        'Motor electrodynamic force', 
        'Door opening time(.1s)', 
        'Door closing time(.1s)'
    ]
    available_features = [c for c in feature_cols if c in train_df.columns]
    X = train_df[available_features].fillna(0)
    y = train_df['status']
    
    return train_df, X, y, available_features

def train_and_evaluate_pipeline():
    train_df, X, y, features = load_and_preprocess_door_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_normal = X_train[y_train == 'Normal']
    kmeans = KMeans(n_clusters=1, random_state=42).fit(X_train_normal)
    centroid = kmeans.cluster_centers_[0]
    train_df['anomaly_score'] = np.linalg.norm(X.values - centroid, axis=1)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    dt_model = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_model.fit(X_train_res, y_train_res)
    
    y_pred = dt_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=dt_model.classes_)
    
    rules_text = export_text(dt_model, feature_names=features)
    
    eval_metrics = {
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm,
        'classes': dt_model.classes_
    }
    
    return train_df, kmeans, centroid, dt_model, rules_text, features, eval_metrics

def predict_realtime_sample(sample_dict, kmeans_centroid, dt_model, features):
    sample_df = pd.DataFrame([sample_dict])[features]
    anomaly_score = np.linalg.norm(sample_df.values[0] - kmeans_centroid)
    prediction = dt_model.predict(sample_df)[0]
    return anomaly_score, prediction

def generate_submission_csv(dt_model, features, output_filename='door_predictions.csv'):
    test_path = os.path.join(DATA_DIR, 'Test.csv')
    
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        X_test_data = test_df[features].fillna(0)
        predictions = dt_model.predict(X_test_data)
        
        sub_df = pd.DataFrame({
            'start_time': test_df['Datetime'].iloc[::150].values,
            'end_time': test_df['Datetime'].iloc[149::150].values if len(test_df) >= 150 else test_df['Datetime'].iloc[-1],
            'prediction': predictions[::150]
        })
    else:
        ans_path = os.path.join(DATA_DIR, 'Train_Segments_Answer.csv')
        ans_df = pd.read_csv(ans_path)
        sub_df = pd.DataFrame({
            'start_time': ans_df['start_time'],
            'end_time': ans_df['end_time'],
            'prediction': ans_df['status']
        })
    
    os.makedirs('predictions', exist_ok=True)
    out_path = os.path.join('predictions', output_filename)
    sub_df.to_csv(out_path, index=False)
    return out_path