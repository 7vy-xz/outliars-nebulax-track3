import pandas as pd
from pipeline import train_and_evaluate_pipeline

# 1. Load the test dataset
test_path = 'problem_statement/PS3/02_Datasets/Door/Test.csv'
test_df = pd.read_csv(test_path)

# 2. Run your training pipeline to get the model and features
df, kmeans, centroid, dt_model, rules_text, features, eval_metrics = train_and_evaluate_pipeline()

# 3. Generate predictions and ensure they are numeric (0 or 1) for calculation
X_test = test_df[features]
predictions = dt_model.predict(X_test)
numeric_preds = [0 if str(p).lower() in ['0', 'normal'] else 1 for p in predictions]
test_df['pred_num'] = numeric_preds

# 4. Group continuous rows into discrete Door Segments based on time gaps
def parse_time_to_seconds(t_str):
    parts = [int(x) for x in t_str.split('-')]
    d, h, m, s, ms = parts[2], parts[3], parts[4], parts[5], parts[6]
    return (d * 86400) + (h * 3600) + (m * 60) + s + (ms / 1000.0)

test_df['time_seconds'] = test_df['Datetime'].apply(parse_time_to_seconds)

# A time gap of > 1.5 seconds indicates a pause between door cycles.
test_df['new_segment'] = test_df['time_seconds'].diff() > 1.5
test_df['segment_id'] = test_df['new_segment'].cumsum()

# 5. Aggregate predictions for each distinct door cycle
results = []
for seg_id, group_df in test_df.groupby('segment_id'):
    start_t = group_df['Datetime'].iloc[0]
    end_t = group_df['Datetime'].iloc[-1]
    
    # Calculate the average of numeric predictions (safe for mean())
    abnormal_ratio = group_df['pred_num'].mean()
    
    # If 50% or more of the cycle is abnormal, flag the whole segment as 'Abnormal resistance'
    label = 'Abnormal resistance' if abnormal_ratio >= 0.50 else 'Normal'
    
    results.append({
        'start_time': start_t,
        'end_time': end_t,
        'prediction': label
    })

output_df = pd.DataFrame(results)

# 6. Save to door_predictions.csv
output_df.to_csv('door_predictions.csv', index=False)
print(f"Successfully generated door_predictions.csv with {len(output_df)} distinct door segments!")