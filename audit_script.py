import json
import pandas as pd
import numpy as np
import sys
import glob

print("Checking JSON metrics")
try:
    with open("results/scaled_inference_summary.json", "r") as f:
        metrics = json.load(f)
        print("JSON metrics keys:", metrics.keys())
        print("Model names:", metrics.get("models", {}).keys())
except Exception as e:
    print(e)

print("Checking CSV predictions")
try:
    df = pd.read_csv("results/scaled_predictions.csv")
    print("CSV Shape:", df.shape)
    for col in df.columns:
        if col.endswith('_pred'):
            y_pred = df[col]
            y_true = df['label']
            acc = np.mean(y_pred == y_true)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
            tpr = tp / (tp + fn) if (tp+fn) > 0 else 0
            print(f"{col}: Acc: {acc:.4f}, FPR: {fpr:.4f}, TPR: {tpr:.4f}")
except Exception as e:
    print(e)

