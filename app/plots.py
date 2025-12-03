# app/plots.py
"""
Plotting helper functions that return matplotlib.figure objects.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

sns.set_style("whitegrid")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0,1], [0,1], '--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    plt.tight_layout()
    return fig

def plot_precision_recall(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(6, max(3, top_n*0.4)))
    y_pos = np.arange(len(top_feats))
    ax.barh(y_pos, top_vals[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances (Random Forest)")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df_numeric):
    corr = df_numeric.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    return fig
