import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_model_comparison(results_df, series, predictions, window_size=30):
    """Plot comprehensive model comparison."""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Model Performance Ranking
    plt.subplot(2, 2, 1)
    sns.barplot(data=results_df.sort_values('SMAPE'), x='SMAPE', y='Model')
    plt.title('Model Performance Ranking')
    
    # 2. Stability Analysis
    plt.subplot(2, 2, 2)
    sns.barplot(data=results_df.sort_values('Stability'), x='Stability', y='Model')
    plt.title('Model Stability (SMAPE Std Dev)')
    
    # 3. Forecasts vs Actual
    plt.subplot(2, 2, 3)
    series[-window_size:].plot(label='Actual', linewidth=6, color='black')
    for name, pred in predictions.items():
        pred.plot(label=name, alpha=0.7)
    plt.title('Forecasts vs Actual')
    plt.legend(bbox_to_anchor=(1.05, 1))
    
    # 4. Cost vs Accuracy
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['Cost'], results_df['SMAPE'])
    for _, row in results_df.iterrows():
        plt.annotate(row['Model'], (row['Cost'], row['SMAPE']))
    plt.xlabel('Cost ($/hour)')
    plt.ylabel('SMAPE')
    plt.title('Cost vs Accuracy Trade-off')
    
    plt.tight_layout()
    return fig