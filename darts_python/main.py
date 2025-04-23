from config import MODELS_CONFIG
from utils import generate_synthetic_data, save_model, load_model, evaluate_stability
from model_selection import rolling_cv
from visualization import plot_model_comparison
from darts.metrics import smape
import pandas as pd
import numpy as np

def main():
    # 1. Load and transform data
    series = generate_synthetic_data('2022-01-01', '2025-04-09')
    train, test = series[:-28], series[-28:]
    
    # 2. Find best models
    results = []
    best_models = {}
    predictions = {}
    
    for name, config in MODELS_CONFIG.items():
        try:
            # Find best parameters
            best_params, _ = rolling_cv(config['model_class'], config['grid'], train)
            
            # Train model with best parameters
            model = config['model_class'](**best_params, **config['kwargs'])
            model.fit(train)
            
            # Save model
            save_model(model, name)
            best_models[name] = model
            
            # Make predictions
            pred = model.predict(len(test))
            predictions[name] = pred
            
            # Evaluate stability
            stability_smapes, _ = evaluate_stability(model, series)
            
            # Store results
            results.append({
                'Model': name,
                'SMAPE': smape(test, pred),
                'Stability': np.std(stability_smapes),
                'Cost': config['inference_cost_per_hour']
            })
            
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # 5. Visualize results
    fig = plot_model_comparison(results_df, series, predictions)
    fig.savefig('model_comparison.png')
    
    # Print detailed results
    print("\nModel Performance Summary:")
    print(results_df.sort_values('SMAPE').to_string())

if __name__ == "__main__":
    main()