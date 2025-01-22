import sys
import os
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from backend.ml_models.ensemble_predictor import EnsemblePredictor
from backend.ml_models.model_validator import ModelValidator
from backend.data_collectors.stock_data_collector import StockDataCollector
from backend.data_collectors.news_collector import NewsCollector

async def test_model_accuracy():
    """Test the model's prediction accuracy on various stocks"""
    
    # Test stocks from different sectors
    test_stocks = [
        'AAPL',  # Technology
        'JPM',   # Finance
        'JNJ',   # Healthcare
        'XOM',   # Energy
        'WMT',   # Retail
    ]
    
    predictor = EnsemblePredictor()
    validator = ModelValidator()
    
    results = []
    
    print("\nTesting Model Accuracy...")
    print("=" * 50)
    
    for symbol in test_stocks:
        print(f"\nTesting {symbol}...")
        
        try:
            # Train model
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()
            
            train_result = await predictor.train_models(symbol, start_date, end_date)
            if train_result['status'] != 'success':
                print(f"Error training model for {symbol}: {train_result.get('message', 'Unknown error')}")
                continue
                
            # Validate model
            validation_result = await validator.validate_model(symbol, validation_period=30)
            if validation_result['status'] != 'success':
                print(f"Error validating model for {symbol}: {validation_result.get('message', 'Unknown error')}")
                continue
            
            metrics = validation_result['metrics']
            
            # Calculate accuracy
            accuracy = (1 - metrics['mape']) * 100  # Convert MAPE to accuracy
            hit_rate = metrics['hit_rate'] * 100    # Direction accuracy
            
            results.append({
                'symbol': symbol,
                'accuracy': accuracy,
                'hit_rate': hit_rate,
                'rmse': metrics['rmse'],
                'profit_factor': metrics['profit_factor']
            })
            
            print(f"Results for {symbol}:")
            print(f"  Price Accuracy: {accuracy:.2f}%")
            print(f"  Direction Accuracy: {hit_rate:.2f}%")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            
        except Exception as e:
            print(f"Error testing {symbol}: {str(e)}")
    
    if results:
        # Calculate average metrics
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_hit_rate = np.mean([r['hit_rate'] for r in results])
        avg_rmse = np.mean([r['rmse'] for r in results])
        avg_profit_factor = np.mean([r['profit_factor'] for r in results])
        
        print("\nOverall Results:")
        print("=" * 50)
        print(f"Average Price Accuracy: {avg_accuracy:.2f}%")
        print(f"Average Direction Accuracy: {avg_hit_rate:.2f}%")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Average Profit Factor: {avg_profit_factor:.2f}")
        
        if avg_accuracy >= 95:
            print("\n✅ Model meets the 95% accuracy target!")
        else:
            print("\n❌ Model does not meet the 95% accuracy target.")
            print("Recommendations for improvement:")
            print("1. Collect more historical data")
            print("2. Add more technical indicators")
            print("3. Fine-tune model hyperparameters")
            print("4. Consider adding more news sources")
            print("5. Implement additional feature engineering")

if __name__ == "__main__":
    asyncio.run(test_model_accuracy())
