import os
from datetime import datetime
from backend.ml_models.advanced_predictor import AdvancedPredictor
import pandas as pd
from tabulate import tabulate
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_predictions():
    try:
        # Initialize predictor
        predictor = AdvancedPredictor()
        
        # List of stocks to analyze
        stocks = [
            'AAPL',  # Apple
            'MSFT',  # Microsoft
            'GOOGL', # Google
            'AMZN',  # Amazon
            'NVDA',  # NVIDIA
            'META',  # Meta (Facebook)
            'TSLA'   # Tesla
        ]
        
        # Store results
        results = []
        
        print("\nStock Market Prediction Analysis")
        print("=" * 50)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for symbol in stocks:
            try:
                print(f"\nAnalyzing {symbol}...")
                prediction = predictor.predict_comprehensive(symbol)
                
                if prediction == predictor.default_response:
                    print(f"Warning: Could not generate prediction for {symbol}")
                    continue
                
                results.append({
                    'Symbol': symbol,
                    'Prediction': prediction['prediction'].upper(),
                    'Confidence': f"{prediction['confidence']}%",
                    'Next Day': f"${prediction['next_day']}",
                    'Return': f"{prediction['predicted_return']}%"
                })
                
                # Print detailed analysis
                print("\nTechnical Indicators:")
                for indicator, value in prediction['technical_indicators'].items():
                    print(f"  {indicator}: {value}")
                
                print("\nAnalysis Points:")
                for point in prediction['analysis']:
                    print(f"  • {point}")
                
                print("\n" + "-" * 50)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                traceback.print_exc()
                continue
        
        if not results:
            logger.error("No predictions were generated successfully")
            return
        
        # Create summary table
        print("\nSummary of Predictions:")
        print(tabulate(results, headers='keys', tablefmt='grid'))
        
        # Save results to CSV
        df = pd.DataFrame(results)
        filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        
    except Exception as e:
        logger.error(f"Fatal error in prediction script: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_predictions()
