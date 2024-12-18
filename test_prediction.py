import asyncio
import os
from dotenv import load_dotenv
from backend.ml_models.advanced_predictor import AdvancedPredictor

# Load environment variables
load_dotenv()
news_api_key = os.getenv('NEWS_API_KEY')

async def test_prediction():
    try:
        # Initialize predictor
        predictor = AdvancedPredictor(news_api_key=news_api_key)
        
        # Test stock symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in symbols:
            print(f"\nAnalyzing {symbol}...")
            
            # Get comprehensive prediction
            prediction = await predictor.predict_comprehensive(symbol)
            
            print(f"Prediction Results for {symbol}:")
            print("=" * 50)
            print(f"Overall Prediction: {prediction['prediction']}")
            print(f"Confidence: {prediction['confidence']:.2f}")
            
            # News Analysis
            print("\nNews Analysis:")
            print("-" * 30)
            for evidence in prediction['evidence'][:3]:  # Show top 3 news items
                print(f"- {evidence['title']}")
                print(f"  Impact: {evidence['impact_score']:.2f}")
                print(f"  Sentiment: {evidence['sentiment']}")
            
            # Technical Analysis
            print("\nTechnical Indicators:")
            print("-" * 30)
            for indicator, value in prediction['technical_indicators'].items():
                print(f"{indicator}: {value:.2f}")
            
            # Save models
            predictor.save_models()
            print("\nModels saved successfully!")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_prediction())
