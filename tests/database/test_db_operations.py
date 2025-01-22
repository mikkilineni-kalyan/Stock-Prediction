import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

from backend.database.models import (
    StockData, Feature, FeatureValue, Model, 
    Prediction, ModelPerformance, TradingSignal
)

class TestStockData:
    def test_insert_stock_data(self, db_session, sample_stock_data):
        """Test inserting stock data records"""
        for data in sample_stock_data:
            stock_data = StockData(**data)
            db_session.add(stock_data)
        db_session.commit()
        
        # Verify data was inserted
        records = db_session.query(StockData).all()
        assert len(records) == len(sample_stock_data)
        
        # Verify data integrity
        first_record = records[0]
        assert first_record.symbol == 'AAPL'
        assert isinstance(first_record.timestamp, datetime)
        assert isinstance(first_record.volume, float)
    
    def test_query_stock_data_by_symbol(self, db_session, sample_stock_data):
        """Test querying stock data by symbol"""
        # Insert test data
        for data in sample_stock_data:
            db_session.add(StockData(**data))
        db_session.commit()
        
        # Query by symbol
        results = db_session.query(StockData).filter_by(symbol='AAPL').all()
        assert len(results) == len(sample_stock_data)
    
    def test_query_stock_data_by_date_range(self, db_session, sample_stock_data):
        """Test querying stock data within a date range"""
        # Insert test data
        for data in sample_stock_data:
            db_session.add(StockData(**data))
        db_session.commit()
        
        # Query by date range
        start_date = datetime.utcnow() - timedelta(days=5)
        end_date = datetime.utcnow()
        
        results = db_session.query(StockData).filter(
            StockData.timestamp.between(start_date, end_date)
        ).all()
        
        assert len(results) == 6  # 5 days + current day

class TestFeatures:
    def test_create_feature(self, db_session, sample_feature_data):
        """Test creating a new feature"""
        feature = Feature(**sample_feature_data)
        db_session.add(feature)
        db_session.commit()
        
        # Verify feature was created
        saved_feature = db_session.query(Feature).first()
        assert saved_feature.name == sample_feature_data['name']
        assert saved_feature.parameters == sample_feature_data['parameters']
    
    def test_feature_unique_name(self, db_session, sample_feature_data):
        """Test that feature names must be unique"""
        # Create first feature
        feature1 = Feature(**sample_feature_data)
        db_session.add(feature1)
        db_session.commit()
        
        # Try to create second feature with same name
        feature2 = Feature(**sample_feature_data)
        db_session.add(feature2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()

class TestModels:
    def test_create_model(self, db_session, sample_model_data):
        """Test creating a new model"""
        model = Model(**sample_model_data)
        db_session.add(model)
        db_session.commit()
        
        # Verify model was created
        saved_model = db_session.query(Model).first()
        assert saved_model.name == sample_model_data['name']
        assert saved_model.parameters == sample_model_data['parameters']
    
    def test_model_predictions(self, db_session, sample_model_data, sample_prediction_data):
        """Test creating model predictions"""
        # Create model
        model = Model(**sample_model_data)
        db_session.add(model)
        db_session.commit()
        
        # Add predictions
        for pred_data in sample_prediction_data:
            prediction = Prediction(model_id=model.id, **pred_data)
            db_session.add(prediction)
        db_session.commit()
        
        # Verify predictions
        predictions = db_session.query(Prediction).filter_by(model_id=model.id).all()
        assert len(predictions) == len(sample_prediction_data)
    
    def test_model_performance_tracking(self, db_session, sample_model_data):
        """Test tracking model performance metrics"""
        # Create model
        model = Model(**sample_model_data)
        db_session.add(model)
        db_session.commit()
        
        # Add performance metrics
        metrics = [
            ('accuracy', 0.85),
            ('precision', 0.83),
            ('recall', 0.87)
        ]
        
        for metric_name, value in metrics:
            performance = ModelPerformance(
                model_id=model.id,
                metric_name=metric_name,
                metric_value=value,
                symbol='AAPL',
                window='daily'
            )
            db_session.add(performance)
        db_session.commit()
        
        # Verify metrics
        saved_metrics = db_session.query(ModelPerformance).filter_by(model_id=model.id).all()
        assert len(saved_metrics) == len(metrics)

class TestTradingSignals:
    def test_create_trading_signal(self, db_session):
        """Test creating trading signals"""
        signal_data = {
            'symbol': 'AAPL',
            'timestamp': datetime.utcnow(),
            'signal_type': 'buy',
            'strength': 0.85,
            'confidence': 0.9,
            'factors': {
                'technical': 0.8,
                'sentiment': 0.9,
                'fundamental': 0.85
            }
        }
        
        signal = TradingSignal(**signal_data)
        db_session.add(signal)
        db_session.commit()
        
        # Verify signal
        saved_signal = db_session.query(TradingSignal).first()
        assert saved_signal.symbol == signal_data['symbol']
        assert saved_signal.signal_type == signal_data['signal_type']
        assert saved_signal.factors == signal_data['factors']
    
    def test_query_recent_signals(self, db_session):
        """Test querying recent trading signals"""
        # Create multiple signals
        base_time = datetime.utcnow()
        signals = []
        for i in range(5):
            signal = TradingSignal(
                symbol='AAPL',
                timestamp=base_time - timedelta(hours=i),
                signal_type='buy' if i % 2 == 0 else 'sell',
                strength=0.8 + i * 0.02,
                confidence=0.85 + i * 0.01,
                factors={'technical': 0.8, 'sentiment': 0.9}
            )
            signals.append(signal)
        
        db_session.add_all(signals)
        db_session.commit()
        
        # Query last 3 signals
        recent_signals = db_session.query(TradingSignal)\
            .order_by(TradingSignal.timestamp.desc())\
            .limit(3)\
            .all()
        
        assert len(recent_signals) == 3
        assert recent_signals[0].timestamp > recent_signals[1].timestamp
