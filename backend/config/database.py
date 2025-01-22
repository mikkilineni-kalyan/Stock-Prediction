from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()

class Stock(Base):
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False)
    company_name = Column(String(255))
    sector = Column(String(100))
    industry = Column(String(100))
    last_updated = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    prices = relationship("StockPrice", back_populates="stock")
    news = relationship("NewsArticle", back_populates="stock")
    predictions = relationship("Prediction", back_populates="stock")

class StockPrice(Base):
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    adjusted_close = Column(Float)
    volume = Column(Integer)
    
    # Technical indicators
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    rsi_14 = Column(Float)
    macd = Column(Float)
    
    # Relationships
    stock = relationship("Stock", back_populates="prices")

class NewsArticle(Base):
    __tablename__ = 'news_articles'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    source = Column(String(100))
    url = Column(String(1000))
    published_at = Column(DateTime)
    sentiment_score = Column(Float)
    impact_score = Column(Float)  # 1-5 rating
    
    # Relationships
    stock = relationship("Stock", back_populates="news")

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_type = Column(String(50))  # hourly, daily, weekly
    predicted_price = Column(Float)
    confidence_score = Column(Float)
    actual_price = Column(Float, nullable=True)
    features_used = Column(Text)  # JSON string of features
    model_version = Column(String(50))
    accuracy = Column(Float, nullable=True)
    
    # Relationships
    stock = relationship("Stock", back_populates="predictions")

def init_db():
    """Initialize the database with tables"""
    # Get database URL from environment variable or use default
    database_url = os.getenv('DATABASE_URL', 'postgresql://localhost/stock_prediction')
    
    # Create engine
    engine = create_engine(database_url)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    Session = sessionmaker(bind=engine)
    return Session()

# Create session instance
session = init_db()
