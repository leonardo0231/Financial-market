import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
from functools import wraps

from ..utils.circuit_breaker import api_circuit_breaker
from ..utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class FinnhubClient:
    """Finnhub API client with rate limiting and caching"""
    
    def __init__(self, api_key: str, cache_manager: Optional[CacheManager] = None):
        """Initialize Finnhub client"""
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.cache_manager = cache_manager
        
        # Rate limiting: 150 req/min for basic plan
        self.rate_limit = 150
        self.rate_window = 60  # seconds
        self.request_times = []
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'X-Finnhub-Token': self.api_key
        })
    
    def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Remove old requests outside the window
        self.request_times = [t for t in self.request_times 
                            if current_time - t < self.rate_window]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.rate_limit:
            sleep_time = self.rate_window - (current_time - self.request_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(current_time)
    
    @api_circuit_breaker.decorator
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and circuit breaker"""
        self._rate_limit_check()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub API error: {e}")
            raise
    
    def get_forex_rates(self, symbol: str = "OANDA:XAU_USD") -> Optional[Dict]:
        """Get real-time forex rates"""
        # Check cache first
        if self.cache_manager:
            cache_key = f"finnhub:forex:{symbol}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
        
        try:
            data = self._make_request("quote", {"symbol": symbol})
            
            # Cache the result
            if self.cache_manager and data:
                self.cache_manager.set(f"finnhub:forex:{symbol}", data, ttl=10)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching forex rates for {symbol}: {e}")
            return None
    
    def get_market_news(self, category: str = "forex") -> List[Dict]:
        """Get market news"""
        # Check cache first
        if self.cache_manager:
            cache_key = f"finnhub:news:{category}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
        
        try:
            data = self._make_request("news", {
                "category": category,
                "minId": 0
            })
            
            # Filter for relevant news
            gold_keywords = ['gold', 'xau', 'precious metal', 'commodity', 'dollar', 'usd', 'fed', 'inflation']
            relevant_news = [
                news for news in data
                if any(keyword in news.get('headline', '').lower() or 
                      keyword in news.get('summary', '').lower() 
                      for keyword in gold_keywords)
            ]
            
            # Cache the result
            if self.cache_manager and relevant_news:
                self.cache_manager.set(f"finnhub:news:{category}", relevant_news, ttl=300)
            
            return relevant_news[:10]  # Return top 10 news items
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []
    
    def get_economic_calendar(self) -> List[Dict]:
        """Get economic calendar events"""
        # Check cache first
        if self.cache_manager:
            cache_key = "finnhub:economic_calendar"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
        
        try:
            # Get events for next 7 days
            from_date = datetime.now().strftime("%Y-%m-%d")
            to_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            
            data = self._make_request("calendar/economic", {
                "from": from_date,
                "to": to_date
            })
            
            # Filter for high impact events
            high_impact_events = [
                event for event in data.get('economicCalendar', [])
                if event.get('impact', 0) >= 2  # Medium to high impact
            ]
            
            # Cache the result
            if self.cache_manager and high_impact_events:
                self.cache_manager.set("finnhub:economic_calendar", high_impact_events, ttl=3600)
            
            return high_impact_events
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return []
    
    def get_market_sentiment(self, symbol: str = "XAUUSD") -> Optional[Dict]:
        """Get market sentiment indicators"""
        # Check cache first
        if self.cache_manager:
            cache_key = f"finnhub:sentiment:{symbol}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
        
        try:
            # Get news sentiment
            news = self.get_market_news()
            
            # Simple sentiment analysis based on news
            positive_words = ['rise', 'gain', 'up', 'high', 'bullish', 'strong', 'growth']
            negative_words = ['fall', 'drop', 'down', 'low', 'bearish', 'weak', 'decline']
            
            sentiment_score = 0
            for article in news:
                text = (article.get('headline', '') + ' ' + article.get('summary', '')).lower()
                sentiment_score += sum(1 for word in positive_words if word in text)
                sentiment_score -= sum(1 for word in negative_words if word in text)
            
            sentiment = {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'sentiment': 'bullish' if sentiment_score > 2 else 'bearish' if sentiment_score < -2 else 'neutral',
                'news_count': len(news),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            if self.cache_manager:
                self.cache_manager.set(f"finnhub:sentiment:{symbol}", sentiment, ttl=600)
            
            return sentiment
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return None