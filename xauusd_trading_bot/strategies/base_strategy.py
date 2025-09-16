import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """Standardized strategy signal output"""
    signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    patterns_found: List[str] = None
    analysis: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.patterns_found is None:
            self.patterns_found = []
        if self.analysis is None:
            self.analysis = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str = None):
        """Initialize base strategy"""
        self.name = name or self.__class__.__name__
        self.enabled = True
        self.weight = 1.0
        self.params = {}
        self._last_signal = None
        self._performance_metrics = {
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0,
            'win_rate': 0.0
        }
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyze market data and generate trading signal
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with signal information
        """
        pass
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data"""
        if df is None or df.empty:
            logger.warning(f"{self.name}: Empty dataframe provided")
            return False
        
        # Essential OHLC columns
        essential_columns = ['open', 'high', 'low', 'close']
        missing_essential = [col for col in essential_columns if col not in df.columns]
        
        if missing_essential:
            logger.error(f"{self.name}: Missing essential columns: {missing_essential}")
            return False
            
        # Volume column (accept either volume or tick_volume)
        if 'volume' not in df.columns and 'tick_volume' not in df.columns:
            logger.error(f"{self.name}: Missing volume data (need either 'volume' or 'tick_volume')")
            return False
        
        if len(df) < self.get_minimum_bars():
            logger.warning(f"{self.name}: Insufficient data. Need at least {self.get_minimum_bars()} bars")
            return False
        
        return True
    
    def get_minimum_bars(self) -> int:
        """Get minimum number of bars required for analysis"""
        return 50  # Default, can be overridden
    
    def update_parameters(self, params: Dict) -> None:
        """Update strategy parameters"""
        self.params.update(params)
        logger.info(f"{self.name}: Parameters updated: {params}")
    
    def get_parameters(self) -> Dict:
        """Get current strategy parameters"""
        return self.params.copy()
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable strategy"""
        self.enabled = enabled
        logger.info(f"{self.name}: {'Enabled' if enabled else 'Disabled'}")
    
    def set_weight(self, weight: float) -> None:
        """Set strategy weight for signal combination"""
        self.weight = max(0.0, min(1.0, weight))
        logger.info(f"{self.name}: Weight set to {self.weight}")
    
    def record_performance(self, signal_result: str) -> None:
        """Record strategy performance"""
        self._performance_metrics['total_signals'] += 1
        
        if signal_result == 'win':
            self._performance_metrics['winning_signals'] += 1
        elif signal_result == 'loss':
            self._performance_metrics['losing_signals'] += 1
        
        if self._performance_metrics['total_signals'] > 0:
            self._performance_metrics['win_rate'] = (
                self._performance_metrics['winning_signals'] / 
                self._performance_metrics['total_signals']
            )
    
    def get_performance_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        return self._performance_metrics.copy()
    
    def _create_signal(
        self,
        signal: str,
        confidence: float,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        patterns: Optional[List[str]] = None,
        analysis: Optional[Dict] = None
    ) -> Dict:
        """Create standardized signal dictionary"""
        signal_obj = StrategySignal(
            signal=signal,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            patterns_found=patterns or [],
            analysis=analysis or {}
        )
        
        # Calculate risk-reward ratio
        if stop_loss and take_profit and entry_price:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            if risk > 0:
                signal_obj.risk_reward_ratio = reward / risk
        
        self._last_signal = signal_obj
        
        return {
            'strategy': self.name,
            'signal': signal_obj.signal,
            'confidence': signal_obj.confidence,
            'entry_price': signal_obj.entry_price,
            'stop_loss': signal_obj.stop_loss,
            'take_profit': signal_obj.take_profit,
            'risk_reward_ratio': signal_obj.risk_reward_ratio,
            'patterns': signal_obj.patterns_found,
            'analysis': signal_obj.analysis,
            'timestamp': signal_obj.timestamp.isoformat()
        }
    
    def _neutral_signal(self, reason: str = "No clear signal") -> Dict:
        """Create neutral signal"""
        return self._create_signal(
            signal='NEUTRAL',
            confidence=0.0,
            analysis={'reason': reason}
        )