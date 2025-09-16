from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AnalysisResponse(BaseModel):
    signal: str
    strength: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: Optional[str] = None
    risk_parameters: Optional[Dict[str, Any]] = None

class SignalPayload(BaseModel):
    signal_id: str
    symbol: str
    signal_data: Dict[str, Any]
    created_at: str

class RiskCheckPayload(BaseModel):
    signal_id: str
    approved: bool
    risk_score: float
    risk_checks: Dict[str, Any]
    signal_data: Dict[str, Any]
    current_positions: int
    emergency_stop_active: bool
    decision_time: str
    reason: str

class TradeRequest(BaseModel):
    symbol: str = "XAUUSD"
    signal: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: float = 0.01

class TradeResult(BaseModel):
    execution_id: str
    symbol: str
    success: bool
    ticket: Optional[int] = None
    volume: Optional[float] = None
    price: Optional[float] = None
    error: Optional[str] = None
    retcode: Optional[int] = None
    executed_at: Optional[str] = None
    failed_at: Optional[str] = None
    message: Optional[str] = None