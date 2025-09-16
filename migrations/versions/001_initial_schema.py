"""Initial schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2024-12-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial schema for trading bot"""
    
    # Create trades table
    op.create_table('trades',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('trade_id', sa.String(length=50), nullable=False),
        sa.Column('ticket', sa.String(length=50), nullable=True),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('signal_type', sa.String(length=10), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('stop_loss', sa.Float(), nullable=True),
        sa.Column('take_profit', sa.Float(), nullable=True),
        sa.Column('exit_price', sa.Float(), nullable=True),
        sa.Column('entry_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('exit_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('profit_loss', sa.Float(), nullable=True),
        sa.Column('profit_loss_pips', sa.Float(), nullable=True),
        sa.Column('commission', sa.Float(), nullable=True),
        sa.Column('swap', sa.Float(), nullable=True),
        sa.Column('strategy_name', sa.String(length=50), nullable=False),
        sa.Column('strategy_signal_strength', sa.Float(), nullable=True),
        sa.Column('strategy_confidence', sa.Float(), nullable=True),
        sa.Column('risk_percent', sa.Float(), nullable=True),
        sa.Column('risk_amount', sa.Float(), nullable=True),
        sa.Column('risk_reward_ratio', sa.Float(), nullable=True),
        sa.Column('market_volatility', sa.Float(), nullable=True),
        sa.Column('spread', sa.Float(), nullable=True),
        sa.Column('extra', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for trades
    op.create_index('idx_trades_symbol_time', 'trades', ['symbol', 'entry_time'])
    op.create_index('idx_trades_strategy_status', 'trades', ['strategy_name', 'status'])
    op.create_index('idx_trades_performance', 'trades', ['profit_loss', 'risk_reward_ratio'])
    op.create_index(op.f('ix_trades_ticket'), 'trades', ['ticket'], unique=True)
    op.create_index(op.f('ix_trades_trade_id'), 'trades', ['trade_id'], unique=True)
    op.create_index(op.f('ix_trades_symbol'), 'trades', ['symbol'])
    op.create_index(op.f('ix_trades_strategy_name'), 'trades', ['strategy_name'])
    op.create_index(op.f('ix_trades_entry_time'), 'trades', ['entry_time'])
    op.create_index(op.f('ix_trades_exit_time'), 'trades', ['exit_time'])
    op.create_index(op.f('ix_trades_status'), 'trades', ['status'])
    
    # Create signals table
    op.create_table('signals',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('signal_id', sa.String(length=100), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('signal_type', sa.String(length=10), nullable=False),
        sa.Column('strategy_name', sa.String(length=50), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('stop_loss', sa.Float(), nullable=False),
        sa.Column('take_profit', sa.Float(), nullable=False),
        sa.Column('market_condition', sa.String(length=50), nullable=True),
        sa.Column('volatility', sa.Float(), nullable=True),
        sa.Column('trend_direction', sa.String(length=20), nullable=True),
        sa.Column('executed', sa.Boolean(), nullable=True),
        sa.Column('execution_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.Column('suggested_volume', sa.Float(), nullable=True),
        sa.Column('risk_reward_ratio', sa.Float(), nullable=True),
        sa.Column('trade_id', sa.String(length=50), nullable=True),
        sa.Column('technical_indicators', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('extra', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['trade_id'], ['trades.trade_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for signals
    op.create_index('idx_signals_strategy_time', 'signals', ['strategy_name', 'created_at'])
    op.create_index('idx_signals_execution', 'signals', ['executed', 'execution_time'])
    op.create_index('idx_signals_strength', 'signals', ['strength', 'confidence'])
    op.create_index(op.f('ix_signals_signal_id'), 'signals', ['signal_id'], unique=True)
    op.create_index(op.f('ix_signals_symbol'), 'signals', ['symbol'])
    op.create_index(op.f('ix_signals_strategy_name'), 'signals', ['strategy_name'])
    op.create_index(op.f('ix_signals_executed'), 'signals', ['executed'])
    
    # Create performance table
    op.create_table('performance',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_type', sa.String(length=20), nullable=False),
        sa.Column('total_trades', sa.Integer(), nullable=True),
        sa.Column('winning_trades', sa.Integer(), nullable=True),
        sa.Column('losing_trades', sa.Integer(), nullable=True),
        sa.Column('win_rate', sa.Float(), nullable=True),
        sa.Column('gross_profit', sa.Float(), nullable=True),
        sa.Column('gross_loss', sa.Float(), nullable=True),
        sa.Column('net_profit', sa.Float(), nullable=True),
        sa.Column('profit_factor', sa.Float(), nullable=True),
        sa.Column('starting_balance', sa.Float(), nullable=False),
        sa.Column('ending_balance', sa.Float(), nullable=False),
        sa.Column('max_balance', sa.Float(), nullable=False),
        sa.Column('min_balance', sa.Float(), nullable=False),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('max_drawdown_percent', sa.Float(), nullable=True),
        sa.Column('avg_trade_return', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('sortino_ratio', sa.Float(), nullable=True),
        sa.Column('calmar_ratio', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date', 'period_type', name='uq_performance_date_period')
    )
    
    # Create indexes for performance
    op.create_index('idx_performance_period', 'performance', ['period_type', 'date'])
    op.create_index(op.f('ix_performance_date'), 'performance', ['date'])
    op.create_index(op.f('ix_performance_period_type'), 'performance', ['period_type'])


def downgrade() -> None:
    """Drop all tables"""
    op.drop_table('performance')
    op.drop_table('signals')
    op.drop_table('trades')