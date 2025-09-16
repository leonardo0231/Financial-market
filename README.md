# XAU/USD AI Trading Bot

An advanced AI-powered trading bot for XAU/USD (Gold) trading on MetaTrader 5, built with n8n workflow automation and Python.

## Features

- **Three Professional Trading Strategies**:
  - Al Brooks Price Action
  - Linda Raschke Pattern Trading
  - ICT Smart Money Concepts

- **Flexible Strategy Management**:
  - Use strategies individually or in combination
  - Weighted, majority, or unanimous signal combination
  - Real-time strategy switching via Telegram

- **Advanced Risk Management**:
  - Dynamic position sizing based on account equity
  - Multiple risk tiers
  - Trailing stop loss
  - Maximum drawdown protection

- **n8n Workflow Automation**:
  - Visual workflow design
  - 70-80% of logic handled by n8n
  - Easy modification without code changes

- **Telegram Bot Control**:
  - Full control dashboard
  - Real-time notifications
  - Performance reports
  - Strategy management

## Architecture

The system uses a microservices architecture with:
- **Python**: Trading strategies and MT5 integration
- **Redis**: Message queue and caching
- **PostgreSQL**: Trade history and analytics
- **Grafana**: Performance monitoring

## Installation

### Prerequisites
- Docker and Docker Compose
- MetaTrader 5 terminal
- Valid broker account with API access

### Quick Start

### Option 1: Docker Compose (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/leonardo0231/Financial-market.git
cd Financial-market