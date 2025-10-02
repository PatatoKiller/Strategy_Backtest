# Strategy_Backtest

A lightweight, modular **Python backtesting framework** for systematic trading strategies.  
The package is designed with **object-oriented architecture** for flexibility, making it easy to implement, test, and compare different strategies, execution policies, and portfolio allocation rules.

---

## Features

- **Modular Design**  
  - `Strategy`: abstract base class for building rule-based or position-based strategies.  
  - `ExecutionPolicy`: flexible execution models (e.g., market orders, stop loss, trailing stop).  
  - `Portfolio`: trade/position tracking, NAV computation, and performance statistics.  
  - `Backtest`: orchestration of data, strategy, portfolio, and evaluation.  

- **Support for Multiple Strategy Types**  
  - Event-based strategies (with explicit entry/exit rules).  
  - Position-based strategies (with time-series of desired weights).  

- **Performance Analysis**  
  - NAV curve and drawdown plotting.  
  - Trade logs and PnL attribution.  
  - Risk metrics (Sharpe ratio, max drawdown, win rate, etc.).

- **Extensible**  
  - Add new strategies by inheriting from `Strategy`.  
  - Plug in different execution rules or portfolio allocation methods.  
  - Designed for research and prototyping of trading models.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/PatatoKiller/Strategy_Backtest.git
cd Strategy_Backtest
pip install -r requirements.txt
```
---

## Project Structure

```text
.
├── models/
│   ├── Strategy.py            # Abstract base class
│   ├── Portfolio.py           # Portfolio and trade tracking
│   ├── ExecutionPolicy.py     # Execution logic
│   ├── Backtest.py            # Backtest orchestration
│   ├── Strategy8.py           # Example strategies (8.1, 8.2)
│   └── ...
├── notebooks/                 # Example Jupyter notebooks
├── data/                      # Sample datasets
├── tests/                     # Unit tests
├── requirements.txt
└── README.md