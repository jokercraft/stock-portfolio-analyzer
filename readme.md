# Stock Portfolio Analyzer

A powerful Streamlit application that helps investors analyze and optimize stock portfolios using advanced financial optimization techniques.

![Stock Portfolio Analyzer](https://stock-portfolizer.streamlit.app/)

## Features

- **Stock Data Analysis**: Fetch and visualize stock data with price action, technical indicators, and buy/sell signals
- **Portfolio Optimization**: Apply Mean-Variance and Black-Litterman optimization methods
- **Technical Indicators**: Weekly MACD, EMAs, and other technical indicators to identify trading opportunities
- **Financial Ratio Analysis**: View critical valuation metrics like P/E, P/B, and EV/EBITDA ratios
- **EBITDA Valuation Model**: Estimate fair value of stocks based on EBITDA projections and historical multiples
- **Sharpe Ratio Maximization**: Optimize portfolio weights to maximize risk-adjusted returns
- **BİST (Borsa Istanbul) Support**: Special analysis capabilities for Turkish stocks

## Requirements

- Python 3.11+
- Required Python packages (see `pyproject.toml`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/stock-portfolio-analyzer.git
   cd stock-portfolio-analyzer
   ```

2. Install the required packages:
   ```bash
   pip install -e .
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. In the application:
   - Add stocks to your portfolio by entering their ticker symbols
   - Select the analysis period (1 year, 2 years, or 5 years)
   - Choose the optimization method (Mean-Variance or Black-Litterman)
   - Set the risk-free rate
   - For Black-Litterman optimization, add your market views and confidence levels
   - Click "Optimize Portfolio" to generate the optimal portfolio weights

## Architecture

The application consists of the following main components:

- `app.py`: Main Streamlit interface
- `stock_data.py`: Functions for fetching and processing stock data
- `portfolio.py`: Portfolio management and optimization logic
- `black_litterman.py`: Implementation of the Black-Litterman model
- `visualization.py`: Data visualization utilities
- `financial_data.py`: Financial data processing for BİST stocks

## Optimization Methods

### Mean-Variance Optimization

The classic Markowitz portfolio optimization, which aims to:
- Maximize the Sharpe ratio (excess return per unit of risk)
- Balance risk and return based on historical performance
- Apply diversification penalty to avoid over-concentration

### Black-Litterman Optimization

An advanced model that:
- Incorporates market equilibrium as a starting point
- Allows investors to express their views on specific assets
- Adjusts the expected returns based on confidence levels
- Uses market capitalization weights as a prior

## Example

```python
# Initialize a portfolio
portfolio = Portfolio(risk_free_rate=0.02)

# Add stocks
portfolio.add_stock("AAPL")
portfolio.add_stock("MSFT")
portfolio.add_stock("AMZN")

# Optimize the portfolio using Mean-Variance method
result = portfolio.optimize_portfolio(period='1y')

# Print the results
print(f"Optimal Weights: {result['weights']}")
print(f"Expected Annual Return: {result['annual_return']:.2%}")
print(f"Expected Annual Volatility: {result['annual_volatility']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
```

## License

MIT License

## Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) for stock data
- [Streamlit](https://streamlit.io/) for the interactive web interface
- [Plotly](https://plotly.com/) for interactive visualizations
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data processing