import streamlit as st
import pandas as pd
from stock_data import StockData
from portfolio import Portfolio
from visualization import Visualizer

# Page configuration
st.set_page_config(
    page_title="Stock Portfolio Analyzer",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = Portfolio()
if 'analysis_period' not in st.session_state:
    st.session_state.analysis_period = '1y'
if 'optimization_method' not in st.session_state:
    st.session_state.optimization_method = 'mean_variance'
if 'views' not in st.session_state:
    st.session_state.views = {}
if 'view_confidences' not in st.session_state:
    st.session_state.view_confidences = {}
if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = 0.02  # Default 2%

# Header
st.title("📈 Stock Portfolio Analyzer")

# Sidebar
with st.sidebar:
    st.header("Portfolio Management")

    # Analysis Settings
    st.subheader("Analysis Settings")

    # Risk-free Rate Input
    risk_free_rate = st.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.risk_free_rate * 100,
        step=0.05,  # 5 basis points steps
        help="The risk-free rate used in portfolio optimization calculations. This is typically based on government bond yields in your country."
    )

    # Convert percentage to decimal and update if changed
    risk_free_rate = risk_free_rate / 100
    if risk_free_rate != st.session_state.risk_free_rate:
        st.session_state.risk_free_rate = risk_free_rate
        st.session_state.portfolio.set_risk_free_rate(risk_free_rate)
        if st.session_state.portfolio.stocks:
            st.warning("Please re-optimize your portfolio with the new risk-free rate")

    # Analysis Period Selection
    analysis_period = st.radio(
        "Select Analysis Period",
        options=list(StockData.VALID_PERIODS.keys()),
        format_func=lambda x: StockData.VALID_PERIODS[x],
        horizontal=True  # Makes the radio buttons appear in a horizontal line
    )

    # Optimization Method Selection
    optimization_method = st.selectbox(
        "Optimization Method",
        options=['mean_variance', 'black_litterman'],
        format_func=lambda x: "Mean-Variance" if x == 'mean_variance' else "Black-Litterman",
        key='opt_method'
    )

    if optimization_method != st.session_state.optimization_method:
        st.session_state.optimization_method = optimization_method
        st.session_state.portfolio.set_optimization_method(optimization_method)
        if st.session_state.portfolio.stocks:
            st.warning("Please re-optimize your portfolio with the new method")

    if analysis_period != st.session_state.analysis_period:
        st.session_state.analysis_period = analysis_period
        if st.session_state.portfolio.stocks:
            st.warning("Please re-optimize your portfolio with the new analysis period")

    # Stock Search
    st.subheader("Add Stocks")
    stock_search = st.text_input("Enter Stock Symbol (e.g., AAPL)")
    if st.button("Add to Portfolio"):
        if stock_search:
            st.session_state.portfolio.add_stock(stock_search.upper())
            st.success(f"Added {stock_search.upper()} to portfolio")

    # Portfolio Stocks
    st.subheader("Current Portfolio")
    for stock in st.session_state.portfolio.stocks:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(stock)
        with col2:
            if st.button("Remove", key=f"remove_{stock}"):
                st.session_state.portfolio.remove_stock(stock)
                st.rerun()

    # Black-Litterman Views
    if optimization_method == 'black_litterman' and st.session_state.portfolio.stocks:
        st.subheader("Market Views")
        st.markdown("""
        Add your expected annual returns for specific stocks.
        - Use percentages (e.g., 10% for 10% expected return)
        - Confidence level indicates how sure you are about your view
        """)

        for stock in st.session_state.portfolio.stocks:
            col1, col2 = st.columns(2)
            with col1:
                view = st.number_input(
                    f"Expected Return (%) for {stock}",
                    min_value=-100.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.05,  # 5 basis points steps
                    format="%.2f",
                    key=f"view_{stock}"
                )
            with col2:
                confidence = st.slider(
                    f"Confidence for {stock}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key=f"conf_{stock}"
                )

            if view != 0.0:
                st.session_state.views[stock] = view / 100.0  # Convert percentage to decimal
                st.session_state.view_confidences[stock] = confidence
            else:
                st.session_state.views.pop(stock, None)
                st.session_state.view_confidences.pop(stock, None)

    # Optimize Portfolio
    if st.button("Optimize Portfolio"):
        if st.session_state.portfolio.stocks:
            views = None
            view_confidences = None

            if optimization_method == 'black_litterman' and st.session_state.views:
                views = [
                    (st.session_state.portfolio.stocks.index(stock), view)
                    for stock, view in st.session_state.views.items()
                ]
                view_confidences = [
                    st.session_state.view_confidences[stock]
                    for stock in st.session_state.views.keys()
                ]

            result = st.session_state.portfolio.optimize_portfolio(
                period=st.session_state.analysis_period,
                views=views,
                view_confidences=view_confidences,
                risk_free_rate=st.session_state.risk_free_rate
            )

            if result:
                st.success("Portfolio optimized successfully!")
            else:
                st.error("Failed to optimize portfolio. Please check the stock data.")
        else:
            st.warning("Add stocks to optimize portfolio")

# Main Content
tab1, tab2 = st.tabs(["Stock Analysis", "Portfolio Analysis"])

with tab1:
    if stock_search:
        try:
            # Stock Data
            stock_data = StockData.get_stock_data(stock_search.upper(), period=st.session_state.analysis_period)
            stock_info = StockData.get_stock_info(stock_search.upper())

            # Stock Info
            st.subheader(f"{stock_info['name']} ({stock_search.upper()})")

            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "P/E Ratio", 
                    f"{stock_info['pe_ratio']:.2f}" if stock_info['pe_ratio'] else "N/A",
                    help="Price to Earnings ratio: A valuation metric comparing a company's stock price to its earnings per share."
                )
            with col2:
                st.metric(
                    "P/B Ratio", 
                    f"{stock_info['pb_ratio']:.2f}" if stock_info['pb_ratio'] else "N/A",
                    help="Price to Book ratio: Compares a company's market value to its book value."
                )
            with col3:
                st.metric(
                    "Beta", 
                    f"{stock_info['beta']:.2f}" if stock_info['beta'] else "N/A",
                    help="Beta measures a stock's volatility compared to the overall market. Beta > 1 indicates higher volatility than the market."
                )
            with col4:
                st.metric(
                    "Dividend Yield", 
                    f"{stock_info['dividend_yield']:.2f}%" if stock_info['dividend_yield'] else "N/A",
                    help="Annual dividend payments as a percentage of stock price."
                )

            # Valuation Analysis
            if 'target_price' in stock_info:
                st.subheader("Valuation Analysis", help="Based on EBITDA and historical multiples")

                vcol1, vcol2, vcol3 = st.columns(3)
                with vcol1:
                    st.metric(
                        "Forward EBITDA (M)", 
                        f"${stock_info['forward_ebitda']/1e6:.1f}M",
                        help="Estimated forward 12-month EBITDA based on recent quarterly averages"
                    )
                with vcol2:
                    st.metric(
                        "EV/EBITDA Multiple", 
                        f"{stock_info['historical_ev_ebitda']:.1f}x",
                        help="Enterprise Value to EBITDA ratio - a key valuation multiple"
                    )
                with vcol3:
                    st.metric(
                        "Target Price", 
                        f"${stock_info['target_price']:.2f}",
                        delta=f"{stock_info['expected_return']:.1f}%",
                        help="Estimated fair value based on forward EBITDA and historical multiples"
                    )

                # Valuation Methodology Explanation
                with st.expander("View Valuation Methodology"):
                    st.markdown("""
                    **Valuation Methodology:**
                    1. Calculate forward EBITDA by averaging recent quarterly EBITDA
                    2. Apply historical EV/EBITDA multiple to estimate target Enterprise Value
                    3. Adjust for debt and cash to derive equity value
                    4. Divide by shares outstanding to get target price
                    5. Compare with current price to estimate potential return

                    Note: This is a simplified valuation model and should be used alongside other analysis methods.
                    """)
            
            # Charts
            data = StockData.calculate_technical_indicators(stock_data)
            st.plotly_chart(Visualizer.create_stock_price_chart(data, stock_search.upper()))      

        except Exception as e:
            st.error(f"Error fetching data for {stock_search}: {str(e)}")

with tab2:
    if st.session_state.portfolio.stocks:
        # Portfolio Statistics
        stats = st.session_state.portfolio.get_portfolio_stats(period=st.session_state.analysis_period)
        if stats:
            st.subheader("Portfolio Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Annual Return", f"{stats['return']*100:.2f}%")
            with col2:
                st.metric("Annual Volatility", f"{stats['volatility']*100:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")

            # Portfolio Composition
            st.subheader("Portfolio Composition")
            weights_chart = Visualizer.create_portfolio_weights_chart(
                st.session_state.portfolio.weights,
                st.session_state.portfolio.stocks
            )
            st.plotly_chart(weights_chart)

            # Weights Table
            st.subheader("Portfolio Weights")
            weights_df = pd.DataFrame({
                'Stock': st.session_state.portfolio.stocks,
                'Weight': [f"{w*100:.2f}%" for w in st.session_state.portfolio.weights]
            })
            st.table(weights_df)
        else:
            st.warning("Unable to calculate portfolio statistics. Try optimizing the portfolio or check if all stocks have valid data.")
    else:
        st.info("Add stocks to your portfolio to see analysis")