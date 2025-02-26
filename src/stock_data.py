import yfinance as yf
import pandas as pd
import numpy as np

from financial_data import Financials

class StockData:
    VALID_PERIODS = {
        '1y': '1 Year',
        '2y': '2 Years',
        '5y': '5 Years'
    }

    @staticmethod
    def safe_float(value):
        """Safely convert value to float, handling None and invalid values"""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def get_stock_data(ticker, period='1y'):
        """Fetch stock data from Yahoo Finance"""
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist

    @staticmethod
    def get_stock_info(ticker):
        """Get stock information and key metrics"""
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Initialize metrics dictionary
        metrics = {
            'name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'pe_ratio': info.get('trailingPE', None),
            'pb_ratio': info.get('priceToBook', None),
            'market_cap': info.get('marketCap', None),
            'dividend_yield': info.get('dividendYield', None),
            'beta': info.get('beta', None),
            'current_price': info.get('currentPrice', None)
        }
        # Get quarterly financials for EBITDA analysis
        try:
            if "." in ticker:
                ticker = ticker.split(".")[0]

                bist_metrics = Financials().process_bist_financials(ticker)
                if bist_metrics:
                    df = bist_metrics[ticker]
                    df = df.set_index('itemCode')
                    
                    # Add financial ratios from BIST data
                    bist_ratios = Financials().get_bist_financial_ratios(ticker)
                    metrics['pe_ratio'] = bist_ratios['pe_ratio'].iloc[0]
                    metrics['pb_ratio'] = bist_ratios['pb_ratio'].iloc[0]
                    last_reporting_date = df.columns[-1]

                    # Calculate EBITDA with growth rate analysis
                    ebitda_analysis = StockData.calculate_quarterly_ebitda_growth(df)
                    
                    # Store EBITDA analysis results in metrics
                    if ebitda_analysis and ebitda_analysis['latest_quarterly_ebitda'] is not None:
                        metrics['latest_quarterly_ebitda'] = ebitda_analysis['latest_quarterly_ebitda']
                        metrics['latest_reporting_date'] = ebitda_analysis['latest_date']
                        
                        if ebitda_analysis['recommended_growth_rate'] is not None:
                            metrics['quarterly_growth_rate'] = ebitda_analysis['recommended_growth_rate'] * 100  # Convert to percentage
                            metrics['growth_rate_type'] = 'YoY' if ebitda_analysis['avg_yoy_growth'] is not None else 'QoQ'
                        else:
                            metrics['quarterly_growth_rate'] = 0
                            metrics['growth_rate_type'] = 'None (using flat projection)'

                        # Project forward EBITDA using growth rates
                        metrics['forward_ebitda'] = StockData.project_forward_ebitda(ebitda_analysis)
                    else:
                        # Fall back to the original calculation method if analysis fails
                        gross_profit_row = df[df['itemDescEng'] == 'GROSS PROFIT (LOSS)'][last_reporting_date].iloc[0]
                        general_admin_exp = df[df['itemDescEng'] == 'General Administrative Expenses (-)'][last_reporting_date].iloc[0]
                        marketing_sales_exp = df[df['itemDescEng'] == 'Marketing Selling & Distrib. Expenses (-)'][last_reporting_date].iloc[0]
                        amortization_exp = df[df['itemDescEng'] == 'Depreciation & Amortization'][last_reporting_date].iloc[0]
                        current_ebidta = float(gross_profit_row) + float(general_admin_exp) + float(marketing_sales_exp)
                        current_ebidta = current_ebidta + float(amortization_exp)
                        
                        metrics['forward_ebitda'] = current_ebidta * (12 / int(last_reporting_date.split('/')[1]))
                        metrics['quarterly_growth_rate'] = 0
                        metrics['growth_rate_type'] = 'None (simple annualization)'
                    
                    # Continue with valuation metrics...
                    if not np.isnan(bist_ratios['ev_ebitda'].iloc[0]):
                        historical_ev_ebitda = bist_ratios['ev_ebitda'].iloc[0]
                        shares_outstanding = info.get('sharesOutstanding', None)
                        total_debt = StockData.safe_float(bist_metrics.get('total_liabilities', 0))
                        cash = StockData.safe_float(bist_metrics.get('cash', 0))
                        
                        if all(v is not None for v in [metrics['forward_ebitda'], historical_ev_ebitda, 
                                                    metrics['current_price'], shares_outstanding]):
                            target_ev = metrics['forward_ebitda'] * (historical_ev_ebitda * 0.8) # Use 80% of historical EV/EBITDA
                            equity_value = target_ev - total_debt + cash
                            target_price = equity_value / shares_outstanding
                            expected_return = ((target_price / metrics['current_price']) - 1) * 100

                            metrics.update({
                            'historical_ev_ebitda': historical_ev_ebitda,
                            'target_price': target_price,
                            'expected_return': expected_return,
                            'enterprise_value': target_ev
                        })
                    else:
                        target_price = StockData.calculate_target_price_with_roe(ticker, 
                                                                            metrics['pe_ratio'], 
                                                                            metrics['pb_ratio'],
                                                                            metrics['current_price'])
                        expected_return = ((target_price / metrics['current_price']) - 1) * 100
                        historical_ev_ebitda = 0.0
                        target_ev = 0.0
                        
                        metrics.update({
                            'historical_ev_ebitda': historical_ev_ebitda,
                            'target_price': target_price,
                            'expected_return': expected_return,
                            'enterprise_value': target_ev,
                            'forward_ebitda': historical_ev_ebitda
                        })        
                else:
                    target_price = StockData.calculate_target_price_with_roe(ticker, 
                                                                                metrics['pe_ratio'], 
                                                                                metrics['pb_ratio'],
                                                                                metrics['current_price'])
                    expected_return = ((target_price / metrics['current_price']) - 1) * 100
                    historical_ev_ebitda = 0.0
                    target_ev = 0.0

                    metrics.update({
                        'historical_ev_ebitda': historical_ev_ebitda,
                        'target_price': target_price,
                        'expected_return': expected_return,
                        'enterprise_value': target_ev,
                        'forward_ebitda': historical_ev_ebitda
                    })
            else:
                # Original valuation logic for non-Turkish stocks
                try:
                    quarterly_data = stock.quarterly_financials
                    if not quarterly_data.empty:
                        recent_ebitdas = quarterly_data.loc['EBITDA'].dropna()
                        if len(recent_ebitdas) >= 4:
                            avg_quarterly_ebitda = recent_ebitdas.mean()
                            metrics['forward_ebitda'] = avg_quarterly_ebitda * 4
                            
                            historical_ev_ebitda = info.get('enterpriseToEbitda', None)
                            shares_outstanding = info.get('sharesOutstanding', None)
                            total_debt = info.get('totalDebt', 0)
                            cash = info.get('totalCash', 0)
                            
                            if all(v is not None for v in [metrics['forward_ebitda'], historical_ev_ebitda, 
                                                        metrics['current_price'], shares_outstanding]):
                                target_ev = metrics['forward_ebitda'] * (historical_ev_ebitda * 0.8) # Use 80% of historical EV/EBITDA
                                equity_value = target_ev - total_debt + cash
                                target_price = equity_value / shares_outstanding
                                expected_return = ((target_price / metrics['current_price']) - 1) * 100
                                
                                metrics.update({
                                    'historical_ev_ebitda': historical_ev_ebitda,
                                    'target_price': target_price,
                                    'expected_return': expected_return,
                                    'enterprise_value': target_ev
                                })
                except Exception as e:
                    print(f"Error calculating valuation metrics: {str(e)}")
        except Exception as e:
            print(f"Error processing BIST financials: {str(e)}")

        return metrics

    @staticmethod
    def calculate_returns(data):
        """Calculate daily returns"""
        return data['Close'].pct_change().dropna()

    @staticmethod
    def calculate_target_price_with_roe(ticker, pe_ratio, pb_ratio, price):
        """Calculate target price using ROE and book value"""
        try:
            roe_rate_raw = Financials().get_roe(ticker)
            avg_pb = StockData.get_avg_pb_ratio(ticker)

            if roe_rate_raw:
                roe_rate = float(roe_rate_raw.split(',')[0])/100
            else:
                roe_rate = None
                return price * (avg_pb / pb_ratio)
            
            implied_pe = avg_pb / roe_rate
            target_price = price * (implied_pe / pe_ratio)

            return target_price
        except Exception as e:
            print(f"Error fetching ROE for {ticker}: {str(e)}")
            return 0.0

    @staticmethod
    def get_avg_pb_ratio(ticker):
        """Get average PB ratio for the last 3 years"""
        ticker = yf.Ticker(f'{ticker}.IS')
    
        # Get annual financial data for balance sheet
        annual_bs = ticker.get_balance_sheet()

        # Get historical prices
        # We'll get 4 years of data to ensure we have all year-end prices
        hist_prices = ticker.history(period="4y")
        
        # Convert timezone-aware index to timezone-naive
        hist_prices.index = hist_prices.index.tz_localize(None)
        
        pb_ratios = []
        
        for date, data in annual_bs.items():
            year_end = date.strftime('%Y-12-31')
            
            # Convert year_end to datetime for comparison
            year_end = pd.to_datetime(year_end)
            
            # Skip if this year is before our price data
            if date < hist_prices.index[0]:
                continue
                
            book_value = data.loc['StockholdersEquity']
            shares = data.loc['ShareIssued']
            
            if shares == 0 or pd.isna(shares) or pd.isna(book_value):
                continue
                
            # Find closest available date before year-end
            available_dates = hist_prices.index[hist_prices.index <= year_end]
            if len(available_dates) == 0:
                continue
                
            closest_date = available_dates[-1]
            year_end_price = hist_prices.loc[closest_date, 'Close']
            
            book_value_per_share = book_value / shares
            pb_ratio = year_end_price / book_value_per_share
            pb_ratios.append(pb_ratio)
        
        if len(pb_ratios) < 1:
            raise ValueError(f"No valid P/B ratios could be calculated for {ticker}")
            
        # Get last 3 years average
        pb_ratios = pb_ratios[:3]
        avg_pb = sum(pb_ratios) / len(pb_ratios)
        
        return float(avg_pb)

    @staticmethod
    def calculate_metrics(returns, period='1y'):
        """Calculate key investment metrics with period adjustment"""
        # Calculate annualization factor based on period
        period_days = {
            '3mo': 63,   # ~63 trading days in 3 months
            '6mo': 126,  # ~126 trading days in 6 months
            '1y': 252,   # ~252 trading days in 1 year
            '2y': 504,   # ~504 trading days in 2 years
            '5y': 1260   # ~1260 trading days in 5 years
        }

        trading_days = period_days.get(period, 252)  # default to 1 year if period not found
        annualization_factor = 252 / trading_days

        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility

        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    @staticmethod
    def calculate_weekly_macd_and_signals(data):
        """
        Calculate MACD and 50 EMA based on weekly data and generate buy signals:
        - Price above weekly 50 EMA
        - Weekly MACD (8,13) crosses above signal
        - MACD crossover occurs below zero
        """
        # Resample to weekly data
        weekly_data = data['Close'].resample('W').last()
        
        # Calculate weekly 50 EMA
        weekly_ema_50 = weekly_data.ewm(span=50, adjust=False).mean()
        
        # Calculate weekly MACD with 5,8 parameters
        ema5 = weekly_data.ewm(span=5, adjust=False).mean()
        ema8 = weekly_data.ewm(span=8, adjust=False).mean()
        weekly_macd = ema5 - ema8
        
        # Calculate signal line (8-period EMA of MACD)
        weekly_signal = weekly_macd.ewm(span=8, adjust=False).mean()
        
        # Create weekly DataFrame with all indicators
        weekly_df = pd.DataFrame({
            'MACD': weekly_macd,
            'Signal_Line': weekly_signal,
            'EMA_50': weekly_ema_50
        })
        
        # Forward fill weekly values to daily timestamps
        daily_indicators = weekly_df.reindex(data.index, method='ffill')
        
        # Add indicators to original DataFrame
        data['MACD'] = daily_indicators['MACD']
        data['Signal_Line'] = daily_indicators['Signal_Line']
        data['Weekly_EMA_50'] = daily_indicators['EMA_50']
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
        
        # Generate buy signals based on conditions
        # 1. Price above weekly 50 EMA
        data['Above_Weekly_50_EMA'] = data['Close'] > data['Weekly_EMA_50']
        
        # 2. MACD crosses above Signal Line
        data['MACD_Crossover'] = (
            (data['MACD'] > data['Signal_Line']) & 
            (data['MACD'].shift(1) <= data['Signal_Line'].shift(1))
        )
        
        # 3. MACD is below zero during crossover
        data['MACD_Below_Zero'] = data['MACD'] < 0
        
        # Combined buy signal - all conditions must be met
        data['Buy_Signal'] = (
            data['Above_Weekly_50_EMA'] & 
            data['MACD_Crossover'] &
            data['MACD_Below_Zero']
        )
        
        return data

    @staticmethod
    def calculate_technical_indicators(data):
        """Calculate various technical indicators including weekly EMA and MACD"""
        # Calculate daily EMAs for display purposes
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate weekly indicators and buy signals
        data = StockData.calculate_weekly_macd_and_signals(data)
        
        return data
    
    @staticmethod
    def calculate_quarterly_ebitda_growth(df):
        """
        Calculate quarter-over-quarter growth rates for EBITDA using Turkish BIST financial data.
        Converts cumulative quarterly data to non-cumulative quarterly values.
        
        Args:
            df (DataFrame): Financial data from BIST (with itemCode and quarterly date columns)
        
        Returns:
            dict: EBITDA analysis including quarterly values and growth rates
        """
        # Get all reporting dates (columns in format 'YYYY/M')
        dates = [col for col in df.columns if '/' in col]
        dates.sort(key=lambda x: (int(x.split('/')[0]), int(x.split('/')[1])))
        
        # Calculate cumulative EBITDA for each reporting period
        ebitda_values = []
        for date in dates:
            try:
                # Extract components for EBITDA calculation
                gross_profit = float(df[df['itemDescEng'] == 'GROSS PROFIT (LOSS)'][date].iloc[0])
                gen_admin = float(df[df['itemDescEng'] == 'General Administrative Expenses (-)'][date].iloc[0])
                marketing = float(df[df['itemDescEng'] == 'Marketing Selling & Distrib. Expenses (-)'][date].iloc[0])
                amortization = float(df[df['itemDescEng'] == 'Depreciation & Amortization'][date].iloc[0])
                
                # Calculate EBITDA
                ebitda = gross_profit + float(gen_admin) + float(marketing) + float(amortization)
                
                # Extract year and month
                year = int(date.split('/')[0])
                month = int(date.split('/')[1])
                quarter = (month + 2) // 3  # Convert month to quarter (1-4)
                
                ebitda_values.append({
                    'year': year,
                    'quarter': quarter,
                    'month': month,
                    'date': date,
                    'cumulative_ebitda': ebitda
                })
            except (ValueError, IndexError, TypeError) as e:
                print(f"Error calculating EBITDA for {date}: {str(e)}")
        
        # Sort values chronologically
        ebitda_values.sort(key=lambda x: (x['year'], x['month']))
        
        # Convert cumulative to quarterly (non-cumulative) values
        quarterly_ebitda = []
        for i, data in enumerate(ebitda_values):
            if data['month'] == 3:  # First quarter (March) - already quarterly
                quarterly_value = data['cumulative_ebitda']
            else:
                # Find the previous cumulative value in the same year
                prev_month = 3 if data['month'] == 6 else 6 if data['month'] == 9 else 9
                prev_data = next(
                    (val for val in ebitda_values 
                    if val['year'] == data['year'] and val['month'] == prev_month),
                    None
                )
                
                if prev_data:
                    quarterly_value = data['cumulative_ebitda'] - prev_data['cumulative_ebitda']
                else:
                    quarterly_value = None  # Can't calculate if previous period is missing
            
            if quarterly_value is not None:
                quarterly_ebitda.append({
                    'year': data['year'],
                    'quarter': data['quarter'],
                    'month': data['month'],
                    'date': data['date'],
                    'quarterly_ebitda': quarterly_value
                })
        
        # Calculate quarter-over-quarter growth rates
        qoq_growth_rates = []
        for i in range(1, len(quarterly_ebitda)):
            current = quarterly_ebitda[i]['quarterly_ebitda']
            previous = quarterly_ebitda[i-1]['quarterly_ebitda']
            
            if previous != 0 and previous is not None and current is not None:
                growth_rate = (current - previous) / abs(previous)
                qoq_growth_rates.append({
                    'from_date': quarterly_ebitda[i-1]['date'],
                    'to_date': quarterly_ebitda[i]['date'],
                    'growth_rate': growth_rate
                })
        
        # Calculate year-over-year growth rates for same quarters
        yoy_growth_rates = []
        for i, data in enumerate(quarterly_ebitda):
            same_quarter_prev_year = next(
                (val for val in quarterly_ebitda 
                if val['year'] == data['year'] - 1 and val['quarter'] == data['quarter']),
                None
            )
            
            if same_quarter_prev_year:
                current = data['quarterly_ebitda']
                previous = same_quarter_prev_year['quarterly_ebitda']
                
                if previous != 0 and previous is not None and current is not None:
                    growth_rate = (current - previous) / abs(previous)
                    yoy_growth_rates.append({
                        'from_date': same_quarter_prev_year['date'],
                        'to_date': data['date'],
                        'growth_rate': growth_rate
                    })
        
        # Get latest quarterly EBITDA
        latest_data = quarterly_ebitda[-1] if quarterly_ebitda else None
        
        # Calculate average growth rates
        avg_qoq_growth = (
            sum(item['growth_rate'] for item in qoq_growth_rates[-4:]) / len(qoq_growth_rates[-4:])
            if len(qoq_growth_rates) >= 4 else
            sum(item['growth_rate'] for item in qoq_growth_rates) / len(qoq_growth_rates)
            if qoq_growth_rates else None
        )
        
        avg_yoy_growth = (
            sum(item['growth_rate'] for item in yoy_growth_rates) / len(yoy_growth_rates)
            if yoy_growth_rates else None
        )
        
        # Use YoY growth if available (better for seasonal businesses), otherwise QoQ
        recommended_growth_rate = avg_qoq_growth if avg_qoq_growth is not None else avg_yoy_growth
        
        return {
            'latest_quarterly_ebitda': latest_data['quarterly_ebitda'] if latest_data else None,
            'latest_quarter': latest_data['quarter'] if latest_data else None,
            'latest_year': latest_data['year'] if latest_data else None,
            'latest_date': latest_data['date'] if latest_data else None,
            'avg_qoq_growth': avg_qoq_growth,
            'avg_yoy_growth': avg_yoy_growth,
            'recommended_growth_rate': recommended_growth_rate,
            'quarterly_values': quarterly_ebitda,
            'qoq_growth_rates': qoq_growth_rates,
            'yoy_growth_rates': yoy_growth_rates
        }

    @staticmethod
    def project_forward_ebitda(ebitda_analysis):
        """
        Project EBITDA for the next 12 months based on quarterly growth trends.
        Uses the average of the last 4 quarters' QoQ growth rates.
        
        Args:
            ebitda_analysis (dict): EBITDA analysis from calculate_quarterly_ebitda_growth
        
        Returns:
            float: Projected forward 12-month EBITDA
        """
        if not ebitda_analysis or not ebitda_analysis['latest_quarterly_ebitda']:
            return None
        
        # Use the average of last 4 quarters' QoQ growth rates
        growth_rate = ebitda_analysis['recommended_growth_rate']
        if growth_rate is None:
            # If no growth rate available, use the last quarter as an estimate
            return ebitda_analysis['latest_quarterly_ebitda'] * 4
        
        # Cap extreme growth rates to reasonable bounds
        if growth_rate > 1.0:  # Cap at 100% growth per quarter
            growth_rate = 1.0
        elif growth_rate < -0.5:  # Minimum -50% per quarter
            growth_rate = -0.5
        
        # Project next 4 quarters
        projected_ebitda = 0
        current_quarterly = ebitda_analysis['latest_quarterly_ebitda']
        
        for _ in range(4):  # Project 4 quarters
            # Apply growth rate to get next quarter's EBITDA
            current_quarterly = current_quarterly * (1 + growth_rate)
            # Ensure we don't project negative EBITDA unless it's already negative
            if ebitda_analysis['latest_quarterly_ebitda'] > 0 and current_quarterly < 0:
                current_quarterly = 0
            projected_ebitda += current_quarterly
        
        return projected_ebitda