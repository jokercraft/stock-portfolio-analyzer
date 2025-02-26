import numpy as np
from scipy.optimize import minimize
import pandas as pd
from stock_data import StockData
from black_litterman import BlackLittermanOptimizer

class Portfolio:
    def __init__(self, risk_free_rate=0.02):
        self.stocks = []
        self.weights = []
        self.risk_free_rate = risk_free_rate  # Now passed as parameter
        self.optimization_method = 'mean_variance'  # default method

    def set_risk_free_rate(self, rate):
        """Update the risk-free rate"""
        self.risk_free_rate = rate

    def set_optimization_method(self, method):
        """Set the optimization method to use"""
        valid_methods = ['mean_variance', 'black_litterman']
        if method not in valid_methods:
            raise ValueError(f"Invalid optimization method. Must be one of {valid_methods}")
        self.optimization_method = method

    def add_stock(self, ticker):
        """Add a stock to the portfolio"""
        if ticker not in self.stocks:
            self.stocks.append(ticker)
            self.weights = [1/len(self.stocks)] * len(self.stocks)

    def remove_stock(self, ticker):
        """Remove a stock from the portfolio"""
        if ticker in self.stocks:
            index = self.stocks.index(ticker)
            self.stocks.pop(index)
            if self.stocks:
                self.weights = [1/len(self.stocks)] * len(self.stocks)
            else:
                self.weights = []

    def get_portfolio_returns(self, period='1y'):
        """Calculate portfolio returns"""
        if not self.stocks:
            return None

        returns_data = []
        for ticker in self.stocks:
            try:
                stock_data = StockData.get_stock_data(ticker, period=period)
                returns = StockData.calculate_returns(stock_data)
                returns_data.append(returns)
            except Exception as e:
                print(f"Error getting data for {ticker}: {str(e)}")
                return None

        return pd.concat(returns_data, axis=1, keys=self.stocks)

    def calculate_risk_based_weights(self, returns):
        """Calculate initial weights based on inverse volatility and return efficiency"""
        # Calculate annualized metrics
        volatilities = returns.std() * np.sqrt(252)
        returns_annual = returns.mean() * 252

        # Handle potential zeros in volatility
        volatilities = volatilities.replace(0, 0.0001)

        # Calculate Sharpe ratio for each asset
        asset_sharpe = (returns_annual - self.risk_free_rate) / volatilities

        # Handle negative Sharpes by adding a constant
        min_sharpe = asset_sharpe.min()
        adjusted_sharpe = asset_sharpe + abs(min(0, min_sharpe)) + 0.1
        
        # Combine inverse volatility and Sharpe ratio for weighting
        efficiency_score = (1/volatilities) * adjusted_sharpe
        weights = efficiency_score / efficiency_score.sum()

        # Ensure weights are valid
        weights = np.clip(weights, 0.01, 0.5)
        return weights / weights.sum()

    def optimize_portfolio(self, period='1y', views=None, view_confidences=None, risk_free_rate=0.02):
        """
        Optimize portfolio weights using selected method

        Parameters:
        - period: Time period for analysis
        - views: Optional list of views for Black-Litterman model
        - view_confidences: Optional confidence levels for views
        """
        returns = self.get_portfolio_returns(period=period)
        if returns is None:
            return None

        if self.optimization_method == 'black_litterman':
            optimizer = BlackLittermanOptimizer(risk_free_rate=self.risk_free_rate)
            result = optimizer.optimize(returns, self.stocks, views, view_confidences)
            if result:
                self.weights = result['weights'].tolist()
                return {
                    'weights': self.weights,
                    'sharpe_ratio': result['sharpe_ratio'],
                    'annual_return': result['expected_return'],
                    'annual_volatility': result['volatility']
                }
            return None

        # Mean-variance optimization (original method)
        n_assets = len(self.stocks)
        
        # Adjust minimum weight based on number of assets
        # If we have many assets, minimum weight needs to be lower
        min_weight = min(0.02, 0.8 / n_assets)  # Ensure feasibility
        max_weight = min(0.4, 1.0 / (n_assets * 0.25))   # Adjust max weight too

        def portfolio_metrics(weights):
            returns_mean = returns.mean()
            cov_matrix = returns.cov()

            portfolio_return = np.sum(returns_mean * weights) * 252
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix * 252, weights))
            )
            return portfolio_return, portfolio_volatility

        def negative_sharpe(weights):
            portfolio_return, portfolio_volatility = portfolio_metrics(weights)
            if portfolio_volatility == 0:
                return 1e6  # Penalty for zero volatility

            # Modified objective function considering both Sharpe ratio and diversification
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility

            # Add diversification penalty
            concentration = np.sum(weights ** 2)  # Herfindahl index
            diversification_penalty = concentration * 0.5

            return -(sharpe - diversification_penalty)  # Negative because we minimize

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        
        # Only add minimum weight constraint if it's feasible
        if min_weight > 0 and n_assets <= 40:  # Skip for large portfolios
            constraints.append({'type': 'ineq', 'fun': lambda x: x - min_weight})

        # Bounds for weights
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        try:
            # Use risk-based weights as starting point
            init_weights = self.calculate_risk_based_weights(returns)
            
            # Ensure initial weights are within bounds and sum to 1
            init_weights = np.clip(init_weights, min_weight, max_weight)
            init_weights = init_weights / np.sum(init_weights)

            result = minimize(
                negative_sharpe,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )

            if result.success:
                # Normalize weights to ensure they sum to 1
                final_weights = result.x / np.sum(result.x)
                self.weights = final_weights.tolist()

                # Calculate final metrics
                final_return, final_vol = portfolio_metrics(final_weights)
                final_sharpe = (final_return - self.risk_free_rate) / final_vol

                return {
                    'weights': self.weights,
                    'sharpe_ratio': final_sharpe,
                    'annual_return': final_return,
                    'annual_volatility': final_vol
                }
            else:
                print(f"Optimization failed: {result.message}. Trying fallback method.")
                
                # FALLBACK METHOD 1: Inverse volatility weighting
                volatilities = returns.std() * np.sqrt(252)
                # Handle zero volatility
                volatilities = volatilities.replace(0, 0.0001)
                
                inv_vol = 1 / volatilities
                fallback_weights = inv_vol / inv_vol.sum()
                
                # Ensure weights are within bounds
                fallback_weights = np.clip(fallback_weights, min_weight, max_weight)
                fallback_weights = fallback_weights / np.sum(fallback_weights)
                
                self.weights = fallback_weights.tolist()
                
                # Calculate metrics with fallback weights
                try:
                    fallback_return, fallback_vol = portfolio_metrics(fallback_weights)
                    fallback_sharpe = (fallback_return - self.risk_free_rate) / fallback_vol
                    
                    return {
                        'weights': self.weights,
                        'sharpe_ratio': fallback_sharpe,
                        'annual_return': fallback_return,
                        'annual_volatility': fallback_vol
                    }
                except Exception as inner_e:
                    print(f"Fallback method also failed: {str(inner_e)}")
                    # FALLBACK METHOD 2: Equal weighting
                    self.weights = [1/n_assets] * n_assets
                    
                    try:
                        equal_weights = np.array(self.weights)
                        eq_return, eq_vol = portfolio_metrics(equal_weights)
                        eq_sharpe = (eq_return - self.risk_free_rate) / eq_vol
                        
                        return {
                            'weights': self.weights,
                            'sharpe_ratio': eq_sharpe,
                            'annual_return': eq_return,
                            'annual_volatility': eq_vol
                        }
                    except:
                        # Last resort: just return the weights without metrics
                        return {
                            'weights': self.weights,
                            'note': 'Unable to calculate metrics'
                        }
        except Exception as e:
            print(f"Error in optimization: {str(e)}")
            # If all else fails, use equal weights
            self.weights = [1/n_assets] * n_assets
            return {
                'weights': self.weights,
                'note': 'Optimization failed, using equal weights'
            }

    def get_portfolio_stats(self, period='1y'):
        """Calculate portfolio statistics"""
        returns = self.get_portfolio_returns(period=period)
        if returns is None or not self.weights:
            return None

        try:
            weights = np.array(self.weights)
            returns_mean = returns.mean()
            cov_matrix = returns.cov()

            portfolio_return = np.sum(returns_mean * weights) * 252
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix * 252, weights))
            )
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

            return {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio
            }
        except Exception as e:
            print(f"Error calculating portfolio stats: {str(e)}")
            return None