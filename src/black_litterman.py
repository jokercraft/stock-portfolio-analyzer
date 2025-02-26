import numpy as np
import pandas as pd
import yfinance as yf

class BlackLittermanOptimizer:
    def __init__(self, risk_free_rate=0.02, market_risk_premium=0.06, tau=0.025):
        """
        Initialize Black-Litterman optimizer

        Parameters:
        - risk_free_rate: Risk-free rate (default: 2%)
        - market_risk_premium: Market risk premium (default: 6%)
        - tau: Uncertainty in prior (default: 0.025)
        """
        self.risk_free_rate = risk_free_rate
        self.market_risk_premium = market_risk_premium
        self.tau = tau

    def get_market_weights(self, tickers):
        """Get market capitalization weights for given tickers"""
        market_caps = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                market_caps[ticker] = stock.info.get('marketCap', 0)
            except:
                market_caps[ticker] = 0

        total_cap = sum(market_caps.values())
        if total_cap == 0:
            # Fallback to equal weights if market cap data unavailable
            return np.array([1/len(tickers)] * len(tickers))

        weights = np.array([market_caps[ticker]/total_cap for ticker in tickers])
        return weights

    def calculate_equilibrium_returns(self, returns, market_weights):
        """Calculate equilibrium returns using CAPM"""
        cov_matrix = returns.cov() * 252  # Annualized covariance matrix
        pi = self.risk_free_rate + self.market_risk_premium * market_weights
        return pi

    def incorporate_views(self, returns, market_weights, views=None, view_confidences=None):
        """
        Incorporate investor views into the Black-Litterman model

        Parameters:
        - returns: DataFrame of historical returns
        - market_weights: Array of market capitalization weights
        - views: List of tuples (asset_index, view_return)
        - view_confidences: List of confidence levels for each view (0 to 1)
        """
        n_assets = len(market_weights)
        cov_matrix = returns.cov().values * 252  # Convert to numpy array

        # Calculate prior (equilibrium) returns
        pi = self.calculate_equilibrium_returns(returns, market_weights)

        if views is None or view_confidences is None or len(views) == 0:
            # If no views provided, return market equilibrium
            return pi, cov_matrix

        # Construct view matrix P
        P = np.zeros((len(views), n_assets))
        q = np.zeros(len(views))
        omega = np.zeros((len(views), len(views)))

        # Process views and confidences
        for i, (asset_idx, view_return) in enumerate(views):
            P[i, asset_idx] = 1
            q[i] = view_return
            
            # Limit confidence values to avoid singular matrix
            # Cap confidence at 0.99 to ensure omega is invertible
            confidence = min(max(view_confidences[i], 0.01), 0.99)
            
            # Add a small constant to ensure numerical stability
            epsilon = 1e-6
            
            # Set omega values based on confidence
            omega[i, i] = (1 - confidence) * cov_matrix[asset_idx, asset_idx] + epsilon

        # Calculate posterior estimates
        try:
            pi_adjusted = self._compute_posterior(pi, P, q, omega, cov_matrix)
            return pi_adjusted, cov_matrix
        except np.linalg.LinAlgError as e:
            print(f"Matrix inversion error: {e}. Falling back to equilibrium returns.")
            return pi, cov_matrix

    def _compute_posterior(self, pi, P, q, omega, cov_matrix):
        """Compute posterior expected returns"""
        tau_V = self.tau * cov_matrix
        
        try:
            # Try standard inversion
            inv_omega = np.linalg.inv(omega)
        except np.linalg.LinAlgError:
            # If standard inversion fails, use pseudo-inverse
            print("Using pseudo-inverse for omega matrix")
            inv_omega = np.linalg.pinv(omega)
            
        try:
            inv_tau_V = np.linalg.inv(tau_V)
        except np.linalg.LinAlgError:
            # If covariance matrix inversion fails, use pseudo-inverse
            print("Using pseudo-inverse for covariance matrix")
            inv_tau_V = np.linalg.pinv(tau_V)

        try:
            # Posterior estimate
            term1_inner = inv_tau_V + P.T @ inv_omega @ P
            term1 = np.linalg.inv(term1_inner)
            term2 = inv_tau_V @ pi + P.T @ inv_omega @ q
            posterior_returns = term1 @ term2
            
            return posterior_returns
        except np.linalg.LinAlgError:
            # If posterior calculation fails, return prior
            print("Error in posterior calculation, returning prior")
            return pi

    def optimize(self, returns, tickers, views=None, view_confidences=None):
        """
        Perform Black-Litterman optimization

        Parameters:
        - returns: DataFrame of historical returns
        - tickers: List of stock tickers
        - views: Optional list of views on specific assets
        - view_confidences: Optional list of confidence levels for views

        Returns:
        - Dictionary containing optimal weights and metrics
        """
        try:
            market_weights = self.get_market_weights(tickers)
            expected_returns, cov_matrix = self.incorporate_views(
                returns, market_weights, views, view_confidences
            )

            def portfolio_metrics(weights):
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_volatility = np.sqrt(
                    np.dot(weights.T, np.dot(cov_matrix, weights))
                )
                return portfolio_return, portfolio_volatility

            def objective(weights):
                ret, vol = portfolio_metrics(weights)
                sharpe = (ret - self.risk_free_rate) / vol
                # Add diversification penalty
                concentration = np.sum(weights ** 2)
                return -(sharpe - 0.5 * concentration)

            # Optimization constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            ]
            
            # Determine minimum weight based on number of assets
            min_weight = min(0.02, 0.8 / len(tickers))
            max_weight = min(0.4, 1.0)
            
            # Only add minimum weight constraint if feasible
            if min_weight > 0 and len(tickers) <= 40:
                constraints.append({'type': 'ineq', 'fun': lambda x: x - min_weight})
                
            bounds = tuple((min_weight, max_weight) for _ in range(len(tickers)))

            from scipy.optimize import minimize
            
            # Try optimization with initial weights from market cap
            initial_weights = market_weights
            
            # Ensure initial weights are valid
            initial_weights = np.clip(initial_weights, min_weight, max_weight)
            initial_weights = initial_weights / np.sum(initial_weights)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )

            if result.success:
                optimal_weights = result.x
                # Normalize to ensure weights sum to 1
                optimal_weights = optimal_weights / np.sum(optimal_weights)
                
                final_return, final_vol = portfolio_metrics(optimal_weights)

                return {
                    'weights': optimal_weights,
                    'expected_return': final_return,
                    'volatility': final_vol,
                    'sharpe_ratio': (final_return - self.risk_free_rate) / final_vol
                }
            else:
                print(f"Black-Litterman optimization failed: {result.message}. Using fallback method.")
                # Fallback to equal weights
                weights = np.array([1/len(tickers)] * len(tickers))
                ret, vol = portfolio_metrics(weights)
                
                return {
                    'weights': weights,
                    'expected_return': ret,
                    'volatility': vol,
                    'sharpe_ratio': (ret - self.risk_free_rate) / vol,
                    'note': 'Optimization failed, using equal weights'
                }
                
        except Exception as e:
            print(f"Error in Black-Litterman optimization: {str(e)}")
            # Fallback to equal weights if all else fails
            fallback_weights = np.array([1/len(tickers)] * len(tickers))
            
            try:
                # Try to calculate metrics with equal weights
                ret = np.sum(market_weights * fallback_weights)
                vol = np.sqrt(np.dot(fallback_weights.T, np.dot(returns.cov().values * 252, fallback_weights)))
                sharpe = (ret - self.risk_free_rate) / vol
                
                return {
                    'weights': fallback_weights,
                    'expected_return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe,
                    'note': 'Optimization error, using equal weights'
                }
            except:
                # If even the metrics calculation fails, just return weights
                return {
                    'weights': fallback_weights.tolist(),
                    'note': 'Complete optimization failure, using equal weights'
                }
        return None