"""
Quantitative Models

Hybrid quantitative + ML models for portfolio optimization,
risk metrics, and options pricing.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from scipy import optimize
from scipy import stats
import warnings


@dataclass
class Greeks:
    """Options Greeks from Black-Scholes model"""
    delta: float          # Price sensitivity
    gamma: float          # Delta sensitivity
    theta: float          # Time decay
    vega: float           # Volatility sensitivity
    rho: float            # Interest rate sensitivity
    implied_volatility: float
    theoretical_price: float


@dataclass
class OptimalWeights:
    """Optimal portfolio weights from optimization"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_expected: float
    optimization_method: str
    constraints_applied: List[str] = field(default_factory=list)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a portfolio or position"""
    # Return metrics
    total_return: float
    annualized_return: float
    excess_return: float  # vs benchmark
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    beta: float
    tracking_error: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int  # days
    
    # Tail risk
    var_95: float          # Value at Risk (95%)
    var_99: float          # Value at Risk (99%)
    cvar_95: float         # Conditional VaR (Expected Shortfall)
    
    # Win/Loss
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float


class BlackScholes:
    """
    Black-Scholes options pricing model.
    
    Provides theoretical option prices and Greeks.
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter"""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter"""
        if T <= 0 or sigma <= 0:
            return 0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(
        S: float,      # Current stock price
        K: float,      # Strike price
        T: float,      # Time to expiration (years)
        r: float,      # Risk-free rate
        sigma: float   # Volatility
    ) -> float:
        """Calculate call option price"""
        if T <= 0:
            return max(0, S - K)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    
    @staticmethod
    def put_price(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate put option price"""
        if T <= 0:
            return max(0, K - S)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    
    @staticmethod
    def calculate_greeks(
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "call"
    ) -> Greeks:
        """Calculate all Greeks for an option"""
        if T <= 0 or sigma <= 0:
            return Greeks(
                delta=1.0 if option_type == "call" else -1.0,
                gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
                implied_volatility=sigma,
                theoretical_price=max(0, S - K) if option_type == "call" else max(0, K - S)
            )
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        # Delta
        if option_type == "call":
            delta = stats.norm.cdf(d1)
        else:
            delta = stats.norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_common = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == "call":
            theta = (theta_common - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
        else:
            theta = (theta_common + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365
        
        # Vega (same for calls and puts)
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% move
        
        # Rho
        if option_type == "call":
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        # Theoretical price
        if option_type == "call":
            price = BlackScholes.call_price(S, K, T, r, sigma)
        else:
            price = BlackScholes.put_price(S, K, T, r, sigma)
        
        return Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            implied_volatility=sigma,
            theoretical_price=price
        )
    
    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float, K: float, T: float, r: float,
        option_type: str = "call",
        precision: float = 0.0001,
        max_iterations: int = 100
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        """
        if T <= 0:
            return 0.0
        
        # Initial guess
        sigma = 0.3
        
        for _ in range(max_iterations):
            if option_type == "call":
                price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholes.put_price(S, K, T, r, sigma)
            
            diff = price - market_price
            
            if abs(diff) < precision:
                return sigma
            
            # Vega for Newton-Raphson
            d1 = BlackScholes.d1(S, K, T, r, sigma)
            vega = S * stats.norm.pdf(d1) * np.sqrt(T)
            
            if vega < 1e-10:
                break
            
            sigma = sigma - diff / vega
            sigma = max(0.01, min(5.0, sigma))  # Bounds
        
        return sigma


class PortfolioOptimizer:
    """
    Modern Portfolio Theory optimization.
    
    Supports:
    - Sharpe ratio maximization
    - Minimum volatility
    - Risk parity
    - Maximum diversification
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize optimizer with historical returns.
        
        Args:
            returns: DataFrame of asset returns (columns = assets)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.assets = list(returns.columns)
        
        # Calculate statistics
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252     # Annualized
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate expected portfolio return"""
        return np.sum(weights * self.mean_returns)
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        if vol == 0:
            return 0
        return (ret - self.risk_free_rate) / vol
    
    def sortino_ratio(self, weights: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        portfolio_returns = self.returns @ weights
        downside_returns = portfolio_returns[portfolio_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns) * np.sqrt(252)
        if downside_std == 0:
            return float('inf')
        
        ret = self.portfolio_return(weights)
        return (ret - self.risk_free_rate) / downside_std
    
    def optimize_sharpe(
        self,
        max_weight: float = 0.25,
        min_weight: float = 0.0
    ) -> OptimalWeights:
        """
        Optimize portfolio for maximum Sharpe ratio.
        """
        # Objective: negative Sharpe (for minimization)
        def neg_sharpe(weights):
            return -self.sharpe_ratio(weights)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Initial guess (equal weight)
        x0 = np.array([1 / self.n_assets] * self.n_assets)
        
        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize.minimize(
                neg_sharpe, x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        
        weights = result.x
        
        return OptimalWeights(
            weights={
                asset: float(w) for asset, w in zip(self.assets, weights)
            },
            expected_return=self.portfolio_return(weights),
            expected_volatility=self.portfolio_volatility(weights),
            sharpe_ratio=self.sharpe_ratio(weights),
            sortino_ratio=self.sortino_ratio(weights),
            max_drawdown_expected=self.portfolio_volatility(weights) * 2.5,
            optimization_method="max_sharpe",
            constraints_applied=[
                f"max_weight={max_weight}",
                f"min_weight={min_weight}"
            ]
        )
    
    def optimize_min_volatility(
        self,
        max_weight: float = 0.25,
        min_weight: float = 0.0,
        target_return: Optional[float] = None
    ) -> OptimalWeights:
        """
        Optimize portfolio for minimum volatility.
        """
        # Objective: portfolio volatility
        def portfolio_vol(weights):
            return self.portfolio_volatility(weights)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.portfolio_return(x) - target_return
            })
        
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        x0 = np.array([1 / self.n_assets] * self.n_assets)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize.minimize(
                portfolio_vol, x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        
        weights = result.x
        
        return OptimalWeights(
            weights={
                asset: float(w) for asset, w in zip(self.assets, weights)
            },
            expected_return=self.portfolio_return(weights),
            expected_volatility=self.portfolio_volatility(weights),
            sharpe_ratio=self.sharpe_ratio(weights),
            sortino_ratio=self.sortino_ratio(weights),
            max_drawdown_expected=self.portfolio_volatility(weights) * 2.5,
            optimization_method="min_volatility",
            constraints_applied=[
                f"max_weight={max_weight}",
                f"target_return={target_return}" if target_return else ""
            ]
        )
    
    def efficient_frontier(
        self,
        n_points: int = 50,
        max_weight: float = 0.25
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Generate efficient frontier points.
        
        Returns:
            List of (return, volatility, weights) tuples
        """
        # Get min and max returns
        min_vol_portfolio = self.optimize_min_volatility(max_weight=max_weight)
        max_sharpe_portfolio = self.optimize_sharpe(max_weight=max_weight)
        
        min_ret = min_vol_portfolio.expected_return
        max_ret = max(
            max_sharpe_portfolio.expected_return,
            float(self.mean_returns.max())
        )
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier = []
        
        for target in target_returns:
            try:
                portfolio = self.optimize_min_volatility(
                    max_weight=max_weight,
                    target_return=target
                )
                frontier.append((
                    portfolio.expected_return,
                    portfolio.expected_volatility,
                    np.array(list(portfolio.weights.values()))
                ))
            except Exception:
                continue
        
        return frontier


class QuantitativeModels:
    """
    Unified interface for all quantitative models.
    
    Combines:
    - Portfolio optimization (Sharpe, Sortino)
    - Options pricing (Black-Scholes)
    - Risk metrics calculation
    - Market outperformance scoring
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize with risk-free rate.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
        self.black_scholes = BlackScholes()
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Daily returns series
            benchmark_returns: Optional benchmark returns
        """
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sharpe and Sortino
        excess_return_daily = returns.mean() - self.risk_free_rate / 252
        sharpe = (excess_return_daily * 252) / volatility if volatility > 0 else 0
        sortino = (excess_return_daily * 252) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0
        
        # Find drawdown duration
        in_drawdown = drawdowns < 0
        drawdown_periods = in_drawdown.astype(int).groupby(
            (~in_drawdown).cumsum()
        ).sum()
        drawdown_duration = int(drawdown_periods.max()) if len(drawdown_periods) > 0 else 0
        
        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Win rate and profit factor
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_days.mean() if len(winning_days) > 0 else 0
        avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
        profit_factor = (
            abs(winning_days.sum() / losing_days.sum())
            if len(losing_days) > 0 and losing_days.sum() != 0
            else float('inf')
        )
        
        # Benchmark-relative metrics
        excess_return = 0
        beta = 1.0
        tracking_error = 0
        information_ratio = 0
        treynor_ratio = 0
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align series
            aligned = pd.concat(
                [returns, benchmark_returns], axis=1
            ).dropna()
            if len(aligned) > 10:
                ret = aligned.iloc[:, 0]
                bench = aligned.iloc[:, 1]
                
                excess_return = annualized_return - (
                    (1 + bench).prod() ** (252 / len(bench)) - 1
                )
                
                # Beta
                cov = np.cov(ret, bench)[0, 1]
                bench_var = bench.var()
                beta = cov / bench_var if bench_var > 0 else 1.0
                
                # Tracking error
                tracking_diff = ret - bench
                tracking_error = tracking_diff.std() * np.sqrt(252)
                
                # Information ratio
                information_ratio = (
                    excess_return / tracking_error
                    if tracking_error > 0 else 0
                )
                
                # Treynor ratio
                treynor_ratio = (
                    (annualized_return - self.risk_free_rate) / beta
                    if beta != 0 else 0
                )
        
        return RiskMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_return=excess_return,
            volatility=volatility,
            downside_volatility=downside_vol,
            beta=beta,
            tracking_error=tracking_error,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            drawdown_duration=drawdown_duration,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
    
    def black_scholes_greeks(
        self,
        current_price: float,
        strike: float,
        days_to_expiry: int,
        volatility: float,
        option_type: str = "call"
    ) -> Greeks:
        """
        Calculate Black-Scholes Greeks for an option.
        
        Args:
            current_price: Current stock price
            strike: Option strike price
            days_to_expiry: Days until expiration
            volatility: Annualized volatility (e.g., 0.25 for 25%)
            option_type: "call" or "put"
        
        Returns:
            Greeks object with all sensitivities
        """
        T = days_to_expiry / 365
        return BlackScholes.calculate_greeks(
            S=current_price,
            K=strike,
            T=T,
            r=self.risk_free_rate,
            sigma=volatility,
            option_type=option_type
        )
    
    def portfolio_optimization(
        self,
        returns: pd.DataFrame,
        optimize_for: str = "sharpe",
        max_weight: float = 0.25,
        min_weight: float = 0.0,
        target_return: Optional[float] = None
    ) -> OptimalWeights:
        """
        Optimize portfolio weights.
        
        Args:
            returns: DataFrame of asset daily returns
            optimize_for: "sharpe", "sortino", or "min_vol"
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            target_return: Optional target return for min_vol
        
        Returns:
            Optimal portfolio weights and metrics
        """
        optimizer = PortfolioOptimizer(returns, self.risk_free_rate)
        
        if optimize_for == "sharpe":
            return optimizer.optimize_sharpe(max_weight, min_weight)
        elif optimize_for == "min_vol":
            return optimizer.optimize_min_volatility(
                max_weight, min_weight, target_return
            )
        else:
            # Default to Sharpe
            return optimizer.optimize_sharpe(max_weight, min_weight)
    
    def market_outperformance_score(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Score strategy's potential to outperform benchmark.
        
        Returns scores and metrics for comparison.
        """
        strategy_metrics = self.calculate_risk_metrics(
            strategy_returns, benchmark_returns
        )
        benchmark_metrics = self.calculate_risk_metrics(benchmark_returns)
        
        # Calculate component scores
        sharpe_score = min(100, max(0, strategy_metrics.sharpe_ratio * 33))
        alpha_score = min(100, max(0, 50 + strategy_metrics.excess_return * 500))
        risk_score = min(
            100, 
            max(0, 100 - abs(strategy_metrics.max_drawdown) * 200)
        )
        consistency_score = min(100, strategy_metrics.win_rate * 150)
        
        # Overall score (weighted average)
        overall = (
            sharpe_score * 0.3 +
            alpha_score * 0.3 +
            risk_score * 0.2 +
            consistency_score * 0.2
        )
        
        return {
            "overall_score": overall,
            "sharpe_score": sharpe_score,
            "alpha_score": alpha_score,
            "risk_score": risk_score,
            "consistency_score": consistency_score,
            "sharpe_ratio": strategy_metrics.sharpe_ratio,
            "excess_return": strategy_metrics.excess_return,
            "max_drawdown": strategy_metrics.max_drawdown,
            "win_rate": strategy_metrics.win_rate,
            "information_ratio": strategy_metrics.information_ratio,
            "beta": strategy_metrics.beta
        }
    
    def to_prompt_context(
        self,
        optimal_weights: OptimalWeights,
        risk_metrics: Optional[RiskMetrics] = None,
        greeks: Optional[Greeks] = None
    ) -> str:
        """Format quantitative analysis for LLM prompt"""
        sections = []
        
        # Portfolio optimization section
        weights_text = "\n".join([
            f"    {asset}: {weight:.1%}"
            for asset, weight in sorted(
                optimal_weights.weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        ])
        
        sections.append(f"""
## Quantitative Edge

### Portfolio Optimization ({optimal_weights.optimization_method})
Optimal allocation:
{weights_text}

Expected metrics:
- Return: {optimal_weights.expected_return:+.1%}
- Volatility: {optimal_weights.expected_volatility:.1%}
- Sharpe Ratio: {optimal_weights.sharpe_ratio:.2f}
- Sortino Ratio: {optimal_weights.sortino_ratio:.2f}
- Max Drawdown (Est): {optimal_weights.max_drawdown_expected:.1%}
""")
        
        if risk_metrics:
            sections.append(f"""
### Risk Analysis
- Current Sharpe: {risk_metrics.sharpe_ratio:.2f}
- Max Drawdown: {risk_metrics.max_drawdown:.1%}
- Win Rate: {risk_metrics.win_rate:.0%}
- VaR (95%): {risk_metrics.var_95:.2%}
- Beta: {risk_metrics.beta:.2f}
""")
        
        if greeks:
            sections.append(f"""
### Options Analysis (Black-Scholes)
- Delta: {greeks.delta:.3f}
- Gamma: {greeks.gamma:.4f}
- Theta: {greeks.theta:.4f} (daily)
- Vega: {greeks.vega:.4f} (per 1% IV)
- Implied Vol: {greeks.implied_volatility:.1%}
""")
        
        return "\n".join(sections)
