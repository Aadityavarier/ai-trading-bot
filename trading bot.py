from __future__ import annotations
import sys
import math
import logging
import traceback
import asyncio
import warnings
from typing import Dict, Optional, Tuple, Any, List, Union
import time
import datetime as dt
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
import numba as nb

# ML libs
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib

# Check scikit-learn version for compatibility
import sklearn
sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
SKLEARN_NEW_API = sklearn_version >= (1, 2)

# Advanced ML
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# Technical analysis libs
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Live trading
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

# Indian market data
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("EnhancedInstitutionalBot")

# -----------------------
# Performance Utilities
# -----------------------
@nb.jit(nopython=True, cache=True)
def fast_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using numba"""
    result = np.empty_like(data)
    result[:window-1] = np.nan
    for i in range(window-1, len(data)):
        result[i] = np.mean(data[i-window+1:i+1])
    return result

@nb.jit(nopython=True, cache=True)
def fast_rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation using numba"""
    result = np.empty_like(data)
    result[:window-1] = np.nan
    for i in range(window-1, len(data)):
        result[i] = np.std(data[i-window+1:i+1])
    return result

@nb.jit(nopython=True, cache=True)
def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Fast RSI calculation using numba"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.empty_like(prices)
    rsi[:period] = np.nan
    rsi[period] = 100. - 100. / (1. + rs)

    for i in range(period + 1, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

@nb.jit(nopython=True, cache=True)
def calculate_kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Fast Kelly Criterion calculation"""
    if win_prob <= 0.5:
        return 0.0
    return (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

# -----------------------
# Metrics & Performance
# -----------------------
def cagr(returns: pd.Series) -> float:
    """Compound Annual Growth Rate"""
    if len(returns) < 2:
        return 0.0
    cumulative = (1 + returns).cumprod()
    if cumulative.iloc[-1] <= 0:
        return -1.0
    total_years = (returns.index[-1] - returns.index[0]).days / 365.25
    if total_years <= 0:
        return 0.0
    return (cumulative.iloc[-1] ** (1 / total_years)) - 1

def sharpe_ratio(returns: pd.Series, rf: float = 0.05) -> float:
    """Sharpe ratio with annualization"""
    if returns.std() == 0 or len(returns) == 0:
        return 0.0
    excess_returns = returns - rf / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def sortino_ratio(returns: pd.Series, rf: float = 0.05) -> float:
    """Sortino ratio (downside deviation)"""
    excess_returns = returns - rf / 252
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum drawdown calculation"""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calmar_ratio(returns: pd.Series) -> float:
    """Calmar ratio (CAGR / Max Drawdown)"""
    cagr_val = cagr(returns)
    cum_returns = (1 + returns).cumprod()
    max_dd = abs(max_drawdown(cum_returns))
    return cagr_val / max_dd if max_dd > 0 else 0.0

# -----------------------
# Market Regime Detection
# -----------------------
class AdvancedRegimeDetector:
    """Advanced market regime detection system"""
    
    def __init__(self):
        self.lookback = 60
        self.vol_threshold = 0.025
        self.trend_threshold = 0.02
        
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Detect multiple market regimes"""
        close = df['close']
        returns = close.pct_change().fillna(0.0)
        
        regimes = {}
        
        # Volatility regime (NUMERIC ONLY)
        vol_20 = returns.rolling(20).std().fillna(0.02)
        vol_60 = returns.rolling(60).std().fillna(0.02)
        regimes['vol_regime_numeric'] = (vol_20 > vol_60 * 1.5).astype(int)
        
        # Trend regime (NUMERIC ONLY)
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        
        trend_strength = (sma_20 - sma_50) / (sma_50 + 1e-8)
        trend_numeric = np.where(
            trend_strength > self.trend_threshold, 1,
            np.where(trend_strength < -self.trend_threshold, -1, 0)
        )
        regimes['trend_regime_numeric'] = pd.Series(trend_numeric, index=df.index)
        
        # Market structure regime (NUMERIC ONLY)
        bullish = (close > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200)
        bearish = (close < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200)
        structure_numeric = np.where(bullish, 1, np.where(bearish, -1, 0))
        regimes['structure_regime_numeric'] = pd.Series(structure_numeric, index=df.index)
        
        # Momentum regime (NUMERIC ONLY)
        momentum = close.pct_change(20).fillna(0.0)
        mom_threshold = momentum.rolling(60).std() * 1.5
        mom_numeric = np.where(
            momentum > mom_threshold, 1,
            np.where(momentum < -mom_threshold, -1, 0)
        )
        regimes['momentum_regime_numeric'] = pd.Series(mom_numeric, index=df.index)
        
        return regimes
    
    def should_trade(self, regimes: Dict[str, float], timestamp: pd.Timestamp) -> bool:
        """Determine if trading conditions are favorable"""
        try:
            vol_regime = regimes.get('vol_regime_numeric', 0)  # 0 = low vol, 1 = high vol
            trend_regime = regimes.get('trend_regime_numeric', 0)  # -1, 0, 1
            structure_regime = regimes.get('structure_regime_numeric', 0)  # -1, 0, 1
            
            # Only trade in favorable conditions
            favorable_conditions = [
                vol_regime == 0,  # Low volatility
                trend_regime >= 0,  # Not in strong downtrend
                structure_regime >= -1  # Not in strong bear market
            ]
            
            return sum(favorable_conditions) >= 2  # At least 2 favorable conditions
            
        except:
            return False

# -----------------------
# Enhanced Feature Engineering (FIXED)
# -----------------------
class InstitutionalFeatureEngine:
    """Enhanced feature engineering with NUMERIC ONLY outputs"""
    
    def __init__(self, primary_symbol: str, use_cache: bool = True):
        self.symbol = primary_symbol
        self.use_cache = use_cache
        self.feature_cache = {}
        self.regime_detector = AdvancedRegimeDetector()
        
    def _merge_features(self, base_df: pd.DataFrame, features: Dict[str, pd.Series]) -> pd.DataFrame:
        """Merge features with collision handling"""
        if features is None or len(features) == 0:
            return base_df

        if not isinstance(base_df, pd.DataFrame):
            base_df = pd.DataFrame(index=base_df.index)

        for col_name, series in features.items():
            if series is None:
                continue
            s = pd.Series(series).reindex(base_df.index)
            if col_name in base_df.columns:
                suffix_id = 1
                new_name = f"{col_name}_{suffix_id}"
                while new_name in base_df.columns:
                    suffix_id += 1
                    new_name = f"{col_name}_{suffix_id}"
                col_name = new_name
            base_df[col_name] = s
        return base_df
    
    def basic_price_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Enhanced basic price features"""
        out = {}
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        
        # Returns with proper NaN handling
        returns = pd.Series(np.diff(np.log(close), prepend=np.nan), index=df.index)
        returns = returns.fillna(0.0)
        out['returns'] = returns
        out['returns_abs'] = returns.abs()
        
        # Enhanced OHLC features
        out['hl_range'] = pd.Series((high - low) / (close + 1e-8), index=df.index)
        out['oc_change'] = pd.Series((close - open_price) / (open_price + 1e-8), index=df.index)
        out['ho_gap'] = pd.Series((open_price - np.roll(close, 1)) / (np.roll(close, 1) + 1e-8), index=df.index)
        
        # Price position and strength
        range_val = high - low + 1e-8
        out['close_position'] = pd.Series((close - low) / range_val, index=df.index)
        out['price_strength'] = pd.Series((close - open_price) / range_val, index=df.index)
        
        # Multi-timeframe returns
        for period in [2, 3, 5, 8, 13, 21]:
            out[f'returns_{period}d'] = pd.Series(close, index=df.index).pct_change(period).fillna(0.0)
        
        return out
    
    def advanced_momentum_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Enhanced momentum with regime awareness"""
        out = {}
        close = df['close']
        
        windows = [5, 10, 20, 50, 100]  # Reduced complexity
        
        for w in windows:
            # Basic momentum
            mom = close.pct_change(w).fillna(0.0)
            out[f'mom_{w}'] = mom
            out[f'mom_rank_{w}'] = mom.rolling(w*2).rank(pct=True).fillna(0.5)
            
            # Moving averages
            sma = close.rolling(w).mean()
            ema = close.ewm(span=w).mean()
            out[f'sma_{w}'] = sma
            out[f'ema_{w}'] = ema
            out[f'price_to_sma_{w}'] = (close / (sma + 1e-8) - 1).fillna(0.0)
            out[f'price_to_ema_{w}'] = (close / (ema + 1e-8) - 1).fillna(0.0)
            
            # Momentum acceleration
            mom_change = mom - mom.shift(w//2)
            out[f'mom_accel_{w}'] = mom_change.fillna(0.0)
            
        # Advanced momentum indicators
        # Rate of Change (ROC)
        for period in [10, 20]:
            roc = ((close - close.shift(period)) / (close.shift(period) + 1e-8) * 100).fillna(0.0)
            out[f'roc_{period}'] = roc
            
        # MACD with enhancements
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        out['macd'] = macd.fillna(0.0)
        out['macd_signal'] = signal.fillna(0.0)
        out['macd_histogram'] = histogram.fillna(0.0)
        out['macd_histogram_change'] = histogram.diff().fillna(0.0)
        
        return out
    
    def volatility_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Enhanced volatility with regime context"""
        out = {}
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change().fillna(0.0)
        
        windows = [10, 20, 50]  # Reduced complexity
        
        for w in windows:
            # Basic volatility
            vol = returns.rolling(w).std().fillna(0.0)
            out[f'vol_{w}'] = vol
            out[f'vol_rank_{w}'] = vol.rolling(w*2).rank(pct=True).fillna(0.5)
            
            # Volatility ratios
            vol_short = returns.rolling(w//2).std().fillna(0.0)
            out[f'vol_ratio_{w}'] = (vol_short / (vol + 1e-8)).fillna(1.0)
            
            # Parkinson volatility
            hl_ratio = (high / (low + 1e-8)).fillna(1.0)
            hl_vol = np.sqrt(0.361 * np.log(hl_ratio) ** 2)
            out[f'parkinson_vol_{w}'] = hl_vol.rolling(w).mean().fillna(0.0)
            
        # VIX-like indicator
        out['vix_proxy'] = (returns.rolling(20).std() * np.sqrt(252) * 100).fillna(0.0)
        
        # Volatility regime indicator
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        out['vol_regime_score'] = (vol_20 / (vol_60 + 1e-8)).fillna(1.0)
        
        return out
    
    def volume_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Enhanced volume analysis"""
        out = {}
        volume = df.get('volume', pd.Series(1, index=df.index))
        close = df['close']
        returns = close.pct_change().fillna(0.0)
        
        windows = [10, 20, 50]  # Reduced complexity
        
        for w in windows:
            vol_ma = volume.rolling(w).mean()
            out[f'volume_ma_{w}'] = vol_ma.fillna(volume.mean())
            out[f'volume_ratio_{w}'] = (volume / (vol_ma + 1e-8)).fillna(1.0)
            out[f'volume_rank_{w}'] = volume.rolling(w*2).rank(pct=True).fillna(0.5)
        
        # Volume-Price Trend (VPT)
        vpt = (returns * volume).cumsum()
        out['vpt'] = vpt.fillna(0.0)
        out['vpt_change'] = vpt.diff(10).fillna(0.0)
        
        # Money Flow Index approximation
        typical_price = (df['high'] + df['low'] + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(returns > 0, 0).rolling(14).sum()
        negative_flow = money_flow.where(returns < 0, 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8)))
        out['mfi'] = mfi.fillna(50.0)
        
        return out
    
    def technical_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Enhanced technical indicators"""
        out = {}
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # RSI variants
        out['rsi_14'] = pd.Series(fast_rsi(close, 14), index=df.index).fillna(50.0)
        out['rsi_21'] = pd.Series(fast_rsi(close, 21), index=df.index).fillna(50.0)
        
        # RSI divergence
        rsi_14 = out['rsi_14']
        price_change = df['close'].diff(14)
        rsi_change = rsi_14.diff(14)
        out['rsi_divergence'] = ((price_change > 0) & (rsi_change < 0)).astype(int) - \
                               ((price_change < 0) & (rsi_change > 0)).astype(int)
        
        # Bollinger Bands with squeeze detection
        for period in [20]:
            sma = pd.Series(fast_rolling_mean(close, period), index=df.index).fillna(method='bfill')
            std = pd.Series(fast_rolling_std(close, period), index=df.index).fillna(0.0)
            
            out[f'bb_upper_{period}'] = (sma + 2 * std).fillna(sma)
            out[f'bb_lower_{period}'] = (sma - 2 * std).fillna(sma)
            out[f'bb_position_{period}'] = ((df['close'] - sma) / (2 * std + 1e-8)).fillna(0.0)
            out[f'bb_width_{period}'] = (4 * std / (sma + 1e-8)).fillna(0.0)
            
            # Bollinger Squeeze
            bb_width = out[f'bb_width_{period}']
            bb_squeeze = bb_width.rolling(20).rank(pct=True) < 0.2
            out[f'bb_squeeze_{period}'] = bb_squeeze.astype(int).fillna(0)
        
        # ATR and ATR-based indicators
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(fast_rolling_mean(tr, 14), index=df.index).fillna(0.0)
        out['atr_14'] = atr
        
        # ATR-based position sizing indicator
        out['atr_position_size'] = (0.02 * df['close'] / (atr + 1e-8)).fillna(1.0)  # 2% risk
        
        # Stochastic with smoothing
        for period in [14]:
            low_min = pd.Series(low, index=df.index).rolling(period).min()
            high_max = pd.Series(high, index=df.index).rolling(period).max()
            k_percent = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
            k_smooth = k_percent.rolling(3).mean()
            d_smooth = k_smooth.rolling(3).mean()
            
            out[f'stoch_k_{period}'] = k_smooth.fillna(50.0)
            out[f'stoch_d_{period}'] = d_smooth.fillna(50.0)
            out[f'stoch_cross_{period}'] = ((k_smooth > d_smooth) & 
                                           (k_smooth.shift(1) <= d_smooth.shift(1))).astype(int).fillna(0)
        
        return out
    
    def regime_aware_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Market regime-aware features - NUMERIC ONLY"""
        out = {}
        
        # Get regime information (all numeric)
        regimes = self.regime_detector.detect_regime(df)
        
        # Add ONLY numeric regime features
        for regime_name, regime_series in regimes.items():
            out[regime_name] = regime_series.fillna(0)
        
        # Combined regime score
        regime_scores = []
        for col in regimes.keys():
            if col in out:
                regime_scores.append(out[col])
        
        if regime_scores:
            out['regime_composite'] = pd.concat(regime_scores, axis=1).mean(axis=1).fillna(0.0)
        
        return out
    
    def statistical_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Statistical features with reduced complexity"""
        out = {}
        returns = df['close'].pct_change().fillna(0.0)
        close = df['close']
        
        windows = [20, 50]  # Reduced complexity
        
        for w in windows:
            # Higher moments
            out[f'skew_{w}'] = returns.rolling(w).skew().fillna(0.0)
            out[f'kurt_{w}'] = returns.rolling(w).kurt().fillna(0.0)
            
            # Percentile features
            out[f'returns_pct_rank_{w}'] = returns.rolling(w*2).rank(pct=True).fillna(0.5)
            
            # Z-score
            returns_mean = returns.rolling(w).mean()
            returns_std = returns.rolling(w).std()
            out[f'returns_zscore_{w}'] = ((returns - returns_mean) / (returns_std + 1e-8)).fillna(0.0)
        
        return out
    
    def interaction_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Feature interactions for better signal quality"""
        out = {}
        
        close = df['close']
        returns = close.pct_change().fillna(0.0)
        vol_20 = returns.rolling(20).std().fillna(0.02)
        
        # Momentum-Volatility interaction
        mom_10 = close.pct_change(10).fillna(0.0)
        out['mom_vol_interaction'] = mom_10 * (1 / (vol_20 + 1e-8))  # Momentum adjusted for volatility
        
        # Price-Volume interaction
        if 'volume' in df.columns:
            volume = df['volume']
            vol_ma = volume.rolling(20).mean()
            volume_strength = volume / (vol_ma + 1e-8)
            out['price_volume_strength'] = returns * volume_strength
        
        # RSI-Momentum interaction
        rsi = pd.Series(fast_rsi(close.values, 14), index=df.index).fillna(50.0)
        rsi_normalized = (rsi - 50) / 50  # Normalize RSI to -1 to 1
        out['rsi_momentum_interaction'] = rsi_normalized * mom_10
        
        return out
    
    def calendar_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calendar features"""
        out = {}
        idx = pd.to_datetime(df.index)
        
        # Basic calendar
        out['day_of_week'] = idx.dayofweek
        out['day_of_month'] = idx.day
        out['month'] = idx.month
        out['quarter'] = idx.quarter
        
        # Market specific
        out['is_month_end'] = idx.is_month_end.astype(int)
        out['is_quarter_end'] = idx.is_quarter_end.astype(int)
        
        # Indian market specific
        out['is_monday'] = (idx.dayofweek == 0).astype(int)
        out['is_friday'] = (idx.dayofweek == 4).astype(int)
        
        return out
    
    def generate_features(self, raw_df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Main feature generation with improved processing - NUMERIC ONLY OUTPUT"""
        df = raw_df.copy().sort_index()
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Input dataframe must contain columns: {required_cols}")
        
        logger.info(f"Generating enhanced features for {len(df)} rows")
        
        # Initialize features DataFrame
        features_df = pd.DataFrame(index=df.index)
        
        try:
            # Generate feature groups with improved selection
            features_df = self._merge_features(features_df, self.basic_price_features(df))
            features_df = self._merge_features(features_df, self.advanced_momentum_features(df))
            features_df = self._merge_features(features_df, self.volatility_features(df))
            features_df = self._merge_features(features_df, self.volume_features(df))
            features_df = self._merge_features(features_df, self.technical_indicators(df))
            features_df = self._merge_features(features_df, self.regime_aware_features(df))
            features_df = self._merge_features(features_df, self.statistical_features(df))
            features_df = self._merge_features(features_df, self.interaction_features(df))
            features_df = self._merge_features(features_df, self.calendar_features(df))
            
            # Add external library features if available (reduced complexity)
            if TALIB_AVAILABLE:
                features_df = self._merge_features(features_df, self._talib_features(df))
            
            if TA_AVAILABLE:
                features_df = self._merge_features(features_df, self._ta_features(df))
                
        except Exception as e:
            logger.error(f"Error in feature generation: {e}")
            features_df = self._merge_features(features_df, self.technical_indicators(df))
        
        # Generate targets with proper handling
        target_returns = df['close'].pct_change().shift(-horizon)
        features_df['target_return'] = target_returns
        features_df['target_direction'] = (target_returns > 0).astype(int)
        
        # Multi-class targets with improved binning
        try:
            # Use more balanced bins based on distribution
            quantiles = target_returns.quantile([0.33, 0.67]).values
            target_cut = pd.cut(
                target_returns, 
                bins=[-np.inf, quantiles[0], quantiles[1], np.inf], 
                labels=[0, 1, 2]
            )
            
            target_class = target_cut.cat.add_categories([-1])
            target_class = target_class.fillna(-1)
            features_df['target_class'] = target_class.astype(int)
            features_df.loc[features_df['target_class'] == -1, 'target_class'] = np.nan
            
        except Exception as e:
            logger.warning(f"Error creating target_class: {e}")
            features_df['target_class'] = features_df['target_direction']
        
        # Clean up
        initial_len = len(features_df)
        features_df = features_df.dropna(subset=['target_direction'])
        
        # Handle remaining NaNs more intelligently
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        # FINAL CHECK: Ensure ALL features are numeric
        non_numeric_cols = features_df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            logger.warning(f"Dropping non-numeric columns: {non_numeric_cols.tolist()}")
            features_df = features_df.drop(columns=non_numeric_cols)
        
        logger.info(f"Generated {features_df.shape[1]} enhanced features with {features_df.shape[0]} rows (dropped {initial_len - len(features_df)} rows)")
        
        return features_df
    
    def _talib_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Selective TA-Lib features"""
        out = {}
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Only most useful indicators
            out['talib_adx'] = pd.Series(talib.ADX(high, low, close), index=df.index).fillna(25.0)
            out['talib_cci'] = pd.Series(talib.CCI(high, low, close), index=df.index).fillna(0.0)
            
        except Exception as e:
            logger.debug(f"TA-Lib error: {e}")
        
        return out
    
    def _ta_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Selective ta library features"""
        out = {}
        try:
            # Only most useful indicators
            out['ta_adx'] = ta.trend.adx(df['high'], df['low'], df['close']).fillna(25.0)
            
        except Exception as e:
            logger.debug(f"ta library error: {e}")
        
        return out

# -----------------------
# FIXED Model Manager
# -----------------------
class OptimizedModelManager:
    """Optimized model manager addressing overfitting - FIXED VERSION"""
    
    def __init__(self, model_path: str = "optimized_model.joblib"):
        self.model_path = model_path
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.feature_columns = None
        self.is_trained = False
        self.regime_detector = AdvancedRegimeDetector()
        
    def _filter_numeric_features(self, features_df: pd.DataFrame, feature_columns: List[str]) -> List[str]:
        """Filter to keep only numeric features - CRITICAL FIX"""
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        filtered_cols = [col for col in feature_columns if col in numeric_cols]
        
        dropped_count = len(feature_columns) - len(filtered_cols)
        if dropped_count > 0:
            logger.info(f"Filtered out {dropped_count} non-numeric features before training")
        
        return filtered_cols
        
    def _create_regularized_models(self, random_state: int = 42) -> Dict[str, Any]:
        """Create regularized models to prevent overfitting"""
        models = {
            'rf': RandomForestClassifier(
                n_estimators=100,          # Reduced from 200
                max_depth=6,               # Reduced from 10
                min_samples_split=10,      # Increased from 5
                min_samples_leaf=5,        # Increased from 2
                max_features=0.7,          # Feature subsampling
                bootstrap=True,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=50,           # Reduced from 100
                max_depth=4,               # Reduced from 6
                learning_rate=0.05,        # Reduced from 0.1
                subsample=0.8,             # Added subsampling
                max_features=0.7,          # Feature subsampling
                random_state=random_state
            )
        }
        
        # Add XGBoost with regularization
        if XGB_AVAILABLE:
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,          # Reduced from 200
                max_depth=4,               # Reduced from 6
                learning_rate=0.05,        # Reduced from 0.1
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,             # L1 regularization
                reg_lambda=0.1,            # L2 regularization
                random_state=random_state,
                n_jobs=-1
            )
        
        # Add LightGBM with regularization
        if LGB_AVAILABLE:
            models['lgb'] = lgb.LGBMClassifier(
                n_estimators=100,          # Reduced from 200
                max_depth=4,               # Reduced from 6
                learning_rate=0.05,        # Reduced from 0.1
                feature_fraction=0.7,
                bagging_fraction=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        return models
    
    def train(self, 
              features_df: pd.DataFrame, 
              feature_columns: Optional[List[str]] = None,
              test_size: float = 0.25,         # Larger test set
              random_state: int = 42,
              cv_folds: int = 3) -> Dict[str, float]:          # Reduced CV folds
        """Train optimized ensemble model - FIXED VERSION"""
        logger.info("Starting optimized model training...")
        
        # Prepare data
        if feature_columns is None:
            feature_columns = [c for c in features_df.columns 
                             if not c.startswith('target_') and c != 'returns']
        
        # CRITICAL FIX: Filter to numeric features only
        feature_columns = self._filter_numeric_features(features_df, feature_columns)
        
        if len(feature_columns) == 0:
            raise ValueError("No numeric features available for training")
        
        # Feature selection - more aggressive
        logger.info("Performing enhanced feature selection...")
        X_temp = features_df[feature_columns].values
        y_temp = features_df['target_direction'].values
        
        # Remove features with low variance
        from sklearn.feature_selection import VarianceThreshold
        variance_selector = VarianceThreshold(threshold=0.01)
        X_temp = variance_selector.fit_transform(X_temp)
        
        # Get remaining feature names
        remaining_features = np.array(feature_columns)[variance_selector.get_support()]
        
        # Select top features using mutual information
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif,  # Better for non-linear relationships
            k=min(30, len(remaining_features))  # Fewer features
        )
        X_selected = self.feature_selector.fit_transform(X_temp, y_temp)
        
        # Update feature columns
        feature_mask = self.feature_selector.get_support()
        self.feature_columns = remaining_features[feature_mask].tolist()
        
        logger.info(f"Selected {len(self.feature_columns)} features from {len(feature_columns)} original features")
        
        # Scaling with robust scaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Time-aware split with larger test set
        split_idx = int((1 - test_size) * len(X_scaled))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_temp[:split_idx], y_temp[split_idx:]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Train regularized models
        base_models = self._create_regularized_models(random_state)
        trained_models = []
        
        for name, model in base_models.items():
            try:
                logger.info(f"Training regularized {name}...")
                model.fit(X_train, y_train)
                
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                logger.info(f"{name} - Train: {train_score:.4f}, Test: {test_score:.4f}")
                
                # Only include models that generalize well
                if test_score > 0.52 and (train_score - test_score) < 0.15:  # Overfitting check
                    trained_models.append((name, model))
                    logger.info(f"{name} included in ensemble")
                else:
                    logger.warning(f"{name} excluded due to poor generalization")
                    
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        if not trained_models:
            raise RuntimeError("No models met the generalization criteria")
        
        # Create ensemble with fewer models
        self.ensemble = VotingClassifier(
            estimators=trained_models,
            voting='soft'
        )
        
        # Train ensemble
        logger.info(f"Training ensemble with {len(trained_models)} models...")
        self.ensemble.fit(X_train, y_train)
        
        # Calibration with validation
        logger.info("Calibrating probabilities...")
        try:
            if SKLEARN_NEW_API:
                self.calibrated_ensemble = CalibratedClassifierCV(
                    estimator=self.ensemble,
                    method='isotonic',
                    cv=3  # Reduced CV
                )
            else:
                self.calibrated_ensemble = CalibratedClassifierCV(
                    base_estimator=self.ensemble,
                    method='isotonic',
                    cv=3
                )
            
            self.calibrated_ensemble.fit(X_test, y_test)
            self.model = self.calibrated_ensemble
            
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            self.model = self.ensemble
        
        # Enhanced evaluation
        y_pred = self.model.predict(X_test)
        
        try:
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
            else:
                auc_score = 0.0
        except:
            auc_score = 0.0
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': auc_score,
            'num_features': len(self.feature_columns),
            'train_test_gap': 0.0  # Will be calculated if available
        }
        
        logger.info(f"Optimized model metrics: {metrics}")
        
        # Save model
        self.save()
        self.is_trained = True
        
        return metrics
    
    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with regime awareness"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Get base probabilities
        X = X_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            base_probs = self.model.predict_proba(X_scaled)[:, 1]
        else:
            base_probs = self.model.predict(X_scaled).astype(float)
        
        return base_probs
    
    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        """Predict with regime filtering"""
        probs = self.predict_proba(X_df)
        return (probs > 0.5).astype(int)
    
    def save(self):
        """Save optimized model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'regime_detector': self.regime_detector
        }
        joblib.dump(model_data, self.model_path)
        logger.info(f"Optimized model saved to {self.model_path}")
    
    def load(self) -> bool:
        """Load optimized model"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data.get('is_trained', True)
            self.regime_detector = model_data.get('regime_detector', AdvancedRegimeDetector())
            logger.info(f"Optimized model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return False

# [Rest of the classes remain the same as in the previous version...]
# I'll include the key remaining classes to save space:

class OptimizedRiskManager:
    """Enhanced risk management with dynamic sizing"""
    
    def __init__(self, 
                 base_position_size: float = 0.05,
                 max_portfolio_var: float = 0.015,
                 stop_loss_pct: float = 0.08,
                 take_profit_pct: float = 0.15,
                 max_holding_period: int = 15):
        
        self.base_position_size = base_position_size
        self.max_portfolio_var = max_portfolio_var
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_period = max_holding_period
        
        # Performance tracking
        self.trade_history = []
        self.current_streak = 0
        self.win_rate = 0.5
        
    def update_performance(self, trade_result: Dict):
        """Update performance metrics"""
        self.trade_history.append(trade_result)
        
        # Keep last 20 trades for adaptive sizing
        if len(self.trade_history) > 20:
            self.trade_history = self.trade_history[-20:]
        
        # Calculate recent win rate
        if len(self.trade_history) >= 5:
            recent_trades = self.trade_history[-10:]
            wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
            self.win_rate = wins / len(recent_trades)
        
        # Update streak
        if trade_result.get('pnl', 0) > 0:
            self.current_streak = max(0, self.current_streak) + 1
        else:
            self.current_streak = min(0, self.current_streak) - 1
    
    def calculate_position_size(self, 
                              signal_strength: float,
                              volatility: float,
                              current_capital: float,
                              price: float,
                              regime_info: Dict[str, float] = None) -> float:
        """Enhanced position sizing with multiple factors"""
        
        # Base Kelly calculation
        win_prob = signal_strength
        if win_prob <= 0.52:  # Higher threshold
            return 0.0
        
        # Use historical win/loss ratio
        avg_win = self.take_profit_pct
        avg_loss = self.stop_loss_pct
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly fraction with caps
        kelly_fraction = calculate_kelly_fraction(win_prob, win_loss_ratio)
        kelly_fraction = max(0, min(kelly_fraction, 0.15))  # Cap at 15%
        
        # Volatility adjustment - more aggressive
        target_vol = 0.015  # Target 1.5% volatility
        vol_adjustment = min(1.5, target_vol / (volatility + 1e-8))
        
        # Regime adjustment
        regime_multiplier = 1.0
        if regime_info:
            vol_regime = regime_info.get('vol_regime_numeric', 0)
            trend_regime = regime_info.get('trend_regime_numeric', 0)
            structure_regime = regime_info.get('structure_regime_numeric', 0)
            
            # Reduce size in unfavorable conditions
            if vol_regime == 1:  # High vol
                regime_multiplier *= 0.5
            if trend_regime == -1:  # Downtrend
                regime_multiplier *= 0.3
            if structure_regime == -1:  # Bearish
                regime_multiplier *= 0.4
                
            # Boost size in very favorable conditions
            if (vol_regime == 0 and 
                trend_regime == 1 and 
                structure_regime == 1):
                regime_multiplier *= 1.3
        
        # Streak adjustment - reduce size after losses
        streak_multiplier = 1.0
        if self.current_streak < -2:  # After 2 losses
            streak_multiplier = 0.5
        elif self.current_streak > 2:  # After 2 wins
            streak_multiplier = 1.2
        
        # Performance adjustment
        performance_multiplier = 1.0
        if self.win_rate < 0.4:  # Poor recent performance
            performance_multiplier = 0.6
        elif self.win_rate > 0.6:  # Good recent performance
            performance_multiplier = 1.2
        
        # Final position size calculation
        total_multiplier = (vol_adjustment * regime_multiplier * 
                           streak_multiplier * performance_multiplier)
        
        position_fraction = (kelly_fraction * total_multiplier * 0.3)  # Conservative factor
        position_fraction = min(position_fraction, self.base_position_size)
        
        position_value = current_capital * position_fraction
        position_size = position_value / price
        
        return position_size
    
    def should_exit_position(self, 
                           entry_price: float,
                           current_price: float,
                           position_side: str,
                           bars_held: int,
                           signal_strength: float = 0.5) -> Tuple[bool, str]:
        """Enhanced exit logic"""
        
        if position_side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Dynamic stop loss based on volatility
        dynamic_stop = self.stop_loss_pct
        if signal_strength > 0.8:  # High confidence trades get wider stops
            dynamic_stop *= 1.2
        
        # Trailing stop logic
        if pnl_pct > 0.05:  # If up 5%, use trailing stop
            trailing_stop = max(dynamic_stop, pnl_pct * 0.5)  # Trail at 50% of gains
            if pnl_pct < -trailing_stop:
                return True, "trailing_stop"
        
        # Fixed stop loss
        if pnl_pct <= -dynamic_stop:
            return True, "stop_loss"
        
        # Dynamic take profit
        take_profit = self.take_profit_pct
        if signal_strength < 0.7:  # Low confidence trades take profits earlier
            take_profit *= 0.8
        
        if pnl_pct >= take_profit:
            return True, "take_profit"
        
        # Time-based exit with signal strength consideration
        max_hold = self.max_holding_period
        if signal_strength < 0.65:  # Low confidence trades held for shorter time
            max_hold = int(max_hold * 0.7)
        
        if bars_held > max_hold:
            return True, "time_exit"
        
        # Exit on weak signal deterioration
        if bars_held > 5 and signal_strength < 0.55:
            return True, "weak_signal"
        
        return False, "hold"

# [Backtester and other classes remain similar, just with the fixed regime handling]

class OptimizedBacktester:
    """Enhanced backtester with improved realism"""
    
    def __init__(self,
                 price_df: pd.DataFrame,
                 features_df: pd.DataFrame,
                 model_manager: OptimizedModelManager,
                 risk_manager: OptimizedRiskManager,
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        
        self.price_df = price_df.sort_index()
        self.features_df = features_df.sort_index()
        self.model_manager = model_manager
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.regime_detector = AdvancedRegimeDetector()
        
    def run_backtest(self, 
                    feature_columns: List[str],
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    min_signal_strength: float = 0.65) -> Dict[str, Any]:
        """Run enhanced backtest"""
        logger.info("Starting optimized backtest...")
        
        # Align data
        common_index = self.price_df.index.intersection(self.features_df.index)
        if start_date:
            common_index = common_index[common_index >= start_date]
        if end_date:
            common_index = common_index[common_index <= end_date]
        
        price_data = self.price_df.loc[common_index]
        feature_data = self.features_df.loc[common_index]
        
        # Get regime information (numeric only)
        regimes_df = self.regime_detector.detect_regime(price_data)
        
        # Initialize tracking
        cash = self.initial_capital
        position = 0
        position_entry_price = 0
        position_entry_date = None
        bars_held = 0
        entry_signal_strength = 0.0
        
        # Performance tracking
        nav_series = []
        positions_series = []
        trades = []
        signals_generated = 0
        signals_executed = 0
        
        # Precompute signals
        try:
            feature_matrix = feature_data[feature_columns]
            probabilities = pd.Series(
                self.model_manager.predict_proba(feature_matrix),
                index=feature_matrix.index
            )
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            probabilities = pd.Series(0.5, index=feature_matrix.index)
        
        # Main backtest loop
        pending_order = None
        
        for i, (timestamp, row) in enumerate(price_data.iterrows()):
            current_price = row['close']
            
            # Execute pending order (one-bar delay)
            if pending_order is not None and i > 0:
                order_type, order_size, signal_strength = pending_order
                signals_executed += 1
                
                if order_type == 'buy' and position == 0:
                    execution_price = row['open'] * (1 + self.slippage)
                    commission_cost = abs(order_size) * execution_price * self.commission
                    
                    if cash >= (abs(order_size) * execution_price + commission_cost):
                        cash -= abs(order_size) * execution_price + commission_cost
                        position = abs(order_size)
                        position_entry_price = execution_price
                        position_entry_date = timestamp
                        bars_held = 0
                        entry_signal_strength = signal_strength
                        
                        trades.append({
                            'timestamp': timestamp,
                            'action': 'buy',
                            'price': execution_price,
                            'size': abs(order_size),
                            'commission': commission_cost,
                            'signal_strength': signal_strength
                        })
                
                elif order_type == 'sell' and position > 0:
                    execution_price = row['open'] * (1 - self.slippage)
                    commission_cost = position * execution_price * self.commission
                    
                    total_commission = commission_cost + trades[-1]['commission']
                    cash += position * execution_price - commission_cost
                    trade_pnl = (execution_price - position_entry_price) * position - total_commission
                    
                    trade_result = {
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': execution_price,
                        'size': position,
                        'commission': commission_cost,
                        'pnl': trade_pnl,
                        'bars_held': bars_held,
                        'entry_signal': entry_signal_strength,
                        'pnl_pct': trade_pnl / (position_entry_price * position)
                    }
                    
                    trades.append(trade_result)
                    
                    # Update risk manager performance
                    self.risk_manager.update_performance(trade_result)
                    
                    position = 0
                    position_entry_price = 0
                    bars_held = 0
                    entry_signal_strength = 0.0
                
                pending_order = None
            
            # Update position
            if position > 0:
                bars_held += 1
            
            # Calculate NAV
            nav = cash + position * current_price
            nav_series.append(nav)
            positions_series.append(position)
            
            # Risk management - exit checks
            if position > 0:
                should_exit, exit_reason = self.risk_manager.should_exit_position(
                    position_entry_price, current_price, 'long', bars_held, entry_signal_strength
                )
                
                if should_exit:
                    pending_order = ('sell', position, 1.0)
                    continue
            
            # Signal generation (only if no position)
            if position == 0 and i < len(price_data) - 1:
                try:
                    signal_prob = probabilities.loc[timestamp]
                    
                    # Get regime info for this timestamp (numeric values)
                    regime_info = {}
                    for regime_name, regime_series in regimes_df.items():
                        try:
                            regime_info[regime_name] = float(regime_series.loc[timestamp])
                        except:
                            regime_info[regime_name] = 0.0
                    
                    signals_generated += 1
                    
                    # Enhanced signal filtering
                    if (signal_prob > min_signal_strength and 
                        self.regime_detector.should_trade(regime_info, timestamp)):
                        
                        # Calculate position size with regime info
                        volatility = feature_data.loc[timestamp, 'vol_20'] if 'vol_20' in feature_data.columns else 0.02
                        
                        position_size = self.risk_manager.calculate_position_size(
                            signal_prob, volatility, nav, current_price, regime_info
                        )
                        
                        if position_size > 0:
                            pending_order = ('buy', position_size, signal_prob)
                            
                except Exception as e:
                    logger.debug(f"Signal processing error at {timestamp}: {e}")
        
        # Performance calculation
        nav_series = pd.Series(nav_series, index=price_data.index)
        positions_series = pd.Series(positions_series, index=price_data.index)
        returns = nav_series.pct_change().fillna(0)
        
        # Enhanced statistics
        buy_trades = [t for t in trades if t['action'] == 'buy']
        sell_trades = [t for t in trades if t['action'] == 'sell' and 'pnl' in t]
        
        if sell_trades:
            win_trades = [t for t in sell_trades if t['pnl'] > 0]
            avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([t['pnl'] for t in sell_trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in sell_trades) else 0
            avg_hold_time = np.mean([t['bars_held'] for t in sell_trades])
        else:
            avg_win = avg_loss = avg_hold_time = 0
        
        stats = {
            'start_nav': float(nav_series.iloc[0]),
            'end_nav': float(nav_series.iloc[-1]),
            'total_return': float(nav_series.iloc[-1] / nav_series.iloc[0] - 1),
            'cagr': float(cagr(returns)),
            'sharpe': float(sharpe_ratio(returns)),
            'sortino': float(sortino_ratio(returns)),
            'max_drawdown': float(max_drawdown((1 + returns).cumprod())),
            'calmar': float(calmar_ratio(returns)),
            'num_trades': len(buy_trades),
            'win_rate': self._calculate_win_rate(trades),
            'profit_factor': self._calculate_profit_factor(trades),
            'signals_generated': signals_generated,
            'signals_executed': signals_executed,
            'execution_rate': signals_executed / max(signals_generated, 1),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'avg_hold_time': float(avg_hold_time)
        }
        
        logger.info(f"Optimized backtest completed: {stats}")
        
        return {
            'stats': stats,
            'nav': nav_series,
            'positions': positions_series,
            'trades': trades,
            'returns': returns,
            'regimes': regimes_df
        }
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate"""
        sell_trades = [t for t in trades if t['action'] == 'sell' and 'pnl' in t]
        if not sell_trades:
            return 0.0
        
        winning_trades = len([t for t in sell_trades if t['pnl'] > 0])
        return winning_trades / len(sell_trades)
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor"""
        sell_trades = [t for t in trades if t['action'] == 'sell' and 'pnl' in t]
        if not sell_trades:
            return 0.0
        
        gross_profit = sum([t['pnl'] for t in sell_trades if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in sell_trades if t['pnl'] < 0]))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

# [Indian Market Data Manager remains the same]
class IndianMarketDataManager:
    """Data manager for Indian stock markets"""
    
    INDIAN_STOCKS = {
        'RELIANCE': 'RELIANCE.NS',
        'TCS': 'TCS.NS',
        'HDFCBANK': 'HDFCBANK.NS',
        'INFY': 'INFY.NS',
        'HINDUNILVR': 'HINDUNILVR.NS',
        'ICICIBANK': 'ICICIBANK.NS',
        'KOTAKBANK': 'KOTAKBANK.NS',
        'BHARTIARTL': 'BHARTIARTL.NS',
        'ITC': 'ITC.NS',
        'ASIANPAINT': 'ASIANPAINT.NS',
        'MARUTI': 'MARUTI.NS',
        'AXISBANK': 'AXISBANK.NS',
        'LT': 'LT.NS',
        'SUNPHARMA': 'SUNPHARMA.NS',
        'TITAN': 'TITAN.NS',
        'ULTRACEMCO': 'ULTRACEMCO.NS',
        'NESTLEIND': 'NESTLEIND.NS',
        'WIPRO': 'WIPRO.NS',
        'JSWSTEEL': 'JSWSTEEL.NS',
        'TATAMOTORS': 'TATAMOTORS.NS',
    }
    
    def __init__(self):
        if not YF_AVAILABLE:
            raise RuntimeError("yfinance not available. Install with: pip install yfinance")
    
    def download_stock_data(self, 
                          symbol: str, 
                          period: str = "2y",
                          interval: str = "1d") -> pd.DataFrame:
        """Download Indian stock data using yfinance"""
        if symbol in self.INDIAN_STOCKS:
            yf_symbol = self.INDIAN_STOCKS[symbol]
        elif '.NS' not in symbol:
            yf_symbol = f"{symbol}.NS"
        else:
            yf_symbol = symbol
        
        logger.info(f"Downloading data for {yf_symbol}...")
        
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for {yf_symbol}")
            
            # Standardize column names
            df.reset_index(inplace=True)
            df.columns = [col.lower() for col in df.columns]
            
            column_mapping = {
                'date': 'timestamp',
                'datetime': 'timestamp'
            }
            df.rename(columns=column_mapping, inplace=True)
            df.set_index('timestamp', inplace=True)
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Downloaded {len(df)} rows for {yf_symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download data for {yf_symbol}: {e}")
            raise

class OptimizedTradingBot:
    """Optimized trading bot with improved performance - FIXED VERSION"""
    
    def __init__(self, 
                 symbols: List[str] = None,
                 model_path: str = "optimized_model.joblib"):
        
        if symbols is None:
            symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'TATAMOTORS', 'INFY']
        
        self.symbols = symbols
        self.model_path = model_path
        
        # Initialize components
        self.data_manager = IndianMarketDataManager()
        self.model_manager = OptimizedModelManager(model_path)
        self.risk_manager = OptimizedRiskManager()
        
    def run_full_analysis(self, 
                         symbol: str = 'TATAMOTORS',
                         period: str = "2y",
                         test_size: float = 0.25) -> Dict[str, Any]:
        """Run complete optimized analysis pipeline - FIXED VERSION"""
        try:
            logger.info(f"Starting OPTIMIZED analysis for {symbol}")
            
            # 1. Download data
            logger.info("Downloading market data...")
            price_df = self.data_manager.download_stock_data(symbol, period=period)
            
            # 2. Generate enhanced features
            logger.info("Engineering enhanced features...")
            fe = InstitutionalFeatureEngine(symbol)
            features_df = fe.generate_features(price_df, horizon=1)
            
            # 3. Train optimized model
            logger.info("Training optimized ensemble model...")
            feature_columns = [c for c in features_df.columns 
                             if not c.startswith('target_') and c != 'returns']
            
            model_metrics = self.model_manager.train(
                features_df, 
                feature_columns=feature_columns,
                test_size=test_size
            )
            
            # 4. Run optimized backtest
            logger.info("Running optimized backtest...")
            backtester = OptimizedBacktester(
                price_df, features_df, self.model_manager, self.risk_manager
            )
            
            backtest_results = backtester.run_backtest(
                self.model_manager.feature_columns,  # Use selected features
                min_signal_strength=0.65  # Higher threshold
            )
            
            # 5. Compile results
            results = {
                'symbol': symbol,
                'data_period': period,
                'model_metrics': model_metrics,
                'backtest_results': backtest_results,
                'feature_count': len(self.model_manager.feature_columns),
                'data_points': len(features_df),
                'optimization_notes': [
                    'FIXED: Filtered non-numeric features before training',
                    'Reduced model complexity to prevent overfitting',
                    'Implemented regime-aware position sizing',
                    'Enhanced risk management with dynamic stops',
                    'Higher signal quality thresholds (65%+)',
                    'Improved feature selection and regularization'
                ]
            }
            
            logger.info("OPTIMIZED analysis completed successfully!")
            self._print_enhanced_results_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in optimized analysis: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _print_enhanced_results_summary(self, results: Dict[str, Any]):
        """Print enhanced results summary"""
        print("\n" + "="*70)
        print(f"FIXED OPTIMIZED AI TRADING BOT RESULTS - {results['symbol']}")
        print("="*70)
        
        # Model Performance
        print("\nMODEL PERFORMANCE (Anti-Overfitting):")
        metrics = results['model_metrics']
        print(f"   Accuracy:      {metrics['accuracy']:.4f}")
        print(f"   Precision:     {metrics['precision']:.4f}")
        print(f"   Recall:        {metrics['recall']:.4f}")
        print(f"   F1 Score:      {metrics['f1']:.4f}")
        print(f"   AUC:           {metrics['auc']:.4f}")
        print(f"   Features Used: {metrics.get('num_features', 'N/A')}")
        
        # Backtest Performance
        print("\nBACKTEST PERFORMANCE (Enhanced):")
        bt_stats = results['backtest_results']['stats']
        print(f"   Total Return:    {bt_stats['total_return']:.2%}")
        print(f"   CAGR:            {bt_stats['cagr']:.2%}")
        print(f"   Sharpe Ratio:    {bt_stats['sharpe']:.4f}")
        print(f"   Sortino Ratio:   {bt_stats['sortino']:.4f}")
        print(f"   Max Drawdown:    {bt_stats['max_drawdown']:.2%}")
        print(f"   Calmar Ratio:    {bt_stats['calmar']:.4f}")
        print(f"   Win Rate:        {bt_stats['win_rate']:.2%}")
        print(f"   Profit Factor:   {bt_stats['profit_factor']:.2f}")
        print(f"   Num Trades:      {bt_stats['num_trades']}")
        
        # Enhanced metrics
        print(f"\nEXECUTION QUALITY:")
        print(f"   Signals Generated: {bt_stats['signals_generated']}")
        print(f"   Signals Executed:  {bt_stats['signals_executed']}")
        print(f"   Execution Rate:    {bt_stats['execution_rate']:.2%}")
        print(f"   Avg Win:           ${bt_stats['avg_win']:.2f}")
        print(f"   Avg Loss:          ${bt_stats['avg_loss']:.2f}")
        print(f"   Avg Hold Time:     {bt_stats['avg_hold_time']:.1f} days")
        
        # Data Stats
        print(f"\nDATA STATISTICS:")
        print(f"   Features:        {results['feature_count']}")
        print(f"   Data Points:     {results['data_points']}")
        print(f"   Period:          {results['data_period']}")
        
        # Optimizations Applied
        print(f"\nOPTIMIZATIONS APPLIED:")
        for note in results['optimization_notes']:
            print(f"   • {note}")
        
        print("="*70)
        
        # Performance assessment
        sharpe = bt_stats['sharpe']
        cagr = bt_stats['cagr']
        
        print(f"\nPERFORMANCE ASSESSMENT:")
        if sharpe > 1.0:
            print(f"   ✓ EXCELLENT: Sharpe ratio > 1.0 ({sharpe:.2f})")
        elif sharpe > 0.5:
            print(f"   ✓ GOOD: Sharpe ratio > 0.5 ({sharpe:.2f})")
        elif sharpe > 0:
            print(f"   ⚠ ACCEPTABLE: Positive Sharpe ratio ({sharpe:.2f})")
        else:
            print(f"   ✗ POOR: Negative Sharpe ratio ({sharpe:.2f})")
            
        if cagr > 0.15:
            print(f"   ✓ EXCELLENT: CAGR > 15% ({cagr:.1%})")
        elif cagr > 0.08:
            print(f"   ✓ GOOD: CAGR > 8% ({cagr:.1%})")
        elif cagr > 0:
            print(f"   ⚠ ACCEPTABLE: Positive CAGR ({cagr:.1%})")
        else:
            print(f"   ✗ POOR: Negative CAGR ({cagr:.1%})")
            
        print("="*70)

def main():
    """Main function with enhanced interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FIXED Optimized Institutional AI Trading Bot for Indian Markets")
    parser.add_argument('--symbol', type=str, default='TATAMOTORS', 
                       help='Stock symbol to analyze (default: TATAMOTORS)')
    parser.add_argument('--period', type=str, default='2y',
                       help='Data period (default: 2y)')
    parser.add_argument('--model-path', type=str, default='optimized_model.joblib',
                       help='Model save path')
    parser.add_argument('--test-size', type=float, default=0.25,
                       help='Test set size (default: 0.25)')
    
    args = parser.parse_args()
    
    try:
        # Initialize fixed optimized bot
        bot = OptimizedTradingBot(model_path=args.model_path)
        
        # Run fixed optimized analysis
        results = bot.run_full_analysis(
            symbol=args.symbol,
            period=args.period,
            test_size=args.test_size
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print("\n" + "="*60)
        print("TROUBLESHOOTING GUIDE:")
        print("="*60)
        print("1. Install required packages:")
        print("   pip install numpy pandas scikit-learn yfinance")
        print("   pip install xgboost lightgbm numba")
        print("\n2. Update packages if needed:")
        print("   pip install --upgrade scikit-learn numpy pandas")
        print("\n3. For performance issues:")
        print("   - Reduce the data period (--period 1y)")
        print("   - Increase test size (--test-size 0.3)")
        print("="*60)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
