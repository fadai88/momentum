import pandas as pd
import numpy as np
from scipy import stats
from heapq import nlargest

# Load and preprocess data
close_prices = pd.read_csv("usdt_price_data_updated.csv")
close_prices['timestamp'] = pd.to_datetime(close_prices['timestamp'])
close_prices.set_index('timestamp', inplace=True)
fan_tokens = ['SANTOS', 'ACM', 'BAR', 'ASR', 'PSG', 'PORTO', 'LAZIO', 'JUV', 'CITY', 'ATM', 'ACM']
close_prices = close_prices.drop(columns=[c for c in fan_tokens if c in close_prices.columns])

# Define functions
def slope(ts):
    ts = ts.dropna()
    if len(ts) < 2:
        return np.nan
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    if np.any(np.isinf(log_ts)) or np.any(np.isnan(log_ts)):
        return np.nan
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(slope), 365) - 1) * 100
    return np.log1p(abs(annualized_slope)) * np.sign(annualized_slope) * (r_value ** 2)

def information_discreteness(df):
    df = df.dropna()
    if len(df) < 2:
        return np.nan
    returns = df.pct_change(fill_method=None).dropna()
    if len(returns) == 0:
        return np.nan
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    negative_positive_diff = (negative_returns.count() / len(df)) - (positive_returns.count() / len(df))
    cumul_return = df.iloc[-1] / df.iloc[0] - 1
    sign = lambda x: 1 if x > 0 else -1 if x < 0 else 0
    return sign(cumul_return) * negative_positive_diff

# Trading strategy function
def crypto_momentum_double_sorted(df, lookback, last_days, holding_days, threshold, number_of_tokens=10, commission=0.001):
    weekly_returns = []
    btc_price = df.iloc[:, 0]
    for i in range(lookback, len(df) - holding_days + 1, holding_days):
        if btc_price.iloc[i] / btc_price.iloc[i-lookback] - 1 > threshold:
            total = 0
            past_returns = {}
            id_values = {}
            # Calculate momentum and ID for the lookback period
            for col in df.columns[2:]:
                if not np.isnan(df[col].iloc[i]):
                    ts = df[col].iloc[i-lookback:i-last_days]
                    if len(ts.dropna()) >= 2:
                        past_returns[col] = slope(ts)
                        id_values[col] = information_discreteness(ts)
            
            # Sort by momentum into quintiles
            momentum_quintiles = pd.qcut(list(past_returns.values()), 5, labels=False, duplicates='drop')
            momentum_dict = {col: quintile for col, quintile in zip(past_returns.keys(), momentum_quintiles)}
            
            # Sort by ID within high-momentum quintile (4) into quintiles
            high_mom_tokens = [col for col in past_returns if momentum_dict[col] == 4]
            if high_mom_tokens:
                id_quintiles = pd.qcut([id_values[col] for col in high_mom_tokens], 5, labels=False, duplicates='drop')
                id_dict = {col: quintile for col, quintile in zip(high_mom_tokens, id_quintiles)}
                low_id_tokens = [col for col in high_mom_tokens if id_dict[col] == 0]  # Select low-ID (0)
                
                # Rank by momentum within high-momentum, low-ID tokens
                slope_ranks = {col: past_returns[col] for col in low_id_tokens}
                top_tokens = nlargest(number_of_tokens, slope_ranks, key=slope_ranks.get)
                
                for coin in top_tokens:
                    try:
                        weekly_return = (df[coin].iloc[i+holding_days-1] * (1-commission)) / (df[coin].iloc[i] * (1+commission)) - 1
                        total += weekly_return
                    except:
                        pass
                avg_weekly_return = total / number_of_tokens if top_tokens else 0
            else:
                avg_weekly_return = 0
            weekly_returns.append(avg_weekly_return)
        else:
            weekly_returns.append(0)
    return [x for x in weekly_returns if str(x) != 'nan']

# Example usage
lookback = 35
holding_days = 7
threshold = 0.05
number_of_tokens = 10
commission = 0.001

weekly_returns = crypto_momentum_double_sorted(close_prices, lookback, 0, holding_days, threshold, number_of_tokens, commission)

# Calculate performance metrics
def geom_return(returns, holding_days):
    returns = [i + 1 for i in returns]
    cumulative_returns = np.cumprod(returns)
    geometric_return = cumulative_returns[-1] ** (1/len(cumulative_returns)) - 1
    return (1 + geometric_return) ** (365/holding_days) - 1

def calculate_max_drawdown(returns):
    returns = [i + 1 for i in returns]
    cumulative_returns = np.cumprod(returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return np.min(drawdown)

weekly_returns = [x for x in weekly_returns if str(x) != 'nan']
annual_return = geom_return(weekly_returns, holding_days)
mdd = calculate_max_drawdown(weekly_returns)

print(f"Annualized Return: {annual_return:.2%}")
print(f"Maximum Drawdown: {mdd:.2%}")