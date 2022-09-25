import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
from functions import start_of_year

# Define variables
tickers = ['VBR', 'TLT', 'TIP', 'QQQ']
benchmark = ['SPY']
start = '2005-01-01'
end = '2022-01-01'
portfolio_value = 5000  # Amount in dollars for initial portfolio value

# Get and process data
# Ticker data
prices = yf.download(tickers=tickers, start=start, end=end)['Adj Close']
daily_ret = np.log(prices / prices.shift(1))[1:]
daily_ret_col = list(daily_ret.columns)
monthly_ret = daily_ret.groupby(pd.Grouper(freq='M')).apply(np.sum)

# Benchmark data
prices_benchmark = yf.download(tickers=benchmark, start=start, end=end)['Adj Close']
prices_benchmark_daily_ret = np.log(prices_benchmark / prices_benchmark.shift(1))[1:].to_frame()

# Define trading dates
month_end = pd.date_range(start, end, freq='M').strftime('%Y-%m-%d').tolist()
month_start = [start_of_year(x) for x in month_end]
date_range = list(zip(month_start, month_end))
trading_month_start = []
trading_month_end = []
nyse = mcal.get_calendar('NYSE')
for k, v in date_range:
    nyse_trading_date_range = nyse.schedule(k, v)
    nyse_trading_date_range_index = mcal.date_range(nyse_trading_date_range, frequency='1D') \
        .strftime('%Y-%m-%d') \
        .tolist()
    trading_month_start.append(nyse_trading_date_range_index[0])
    trading_month_end.append(nyse_trading_date_range_index[-1])
trading_date_range = list(zip(trading_month_start, trading_month_end))

# Calculate momentum
momentum = pd.DataFrame()
for col in daily_ret:
    momentum[col + '_one'] = daily_ret[col].rolling(window=21).sum()
    momentum[col + '_three'] = daily_ret[col].rolling(window=63).sum()
    momentum[col + '_six'] = daily_ret[col].rolling(window=126).sum()
momentum.dropna(inplace=True)
prices = prices.iloc[126:]
daily_ret = daily_ret[126:]
prices_benchmark_daily_ret = prices_benchmark_daily_ret[126:]
momentum['QQQ'] = momentum.iloc[:, :3].mean(axis=1)
momentum['VBR'] = momentum.iloc[:, 9:12].mean(axis=1)

# Daily signals
condition_1 = [(momentum['QQQ'] > momentum['VBR']) & (momentum['QQQ'] > 0)]
momentum['signal1'] = np.select(condition_1, ['QQQ'])
condition_2 = [(momentum['VBR'] > momentum['QQQ']) & (momentum['VBR'] > 0)]
momentum['signal2'] = np.select(condition_2, ['VBR'])
condition_3 = [(momentum['signal1'] == str(0)) & (momentum['signal2'] == str(0))]
momentum['signal3'] = np.select(condition_3, ['bond'])
condition_4 = [(momentum['TLT_one'] > momentum['TIP_one'])]
momentum['signal4'] = np.select(condition_4, ['TLT'])
condition_5 = [(momentum['TIP_one'] > momentum['TLT_one'])]
momentum['signal5'] = np.select(condition_5, ['TLT'])
cols = ['signal1', 'signal2', 'signal3', 'signal4', 'signal5']
momentum[cols] = momentum[cols].replace({str(0): ''})
momentum['bond'] = (momentum['signal4'] + momentum['signal5'])
momentum['stock'] = (momentum['signal1'] + momentum['signal2'])
momentum.loc[momentum['signal3'] == 'bond', 'stock'] = momentum['bond']

# Get monthly signal
prices_index = prices.index.to_list()
prices_index_dataset = []
for idx, val in enumerate(prices_index):
    for x in trading_month_start:
        if val.strftime('%Y-%m-%d') == x:
            prices_index_dataset.append(momentum.iloc[[idx], [-1]])
result = np.reshape(prices_index_dataset, (np.shape(prices_index_dataset)[0], np.shape(prices_index_dataset)[1]))
result_df = pd.DataFrame(result, columns=['signal'])
result_df.index = trading_month_start[-len(result_df):]

# Expand monthly signal to daily
nyse_trading_date_range = nyse.schedule(trading_month_start[0], trading_month_end[-1])
nyse_trading_date_range_index = mcal.date_range(nyse_trading_date_range, frequency='1D') \
    .strftime('%Y-%m-%d') \
    .tolist()
result_df = result_df.reindex(nyse_trading_date_range_index, method='ffill').reset_index(drop=True)
daily_ret = daily_ret.reset_index(drop=True)
final_df = pd.concat([daily_ret, result_df], axis=1)[-len(nyse_trading_date_range_index):]
final_df.index = nyse_trading_date_range_index

# Convert signal to daily ret
final_df['signal'] = np.where(final_df['signal'] == 'VBR', final_df['VBR'], final_df['signal'])
final_df['signal'] = np.where(final_df['signal'] == 'QQQ', final_df['QQQ'], final_df['signal'])
final_df['signal'] = np.where(final_df['signal'] == 'TLT', final_df['TLT'], final_df['signal'])
final_df['signal'] = np.where(final_df['signal'] == 'TIP', final_df['TLT'], final_df['signal'])
final_df['signal'] += 1

# Create portfolio column
final_df['Portfolio Value'] = 0
final_df.at[nyse_trading_date_range_index[0], 'Portfolio Value'] = portfolio_value
final_df = final_df.reset_index(drop=False)
final_df['Portfolio Value'] = final_df['Portfolio Value'].replace({np.nan: 0})
final_df.loc[final_df['signal'].isna(), ['signal']] = 1
for i, row in final_df.iterrows():
    if i == 0:
        final_df.loc[i, 'Portfolio Value'] = final_df['Portfolio Value'].iat[0]
    else:
        final_df.loc[i, 'Portfolio Value'] = final_df.loc[i, 'signal'] * final_df.loc[i - 1, 'Portfolio Value']
final_df.drop(final_df.index[-1], inplace=True)
final_df.to_csv('final.csv')

# Create benchmark column
prices_benchmark_daily_ret.columns = ['SPY']
prices_benchmark_daily_ret = prices_benchmark_daily_ret.reset_index(drop=False)
benchmark_df = pd.concat([final_df, prices_benchmark_daily_ret], axis=1)
benchmark_df['SPY'] += 1
benchmark_df['Market Returns'] = 0
benchmark_df.at[0, 'Market Returns'] = portfolio_value
benchmark_df['Market Returns'] = benchmark_df['Market Returns'].replace({np.nan: 0})
for i, row in benchmark_df.iterrows():
    if i == 0:
        benchmark_df.loc[i, 'Market Returns'] = benchmark_df['Market Returns'].iat[0]
    else:
        benchmark_df.loc[i, 'Market Returns'] = benchmark_df.loc[i, 'SPY'] * benchmark_df.loc[i - 1, 'Market Returns']
final_df = benchmark_df.set_index('Date')
benchmark_df = benchmark_df.set_index('Date')
final_df['Portfolio Value'].plot(label='Portfolio Value')
benchmark_df['Market Returns'].plot(label='SPY')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Calculate portfolio statistics
# Calculate max drawdown
rolling_max = final_df['Portfolio Value'].rolling(252, min_periods=1).max()
daily_drawdown = final_df['Portfolio Value']/rolling_max - 1.0
max_daily_drawdown = daily_drawdown.rolling(252, min_periods=1).min()
daily_drawdown.plot()
plt.xticks(rotation=45)
plt.title('Portfolio Max Drawdown')
plt.show()
print('----------------------------------------')
print('Max portfolio drawdown: {:.2%}'.format(round((daily_drawdown.min()), 2)))

# Calculate portfolio return statistics
# Annual portfolio returns
final_df['signal'] -= 1
portfolio_annual_return = final_df['signal'].rolling(252).sum().mean()
print('Average annual portfolio return: {:.2%}'.format(portfolio_annual_return))

# Portfolio Sharpe
portfolio_sharpe = final_df['signal'].mean() / final_df['signal'].std()
portfolio_sharpe_annualised = (250**0.5) * portfolio_sharpe
print('Portfolio Sharpe ratio: {:.2}'.format(portfolio_sharpe_annualised))
print('----------------------------------------')

# Cumulative returns graph
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.hist(final_df['signal'], bins=60)
ax1.set_xlabel('Portfolio Returns')
ax1.set_ylabel('Freq')
ax1.set_title('Portfolio Returns Histogram')
plt.show()
