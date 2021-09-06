import requests, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

# saquen su API key en https://developer.tdameritrade.com/content/getting-started
c_key = 'AHSA7C8FWZZFRG1W15WOJLQY8WRSCOVI'
plt.style.use('dark_background')


def options(symbol):
    params = {'apikey': c_key, 'symbol': symbol}
    endpoint = 'https://api.tdameritrade.com/v1/marketdata/chains'
    r = requests.get(url=endpoint, params=params)
    return r.json()


def optionsDF(chain, k_min=0, k_max=1000, ttm_min=10, ttm_max=700):
    v_calls = list(chain['callExpDateMap'].values())
    v_puts = list(chain['putExpDateMap'].values())
    calls, puts = [], []

    for i in range(len(v_calls)):
        v = list(v_calls[i].values())
        for j in range(len(v)):
            calls.append(v[j][0])

    for i in range(len(v_puts)):
        v = list(v_puts[i].values())
        for j in range(len(v)):
            puts.append(v[j][0])

    contracts = pd.concat([pd.DataFrame(calls), pd.DataFrame(puts)])
    tabla = contracts.loc[contracts.daysToExpiration > 0].copy()
    tabla['ticker'] = chain['symbol']
    df_ok = tabla.loc[(tabla['strikePrice'] > k_min) & (tabla['strikePrice'] < k_max)]
    df_ok = df_ok.loc[(df_ok['daysToExpiration'] > ttm_min) & (df_ok['daysToExpiration'] < ttm_max)]
    return df_ok


def prepararMalla(columna, df, leg=None):
    if leg:
        df = df.loc[df['putCall'] == leg].copy()

    df_ok = df.loc[:, ['strikePrice', 'daysToExpiration', columna]]
    df_ok = df_ok.replace('NaN', np.nan).dropna()
    x_q = len(df_ok['strikePrice'].unique())
    y_q = len(df_ok['daysToExpiration'].unique())
    x1 = np.linspace(df_ok['strikePrice'].min(), df_ok['strikePrice'].max(), x_q)
    y1 = np.linspace(df_ok['daysToExpiration'].min(), df_ok['daysToExpiration'].max(), y_q)
    X, Y = np.meshgrid(x1, y1)
    Z = interpolate.griddata((df_ok['strikePrice'], df_ok['daysToExpiration']), df_ok[columna], (X, Y))
    return X, Y, Z, df_ok


def grafCols(cols, leg=None, size=(16, 6)):
    fig = plt.figure(figsize=size)
    ax = [fig.add_subplot(1, len(cols), i + 1, projection='3d') for i in range(len(cols))]
    for i in range(len(cols)):
        col = cols[i]
        df_greeks = data.copy()
        X, Y, Z, df = prepararMalla(col, df_greeks, leg=leg)
        ax[i].plot_wireframe(X, Y, Z, color='orange', lw=2)
        ax[i].set_title(f'{col} ({leg})', fontsize=16, color='w')
        ax[i].set_xlabel('Strikes', fontsize=16, color='w')
        ax[i].set_ylabel('TTM', fontsize=16, color='w')
        ax[i].set_zlabel(col, fontsize=16, color='w')
        ax[i].w_xaxis.set_pane_color((0, 0, 0, 0))
        ax[i].w_yaxis.set_pane_color((0, 0, 0, 0))
        ax[i].w_zaxis.set_pane_color((0, 0, 0, 0))


ticker = 'AAPL'
data = optionsDF(options(ticker), k_min=100, k_max=200, ttm_min=10, ttm_max=180)
grafCols(['delta', 'vega'], 'CALL')
# grafCols(['gamma','theta'], 'CALL')
# grafCols(['rho','volatility'], 'CALL')
# grafCols(['percentChange','openInterest'], 'CALL')

# grafCols(['theoreticalOptionValue','timeValue'], 'CALL')
# grafCols(['timeValue','theoreticalOptionValue'], 'PUT')
plt.suptitle('Sensibilidad y griegas opciones de AAPL')
plt.show()





import yfinance as yf

ggal = yf.download('GGAL.BA', auto_adjust=True, start="2021-01-01")['Close']
sigma = ggal.pct_change().rolling(40).std().mul(250**0.5).mul(100)
sigma.plot(figsize=(12,6), title="Volatilidad de GGAL 40 días en ARS")