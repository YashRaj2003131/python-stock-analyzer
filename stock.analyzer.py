import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

while True:

    # User input
    symbol = input('\nEnter stock symbol (e.g. RELIANCE.NS): ')

    # Create ticker object
    stock = yf.Ticker(symbol)

    # Company information
    info = stock.info

    print('\n===== STOCK ANALYZER =====')
    print('What do you want to know about', symbol)
    print('1. Company Information')
    print('2. Fundamental Analysis')
    print('3. Technical Analysis')
    print('4. Risk Analysis')
    print('5. CAPM Analysis')
    print('6. Monte Carlo Simulation')
    print('7. Exit')

    choice = input('Enter your choice: ')

    if choice == '1':

        company_name = info.get('longName')
        current_price = info.get('currentPrice')
        sector = info.get('sector')
        industry = info.get('industry')
        website = info.get('website')
        market_cap = info.get('marketCap')
        summary = info.get('longBusinessSummary')

        print('\n===== COMPANY INFORMATION =====')
        print('Company Name :', company_name)
        print('Current Price :', current_price)
        print('Sector :', sector)
        print('Industry :', industry)
        print('Website :', website)

        if market_cap:
            print('Market Cap : ₹', round(market_cap / 1e9, 2), 'Billion')

        print('\nBusiness Summary:')
        print(summary)
        break

    elif choice == '2':

        print('\n===== FUNDAMENTAL ANALYSIS =====')

        pe_rato = info.get('trailingPE')
        eps = info.get('trailingEps')
        dividemd_yield = info.get('dividendYield')
        high_52 = info.get('fiftyTwoWeekHigh')
        low_52 = info.get('fiftyTwoWeekLow')
        book_value = info.get('bookValue')
        roe = info.get('returnOnEquity')
        debt_to_equity = info.get('debtToEquity')

        print('P/E Ratio      :', pe_rato)
        print('EPS            :',eps)

        if dividemd_yield:
            print('Dividend Yield:', round(dividemd_yield * 100, 2),'%')
        else:
            print('Dividend Yield: N/A',)

        print('52 Week High   :', high_52)
        print('52 Week Low    :', low_52)
        print('Book Value.    :',book_value)

        if roe:
            print('ROE            :', round(roe * 100, 2), '%')
        else:
            print('ROE            : N/A')

        print('Debt to Equity :', debt_to_equity)
        break

    elif choice == '3':
        print('\n===== TECHNICAL ANALYSIS =====')

        data = stock.history(period = '1y')

        close_price = data['Close']

        ma_50 = close_price.rolling(window=50).mean().iloc[-1]
        ma_200 = close_price.rolling(window=200).mean().iloc[-1]

        current_price = close_price.iloc[-1]

        print('Current Price    :', round(current_price, 2))
        print('50-Day MA        :', round(ma_50, 2))
        print('200-Day MA       :', round(ma_200, 2))

        if current_price > ma_50:
            print('Trend vs 50-Day MA : Buillish')
        else:
            print('Trend vs 50-Day MA : Bearish')

        if current_price > ma_200:
            print('Trend vs 200-Day MA : Buillish')
        else:
            print('Trend vs 200-Day MA : Bearish')


        break

    elif choice == '4':
         
        print('\n===== RISK ANALYSIS =====')

        data = stock.history(period = '5y')

        returns = data['Close'].pct_change(fill_method=None).dropna()

        annual_return = returns.mean() * 250

        variance = returns.var()

        std_dev = returns.std()

        annual_volatility = std_dev * np.sqrt(250)

        if annual_volatility < 0.20:
            risk_level = 'LOW'
        elif annual_volatility < 0.35:
            risk_level = 'MODERATE'
        else:
            risk_level = 'HIGH'

        print('Annual Return          :', round(annual_return * 100, 2), '%')
        print('Variance               :', round(variance * 100, 2), '%')
        print('Daily Std Deviation    :', round(std_dev * 100, 2), '%')
        print('Annual Volatility      :', round(annual_volatility * 100, 2), '%')
        print('Risk Indicator         :', risk_level)
        break

    elif choice == '5':
        
        stock_data = stock.history(period='5y')
        stock_data.index = stock_data.index.tz_localize(None)

        market_data = yf.download('^NSEI', period='5y', progress=False)['Close']

        market_data.index = market_data.index.tz_localize(None)

        stock_return = stock_data['Close'].pct_change(fill_method=None)
        market_return = market_data.pct_change(fill_method=None)

        data = pd.concat(
            [stock_return, market_return],
            axis=1
        ).dropna()

        data.columns = ['Stock', 'Market']

        beta = (
            data['Stock'].cov(data['Market'])
            /
            data['Market'].var()
        )

        risk_free_rate = 0.07
        market_return = 0.12

        expected_return = (
            risk_free_rate
            +
            beta * (market_return - risk_free_rate)
        )

        if beta < 0.8:
            risk_category = 'Defensive'
        elif beta < 1.2:
            risk_category = 'Moderate'
        else:
            risk_category = 'Aggressive'

        if beta < 0.8:
            sensitivity = 'Low'
        elif beta < 1.2:
            sensitivity = 'Moderate'
        else:
            sensitivity = 'High'

        actual_return = stock_return.mean() * 250

        alpha = actual_return - expected_return

        if alpha > 0:
            valuation = 'Undervalued'
        elif alpha < 0:
            valuation = 'Overvalued'
        else:
            valuation = 'Fairly Valued'



        print('\n===== CAPM ANALYSIS =====')

        print('Beta                :', f'{beta:.2f}')
        print('Risk-Free Rate      :', round(risk_free_rate * 100, 2), '%')
        print('Market Return       :', market_return * 100, '%')
        print('Expected Return     :', round(expected_return * 100, 2), '%')
        print('Actual Annual Return:', round(actual_return * 100, 2), '%')
        print('Alpha               :', round(alpha * 100, 2), '%')
        print('Risk Cayegory       :', risk_category)
        print('Market Sensitivity. :', sensitivity)
        print('Valuation Status    :', valuation)
        break

    elif choice == '6':
        data = stock.history(period = '5y')

        close_price = data['Close']

        log_return = np.log(1 + close_price.pct_change(fill_method=None))

        u = log_return.mean()
        var = log_return.var()

        drift = u - (0.5 * var)

        stdev = log_return.std()

        days = 252
        simulations = 1000

        Z = np.random.standard_normal((days, simulations))

        daily_returns = np.exp(drift + stdev * Z)

        price_list = np.zeros_like(daily_returns)

        price_list[0] = close_price.iloc[-1]

        for t in range(1, days):
            price_list[t] = price_list[t - 1] * daily_returns[t]

        final_prices = price_list[-1]

        average_future_price = np.mean(final_prices)

        expected_return = (
            (average_future_price - close_price.iloc[-1])
            /
            close_price.iloc[-1]
        ) * 100

        if expected_return > 15:
            outlook = 'Strongly Bullish'
        elif expected_return > 5:
            outlook = 'Bullish'
        elif expected_return > 0:
            outlook = 'Slightly Bullish'
        elif expected_return > -5:
            outlook = 'Neutral'
        else:
            outlook = 'Bearish'

        print('\n===== MONTE CARLO SIMULATION =====')

        print('Current Price          :', round(close_price.iloc[-1], 2))
        print('Average Future Price   :', round(np.mean(final_prices), 2))
        print('Minimum Future Price   :', round(np.min(final_prices), 2))
        print('Maximun Future Price   :', round(np.max(final_prices), 2))
        print('Expected Return        :', round(expected_return,2),'%')
        print('Market Outlook         :', outlook)

        plt.figure(figsize=(10, 6))
        plt.plot(price_list)
        plt.title(f'Monte Carlo Simulation - {symbol}')
        plt.xlabel('Trading Days')
        plt.ylabel('Stock Price')
        plt.grid(True)
        plt.show()
        break

    else:
        print('Feature coming soon...')