def stock_plot(start_date, end_date, stock_input, date_length):
    from mplfinance.original_flavor import candlestick2_ohlc
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    from pykrx import stock
    from datetime import date
    import pandas as pd

    if type(start_date) == type(date.today()):
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
    df = stock.get_market_ohlcv(start_date, end_date, stock_input)
    if len(df) == date_length:
        df2 = df.iloc[[-1]]
        df = pd.concat([df, df2, df2], axis=0)
        for i in range(0, 3):
            df.iloc[len(df) - 2, i] = df.iloc[len(df) - 1, 3]
            df.iloc[len(df) - 1, i] = df.iloc[len(df) - 1, 3]
    else:
        pass

    df['MA5'] = df['종가'].rolling(5).mean()
    df['MA10'] = df['종가'].rolling(10).mean()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    index = df.index.astype('str') # 캔들스틱 x축이 str로 들어감

    # 이동평균선 그리기
    ax.plot(index, df['MA5'], label='MA5', linewidth=0.7)
    ax.plot(index, df['MA10'], label='MA10', linewidth=0.7)

    # X축 티커 숫자 20개로 제한
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))

    # 그래프 title과 축 이름 지정
    ax.set_title(stock.get_market_ticker_name(stock_input), fontsize=22)
    ax.set_xlabel('Date')

    # 캔들차트 그리기
    candlestick2_ohlc(ax, df['시가'], df['고가'],
                      df['저가'], df['종가'],
                      width=0.5, colorup='r', colordown='b')
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig('image/test_graph_{0}.png'.format(stock_input),
                facecolor='#eeeeee',
                edgecolor='black',
                format='png', dpi=200)
    plt.cla()
    plt.clf()
    plt.close()