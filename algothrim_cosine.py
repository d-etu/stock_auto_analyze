import pandas as pd
from pykrx import stock
import function
from datetime import datetime
from dateutil.relativedelta import relativedelta

cos_num = 0.9
stock_input = "417970"
start_date = "2024-12-30"
end_date = (datetime.strptime(start_date, "%Y-%m-%d") + relativedelta(months=1)).strftime("%Y-%m-%d")

num_stock = [200, 512, 824, 1136, 1448]

df = pd.read_csv("merge/merge_end.csv", encoding='UTF8', index_col=0)
df = df.drop(columns='Unnamed: 0', axis=0)
df.dropna(axis=0, how='all', inplace=True)
df['일자'] = pd.to_datetime(df['일자'], format="%Y%m%d")
df = df.set_index('일자')

df_high = pd.read_csv("merge/merge_high.csv", encoding='UTF8', index_col=0)
df_high= df_high.drop(columns='Unnamed: 0', axis=0)
df_high.dropna(axis=0, how='all', inplace=True)
df_high['일자'] = pd.to_datetime(df_high['일자'], format="%Y%m%d")
df_high = df_high.set_index('일자')

df_vol = pd.read_csv("merge/merge_vol.csv", encoding='UTF8', index_col=0)
df_vol['일자'] = pd.to_datetime(df_vol['일자'], format="%Y%m%d")
df_vol = df_vol.set_index('일자')

tickers = stock.get_market_ticker_list(market="ALL")


def analysis(num):
    stock_amount = -1
    info_save = [[0] * 11 for i in range(41)]

    exist_data = stock.get_market_ohlcv(start_date, end_date, stock_input)
    date_length = len(exist_data)

    exist_data_high = exist_data.loc[:, "고가"].reset_index(drop=True)
    exist_data_high = (exist_data_high - exist_data_high.min()) / (exist_data_high.max() - exist_data_high.min())
    exist_data_end = exist_data.loc[:, "종가"].reset_index(drop=True)
    exist_data_end = (exist_data_end - exist_data_end.min()) / (exist_data_end.max() - exist_data_end.min())
    exist_data.reset_index(inplace=True)

    for i in range(date_length):
        if exist_data_high[i] == 0:
            exist_data_high[i] = exist_data_end[i]

    exist_data_vol = exist_data.loc[:, "거래량"].reset_index(drop=True)
    exist_data_vol = (exist_data_vol - exist_data_vol.min()) / (exist_data_vol.max() - exist_data_vol.min())

    while True:
        for y in range(len(df_vol.columns)):
            if df_vol.columns[y] not in tickers:
                continue
            for x in range(num, num+312):
                new_data_end = df.iloc[x:x + int(date_length), y].reset_index(drop=True)
                if len(new_data_end) < date_length or new_data_end.isnull().sum() > 0:
                    continue

                new_data_vol_200 = df_vol.iloc[x + int(date_length) - 200:x + int(date_length), y].reset_index(
                    drop=True).to_list()

                max_value = max(new_data_vol_200)  # 최대값을 한 번만 계산
                if any(new_data_vol_200[i] == max_value for i in range(180, 200)):
                    pass
                else:
                    continue

                #종가 유사도 계산
                new_data_end = (new_data_end - new_data_end.min()) / (new_data_end.max() - new_data_end.min())
                cos_end = function.cosine_acculate(exist_data_end, new_data_end)
                if cos_end < cos_num or type(cos_end) == type('a'):
                    continue

                #거래량 유사도 계산
                new_data_vol = df_vol.iloc[x:x + int(date_length), y].reset_index(drop=True)
                if new_data_vol.isnull().sum() > 5:
                    continue
                new_data_vol = (new_data_vol - new_data_vol.min()) / (new_data_vol.max() - new_data_vol.min())
                cos_vol = function.cosine_acculate(exist_data_vol, new_data_vol)
                if cos_vol < cos_num or type(cos_vol) == type('a'):
                    continue

                #고가 유사도 계산
                new_data_high = df_high.iloc[x:x + int(date_length), y].reset_index(drop=True)
                new_data_high = (new_data_high - new_data_high.min()) / (new_data_high.max() - new_data_high.min())
                cos_high = function.cosine_acculate(exist_data_high, new_data_high)
                if cos_high < cos_num or type(cos_high) == type('a'):
                    continue

                new_percent_5 = function.average_move_calculation(new_data_end, 5)
                exist_percent_5 = function.average_move_calculation(exist_data_end, 5)
                if abs(new_percent_5 - exist_percent_5) >= 0.3:
                    continue

                if len(new_data_end) < 10:
                    new_data_end_10 = df.iloc[x + int(date_length) - 10: x + int(date_length), y].reset_index(drop=True)
                    new_data_end_10 = (new_data_end_10 - new_data_end_10.min()) / (
                                new_data_end_10.max() - new_data_end_10.min())
                    start_date_2 = (datetime.strptime(end_date, "%Y-%m-%d")
                                    - relativedelta(days=20)).strftime("%Y-%m-%d")
                    exist_data_2 = stock.get_market_ohlcv(start_date_2, end_date, stock_input)
                    exist_data_end_2 = exist_data_2.loc[:, "종가"].reset_index(drop=True)
                    exist_data_end_2 = (exist_data_end_2 - exist_data_end_2.min()) / (
                            exist_data_end_2.max() - exist_data_end_2.min())
                    new_percent_10 = function.average_move_calculation(new_data_end_10, 10)
                    exist_percent_10 = function.average_move_calculation(exist_data_end_2, 10)
                else:
                    new_percent_10 = function.average_move_calculation(new_data_end, 10)
                    exist_percent_10 = function.average_move_calculation(exist_data_end, 10)
                if abs(new_percent_10 - exist_percent_10) >= 0.3:
                    continue

                if x + int(date_length) < len(df.index):
                    stock_amount += 1
                    if stock_amount > 30:
                        break
                    rate_return = [0, 0]
                    print(stock.get_market_ticker_name(stock_input), stock_amount)
                    rate_return[0] = (df_high.iloc[x + int(date_length), y] /
                                      df.iloc[x + int(date_length) - 1, y]) * 100 - 100
                    rate_return[1] = (df_high.iloc[x + int(date_length) + 1, y] /
                                      df.iloc[x + int(date_length) - 1, y]) * 100 - 100
                    info_save = function.save_xy(info_save, df, df_high, x, y, stock_amount, date_length, cos_end,
                                                 cos_vol, cos_high, rate_return[0], rate_return[1])

                if stock_amount > 40:
                    break
            if stock_amount > 40:
                break
        if stock_amount <= 42:
            break

    if stock_amount != -1:
        info_save = [row for row in info_save if row[2] != 0]
        return stock_input, stock_amount, info_save, start_date, end_date, date_length
    else:
        return 0

if __name__ == '__main__':
    import concurrent.futures
    import time

    start = time.time()
    procs = []

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=5)
    for i in num_stock:
        procs.append(pool.submit(analysis, i))

    stock_amount_total = 0
    info_save_total = pd.DataFrame()
    date_length = 0

    for p in concurrent.futures.as_completed(procs):
        if p.result() != 0:
            stock_input, stock_amount, info_save, start_date, end_date, date_length = p.result()
            stock_amount_total = stock_amount_total + stock_amount + 1
            info_save = pd.DataFrame(info_save)

            if info_save_total.empty:
                info_save_total = info_save
            else:
                info_save_total = pd.concat([info_save, info_save_total])

    if not info_save_total.empty:
        function.print_stock_analysis(stock_input, stock_amount_total-1, info_save_total,
                                  start_date, end_date, date_length, "cosine")
    end = time.time()

    print("수행시간: %f 초" % (end - start))

