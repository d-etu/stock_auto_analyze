import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()

#본인의 api키, 텔레그램 token, id로 수정하시면 됩니다
#OPEN_DART_API_KEY = os.getenv('OPEN_DART_API_KEY') #재무제표 분석 사용 안하실 경우 필요 없습니다
telegram_token = os.getenv('telegram_token')
telegram_id = os.getenv('telegram_id')

def cosine_similarity(x, y):
    import numpy as np
    return np.nan_to_num(np.dot(x, y)/(np.sqrt(np.dot(x, x))*np.sqrt(np.dot(y, y))), copy=False)

def cosine_acculate(base, target):
    import numpy as np
    cos_similarity = cosine_similarity(base, target)
    np.nan_to_num(cos_similarity, copy=False)
    return cos_similarity

def dtw_distance(base, target):
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    import pandas as pd
    base = pd.DataFrame(base).reset_index()  # 함수 사용하려면 필요한 과정
    target = pd.DataFrame(target).reset_index()

    if base.isna().values.any() or target.isna().values.any():
        base = base.fillna(0)
        target = target.fillna(0)

    distance, _ = fastdtw(base, target, dist=euclidean)
    return distance

def average_move_calculation(data_end, count):
    moving_average = 0
    date_length = len(data_end)
    if count == 5:
        for i in range(date_length-5, date_length):
            moving_average += data_end.iloc[i]/5
        differ_percent = data_end.iloc[date_length-1] / moving_average
        return differ_percent
    elif count == 10:
        for i in range(date_length-10, date_length):
            moving_average += data_end.iloc[i]/10
        differ_percent = data_end.iloc[date_length - 1] / moving_average
        return differ_percent
    elif count == 20:
        for i in range(date_length-10, date_length):
            moving_average += data_end.iloc[i]/10
        differ_percent = data_end.iloc[date_length - 1] / moving_average
        return differ_percent

def average_move_low_calculation(data_end, data_low, count):
    moving_average = 0
    date_length = len(data_end)
    if count == 5:
        for i in range(date_length-5, date_length):
            moving_average += data_end.iloc[i]/5
        differ_percent = data_low.iloc[date_length-1] / moving_average
        return differ_percent
    elif count == 10:
        for i in range(date_length-10, date_length):
            moving_average += data_end.iloc[i]/10
        try:
            differ_percent = data_low.iloc[date_length -1] / moving_average
        except:
            differ_percent = 10
        return differ_percent
    elif count == 20:
        for i in range(date_length-10, date_length):
            moving_average += data_end.iloc[i]/10
        differ_percent = data_low.iloc[date_length - 1] / moving_average
        return differ_percent

def adjust_corr(len_y, y, corr_end_low):
    corr_end_low = round(corr_end_low, 3)
    if len_y/2 <= y:
        if 0.98 > corr_end_low >=0.95:
            return 0.02
        elif corr_end_low>=0.98:
            return 0.05
        else:
            return 0.03
    elif len_y/4 < y < len_y/2:
        if corr_end_low >= 0.95:
            return 0.03
        else:
            return 0.05
    elif len_y/4 >= y:
        if corr_end_low <= 0.85:
            return 0.1
        elif corr_end_low==0.9:
            return 0.05
        elif corr_end_low==0.95:
            return 0.03

def save_xy(info_save, df, df_high, x, y, stock_amount, date_length, corr_end, corr_vol, corr_high, day1_return, day2_return) :
    from pykrx import stock
    info_save[stock_amount][0] = df.index[x].date()
    info_save[stock_amount][1] = df.index[x+int(date_length)-1].date()
    info_save[stock_amount][2] = stock.get_market_ticker_name(df.columns[y])
    info_save[stock_amount][3] = round((corr_end + corr_vol * 3 + corr_high * 3)/3, 3)
    info_save[stock_amount][4] = round(corr_high, 3)
    info_save[stock_amount][5] = round((df_high.iloc[x + int(date_length), y] / df.iloc[x + int(date_length) - 1, y]) * 100 - 100, 2)
    info_save[stock_amount][6] = round((df_high.iloc[x + int(date_length)+1, y] / df.iloc[x + int(date_length) - 1, y]) * 100 - 100, 2)
    info_save[stock_amount][7] = df.columns[y]
    info_save[stock_amount][8] = df.index[x+int(date_length)+1].date()
    info_save[stock_amount][9] = day1_return
    info_save[stock_amount][10] = day2_return
    return info_save

def print_stock_analysis(stock_input, stock_amount, info_save, start_date, end_date, date_length, algorithm_type):
    from pykrx import stock
    import time
    if type(info_save) == type(list()):
        info_save = pd.DataFrame(info_save)
    #유사관계 알고리즘 종류별 분류
    if algorithm_type == "cosine":
        info_save = info_save.sort_values(by=[3], ascending=False)
    elif algorithm_type == "dtw":
        info_save = info_save.sort_values(by=[3], ascending=True)
    if stock_amount > 4:
        info_save = info_save[:5]
        stock_amount = 4
    import numpy as np

    info_save_np = np.array(info_save)  # 리스트를 NumPy 배열로 변환
    return_day1 = sum(info_save_np[:, 9])
    return_day2 = sum(info_save_np[:, 10])
    print("종목이름: {0} 1일 평균 수익률 : {1}% 2일 평균 수익률 : {2}% 종목 개수 : {3}".format(
        stock.get_market_ticker_name(stock_input), round(return_day1 / (stock_amount+1), 2), round(return_day2 / (stock_amount+1), 2), stock_amount+1))
    telegram_send_text("종목이름: {0} 1일 평균 수익률 : {1}% 2일 평균 수익률 : {2}% 종목 개수 : {3}".format(
        stock.get_market_ticker_name(stock_input), round(return_day1 / (stock_amount+1), 2), round(return_day2 / (stock_amount+1), 2), stock_amount+1))

    for i in range(0, stock_amount+1):
        print("비교 종목 : {0} 날짜 범위 : {1}~{2}, 종목 이름 : {3}, 종합 상관 계수:{4}, 고가 상관계수:{5}, 1일 수익률 : {6}% 2일 수익률 : {7}%".format(stock.get_market_ticker_name(stock_input),
              info_save.iloc[i][0], info_save.iloc[i][1], info_save.iloc[i][2],info_save.iloc[i][3], info_save.iloc[i][4], info_save.iloc[i][5], info_save.iloc[i][6]))
        telegram_send_text("비교 종목 : {0} 날짜 범위 : {1}~{2}, 종목 이름 : {3}, 상관 계수:{4}, 고가 상관계수:{5}, 1일 수익률:{6}% 2일 수익률 : {7}%".format(stock.get_market_ticker_name(stock_input),
                            info_save.iloc[i][0], info_save.iloc[i][1], info_save.iloc[i][2],info_save.iloc[i][3], info_save.iloc[i][4], info_save.iloc[i][5], info_save.iloc[i][6]))
        telegram_send_image(start_date, end_date, stock_input, date_length)
        telegram_send_image(info_save.iloc[i][0], info_save.iloc[i][8], info_save.iloc[i][7], date_length)
        time.sleep(1)


async def telegram_text(message):  # 실행시킬 함수명 임의지정
    import telegram
    token = telegram_token
    chat_id = telegram_id
    bot = telegram.Bot(token=token)
    await bot.send_message(chat_id, message)

def telegram_send_text(message):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(telegram_text(message)) #봇 실행하는 코드

def telegram_send_image(start_date, end_date, stock_input, date_length):
    import asyncio
    import plot
    plot.stock_plot(start_date, end_date, stock_input, date_length)
    async def telegram_image():
        import telegram
        token = telegram_token
        chat_id = telegram_id
        bot = telegram.Bot(token=token)
        send_image = 'image/test_graph_{0}.png'.format(stock_input)
        await bot.send_photo(chat_id=chat_id, photo=open(send_image, 'rb'))
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(telegram_image()) #봇 실행하는 코드

def music_sound(): #검색 종료 후 노래를 듣고 싶으시면 함수 호출하시면 됩니다
    import winsound
    #  도,레,미,파,솔,라,시  Hz
    so1 = {'do': 261, 're': 293, 'mi': 329, 'pa': 349, 'sol': 391, 'ra': 440, 'si': 493}

    mel = ['do', 'mi', 'mi', 'mi', 'sol', 'sol', 're', 'pa', 'pa', 'ra', 'si', 'si']
    dur = [4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2]

    mel2 = ['sol', 'do', 'ra', 'pa', 'mi', 'do', 're']
    dur2 = [1, 1, 1, 1, 1, 1, 1]

    music = zip(mel, dur)
    music2 = zip(mel2, dur2)

    for melody, duration in music:
        winsound.Beep(so1[melody], 1000 // duration)

    for melody, duration in music2:
        winsound.Beep(so1[melody], 1000 // duration)


# 재무제표 관련 코드입니다. 직접 수정하셔서 사용하시면 됩니다.
'''
def financial_calculate(stock_input):
    import OpenDartReader
    import pandas as pd
    import time
    api_key = OPEN_DART_API_KEY  # OpenDart API 에서 받는 KEY 입력
    print("계산 시작")
    # 얻고자 하는 종목명 리스트 형태로 입력
    dart = OpenDartReader(api_key)
    stocks = stock_input
    # '11013'=1분기보고서, '11012' =반기보고서, '11014'=3분기보고서, '11011'=사업보고서

    # 데이터를 얻기위한 반복문 시작
    start_year = 2019
    end_year = 2022

    # 더미 리스트 초기화. 1 ~ 4 분기 데이터를 합할 예정이므로 4 크기 만큼의 리스트 선언.
    liquid_asset = [0, 0, 0, 0]  # 유동자산
    inventory = [0, 0, 0, 0]  # 재고자산
    receivable = [0, 0, 0, 0]  # 매출채권
    non_liquid_asset = [0, 0, 0, 0]  # 비유동자산
    pay = [0, 0, 0, 0]  # 판관비(임금)
    cfo = [0, 0, 0, 0]  # 영업활동현금흐름
    cfi = [0, 0, 0, 0]  # 투자활동현금흐름
    fcf = [0, 0, 0, 0]  # 잉여현금흐름 : 편의상 영업활동 - 투자활동 현금흐름으로 계산
    score = [0, 0, 0, 0]

    for i in range(start_year, end_year + 1):  # OpenDart는 2015년부터 정보를 제공한다.
        df1 = pd.DataFrame()  # Raw Data
        j = i - start_year
        if str(type(dart.finstate_all(stocks, i, reprt_code='11011', fs_div='CFS'))) == "<class 'NoneType'>":
            pass
        elif dart.finstate_all(stocks, i, reprt_code='11011', fs_div='CFS').empty == True:
            pass
        # 타입이 NoneType 이 아니면 읽어온다.
        else:
            df1 = dart.finstate_all(stocks, i, reprt_code='11011', fs_div='CFS')
            # 재무상태표 부분
            condition = (df1.sj_nm == '재무상태표') & ((df1.account_nm == '유동자산') | (df1.account_id == 'ifrs-full_CurrentAssets'))
            condition_2 = (df1.sj_nm == '재무상태표') & (df1.account_nm == '재고자산')
            condition_3 = (df1.sj_nm == '재무상태표') & ((df1.account_id == 'dart_ShortTermTradeReceivable')
                                                    | (df1.account_id == 'ifrs-full_TradeAndOtherCurrentReceivables')
                                                   | (df1.account_nm == '매출채권및기타유동자산')
                                                   | (df1.account_nm == '매출채권'))  # 매출채권
            condition_4 = (df1.sj_nm == '재무상태표') & ((df1.account_nm == '비유동자산') | (df1.account_id == 'ifrs-full_NoncurrentAssets'))
            # 손익계산서 부분
            condition_5 = ((df1.sj_nm == '손익계산서') | (df1.sj_nm == '포괄손익계산서')) & (
                        df1.account_id == 'dart_TotalSellingGeneralAdministrativeExpenses')
            # 현금흐름표 부분
            condition_6 = (df1.sj_nm == '현금흐름표') & (df1.account_id == 'ifrs-full_CashFlowsFromUsedInOperatingActivities')
            condition_7 = (df1.sj_nm == '현금흐름표') & (df1.account_id == 'ifrs-full_CashFlowsFromUsedInInvestingActivities')

            liquid_asset[j] = int(df1.loc[condition].iloc[0]['thstrm_amount']) / 100000000
            try:
                inventory[j] = int(df1.loc[condition_2].iloc[0]['thstrm_amount'])
            except:
                inventory[j] = 10
            receivable[j] = int(df1.loc[condition_3].iloc[0]['thstrm_amount']) / 100000000
            non_liquid_asset[j] = int(df1.loc[condition_4].iloc[0]['thstrm_amount']) / 100000000
            pay[j] = int(df1.loc[condition_5].iloc[0]['thstrm_amount']) / 100000000
            cfo[j] = int(df1.loc[condition_6].iloc[0]['thstrm_amount']) / 100000000
            cfi[j] = int(df1.loc[condition_7].iloc[0]['thstrm_amount']) / 100000000
            fcf[j] = (cfo[j] - cfi[j])
            # 유동자산 대비 재고자산
            score_inven = liquid_asset[j] / inventory[j]
            if score_inven <= 2:
                score_inven = -1.5
            elif score_inven <= 3:
                score_inven = -0.5
            elif score_inven <= 5:
                score_inven = 1
            else:
                score_inven = 1.5
            print(score_inven)
            # 유동자산 대비 매출채권
            score_receive = liquid_asset[j] / receivable[j]
            if score_receive <= 2:
                score_receive = -1.5
            elif score_receive <= 3:
                score_receive = -0.5
            elif score_receive <= 5:
                score_receive = 1
            else:
                score_receive = 1.5
            print(score_receive)
            # 유동자산 대비 비유동자산
            score_nonliq = liquid_asset[j] / non_liquid_asset[j]
            if score_nonliq >= 1.5:
                score_nonliq = 1.5
            elif score_nonliq >= 1:
                score_nonliq = 0.75
            elif score_nonliq >= 0.75:
                score_nonliq = 0
            elif score_nonliq >= 0.5:
                score_nonliq = -0.75
            else:
                score_nonliq = -1.5
            print(score_nonliq)
            # 잉여현금흐름
            if fcf[j] <= -40:
                score_alpha = -2.5
            elif fcf[j] <= 0:
                score_alpha = -1.25
            elif fcf[j] <= 40:
                score_alpha = 0
            elif fcf[j] <= 80:
                score_alpha = 1.25
            else:
                score_alpha = 2.5
            print(score_alpha)
            # 임금 상승
            if j >= 1 and pay[j - 1] != 0:
                score_pay = 0.5
            elif j == 1:
                score_pay = 0
            else:
                score_pay = -0.5
            print(score_pay)
            score[j] = score_inven + score_receive + score_nonliq + score_alpha + score_pay
        # 너무 잦은 요청이 있을 경우 OpenDart API 측에서 IP 를 차단하니 텀을 둔다.
        time.sleep(0.7)


    if mean_score >= 5:
        mean_score = 5

    return("{0}의 점수는 5점 만점에 {1}점입니다.".format(stocks, round(mean_score, 1)))
'''