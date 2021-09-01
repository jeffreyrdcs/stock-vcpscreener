'''
Math/Stat function library for VCP screener
'''

def compute_ad_value(in_dict):
    '''
    Compute the Advance-decline index
    '''
    ad_val = in_dict['adv'] - in_dict['decl']
    ad_per = (in_dict['adv'] - in_dict['decl'])/(in_dict['adv'] + in_dict['decl'])   #Net Breadth
    return [ad_val, ad_per]


def compute_nhnl_value(in_dict):
    '''
    Compute the New highs-lows index
    '''
    nhnl_val = in_dict['new_high'] - in_dict['new_low']
    return nhnl_val


def compute_rs_rating(in_df):
    '''
    Compute three different RS rating.
    The input df need to have 12m (>252 row) of data
    RS 1 and RS2 = 3 & 6 months, http://forums.worden.com/default.aspx?g=posts&t=14007
    RS 3 is my own definition
    '''
    df = in_df.copy()

    current_close = df["Adj Close"][-1]
    ytd_close = df["Adj Close"][-2]
    close_3m = df["Adj Close"][-63]
    close_6m = df["Adj Close"][-126]
    close_9m = df["Adj Close"][-189]
    # Most of the time it doesnt really have 250 rows because of missing dates. Used 245 instead.
    close_12m = df["Adj Close"][-245]

    # Compute RS rating of the stock (3 & 6 months, http://forums.worden.com/default.aspx?g=posts&t=14007)
    rs_rating = (((current_close - close_3m)/close_3m) * 40 + ((current_close - close_6m)/close_6m) * 20 + \
                ((current_close - close_9m)/close_9m) * 20 +((current_close - close_12m)/close_12m) * 20)

    # 6 months weighted RS Rating
    rs_rating2 = (((current_close - close_6m)/close_6m) * 40 + \
                ((current_close - close_9m)/close_9m) * 20 +((current_close - close_12m)/close_12m) * 20)

    # RS rating 3
    wi = 1.0 / (len(df.index) + 63)
    # Doesnt matter - dj['Adj Close'] or not since it is just a constant
    df['Weighted Ratio'] = wi * ((current_close - df['Adj Close']) / df['Adj Close'])
    df.iloc[-63:, 6] = df['Weighted Ratio'].iloc[-63:] * 2.0
    rs_rating3 = df['Weighted Ratio'].sum() * 100

    return rs_rating, rs_rating2, rs_rating3


def compute_rs_rank(in_dforg):
    '''
    Compute RS rank from the RS rating column of the dataframe
    Return a sorted dataframe
    '''
    df = in_dforg.copy()

    if len(df.index) > 0:
        df['RS Rank'] = round((df.loc[:, 'RS Rating'].rank(pct=True)*1),5)
        df['RS Rank 3'] = round((df.loc[:, 'RS Rating 3'].rank(pct=True)*1),5)
        df.sort_values(by=['RS Rank 3'], axis=0, inplace=True, ascending=False)
    else:
        df = pd.DataFrame(columns=['Stock', 'Index', 'RS Rating', 'RS Rating 2', 'RS Rating 3', 'RS Rank', 'RS Rank 3'])
    return df

