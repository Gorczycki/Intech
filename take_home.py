import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

#add function for deletion of current .db to replace with updated csv
#assume portfolio equities were purchased at Day 1 price
#could define connector sqlite3.connect() function


#makes initial .db file, imports all data from csv to .db
#is "prices" somehow the "master" variable?
def create_df_from_file(csv_path, db_path):

    df = pd.read_csv(csv_path)

    df = df.rename(columns = {'Security Index': 'ticker', 'Portfolio Shares': 'portfolio_shares', 
                              'Index Shares': 'index_shares'}) #re-naming headers

    price_columns = [col for col in df.columns if col.startswith("Price on Date")] #date extraction
    price_columns = sorted(price_columns, key=lambda col: int(col.replace("Price on Date ", ""))) #ensures days are still chronological after Pandas

    revised_df = df.melt(id_vars=["ticker", "portfolio_shares", "index_shares"], value_vars=price_columns,
                         var_name="day_label", value_name = "price") #table lay-out is now price-vertical, not price-horizontal
    
    #i.e ticker     price
    #    STOCK1     211.92
    #    STOCK1     211.86
    #H

    #simplify by dropping/converting "price on date X" to integer date (1,2,3,...,21) and "day":
    revised_df["day"] = revised_df["day_label"].str.replace("Price on Date ", "").astype(int)
    revised_df = revised_df.drop(columns=["day_label"]) 


    connector = sqlite3.connect(db_path)
    connector.execute("""CREATE TABLE IF NOT EXISTS prices (ticker TEXT, portfolio_shares REAL, 
                   index_shares REAL, day INTEGER, price REAL); """)
    connector.commit()


    revised_df.to_sql("prices", connector, if_exists="append", index = False) #transfer of Pandas to .db SQL. Table is sorted by day. Day 1 for all equities, followed by day 2

    connector.close()





#Which stocks had the largest relative returns? Meaning share-ownership # agnostic
#Single-stock returns relative to the index return over the period
#stock x outperformed index by x percent

#Make table of 10 best and 10 worst with percentage
def relative_returns():
    connector = sqlite3.connect("equities.db")
    df = pd.read_sql_query("SELECT * FROM prices", connector)
    connector.close()

    indexx_return = index_return()
    tickerrs = df["ticker"].unique()

    rankings = []

    for t in tickerrs:
        initial_val = df.loc[(df["day"] == 1) & (df["ticker"] == t), 'price'].values[0] #sorts on both ticker X and Day Y to find price Z
       
        day_final = df["day"].nunique() #determines final day number
        final_val = df.loc[(df["day"] == day_final) & (df["ticker"] == t), 'price'].values[0] #grabs price on final day for ticker X

        pctt = (final_val - initial_val)/initial_val * 100
        rel_return = pctt - indexx_return #"gained 2.3% over index"
        rel_return = round(rel_return,2)
        rankings.append((t, rel_return)) #two-tuple

    rankings = sorted(rankings, key=lambda x: x[1], reverse = True) #key=lambda x: x[1] sorts on the 2nd [0,1] tuple


    return rankings








#What was the aggregate return of the portfolio?
def portfolio_return():
    connector = sqlite3.connect("equities.db")
    df = pd.read_sql_query("SELECT * FROM prices WHERE portfolio_shares > 0", connector)
    connector.close()

    initial_market_value = 0
    initial_day = df.loc[df["day"] == 1]
    initial_market_value = (initial_day["price"] * initial_day["portfolio_shares"]).sum() #.sum() loops through

    number_days = df["day"].nunique() #day_counter
    final_day = df.loc[df["day"] == number_days]
    final_market_value = (final_day["price"] * final_day["portfolio_shares"]).sum()

    pct_return = (final_market_value - initial_market_value)/initial_market_value * 100

    return pct_return 




#What was the aggregate return of the index?
def index_return():
    connector = sqlite3.connect("equities.db")
    df = pd.read_sql_query("SELECT * FROM prices", connector)
    connector.close()

    day1 = df.loc[df["day"] == 1]
    initial_val = (day1["index_shares"] * day1["price"]).sum()

    num_days = df["day"].nunique() #counts # of days
    day_final = df.loc[df["day"] == num_days]
    final_val = (day_final['index_shares'] * day_final['price']).sum()

    pct_return = (final_val - initial_val)/initial_val * 100
    pct_return = round(pct_return,2)

    return pct_return





#What were the largest single day contributors to performance, similarly the largest single day detractors?
#Assuming contribution to portfolio performance (not index-wide), and assuming we use the scalar of number of shares as exposure to contribution/detraction.
#"Which individual names positively or negatively contributed to performance of the portfolio relative to the index on a given day"
#stock37 on day X, by Y% with Z shares
#Constant proportion of shares from day 1 to 21


#Determines ranking of contribution of individual names to performance of portfolio, adjusting for index.
#gives top 15 best, worst
def contributors():
    connector = sqlite3.connect("equities.db")
    df = pd.read_sql_query("SELECT * FROM prices", connector)
    connector.close()

    number_days = df['day'].nunique() #counter
    number_tickers = df['ticker'].unique() #counter

    rankings = []

    #index_returnn = index_return()
    #1,2,3,4,5,6,7
    #1,2
    #2,3
    for tickerr in number_tickers:
        for dayy in range(1, number_days):
            day_data = df[df['day'] == dayy]

            total_pf_val = (day_data['portfolio_shares'] * day_data['price']).sum() #market value per-day
            total_ind_val = (day_data['index_shares'] * day_data['price']).sum()

            share_count = df.loc[(df['day'] == dayy) & (df['ticker'] == tickerr), 'portfolio_shares'].values[0] # num of pf shares of ticker X
            pf_single_val1 = df.loc[(df['day'] == dayy) & (df['ticker'] == tickerr), 'price'].values[0] #price of ticker X on day Y
            pf_single_val2 = df.loc[(df['day'] == dayy+1) & (df['ticker'] == tickerr), 'price'].values[0] #price of ticker X on day Y+1
            weighting = (share_count * pf_single_val1) / total_pf_val # weighting = (price * shares) / portfolio_market_value
            perf_pf = weighting * ((pf_single_val2 - pf_single_val1) / pf_single_val1) # performance, weighting scaled by percentage gain

            share_count2 = df.loc[(df['day'] == dayy) & (df['ticker'] == tickerr), 'index_shares'].values[0] # num of ind shares of ticker X
            weighting2 = (share_count2 * pf_single_val1) / total_ind_val #index share %
            perf_ind = weighting2 * ((pf_single_val2 - pf_single_val1) / pf_single_val1)
            relative_perf = (perf_pf - perf_ind) * 100
            relative_perf = round(relative_perf, 5)
            rankings.append(((tickerr, dayy), relative_perf))

    rankings = sorted(rankings, key=lambda x: x[1], reverse = True)

    best = rankings[:15]
    worst = rankings[-15:]
    worst = sorted(worst, key=lambda x: x[1])

    return best, worst


def creative():
    #weight_old = price_i(changes) * share_count / pf_value(changes)
    #weight_new = 
    #back_to_old weightings: share count constant, 

    #record initial weights
    #get final weights
    #STOCK10: (35$ * 100) / 5,000,000
    # -> (42*100) / 5,000,000 + gains_from_10
    #old: 42-35 = 7$ * 100 shares = 700$ gain 

    connector = sqlite3.connect("equities.db")
    df = pd.read_sql_query("SELECT * FROM prices WHERE portfolio_shares > 0", connector)
    connector.close()

    weights_initial = []
    weights_final = []
    num_days = df["day"].nunique()
    num_tickers = df["ticker"].unique()

    initial_day = df.loc[df["day"] == 1]
    initial_pf_value = (initial_day["price"] * initial_day["portfolio_shares"]).sum() #market value per-day

    final_day = df.loc[df["day"] == num_days]
    final_pf_value = (final_day["price"] * final_day["portfolio_shares"]).sum()

    #initial_weight formula : (price * share_count) / (pf_value), pf_value = \sum (shares_i * price_i)
    for t in num_tickers:
        val = df.loc[(df["day"] == 1) & (df["ticker"] == t), 'price'].values[0]
        weight = val / initial_pf_val
        val2 = df.loc[(df["day"] == num_days) & [df["ticker"] == t], 'price'].values[0]
        weight2 = val2 / final_pf_value
        weights_initial.append(weight) #initial weighting
        weights_final.append(weight2)        
        delta = abs(weight2 - weight) #percentage difference #shares needed, = final_price / final_pf 
        final = (delta*final_pf_value) / val2
        final = int(final)

            

    #vector of shares, choose BUY or SELL, find # shares to revert

    



def main():
    print(f"Portfolio Return: {portfolio_return():+.2f}%")
    print(f"Index Return: {index_return():+.2f}%")
    print(f"Relative Performance: {portfolio_return() - index_return():+.2f}%")
    print()
    
    returns = relative_returns()
    print("Top 10 equities vs IDX:")
    for ticker, ret in returns[:10]:
        print(f"  {ticker}: {ret:+.2f}%")

    print('\n')
    
    print("Worst 10 equities vs IDX:")
    for ticker, ret in returns[-10:]:
        print(f"  {ticker}: {ret:+.2f}%")
    print()
    


    best_contrib, worst_contrib = contributors()
    print("Best single-day performers:")
    for (ticker, day), contrib in best_contrib:
        print(f"  {ticker} (Day {day}): {contrib:+.2f}%")

    print('\n')
    
    print("Worst single-day performers:")
    for (ticker, day), contrib in worst_contrib:
        print(f"  {ticker} (Day {day}): {contrib:+.2f}%")




if __name__ == "__main__":
    main()
