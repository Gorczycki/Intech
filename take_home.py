import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

#add function for deletion of current .db to replace with updated csv
#assume portfolio equities were purchased at Day 1 price
#add total cash spent for portfolio


#makes initial .db file, imports all data from csv to .db
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

    #simplify by dropping/converting "price on date X" to integer date (1,2,3,...,21):
    revised_df["day"] = revised_df["day_label"].str.replace("Price on Date ", "").astype(int)
    revised_df = revised_df.drop(columns=["day_label"]) 


    connector = sqlite3.connect(db_path)
    connector.execute("""CREATE TABLE IF NOT EXISTS prices (ticker TEXT, portfolio_shares REAL, 
                   index_shares REAL, day INTEGER, price REAL); """)
    connector.commit()


    revised_df.to_sql("prices", connector, if_exists="append", index = False) #transfer of Pandas to .db SQL. Table is sorted by day. Day 1 for all equities, followed by day 2

    connector.close()







#Which stocks had the largest relative returns? Meaning share-ownership # agnostic
def rel_returns():
    rankings = []
    pct = 0
    tickers = []
    pcts = []
    connector = sqlite3.connect("equities.db")
    df = pd.read_sql_query("SELECT * FROM prices", connector)
    connector.close()
    for ticker, group in df.groupby("ticker"):
        initial_quote = group.loc[group['day'] == 1, 'price'].values[0]
        ending_quote = group.loc[group['day'] == 21, 'price'].values[0]
        pct = (ending_quote - initial_quote)/initial_quote * 100
        pct = round(pct,2)
        tickers.append(ticker)
        pcts.append(pct)
        rankings.append((ticker, pct))
    
    rankings = sorted(rankings, key=lambda x: x[1], reverse=True)

    return rankings







#What was the aggregate return of the portfolio?
def portfolio_return():
    connector = sqlite3.connect("equities.db")
    df_temp = pd.read_sql_query("SELECT * FROM prices WHERE portfolio_shares > 0", connector)    
    connector.close()
    percentage_return = 0
    starting_market_value = 0
    ending_MV = 0
    number_days = df_temp["day"].nunique()
    for ticker, group in df_temp.groupby("ticker"):
        day1_row = group.loc[group['day'] == 1]
        day21_row = group.loc[group['day'] == number_days]
        starting_market_value = starting_market_value + day1_row['price'].values[0] * day1_row['portfolio_shares'].values[0] #summing
        ending_MV = ending_MV + day21_row['price'].values[0] * day1_row['portfolio_shares'].values[0] #summing

        
    percentage_return = (ending_MV - starting_market_value)/starting_market_value * 100 #to percent
    percentage_return = round(percentage_return, 2)

    return percentage_return
  




#What was the aggregate return of the index?
def index_return():
    connector = sqlite3.connect("equities.db")
    df_temp = pd.read_sql_query("SELECT * FROM prices", connector)
    connector.close()
    percentage_return = 0
    starting_market_value = 0
    ending_MV = 0
    number_days = df_temp["day"].nunique()
    for ticker, group in df_temp.groupby("ticker"):
        day1_row = group.loc[group['day'] == 1]
        day21_row = group.loc[group['day'] == number_days]
        starting_market_value = starting_market_value + day1_row['price'].values[0] * day1_row['index_shares'].values[0]
        ending_MV = ending_MV + day21_row['price'].values[0] * day1_row['index_shares'].values[0]

    percentage_return = (ending_MV - starting_market_value)/starting_market_value * 100 #percent
    percentage_return = round(percentage_return, 2)

    return percentage_return







#What were the largest single day contributors to performance, similarly the largest single day detractors?
#Assuming contribution to portfolio performance (not index-wide), and assuming we use the scalar of number of shares as exposure to contribution/detraction.

#stock37 on day X, by Y% with Z shares
#If we are scaling by shares owned, then we sort by dollar-amount Day-over-Day move?
#Constant proportion of shares from day 1 to 21
def contributors():
    rankings = []
    dollar_change = 0
    #change = 0
    connector = sqlite3.connect("equities.db")
    df_temp = pd.read_sql_query("SELECT * FROM prices WHERE portfolio_shares > 0", connector)
    connector.close()
    number_days = df_temp["day"].nunique() #grabbing total number of days
    for ticker, group in df_temp.groupby("ticker"):
        group = group.sort_values("day")
        shares = group["portfolio_shares"].iloc[0]
        for d in range(1, number_days):
                day = group.loc[group['day'] == d]
                day_ahead = group.loc[group['day'] == d+1]
                #change = (day_ahead['price'].values[0] - day['price'].values[0])/day['price'].values[0]
                #dollar_change = change * shares
                dollar_change = (day_ahead['price'].values[0] - day['price'].values[0]) * day['portfolio_shares'].values[0]
                dollar_change = round(dollar_change,2)
                rankings.append(((ticker, d), dollar_change))

    rankings = sorted(rankings, key=lambda x: x[1], reverse=True)

    top_ten = []
    bottom_ten = []

    top_ten = rankings[:10]
    bottom_ten = rankings[-10:]


    return top_ten, bottom_ten





def contributor_visualizer():
    best, worst = contributors()
    best_df = pd.DataFrame(best, columns = ['TickerDay', 'DollarChange'])
    worst_df = pd.DataFrame(worst, columns = ['TickerDay', 'DollarChange'])
    best_df[['Ticker', 'Day']] = pd.DataFrame(best_df['TickerDay'].tolist(), index=best_df.index)
    worst_df[['Ticker', 'Day']] = pd.DataFrame(worst_df['TickerDay'].tolist(), index=worst_df.index)
    best_df = best_df.drop(columns=['TickerDay'])
    worst_df = worst_df.drop(columns=['TickerDay'])

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.bar(best_df['Ticker'] + " (Day " + best_df['Day'].astype(str) + ")", best_df['DollarChange'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Single-Day Contributors')

    plt.subplot(1,2,2)
    plt.bar(worst_df['Ticker'] + " (Day " + worst_df['Day'].astype(str) + ")", worst_df['DollarChange'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Bottom 10 Single-Day Contributors')

    plt.tight_layout()
    plt.show()






def portfolio_index_visualizer():
    connector = sqlite3.connect("equities.db")
    df_temp = pd.read_sql_query("SELECT * FROM prices WHERE portfolio_shares > 0", connector)
    connector.close()

    MV = 0
    daily_values = []

    number_days = df_temp["day"].nunique() #grabbing total number of days
    for d in range(1, number_days+1):
        day_DF = df_temp.loc[df_temp['day'] == d]
        MV = (day_DF['price'] * day_DF['portfolio_shares']).sum()
        daily_values.append(MV)
    
    scaled_values = [v / 1e6 for v in daily_values]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, number_days + 1), scaled_values, marker='o', linestyle='-', color='blue')
    plt.xlabel("Day")
    plt.ylabel("Value $MM")
    plt.title("Daily Portfolio Market Value")
    plt.grid(True)
    plt.xticks(range(1, number_days + 1))
    plt.tight_layout()
    plt.show()

    return daily_values






def main():
    portfolio_index_visualizer()


if __name__ == "__main__":
    main()