import numpy as np
import matplotlib.pyplot as plt
import iisignature
#import esig
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import root_mean_squared_error
import pandas as pd
from bokeh.layouts import row
from bokeh.plotting import figure, show
import itertools

def read_GE():
    dr = pd.read_csv('/Users/tylergorczycki/Documents/Intech2/iisig38/GEreversed.csv')
    S = dr['close'].values
    return S

def get_QV_price(S, days):
    QV_hat_price = [np.sum(np.diff(S[:i+1]) ** 2) for i in range(1, days)] #cumulative computation
    QV_hat_price = np.array(QV_hat_price).flatten()
    time_days = np.linspace(0, 1, days-1)  #length adjusted

    r = figure(width=500, height=350, title='Quadratic variation (QV)', x_axis_label='Time', y_axis_label='Cum. squared price delta')
    r.line(time_days, QV_hat_price, line_color='tomato')

    return QV_hat_price, time_days, r

def get_BM_price(S, QV_hat_price, time_days):
    sqrt_QV_hat = np.sqrt(QV_hat_price)
    new_increments_BM = [np.diff(S)[k] / sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat))] #computation
    B_daily_ext = np.cumsum(new_increments_BM)
    B_daily_ext = np.insert(B_daily_ext, 0, 0)

    s = figure(width=500, height=350, title='Underlying Brownian motion', x_axis_label='Time', y_axis_label='Cum. normalized price increments')
    s.line(time_days, B_daily_ext[1:], line_color='tomato')

    return B_daily_ext, s

def sig_keys(B_hat, N, order_model):
    keys = ["()"]
    nbr_components = B_hat.shape[1]
    for order in range(1, order_model + 1):
        for idx in itertools.product(range(1,nbr_components + 1), repeat = order):
            keys.append(str(idx).replace(" ", ""))

    #  prints () (1) (2) (1,1) (1,2) (2,1) (2,2) want this
    return keys

def Signature(B_hat, N, order_model):
    #keys = esig.sigkeys(nbr_components, order_model).strip().split(" ")
    #  prints () (1) (2) (1,1) (1,2) (2,1) (2,2) want this
    keys = sig_keys(B_hat, N, order_model)
    S = np.array([1])
    sig_df = pd.DataFrame(columns=keys)
    for i in range(1, N+1):
        B_hat_Nth = B_hat[:i]
        sig = iisignature.sig(B_hat_Nth, order_model)
        sig = np.hstack((S, sig)) #ordering for display
        temp_df = pd.DataFrame([sig], columns=keys) 
        sig_df = pd.concat([temp_df, sig_df], ignore_index=True) #combining S_t and Signature

    sig_df = sig_df.iloc[::-1].reset_index(drop=True)

    return sig_df, keys

def main():
    S = read_GE()
    days = len(S)
    #S_0 = S[0]
    order_model = 5

    QV_hat_price, time_days, r = get_QV_price(S, days)
    B_daily_ext, s = get_BM_price(S, QV_hat_price, time_days) #function calls
    t_scaled = np.linspace(0, 1, days)  #changed length
    B_hat = np.column_stack((t_scaled, B_daily_ext))
    sig_df, keys = Signature(B_hat, days, order_model)

    reg_sv_x = Lasso(alpha=0.00001, max_iter = 90000).fit(sig_df[keys], S) #Lasso regression on signature terms to S_t prices
    #Coordinate descent, cycling through each \alpha_i
    pred_sv_x = reg_sv_x.predict(sig_df[keys]) #fits coefficients, via matrix multiplication

    print('MSE of the traj with Extrapolated BM', mean_squared_error(S, pred_sv_x, squared = False))
    print('Signature Coefficients:', sig_df)

    p = figure(title="time-series", x_axis_label='Time', y_axis_label='Price', width=800, height=400)
    p.line(t_scaled, S, legend_label="GE Price", line_color="tomato", line_width=1)
    p.line(t_scaled, pred_sv_x, legend_label="Calibrated Model Price", line_color="navy", line_width=1)
    p.legend.location = "bottom_right"
    L = row(s, r, p)
    show(L)

if __name__ == "__main__":
    main()
