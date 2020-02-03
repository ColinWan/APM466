import numpy as np
import pandas as pd
from time import strptime
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from math import exp

data = pd.read_csv("/Users/colinwan/Desktop/UofT_Fourth_Year/APM466/Data_Full.csv")

bond_list = ["CAN 1.5 MAR 20", "CAN 0.75 SEP 20", "CAN 0.75 MAR 21",
             "CAN 0.75 SEP 21", "CAN 0.5 MAR 22", "CAN 2.75 JUN 22",
             "CAN 1.75 MAR 23", "CAN 1.5 JUN 23", "CAN 2.25 MAR 24",
             "CAN 1.5 SEP 24"]

num_bond_time = []
for i in bond_list:
    temp = i.split(" ")
    month = strptime(temp[2], "%b").tm_mon
    num_bond_time.append(np.array([month, int(temp[3]), temp[1]]))

data['Maturity_Year'] = pd.DatetimeIndex(data["Maturity"]).year
data['Maturity_Month'] = pd.DatetimeIndex(data["Maturity"]).month


def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = T * freq
    coupon = coup / 100. * par / freq
    dt = [(i + 1) / freq for i in range(int(periods))]

    ytm_func = lambda y:  sum([coupon/(1+y/freq)**(freq*t) for t in dt]) + par/(1+y/freq)**(periods) - price
    return optimize.newton(ytm_func, guess)

frame_list = []

# Select the bond data we want, given the list that we have chosen to construct yield to maturity.
for bond in num_bond_time:
    cur_data = data[(data.Maturity_Month == int(bond[0])) & (data.Maturity_Year == int("20" + bond[1])) & (data.Coupon == float(bond[2]))]
    # Find what today's date is?
    cur_data['TimeToMaturity'] = (cur_data['Maturity_Year'] - 2020) + (cur_data['Maturity_Month'] - 2) / 12
    cur_data['par'] = 100

    frame_list.append(cur_data)

meta_data = pd.concat(frame_list)

temp = []

# Use Newton Optimization method with

for index, row in meta_data.iterrows():
    yield_rate = bond_ytm(row['Price'], row['par'], row['TimeToMaturity'], row['Coupon'])
    temp.append(pd.Series([row["ISIN"], row["date"], row['TimeToMaturity'], yield_rate, row["Coupon"], row["Price"], row['Maturity_Month'], row['Maturity_Year']]))


yield_df = pd.concat(temp, axis=1).T
yield_df.rename(columns={0:"ISIN", 1:"date", 2:"TimeToMaturity", 3:"Yield_rate", 4:"Coupon", 5:"Price", 6:'Maturity_Month', 7:'Maturity_Year'}, inplace=True)
yield_df['date'] = pd.to_datetime(yield_df.date)

#Dirty Price Addition
for bond in num_bond_time:
    cur_data = yield_df[(yield_df.Maturity_Month == int(bond[0])) & (yield_df.Maturity_Year == int("20" + bond[1])) & (yield_df.Coupon == float(bond[2]))]
    if int(bond[0]) == 3:
        cur_data['Price'] = 136 / 365 * yield_df.Coupon/2 + cur_data['Price']
    elif int(bond[0]) == 6:
        cur_data['Price'] = 45 / 365 * yield_df.Coupon/2 + cur_data['Price']
    elif int(bond[0]) == 9:
        cur_data['Price'] = 136 / 365 * yield_df.Coupon/2 + cur_data['Price']

date = yield_df["date"].unique()

#Plotting For The Yield To Maturity Curve
fig = plt.figure()
for item in date:
    cur_yield = yield_df[yield_df["date"] == item].sort_values(by=["TimeToMaturity"])["Yield_rate"]
    plt.plot(list(range(10)), cur_yield, label=str(item)[:10])
plt.title("Yield")
plt.ylim([0, 0.025])
plt.xlim([0, 9])
plt.legend(fontsize='x-small')
plt.savefig("/Users/colinwan/Desktop/UofT_Fourth_Year/APM466/Yield.png")
plt.show()


spot_rate = np.full([100, 2], 0)
frame_list_2 = []
yield_df['spot'] = 0.0000
yield_df['forward'] = 0.0000

for item in date:
    cur_yield = yield_df[yield_df["date"] == item].sort_values(by=["TimeToMaturity"])
    NumBonds = cur_yield["ISIN"].shape[0]
    for i in range(NumBonds):
        Principal = 100
        payment = cur_yield["Coupon"].iloc[i]/2
        Notional = Principal + payment
        Price = cur_yield["Price"].iloc[i]
        T = cur_yield["TimeToMaturity"].iloc[i]
        CouponPayment = 0
        if i>0:
            for j in range(i - 1):
                spotrate = cur_yield["spot"].iloc[j]
                coupon_time = cur_yield["TimeToMaturity"].iloc[j]
                CouponPayment = CouponPayment + payment*np.exp(-1 * spotrate * coupon_time)
        rate = - np.log((Price - CouponPayment) / Notional) / T

        cur_yield.at[int(cur_yield.iloc[i].name), 'spot'] = rate

    frame_list_2.append(cur_yield)

SpotRateDataFrame = pd.concat(frame_list_2)

#Plotting for Spot Curve
fig = plt.figure()
for item in date:
    cur_yield = SpotRateDataFrame[SpotRateDataFrame["date"] == item].sort_values(by=["TimeToMaturity"])["spot"]
    plt.plot(list(range(8)), cur_yield[2:], label=str(item)[:10])
plt.title("Spot")
plt.ylim([0, 0.025])
plt.xlim([0, 7])
plt.legend(fontsize='x-small')
plt.savefig("/Users/colinwan/Desktop/UofT_Fourth_Year/APM466/Spot.png")

plt.show()

forwardlist = []

for item in date:
    cur_yield = SpotRateDataFrame[SpotRateDataFrame["date"] == item].sort_values(by=["TimeToMaturity"])
    NumBonds = cur_yield["ISIN"].shape[0]
    for i in range(NumBonds):
        if i > 0:
            riti = cur_yield['spot'].iloc[i] * cur_yield['TimeToMaturity'].iloc[i]
            r0t0 = cur_yield['spot'].iloc[0] * cur_yield['TimeToMaturity'].iloc[0]
            forward_rates = (riti - r0t0) / (cur_yield['TimeToMaturity'].iloc[i] - cur_yield['TimeToMaturity'].iloc[0])
            cur_yield.at[int(cur_yield.iloc[i].name), 'forward'] = forward_rates
    forwardlist.append(cur_yield)

forward_rate = pd.concat(forwardlist)

#Forward Curve
fig = plt.figure()
for item in date:
    cur_yield = forward_rate[forward_rate["date"] == item].sort_values(by=["TimeToMaturity"])["forward"]
    plt.plot(list(range(9)), cur_yield[1:], label=str(item)[:10])
plt.title('Forward')
plt.ylim([0, 0.025])
plt.xlim([0, 8])
plt.legend(fontsize='x-small')
plt.savefig("/Users/colinwan/Desktop/UofT_Fourth_Year/APM466/Forward.png")

plt.show()

X_mat_1 = np.full([9, 5], np.nan)
X_mat_2 = np.full([9, 5], np.nan)

bond_names = yield_df.ISIN.unique()
i = 0
for i in range(int(len(bond_names)/2)):
    cur_yield = yield_df[yield_df['ISIN']==bond_names[2*i+1]].sort_values(by=["date"])
    cur_forward = forward_rate[forward_rate['ISIN']==bond_names[2*i+1]].sort_values(by=["date"])
    X_mat_1[:, i] = np.log(cur_yield['Yield_rate'].values.astype('float')[:-1]/cur_yield['Yield_rate'].values.astype('float')[1:])
    X_mat_2[:, i] = np.log(cur_forward['forward'].values.astype('float')[:-1]/cur_forward['forward'].values.astype('float')[1:])


cov_1 = np.cov(X_mat_1.T*100)
cov_2 = np.cov(X_mat_2.T*100)

eig_1 = np.linalg.eig(cov_1)
eig_2 = np.linalg.eig(cov_2)
