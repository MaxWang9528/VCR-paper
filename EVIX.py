import numpy as np
import pandas as pd


class ExpectedVIX:
    def __init__(self, data):
        self.data = data

        self.s = None
        self.m = None
        self.c = None
        self.d = None

    def __str__(self):
        return (f"Speed: {np.round(self.s, 2)}, Mean: {np.round(self.m, 2)} \n"
                f"")

    def save(self, file):
        self.data.to_csv(file)

    def calc_avg_vix(self, days=1):
        self.data["Average VIX Level"] = self.data["VIX Level"].rolling(window=days, min_periods=1).mean()

    def calc_recent_volatility(self, days=1):
        self.data["Recent Volatility"] = np.round(self.data["Change %"].rolling(window=30).std(ddof=1) * (252 ** 0.5), 2)
        self.data["Average Recent Volatility"] = np.round(self.data["Recent Volatility"].rolling(window=days).mean(), 2)

    def calc_next_realized_volatility(self, days=1):
        next_vol = []
        for i in range(len(self.data)):
            future_returns = self.data["Change %"].iloc[i + 1:i + 31]
            if len(future_returns) == 30:
                vol = np.std(future_returns, ddof=1) * (252 ** 0.5)
            else:
                vol = np.nan
            next_vol.append(np.round(vol, 2))

        self.data["Next Realized Volatility"] = next_vol
        self.data["Average Next Realized Volatility"] = (
            np.round(self.data["Next Realized Volatility"].rolling(window=days).mean(), 2)
        )

    def calc_vix_premium(self):
        self.data["Premium"] = self.data["VIX Level"] - self.data["Next Realized Volatility"]
        self.data["Average Premium"] = self.data["Average VIX Level"] - self.data["Average Next Realized Volatility"]

    def calc_mr_volatility(self, days=1):
        self.data["MR Volatility"] = self.data["Recent Volatility"] + self.s/100 * (self.m/100 - self.data["Recent Volatility"])
        self.data["MR Volatility Squared"] = self.data["MR Volatility"] ** 2
        self.data["Average MR Volatility"] = np.round(self.data["MR Volatility"].rolling(window=days).mean(), 2)
        # self.data["Average MR Volatility Squared"] = self.data["Average MR Volatility"] ** 2

    def calc_difference(self, days=1):
        self.data["Difference"] = self.data["VIX Level"] - self.data["MR Volatility"]
        self.data["Difference Squared"] = self.data["Difference"] ** 2
        self.data["Average Difference"] = np.round(self.data["Difference"].rolling(window=days).mean(), 2)
        # self.data["Average Difference Squared"] = self.data["Average Difference"] ** 2


    # linreg stuff
    def get_xy_buckets(self, x_col, y_col, buckets=20):
        df = self.data.dropna(subset=[x_col, y_col]).copy()
        df['Bucket'] = pd.qcut(df[x_col], q=buckets, labels=False)

        bucket_avg = df.groupby('Bucket').agg({
            x_col: 'mean',
            y_col: 'mean'
        }).reset_index()

        x = bucket_avg[x_col]
        y = bucket_avg[y_col]
        return x, y, bucket_avg

    @staticmethod
    def calc_linear_regression(x, y):
        # least squares
        coeffs = np.polyfit(x, y, deg=1)  # [slope, intercept]
        best_fit_line = np.poly1d(coeffs)
        y_pred = best_fit_line(x)

        # r^2
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return coeffs, r_squared, y_pred

    def calc_s_m(self, coeffs):
        a = coeffs[0]
        b = coeffs[1]

        self.s = (1 - a) * 100
        self.m = b / (1 - a)

        