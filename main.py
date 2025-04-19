import numpy as np
import pandas as pd
from DataParser import DataParser
from EVIX import ExpectedVIX
from Grapher import Grapher



def main():
    buckets = 20
    days = 252

    parser = DataParser('SPX.csv', 'VIX.csv', end_date='10/31/2017')
    evix = ExpectedVIX(parser.spx_data)
    grapher = Grapher(evix)

    evix.calc_avg_vix(days=days)
    evix.calc_recent_volatility(days=days)
    evix.calc_next_realized_volatility(days=days)
    evix.calc_vix_premium()

    x, y, bucket_avg = evix.get_xy_buckets("Recent Volatility", "Next Realized Volatility", buckets=buckets)
    coeffs, _, _ = evix.calc_linear_regression(x, y)
    evix.calc_s_m(coeffs)

    evix.calc_mr_volatility(days=days)
    evix.calc_difference(days=days)


    print(evix)
    evix.save("OUT.csv")



    # grapher.scatter_plot("Recent Volatility", "Next Realized Volatility")
    # grapher.bucket_scatter_plot("Recent Volatility", "Next Realized Volatility", buckets=20)
    # grapher.vix_and_next_realized_vol_vs_date(days_label=days)
    grapher.scatter_plot("Difference", "VIX Level")
    grapher.scatter_plot("Average Difference", "Average VIX Level")

    grapher.bucket_scatter_plot("MR Volatility Squared", "Difference Squared", buckets=buckets, outliers=[19])




if __name__ == '__main__':
    main()
