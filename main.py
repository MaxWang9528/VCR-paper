import numpy as np
import pandas as pd
from DataParser import DataParser
from EVIX import ExpectedVIX
from Grapher import Grapher



def main():
    buckets = 20
    days = 252

    # FIRST LINREG -> FIND S AND M OF VOLATILITY MEAN REVERSION WITH RECENT VOL AND NEXT REALIZED VOL
    # parser = DataParser('SPX.csv', 'VIX.csv', end_date='10/31/2017')
    parser = DataParser('SPX.csv', 'VIX.csv')
    evix = ExpectedVIX(parser.spx_data)
    grapher = Grapher(evix)

    evix.calc_vix_cols(days=days)
    evix.calc_recent_volatility(days=days)
    evix.calc_next_realized_volatility(days=days)
    evix.calc_vix_premium()

    x1, y1, _ = evix.get_xy_buckets("Recent Volatility", "Next Realized Volatility", buckets=buckets)
    coeffs1, _, _ = evix.calc_linear_regression(x1, y1)
    evix.calc_s_m(coeffs1)

    # SECOND LINREG -> FIND C AND D WITH MR VOL AND ITS DIFFERENCE TO VIX, TO FIND EVIX AND VCR
    evix.calc_mr_volatility(days=days)
    evix.calc_difference(days=days)
    sampled_df = evix.sample_every_n_days(["Average VIX Level", "Average Difference"], n=60)

    x2, y2, _ = evix.get_xy_buckets("MR Volatility Squared", "Squared Difference", buckets=buckets)
    coeffs2, _, _ = evix.calc_linear_regression(x2, y2, outliers=[19])
    evix.calc_c_d(coeffs2)

    # DECOMPOSITION OF VIX
    evix.calc_variance_premium()
    evix.calc_evix()
    evix.calc_dtm()
    evix.calc_volatility_premium()
    evix.calc_mr_adjustment()
    evix.calc_vcr()

    # PERFORMANCE
    evix.check()
    evix.calc_change_in_recent_volatility()
    evix.calc_vcr_performance()

    print(evix)
    parser.df_to_file(evix.data, "OUT.csv", rounded=False)
    ########## GRAPHS    GRAPHS    GRAPHS    GRAPHS    GRAPHS ##########
    grapher.scatter_plot("Recent Volatility", "Next Realized Volatility")
    grapher.bucket_scatter_plot("Recent Volatility", "Next Realized Volatility", buckets=20)
    grapher.vix_and_next_realized_vol_vs_date(days_label=days)

    grapher.scatter_plot("Average Difference", "Average VIX Level", df=sampled_df)
    grapher.bucket_scatter_plot("MR Volatility Squared", "Squared Difference", buckets=buckets, outliers=[19])

    grapher.vix_decomposition(days=days)
    grapher.vcr_performance(days=252)


if __name__ == '__main__':
    main()
