from matplotlib import pyplot as plt
import mplcursors
import pandas as pd
import numpy as np

class Grapher:
    def __init__(self, evix):
        self.evix = evix
        self.data = evix.data


    def scatter_plot(self, x_col, y_col, df=None):
        if df is None:
            df = self.data.dropna(subset=[x_col, y_col, "Date"])
        else:
            df = df.dropna(subset=[x_col, y_col, "Date"])

        fig, ax = plt.subplots(figsize=(14, 8))
        scatter = ax.scatter(
            df[x_col],
            df[y_col],
            alpha=0.7,
            color="orange",
            edgecolor="k"
        )

        # Tooltip
        mplcursors.cursor(scatter, hover=True).connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"Date: {df['Date'].iloc[sel.index].strftime('%Y-%m-%d')}\n"
                f"{x_col}: {df[x_col].iloc[sel.index]:.4f}\n"
                f"{y_col}: {df[y_col].iloc[sel.index]:.4f}"
            )
        )

        ax.set_title(f"{y_col} vs {x_col} (Interactive Scatter)", fontsize=14)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def bucket_scatter_plot(self, x_col, y_col, buckets=20, outliers=None):
        x, y, bucket_avg = self.evix.get_xy_buckets(x_col, y_col, buckets=buckets)
        if outliers is None:
            outliers = []

        # Identify inlier and outlier indices based on bucket number
        is_outlier = bucket_avg["Bucket"].isin(outliers)
        x_inliers = x[~is_outlier]
        y_inliers = y[~is_outlier]

        # Linear regression on inliers only
        coeffs, r_squared, y_pred = self.evix.calc_linear_regression(x_inliers, y_inliers)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot inliers
        inlier_scatter = ax.scatter(
            x[~is_outlier],
            y[~is_outlier],
            alpha=0.7,
            color="purple",
            edgecolor="k",
            label="Inlier Buckets"
        )

        # Plot outliers
        if is_outlier.any():
            outlier_scatter = ax.scatter(
                x[is_outlier],
                y[is_outlier],
                alpha=0.7,
                color="orange",
                edgecolor="k",
                label="Outlier Buckets"
            )

        # Plot regression line (on inliers only)
        ax.plot(
            x_inliers,
            y_pred,
            color='red',
            linestyle='--',
            label=f"Fit (Excl. Outliers): y = {coeffs[0]:.2f}x + {coeffs[1]:.2f} (R² = {r_squared:.3f})"
        )

        # Tooltip setup
        scatter = inlier_scatter  # mplcursors needs one reference object, it will work for both groups
        mplcursors.cursor(scatter, hover=True).connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"Bucket: {int(bucket_avg['Bucket'].iloc[sel.index])}\n"
                f"{x_col}: {x.iloc[sel.index]:.4f}\n"
                f"{y_col}: {y.iloc[sel.index]:.4f}"
            )
        )

        ax.set_title(f"Avg {y_col} vs Avg {x_col} (Buckets = {buckets})", fontsize=14)
        ax.set_xlabel(f"Average {x_col}")
        ax.set_ylabel(f"Average {y_col}")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


    # unique plots
    def vix_and_next_realized_vol_vs_date(self, days_label=1):
        plt.figure(figsize=(14, 8))

        # Plot the two volatility measures
        plt.plot(self.data["Date"], self.data["Average VIX Level"], label="Average VIX Level", color="blue",
                 linewidth=2)
        plt.plot(self.data["Date"], self.data["Average Next Realized Volatility"],
                 label="Average Next Realized Volatility", color="green", linewidth=2)
        # plt.plot(self.data["Date"], self.data["MR Volatility"],
        #          label="MR Volatility", color="magenta", linewidth=2)

        # Plot the premium (overestimation) line
        plt.plot(self.data["Date"], self.data["Average VIX Premium"], label="VIX Premium", color="red", linewidth=1.5)

        # Horizontal line showing average premium
        avg_premium = self.data["Average VIX Premium"].mean()
        plt.axhline(y=avg_premium, color="red", linestyle=":", linewidth=1.5,
                    label=f"Average VIX Premium ({avg_premium:.2f})")

        plt.title("VIX Level vs. Next Realized Volatility with VIX Premium")
        plt.xlabel("Date")
        plt.ylabel(f"Volatility / VIX Level ({days_label}-Day Average)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def vix_decomposition(self, days=1):
        # Create rolling averages (or just use raw values if days=1)
        data = self.data.copy()

        data["Smooth VIX Level"] = data["VIX Level"].rolling(window=days).mean()
        data["Smooth Recent Volatility"] = data["Recent Volatility"].rolling(window=days).mean()
        data["Smooth MR Adjustment"] = data["MR Adjustment"].rolling(window=days).mean()
        data["Smooth Volatility Premium"] = data["Volatility Premium"].rolling(window=days).mean()
        data["Smooth DTM"] = data["DTM"].rolling(window=days).mean()

        # Drop rows with NaNs (from the rolling windows)
        data.dropna(subset=[
            "Smooth VIX Level", "Smooth Recent Volatility", "Smooth MR Adjustment",
            "Smooth Volatility Premium", "Smooth DTM"
        ], inplace=True)

        plt.figure(figsize=(14, 8))

        # Plot smoothed components
        plt.plot(data["Date"], data["Smooth Recent Volatility"], label="Recent Volatility", color="green")
        plt.plot(data["Date"], data["Smooth MR Adjustment"], label="MR Adjustment", color="purple")
        plt.plot(data["Date"], data["Smooth Volatility Premium"], label="Volatility Premium", color="red")
        plt.plot(data["Date"], data["Smooth DTM"], label="DTM", color="magenta")
        plt.plot(data["Date"], data["Smooth VIX Level"], label="VIX Level", color="blue", linewidth=2)

        plt.title(f"VIX Decomposition Over Time (Smoothed {days}-Day Average)")
        plt.xlabel("Date")
        plt.ylabel("Volatility / VIX Level")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Scatter (invisible) for tooltips
        scatter = plt.scatter(
            data["Date"],
            data["Smooth VIX Level"],
            s=5,
            alpha=0
        )

        # Tooltip with smoothed values
        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            i = sel.index
            sel.annotation.set_text(
                f"Date: {data['Date'].iloc[i]}\n"
                f"Recent Vol: {data['Smooth Recent Volatility'].iloc[i]:.2f}\n"
                f"MR Adj.: {data['Smooth MR Adjustment'].iloc[i]:.2f}\n"
                f"Volatility Premium: {data['Smooth Volatility Premium'].iloc[i]:.2f}\n"
                f"DTM: {data['Smooth DTM'].iloc[i]:.2f}\n"
                f"VIX: {data['Smooth VIX Level'].iloc[i]:.2f}"
            )

        plt.show()



    def vcr_performance(self, days=1):
        data = self.data.copy()
        data.dropna(subset=["VCR", "% Change Recent Volatility", "Date"], inplace=True)

        # Apply rolling averages
        data["Average VCR"] = data["VCR"].rolling(window=days).mean()
        data["Average Realized Change"] = data["% Change Recent Volatility"].rolling(window=days).mean()
        data["Average VCR Performance"] = data["Average Realized Change"] - data["Average VCR"]

        avg_perf = data["Average VCR Performance"].mean()

        plt.figure(figsize=(14, 8))

        # Main lines
        plt.plot(data["Date"], data["Average VCR"], label="VIX-Implied Change (Average VCR)", color="teal", linewidth=2)
        plt.plot(data["Date"], data["Average Realized Change"], label="Realized Volatility Change (Average)",
                 color="orange", linewidth=2)
        plt.plot(data["Date"], data["Average VCR Performance"], label="VCR Performance (Difference)", color="red",
                 linewidth=1, alpha=0.4)

        plt.axhline(avg_perf, color="red", linestyle="--", linewidth=1, alpha=0.8,
                    label=f"Avg VCR Perf: {avg_perf:.2f}")

        self.evix.data["Average VCR Performance"] = data["Average VCR Performance"]

        plt.title(f"VCR vs. Realized Volatility Change ({days}-Day Average)")
        plt.xlabel("Date")
        plt.ylabel("% Volatility Change / VCR % Difference")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Scatter invisible points for tooltips
        scatter = plt.scatter(data["Date"], data["Average VCR Performance"], s=5, alpha=0)

        # Add tooltips
        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            i = sel.index
            sel.annotation.set_text(
                f"Date: {data['Date'].iloc[i]}\n"
                f"Avg VCR: {data['Average VCR'].iloc[i]:.2f}\n"
                f"Avg Realized: {data['Average Realized Change'].iloc[i]:.2f}\n"
                f"Perf (Diff): {data['Average VCR Performance'].iloc[i]:.2f}"
            )

        plt.show()


    def all_performance(self, days=1):
        data = self.data.copy()
        data["Average Realized Change"] = data["% Change Recent Volatility"].rolling(window=days).mean()

        data["Average VCR"] = data["VCR"].rolling(window=days).mean()
        data["Average VIX Level"] = data["VIX Level"].rolling(window=days).mean()
        data["Average MR"] = data["MR Adjustment"].rolling(window=days).mean()
        data["Average Recent Volatility"] = data["Recent Volatility"].rolling(window=days).mean()
        data["Average VolP"] = data["Volatility Premium"].rolling(window=days).mean()

        data["Average VCR Performance"] = np.abs(data["Average VCR"] - data["Average Realized Change"])
        data["Average VIX Performance"] = np.abs(data["Average VIX Level"] - data["Average Realized Change"])
        data["Average MR Performance"] = np.abs(data["Average MR"] - data["Average Realized Change"])
        data["Average RV Performance"] = np.abs(data["Average Recent Volatility"] - data["Average Realized Change"])
        data["Average VolP Performance"] = np.abs(data["Average VolP"] - data["Average Realized Change"])

        plt.figure(figsize=(14, 8))
        plt.plot()

        # performance cols
        # plt.plot(data["Date"], data["Average VCR Performance"], label="VCR", color="teal")
        # plt.plot(data["Date"], data["Average VIX Performance"], label="VIX", color="blueviolet")
        # plt.plot(data["Date"], data["Average MR Performance"], label="MR", color="darkred")

        plt.plot(data["Date"], data["Average VCR"] - data["Average Realized Change"], label="VCR", color="teal")
        plt.plot(data["Date"], data["Average VIX Level"] - data["Average Realized Change"], label="VIX", color="blueviolet")
        plt.plot(data["Date"], data["Average MR"] - data["Average Realized Change"], label="MR", color="darkred")
        plt.plot(data["Date"], data["Average Recent Volatility"] - data["Average Realized Change"], label="RV", color="magenta")
        plt.plot(data["Date"], data["Average VolP"] - data["Average Realized Change"], label="VolP", color="orange")

        # horizontal lines
        mean_vcr_perf = data["Average VCR Performance"].mean()
        mean_vix_perf = data["Average VIX Performance"].mean()
        mean_mr_perf = data["Average MR Performance"].mean()
        mean_rv_perf = data["Average RV Performance"].mean()
        mean_volp_perf = data["Average VolP Performance"].mean()
        # (data["Average MR Performance"]).to_csv("mrperf.csv")
        # (data["Average VIX Performance"]).to_csv("vixperf.csv")
        # (data["Average VCR Performance"]).to_csv("vcrperf.csv")

        # print(data["Average MR Performance"].head(252*2))

        plt.axhline(mean_vcr_perf, color="teal", linestyle="--", linewidth=1.5,
                    label=f"Avg VCR Perf: {mean_vcr_perf:.2f}")
        plt.axhline(mean_vix_perf, color="blueviolet", linestyle="--", linewidth=1.5,
                    label=f"Avg VIX Perf: {mean_vix_perf:.2f}")
        plt.axhline(mean_mr_perf, color="darkred", linestyle="--", linewidth=1.5,
                    label=f"Avg MR Perf: {mean_mr_perf:.2f}")
        plt.axhline(mean_rv_perf, color="magenta", linestyle="--", linewidth=1.5,
                    label=f"Avg MR Perf: {mean_rv_perf:.2f}")
        plt.axhline(mean_volp_perf, color="orange", linestyle="--", linewidth=1.5,
                    label=f"Avg MR Perf: {mean_volp_perf:.2f}")


        plt.title(f"Performance of Different Volatility Metrics ({days}-Day Average)")
        plt.xlabel("Date")
        plt.ylabel("Difference to Next Realized Volatility")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Scatter (invisible) for tooltips
        scatter = plt.scatter(
            data["Date"],
            data["Average VCR"] - data["Average Realized Change"],
            s=5,
            alpha=0
        )

        # Tooltip with performance values
        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            i = sel.index
            sel.annotation.set_text(
                f"Date: {data['Date'].iloc[i]}\n"
                f"VCR Diff: {data['Average VCR'].iloc[i] - data['Average Realized Change'].iloc[i]:.2f}\n"
                f"VIX Diff: {data['Average VIX Level'].iloc[i] - data['Average Realized Change'].iloc[i]:.2f}\n"
                f"MR Diff: {data['Average MR'].iloc[i] - data['Average Realized Change'].iloc[i]:.2f}\n"
                f"RV Diff: {data['Average Recent Volatility'].iloc[i] - data['Average Realized Change'].iloc[i]:.2f}\n"
                f"Realized Change: {data['Average Realized Change'].iloc[i]:.2f}"
            )

        plt.show()


