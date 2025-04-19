from matplotlib import pyplot as plt
import mplcursors

class Grapher:
    def __init__(self, evix):
        self.evix = evix
        self.data = evix.data

    def scatter_plot(self, x_col, y_col, df=None):
        if df is None:
            df = self.data.dropna(subset=[x_col, y_col, "Date"])
        else:
            df = df.dropna(subset=[x_col, y_col, "Date"])

        fig, ax = plt.subplots(figsize=(8, 6))
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

        fig, ax = plt.subplots(figsize=(10, 6))

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
        plt.figure(figsize=(14, 7))

        # Plot the two volatility measures
        plt.plot(self.data["Date"], self.data["Average VIX Level"], label="Average VIX Level", color="blue",
                 linewidth=2)
        plt.plot(self.data["Date"], self.data["Average Next Realized Volatility"],
                 label="Average Next Realized Volatility", color="green", linewidth=2)
        # plt.plot(self.data["Date"], self.data["MR Volatility"],
        #          label="MR Volatility", color="magenta", linewidth=2)

        # Plot the premium (overestimation) line
        plt.plot(self.data["Date"], self.data["Average Premium"], label="VIX Premium", color="red", linewidth=1.5)

        # Horizontal line showing average premium
        avg_premium = self.data["Average Premium"].mean()
        plt.axhline(y=avg_premium, color="red", linestyle=":", linewidth=1.5,
                    label=f"Average Premium ({avg_premium:.2f})")

        plt.title("VIX Level vs. Next Realized Volatility with Premium")
        plt.xlabel("Date")
        plt.ylabel(f"Volatility / VIX Level ({days_label}-Day Average)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()