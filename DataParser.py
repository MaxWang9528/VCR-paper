import numpy as np
import pandas as pd
from datetime import datetime

class DataParser:
    def __init__(self, spx_file, vix_file, start_date=None, end_date=None):
        self.start_date = datetime.strptime(start_date, '%m/%d/%Y').date() if start_date else None
        self.end_date = datetime.strptime(end_date, '%m/%d/%Y').date() if end_date else None
        self.spx_data = self._file_to_df(spx_file)
        self.vix_data = self._file_to_df(vix_file)
        self._combine_spx_vix()
        self._use_core_data()
        self._exclude_dates()
        self.spx_data.dropna(subset=["VIX Level"], inplace=True)

    @staticmethod
    def _file_to_df(file):
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.date
        df['Change %'] = df['Change %'].str.replace('%', '', regex=False).astype(float)
        # df['Change'] = df['Change %'].str.replace('%', '', regex=False).astype(float) * 0.01
        # df['Change'] = np.round(df['Change'], 4)
        # df.drop(columns=['Change %'], inplace=True)
        return df.sort_values('Date')

    def _combine_spx_vix(self):
        vix_subset = self.vix_data[['Date', 'Price']].rename(columns={'Price': 'VIX Level'})
        self.spx_data = self.spx_data.merge(vix_subset, on='Date', how='left')

    def _use_core_data(self):
        self.spx_data = self.spx_data.drop(columns=['Open', 'High', 'Low', 'Vol.'], errors='ignore')

    def _exclude_dates(self):
        if self.start_date:
            self.spx_data = self.spx_data[self.spx_data['Date'] >= self.start_date]
        if self.end_date:
            self.spx_data = self.spx_data[self.spx_data['Date'] <= self.end_date]



