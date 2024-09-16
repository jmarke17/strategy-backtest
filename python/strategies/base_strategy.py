import os
import pandas as pd
import yfinance as yf


class Strategy:

    def __init__(self, name, asset, period_start, period_end, interval):
        """
        Initialize the Strategy class with the given parameters.

        :param name: Name of the strategy.
        :param asset: Asset symbol to fetch data for (e.g., 'O').
        :param period_start: Start date for fetching data (format: 'YYYY-MM-DD').
        :param period_end: End date for fetching data (format: 'YYYY-MM-DD').
        :param interval: Data interval (e.g., '1d', '1h', etc.).
        """
        self.name = name
        self.asset = asset
        self.period_start = period_start
        self.period_end = period_end
        self.interval = interval

    def get_data(self):
        """
        Fetch financial data from Yahoo Finance or load it from a local pickle file if available.

        :return: A pandas DataFrame containing the asset's data.
        """
        try:
            # Define directory to save/load pickle files
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
            os.makedirs(data_dir, exist_ok=True)

            # Generate a file name based on asset, interval, and date range
            file_name = f"{self.asset}_{self.interval}_{self.period_start}_{self.period_end}.pkl"
            file_path = os.path.join(data_dir, file_name)

            # Check if the pickle file exists
            if os.path.exists(file_path):
                # Load data from pickle file
                data = pd.read_pickle(file_path)
                print(f"Data loaded from {file_path}")
            else:
                # Download data from Yahoo Finance
                data = yf.download(
                    tickers=self.asset,
                    start=self.period_start,
                    end=self.period_end,
                    interval=self.interval
                )

                # Check if data is not empty before saving
                if not data.empty:
                    data.to_pickle(file_path)
                    print(f"Data saved to {file_path}")
                else:
                    print(f"No data available for {self.asset} from {self.period_start} to {self.period_end}")

            return data

        except Exception as e:
            print(f"An error occurred while fetching or loading data: {e}")
            return None
