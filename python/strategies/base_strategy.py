import os
import pandas as pd
import yfinance as yf


class Strategy:

    def __init__(self, name, asset, period_start, period_end, interval):
        """
        Initialize the Strategy class with the given parameters.

        :param name: Name of the strategy.
        :param asset: Asset symbol to fetch data for (e.g., 'AAPL').
        :param period_start: Start date for fetching data (format: 'YYYY-MM-DD').
        :param period_end: End date for fetching data (format: 'YYYY-MM-DD').
        :param interval: Data interval (e.g., '1d', '1h', etc.).
        """
        self.name = name
        self.asset = asset
        self.period_start = period_start
        self.period_end = period_end
        self.interval = interval
        self.data = None  # DataFrame to store the asset's data

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
                self.data = pd.read_pickle(file_path)
                print(f"Data loaded from {file_path}")
            else:
                # Download data from Yahoo Finance
                self.data = yf.download(
                    tickers=self.asset,
                    start=self.period_start,
                    end=self.period_end,
                    interval=self.interval
                )

                # Check if data is not empty before saving
                if not self.data.empty:
                    self.data.to_pickle(file_path)
                    print(f"Data saved to {file_path}")
                else:
                    print(f"No data available for {self.asset} from {self.period_start} to {self.period_end}")

            return self.data

        except Exception as e:
            print(f"An error occurred while fetching or loading data: {e}")
            return None

    def calculate_sma(self, period=20, column='Close'):
        """
        Calculate the Simple Moving Average (SMA) and add it to the DataFrame.

        :param period: Number of periods for calculating SMA (default is 20).
        :param column: Column on which to base the SMA calculation (default is 'Close').
        :return: The DataFrame with a new 'SMA_{period}' column.
        """
        try:
            if self.data is None:
                self.get_data()

            sma = self.data[column].rolling(window=period).mean()
            self.data[f'SMA_{period}'] = sma

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating SMA: {e}")
            return None

    def calculate_ema(self, period=20, column='Close'):
        """
        Calculate the Exponential Moving Average (EMA) and add it to the DataFrame.

        :param period: Number of periods for calculating EMA (default is 20).
        :param column: Column on which to base the EMA calculation (default is 'Close').
        :return: The DataFrame with a new 'EMA_{period}' column.
        """
        try:
            if self.data is None:
                self.get_data()

            ema = self.data[column].ewm(span=period, adjust=False).mean()
            self.data[f'EMA_{period}'] = ema

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating EMA: {e}")
            return None

    def calculate_macd(self, slow_period=26, fast_period=12, signal_period=9, column='Close'):
        """
        Calculate the MACD and add it to the DataFrame.

        :param slow_period: Period for the slow EMA (default is 26).
        :param fast_period: Period for the fast EMA (default is 12).
        :param signal_period: Period for the signal line EMA (default is 9).
        :param column: Column on which to base the MACD calculation (default is 'Close').
        :return: The DataFrame with new 'MACD' and 'Signal_Line' columns.
        """
        try:
            if self.data is None:
                self.get_data()

            ema_fast = self.data[column].ewm(span=fast_period, adjust=False).mean()
            ema_slow = self.data[column].ewm(span=slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

            self.data['MACD'] = macd_line
            self.data['Signal_Line'] = signal_line

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating MACD: {e}")
            return None

    def calculate_bollinger_bands(self, period=20, std_dev=2, column='Close'):
        """
        Calculate Bollinger Bands and add them to the DataFrame.

        :param period: Number of periods for moving average and standard deviation (default is 20).
        :param std_dev: Number of standard deviations for the bands (default is 2).
        :param column: Column on which to base the calculation (default is 'Close').
        :return: The DataFrame with new 'Bollinger_Middle', 'Bollinger_Upper', 'Bollinger_Lower' columns.
        """
        try:
            if self.data is None:
                self.get_data()

            sma = self.data[column].rolling(window=period).mean()
            std = self.data[column].rolling(window=period).std()

            self.data['Bollinger_Middle'] = sma
            self.data['Bollinger_Upper'] = sma + (std_dev * std)
            self.data['Bollinger_Lower'] = sma - (std_dev * std)

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating Bollinger Bands: {e}")
            return None

    def calculate_atr(self, period=14):
        """
        Calculate the Average True Range (ATR) and add it to the DataFrame.

        :param period: Number of periods for calculating ATR (default is 14).
        :return: The DataFrame with a new 'ATR' column.
        """
        try:
            if self.data is None:
                self.get_data()

            high_low = self.data['High'] - self.data['Low']
            high_close = (self.data['High'] - self.data['Close'].shift()).abs()
            low_close = (self.data['Low'] - self.data['Close'].shift()).abs()

            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=period).mean()

            self.data['ATR'] = atr

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating ATR: {e}")
            return None

    def calculate_stochastic_oscillator(self, k_period=14, d_period=3):
        """
        Calculate the Stochastic Oscillator and add it to the DataFrame.

        :param k_period: Number of periods for %K (default is 14).
        :param d_period: Number of periods for %D (default is 3).
        :return: The DataFrame with new '%K' and '%D' columns.
        """
        try:
            if self.data is None:
                self.get_data()

            low_min = self.data['Low'].rolling(window=k_period).min()
            high_max = self.data['High'].rolling(window=k_period).max()

            self.data['%K'] = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
            self.data['%D'] = self.data['%K'].rolling(window=d_period).mean()

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating Stochastic Oscillator: {e}")
            return None

    def calculate_cci(self, period=20):
        """
        Calculate the Commodity Channel Index (CCI) and add it to the DataFrame.

        :param period: Number of periods for calculating CCI (default is 20).
        :return: The DataFrame with a new 'CCI' column.
        """
        try:
            if self.data is None:
                self.get_data()

            tp = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mean_dev = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())

            self.data['CCI'] = (tp - sma_tp) / (0.015 * mean_dev)

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating CCI: {e}")
            return None

    def calculate_obv(self):
        """
        Calculate the On-Balance Volume (OBV) and add it to the DataFrame.

        :return: The DataFrame with a new 'OBV' column.
        """
        try:
            if self.data is None:
                self.get_data()

            obv = [0]
            for i in range(1, len(self.data)):
                if self.data['Close'][i] > self.data['Close'][i - 1]:
                    obv.append(obv[-1] + self.data['Volume'][i])
                elif self.data['Close'][i] < self.data['Close'][i - 1]:
                    obv.append(obv[-1] - self.data['Volume'][i])
                else:
                    obv.append(obv[-1])

            self.data['OBV'] = obv

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating OBV: {e}")
            return None

    def calculate_parabolic_sar(self, initial_af=0.02, max_af=0.2):
        """
        Calculate the Parabolic SAR and add it to the DataFrame.

        :param initial_af: Initial acceleration factor (default is 0.02).
        :param max_af: Maximum acceleration factor (default is 0.2).
        :return: The DataFrame with a new 'Parabolic_SAR' column.
        """
        try:
            if self.data is None:
                self.get_data()

            high = self.data['High']
            low = self.data['Low']
            close = self.data['Close']

            # Initialize Parabolic SAR series
            psar = close.copy()
            psar.iloc[0] = close.iloc[0]
            bull = True
            af = initial_af
            ep = high.iloc[0]  # Extreme Point

            # Inner helper function to update AF and EP
            def update_af_ep(is_bull, curr_high, curr_low, extreme_point, accel_factor):
                if is_bull and curr_high > extreme_point:
                    extreme_point = curr_high
                    accel_factor = min(accel_factor + initial_af, max_af)
                elif not is_bull and curr_low < extreme_point:
                    extreme_point = curr_low
                    accel_factor = min(accel_factor + initial_af, max_af)
                return accel_factor, extreme_point

            # Inner helper function to check for trend reversal
            def check_trend_reversal(is_bull, curr_high, curr_low, psar_val, extreme_point, accel_factor):
                if is_bull and curr_low < psar_val:
                    is_bull = False
                    accel_factor = initial_af
                    psar_val = extreme_point
                    extreme_point = curr_low
                elif not is_bull and curr_high > psar_val:
                    is_bull = True
                    accel_factor = initial_af
                    psar_val = extreme_point
                    extreme_point = curr_high
                return is_bull, psar_val, accel_factor, extreme_point

            for i in range(1, len(self.data)):
                # Calculate PSAR
                psar.iloc[i] = psar.iloc[i - 1] + af * (ep - psar.iloc[i - 1])

                # Update AF and EP
                af, ep = update_af_ep(
                    bull, high.iloc[i], low.iloc[i], ep, af
                )

                # Check for trend reversal
                bull, psar.iloc[i], af, ep = check_trend_reversal(
                    bull, high.iloc[i], low.iloc[i], psar.iloc[i], ep, af
                )

            # Add the Parabolic SAR to the DataFrame
            self.data['Parabolic_SAR'] = psar

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating Parabolic SAR: {e}")
            return None

    def calculate_ichimoku_cloud(self):
        """
        Calculate the Ichimoku Cloud and add its components to the DataFrame.

        :return: The DataFrame with new columns for Ichimoku components.
        """
        try:
            if self.data is None:
                self.get_data()

            high = self.data['High']
            low = self.data['Low']
            close = self.data['Close']

            # Tenkan-sen (Conversion Line)
            tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2

            # Kijun-sen (Base Line)
            kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2

            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

            # Senkou Span B (Leading Span B)
            senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)

            # Chikou Span (Lagging Span)
            chikou_span = close.shift(-26)

            self.data['Tenkan_Sen'] = tenkan_sen
            self.data['Kijun_Sen'] = kijun_sen
            self.data['Senkou_Span_A'] = senkou_span_a
            self.data['Senkou_Span_B'] = senkou_span_b
            self.data['Chikou_Span'] = chikou_span

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating Ichimoku Cloud: {e}")
            return None

    def calculate_rsi(self, period=14, column='Close'):
        """
        Calculate the Relative Strength Index (RSI) and add it to the DataFrame.

        :param period: Number of periods for calculating RSI (default is 14).
        :param column: Column on which to base the RSI calculation (default is 'Close').
        :return: The DataFrame with a new 'RSI' column.
        """
        try:
            if self.data is None:
                self.get_data()

            delta = self.data[column].diff(1)
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

            self.data['RSI'] = rsi

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating RSI: {e}")
            return None

    def calculate_technical_indicators(self, indicators=None):
        """
        Calculate selected technical indicators and add them to the DataFrame.

        :param indicators: List of indicator names to calculate. If None, calculates all.
        :return: The DataFrame with new columns for each technical indicator.
        """
        try:
            if self.data is None:
                self.get_data()

            if self.data is None or self.data.empty:
                print("No data available to calculate indicators.")
                return None

            # Default to all indicators if none specified
            if indicators is None:
                indicators = [
                    'sma', 'ema', 'macd', 'bollinger_bands', 'atr',
                    'stochastic_oscillator', 'cci', 'obv', 'parabolic_sar',
                    'ichimoku_cloud', 'rsi'
                ]

            # Map indicator names to methods
            indicator_methods = {
                'sma': self.calculate_sma,
                'ema': self.calculate_ema,
                'macd': self.calculate_macd,
                'bollinger_bands': self.calculate_bollinger_bands,
                'atr': self.calculate_atr,
                'stochastic_oscillator': self.calculate_stochastic_oscillator,
                'cci': self.calculate_cci,
                'obv': self.calculate_obv,
                'parabolic_sar': self.calculate_parabolic_sar,
                'ichimoku_cloud': self.calculate_ichimoku_cloud,
                'rsi': self.calculate_rsi
            }

            # Calculate specified indicators
            for indicator in indicators:
                method = indicator_methods.get(indicator.lower())
                if method:
                    method()
                else:
                    print(f"Indicator '{indicator}' not recognized.")

            return self.data

        except Exception as e:
            print(f"An error occurred while calculating technical indicators: {e}")
            return None
