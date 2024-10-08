from strategies.base_strategy import Strategy


def main():
    """
    Main function to initialize the strategy and fetch financial data.
    """
    try:
        # Initialize the strategy with given parameters
        strategy = Strategy(
            name="RSI",
            asset="AAPL",
            period_start="2023-01-01",
            period_end="2023-12-31",
            interval="1h"
        )

        # Fetch the data using the strategy
        df = strategy.get_data()

        # Calculate only SMA and RSI indicators
        df_with_all_indicators = strategy.calculate_technical_indicators(indicators=[
            'sma',
            'ema',
            'macd',
            'bollinger_bands',
            'atr',
            'stochastic_oscillator',
            'cci',
            'obv',
            'parabolic_sar',
            'ichimoku_cloud',
            'rsi'
        ])

        strategy.plot()

        # Check if the DataFrame is not empty and print the data
        if df_with_all_indicators is not None and not df.empty:
            print(df)
        else:
            print("No data available for the given parameters.")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")


if __name__ == "__main__":
    # Only execute main if this script is run directly
    main()
