from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np

class Trader:
    def calculate_periods(self, state: TradingState) -> (int, int):
        # Calculate short-term and long-term periods based on the number of orders
        num_orders = 4000  # Update this value with the actual number of orders from your CSV file
        
        # Example: Divide the number of orders by a fixed ratio to determine periods
        short_term_period = max(10, num_orders // 100)  # Set a minimum short-term period of 10
        long_term_period = max(100, num_orders // 40)   # Set a minimum long-term period of 100
        
        return short_term_period, long_term_period
    
    def run(self, state: TradingState):
        # Calculate short-term and long-term periods dynamically
        short_term_period, long_term_period = self.calculate_periods(state)

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders = []

            # Calculate moving averages
            short_term_ma = np.mean(order_depth.historical_prices[-short_term_period:])
            long_term_ma = np.mean(order_depth.historical_prices[-long_term_period:])

            # Generate signals based on moving averages
            signal = None
            if short_term_ma > long_term_ma:
                signal = 'BUY'
            elif short_term_ma < long_term_ma:
                signal = 'SELL'

            # Execute trades based on signals
            if signal:
                best_price = order_depth.buy_orders[0][0] if signal == 'SELL' else order_depth.sell_orders[0][0]
                orders.append(Order(product, best_price, -1 if signal == 'SELL' else 1))

            result[product] = orders

        trader_data = "SAMPLE"  # String value holding Trader state data required.
        conversions = 1

        return result, conversions, trader_data
