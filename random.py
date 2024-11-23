from typing import Dict, List, OrderedDict
from datamodel import OrderDepth, TradingState, Order, Symbol, Trade
import math
import pandas as pd
import numpy as np
import statistics


class Trader:
    PRODUCTS = [
        'AMETHYSTS',
        'STARFRUIT',
        'ORCHIDS',
        'GIFT_BASKET',
        'CHOCOLATE',
        'STRAWBERRIES',
        'ROSES',
        'COCONUT',
        'COCONUT_COUPON'
    ]

    DEFAULT_PRICES = {
        'AMETHYSTS': 10000,
        'STARFRUIT': 5000,
        'ORCHIDS': 1000,
        'GIFT_BASKET': 70000,
        'CHOCOLATE': 8000,
        'STRAWBERRIES': 4000,
        'ROSES': 15000,
        'COCONUT': 10000,
        'COCONUT_COUPON': 600
    }

    POSITION_LIMITS = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
        'ORCHIDS': 100,
        'GIFT_BASKET': 60,
        'CHOCOLATE': 250,
        'STRAWBERRIES': 350,
        'ROSES': 60,
        'COCONUT': 300,
        'COCONUT_COUPON': 600

    }

    prices_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [],
                      'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
    mid_prices_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [],
                          'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
    mid_p_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [],
                          'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
    p_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [],
                      'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
    errors_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [],
                      'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
    forecasted_diff_history = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDS": [], "GIFT_BASKET": [], "STRAWBERRIES": [],
                               'CHOCOLATE': [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}

    current_signal = {"AMETHYSTS": "", "STARFRUIT": "None", "ORCHIDS": "None"}

    export_tariffs = {"Min": 1000, "Max": 0, "Second Max": 0}

    spreads_basket = []
    spreads_roses = []
    spreads_chocolates = []

    ratios_basket = []

    etf_prices = []
    etf_returns = []
    nav_prices = []
    nav_returns = []

    coconuts_returns = []
    last_variances = []

    INF = int(1e9)
    LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350,
             'ROSES': 60, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
    POSITION = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0,
                'GIFT_BASKET': 0, 'COCONUT': 0, 'COCONUT_COUPON': 0}
    STARFRUIT_CACHE_SIZE = 5
    AME_RANGE = 2
    ORCHID_MM_RANGE = 5
    starfruit_cache = []

    DIFFERENCE_MEAN = 379.4904833333333
    DIFFERENCE_STD = 76.42438217375009
    PERCENT_OF_STD_TO_TRADE_AT = 0.7
    gift_basket_quantity = 60

    ITERS = 30_000
    COUPON_DIFFERENCE_STD = 13.381768062409492
    COCONUT_DIFFERENCE_STD = 88.75266514702373
    PREV_COCONUT_PRICE = -1
    PREV_COUPON_PRICE = -1
    COCONUT_MEAN = 9999.900983333333
    COCONUT_SUM = 299997029.5
    COUPON_SUM = 19051393.0
    COUPON_Z_SCORE = 0.5

    COCONUT_STORE = []
    COCONUT_STORE_SIZE = 25
    COCONUT_BS_STORE = []

    delta_signs = 1
    time = 0

    COUPON_IV_STORE = []
    COUPON_IV_STORE_SIZE = 25

    def __init__(self) -> None:

        self.ema_prices = dict()
        for product in self.PRODUCTS:
            self.ema_prices[product] = None
        self.ema_param = 0.

        self.window_size = 21

        self.current_pnl = dict()
        self.qt_traded = dict()
        self.pnl_tracker = dict()

        for product in self.PRODUCTS:
            self.current_pnl[product] = 0
            self.qt_traded[product] = 0
            self.pnl_tracker[product] = []

        self.omega = 0.2478
        self.alpha = 8.9738e-03
        self.beta = 0.7572

        self.initial_last_variance = 1.0489105974177813

    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)

    def get_order_book(self, product, state: TradingState):
        market_bids = list((state.order_depths[product].buy_orders).items())
        market_asks = list((state.order_depths[product].sell_orders).items())

        if len(market_bids) > 1:
            bid_price_1, bid_amount_1 = market_bids[0]
            bid_price_2, bid_amount_2 = market_bids[1]

        if len(market_asks) > 1:
            ask_price_1, ask_amount_1 = market_asks[0]
            ask_price_2, ask_amount_2 = market_asks[1]

        bid_price, ask_price = bid_price_1, ask_price_1

        if bid_amount_1 < 5:
            bid_price = bid_price_2
        else:
            bid_price = bid_price_1 + 1

        if ask_amount_1 < 5:
            ask_price = ask_price_2
        else:
            ask_price = ask_price_1 - 1

        return bid_price, ask_price

    def get_best_bid(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders
        # best_bid = max(market_bids)
        best_bid, best_bid_amount = list(market_bids.items())[0]

        return best_bid, best_bid_amount

    def get_best_ask(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        # best_ask = min(market_asks)
        best_ask, best_ask_amount = list(market_asks.items())[0]

        return best_ask, best_ask_amount

    def get_bid2(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 2:
            bid2, bid2_amount = list(market_bids.items())[1]
        else:
            # bid2, bid2_amount = float('-inf'), 0
            bid2, bid2_amount = 0, 0

        return bid2, bid2_amount

    def get_bid3(self, product, state: TradingState):
        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 3:
            bid3, bid3_amount = list(market_bids.items())[2]
        else:
            # bid3, bid3_amount = float('-inf'), 0
            bid3, bid3_amount = 0, 0

        return bid3, bid3_amount

    def get_ask2(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        # best_ask = min(market_asks)
        if len(market_asks) == 2:
            ask2, ask2_amount = list(market_asks.items())[1]
        else:
            # ask2, ask2_amount = float('inf'), 0
            ask2, ask2_amount = 500000, 0

        return ask2, ask2_amount

    def get_ask3(self, product, state: TradingState):
        market_asks = state.order_depths[product].sell_orders
        # best_ask = min(market_asks)
        if len(market_asks) == 3:
            ask3, ask3_amount = list(market_asks.items())[2]
        else:
            # ask3, ask3_amount = float('inf'), 0
            ask3, ask3_amount = 500000, 0

        return ask3, ask3_amount

    def get_mid_price(self, product, state: TradingState):

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = self.DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price

        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask) / 2

    def get_last_price(self, symbol, own_trades: Dict[Symbol, List[Trade]], market_trades: Dict[Symbol, List[Trade]]):
        recent_trades = []
        if symbol in own_trades:
            recent_trades.extend(own_trades[symbol])
        if symbol in market_trades:
            recent_trades.extend(market_trades[symbol])
        recent_trades.sort(key=lambda trade: trade.timestamp)
        last_trade = recent_trades[-1]
        return last_trade.price

    def update_prices_history(self, own_trades: Dict[Symbol, List[Trade]], market_trades: Dict[Symbol, List[Trade]]):
        for symbol in self.PRODUCTS:
            recent_trades = []
            if symbol in own_trades:
                recent_trades.extend(own_trades[symbol])
            if symbol in market_trades:
                recent_trades.extend(market_trades[symbol])

            recent_trades.sort(key=lambda trade: trade.timestamp)

            for trade in recent_trades:
                self.prices_history[symbol].append(trade.price)

            while len(self.prices_history[symbol]) > self.window_size:
                self.prices_history[symbol].pop(0)

    def update_mid_prices_history(self, state):
        for symbol in self.PRODUCTS:
            mid_price = self.get_mid_price(symbol, state)

            self.mid_prices_history[symbol].append(mid_price)

            while len(self.mid_prices_history[symbol]) > self.window_size:
                self.mid_prices_history[symbol].pop(0)

    def update_diff_history(self, diff_history, p_history):
        for symbol in self.PRODUCTS:
            if len(p_history[symbol]) >= 2:
                diff = p_history[symbol][-1] - p_history[symbol][-2]

                diff_history[symbol].append(diff)

            while len(diff_history[symbol]) > 8:
                diff_history[symbol].pop(0)

    def update_ema_prices(self, state: TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in self.PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1 - self.ema_param) * self.ema_prices[product]

        # print(self.ema_prices)

    def calculate_sma(self, product, window_size):
        sma = None
        prices = pd.Series(self.mid_prices_history[product])
        if len(prices) >= window_size:
            window_sum = prices.iloc[-window_size:].sum()
            sma = window_sum / window_size
        return sma

    def calculate_ema(self, product, window_size):
        ema = None
        prices = pd.Series(self.mid_prices_history[product])
        if len(prices) >= window_size:
            ema = prices.ewm(span=window_size, adjust=False).mean().iloc[-1]
        return ema

    def calculate_vwap(self, symbol, own_trades: Dict[Symbol, List[Trade]], market_trades: Dict[Symbol, List[Trade]]):
        vwap = None
        recent_trades = []
        prices = []
        volumes = []
        if symbol in own_trades:
            recent_trades.extend(own_trades[symbol])
        if symbol in market_trades:
            recent_trades.extend(market_trades[symbol])

        recent_trades.sort(key=lambda trade: trade.timestamp)

        for trade in recent_trades:
            prices.append(trade.price)
            volumes.append(trade.quantity)

        data = pd.DataFrame({'prices': prices, 'volumes': volumes})
        vwap = (data['prices'] * data['volumes']).sum() / data['volumes'].sum()
        return vwap

    def calculate_standard_deviation(self, values: List[float]) -> float:
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]

        variance = sum(squared_diffs) / len(values)

        std_dev = math.sqrt(variance)

        return std_dev

    def calculate_order_book_imbalance(self, symbol, state: TradingState):
        if symbol not in state.order_depths:
            return None
        order_book = state.order_depths[symbol]
        bid_volume = sum(order_book.buy_orders.values())
        ask_volume = sum(order_book.sell_orders.values())

        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            imbalance = (bid_volume - ask_volume) / total_volume
            return imbalance
        else:
            print(total_volume)
            return 0

    def basket_strategy(self, state: TradingState) -> List[Order]:
        """
        Buying and Selling based on last trade price vs mean price (ceiling floor version)
        """
        basket_orders = []
        chocolates_orders = []
        strawberries_orders = []
        roses_orders = []

        #### POSITIONS ####
        position_basket = self.get_position('GIFT_BASKET', state)
        position_chocolates = self.get_position('CHOCOLATE', state)
        position_strawberries = self.get_position('STRAWBERRIES', state)
        position_roses = self.get_position('ROSES', state)

        current_position_all = abs(position_basket) + abs(position_chocolates) + abs(position_strawberries) + abs(
            position_roses)

        #### QUANTITIES WE ARE ALLOWED TO TRADE ####
        buy_volume_basket = (self.POSITION_LIMITS['GIFT_BASKET'] - position_basket)
        sell_volume_basket = (- self.POSITION_LIMITS['GIFT_BASKET'] - position_basket)
        buy_volume_chocolates = (self.POSITION_LIMITS['CHOCOLATE'] - position_chocolates)
        sell_volume_chocolates = (- self.POSITION_LIMITS['CHOCOLATE'] - position_chocolates)
        buy_volume_strawberries = (self.POSITION_LIMITS['STRAWBERRIES'] - position_strawberries)
        sell_volume_strawberries = (- self.POSITION_LIMITS['STRAWBERRIES'] - position_strawberries)
        buy_volume_roses = (self.POSITION_LIMITS['ROSES'] - position_roses)
        sell_volume_roses = (- self.POSITION_LIMITS['ROSES'] - position_roses)

        #### MID PRICES ####
        basket_mid_price = self.get_mid_price('GIFT_BASKET', state)
        chocolates_mid_price = self.get_mid_price('CHOCOLATE', state)
        strawberries_mid_price = self.get_mid_price('STRAWBERRIES', state)
        roses_mid_price = self.get_mid_price('ROSES', state)

        #### BIDS, ASKS, VOLUMES ####
        basket_bid, basket_bid_vol = self.get_best_bid('GIFT_BASKET', state)
        basket_ask, basket_ask_vol = self.get_best_ask('GIFT_BASKET', state)
        chocolates_bid, chocolates_bid_vol = self.get_best_bid('CHOCOLATE', state)
        chocolates_ask, chocolates_ask_vol = self.get_best_ask('CHOCOLATE', state)
        strawberries_bid, strawberries_bid_vol = self.get_best_bid('STRAWBERRIES', state)
        strawberries_ask, strawberries_ask_vol = self.get_best_ask('STRAWBERRIES', state)
        roses_bid, roses_bid_vol = self.get_best_bid('ROSES', state)
        roses_ask, roses_ask_vol = self.get_best_ask('ROSES', state)

        strawberries_book_spread = strawberries_ask - strawberries_bid

        #### SECOND BIDS AND ASKS ####
        basket_bid_2, basket_bid_vol_2 = self.get_bid2('GIFT_BASKET', state)
        basket_ask_2, basket_ask_vol_2 = self.get_ask2('GIFT_BASKET', state)
        chocolates_bid_2, chocolates_bid_vol_2 = self.get_bid2('CHOCOLATE', state)
        chocolates_ask_2, chocolates_ask_vol_2 = self.get_ask2('CHOCOLATE', state)
        strawberries_bid_2, strawberries_bid_vol_2 = self.get_bid2('STRAWBERRIES', state)
        strawberries_ask_2, strawberries_ask_vol_2 = self.get_ask2('STRAWBERRIES', state)
        roses_bid_2, roses_bid_vol_2 = self.get_bid2('ROSES', state)
        roses_ask_2, roses_ask_vol_2 = self.get_ask2('ROSES', state)

        #### THIRD BIDS AND ASKS ####
        basket_bid_3, basket_bid_vol_3 = self.get_bid3('GIFT_BASKET', state)
        basket_ask_3, basket_ask_vol_3 = self.get_ask3('GIFT_BASKET', state)
        chocolates_bid_3, chocolates_bid_vol_3 = self.get_bid3('CHOCOLATE', state)
        chocolates_ask_3, chocolates_ask_vol_3 = self.get_ask3('CHOCOLATE', state)
        strawberries_bid_3, strawberries_bid_vol_3 = self.get_bid3('STRAWBERRIES', state)
        strawberries_ask_3, strawberries_ask_vol_3 = self.get_ask3('STRAWBERRIES', state)
        roses_bid_3, roses_bid_vol_3 = self.get_bid3('ROSES', state)
        roses_ask_3, roses_ask_vol_3 = self.get_ask3('ROSES', state)

        nav = 4 * chocolates_mid_price + 6 * strawberries_mid_price + roses_mid_price
        spread_basket = basket_mid_price - nav

        spread_roses = roses_mid_price - 1.3427 * chocolates_mid_price

        spread_chocolates = basket_mid_price - 5.7223 * chocolates_mid_price

        self.spreads_basket.append(spread_basket)
        self.spreads_roses.append(spread_roses)
        self.spreads_chocolates.append(spread_chocolates)

        if len(self.mid_prices_history['GIFT_BASKET']) > 1:
            # intercept, intercept_std, betas, betas_std = self.linear_regression(gift_basket_prices, chocolates_prices, strawberries_prices, roses_prices)
            intercept = 165.3320184325068
            intercept_std = 72.08630428543141
            betas = [3.84317626, 6.17179092, 1.05264383]

            # intercept = -421.4553
            # intercept_std = 47.103
            # betas = [3.8385 , 6.3047, 1.0586]

            beta_chocolates = betas[0]
            beta_strawberries = betas[1]
            beta_roses = betas[2]

            nav = beta_chocolates * chocolates_mid_price + beta_strawberries * strawberries_mid_price + beta_roses * roses_mid_price + intercept
            # print('regression!')
            # print(beta_chocolates)
            # print(beta_strawberries)
            # print(beta_roses)
            # print(intercept)
            spread = basket_mid_price - nav
            print('spread', spread)
            self.spreads_basket.append(spread)
            mean = statistics.mean(self.spreads_basket)
            std = statistics.stdev(self.spreads_basket)

            z_score = (spread - mean) / std

            # if z_score > -0.4 and z_score < 0.4 and current_position_all != 0:
            if spread < intercept_std * 0.1 and spread > intercept_std * (-0.1) and position_basket != 0:
                if position_basket > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_bid, -min(position_basket, basket_bid_vol)))
                    # basket_orders.append(Order('GIFT_BASKET', basket_bid_2, - max(min(position_basket-basket_bid, basket_bid_vol_2), 0)))
                    # basket_orders.append(Order('GIFT_BASKET', basket_bid_3, - max(min(abs(-position_basket)-basket_bid_vol-basket_bid_vol_2, basket_bid_vol_3), 0)))
                elif position_basket < 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_ask, min(-position_basket, -basket_ask_vol)))
                    # basket_orders.append(Order('GIFT_BASKET', basket_ask_2, max(min(-position_basket+basket_ask_vol, -basket_ask_vol_2), 0)))
                    # basket_orders.append(Order('GIFT_BASKET', basket_ask_3, max(min(-position_basket+basket_ask_vol+basket_ask_vol_2, -basket_ask_vol_3), 0)))

            # if z_score > 2: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
            if spread > intercept_std * 1.2:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                qt_basket = min(abs(sell_volume_basket), basket_bid_vol)
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol))
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol))
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol))

                # n_tradable = min(qt_basket, math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries), math.floor(qt_roses/beta_roses ))
                # print(n_tradable)
                n_tradable = 1
                if n_tradable > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_bid, sell_volume_basket))

            # elif z_score < -2: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            if spread < intercept_std * (-1.2):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol))
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol)
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol)
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol)

                # n_tradable = min(qt_basket, math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries), math.floor(qt_roses / beta_roses ))
                # print(n_tradable)
                n_tradable = 1
                if n_tradable > 0:
                    basket_orders.append(Order('GIFT_BASKET', basket_ask, buy_volume_basket))

        if len(self.mid_prices_history['ROSES']) > 1:
            # intercept, intercept_std, betas, betas_std = self.linear_regression(gift_basket_prices, chocolates_prices, strawberries_prices, roses_prices)
            intercept = 24820
            intercept_std = 163.617
            beta_roses = 3.1629

            # intercept = 32260
            # intercept_std = 86.827
            # beta_roses = 2.6492

            nav = beta_roses * roses_mid_price + intercept
            # print('regression!')
            # print(beta_chocolates)
            # print(beta_strawberries)
            # print(beta_roses)
            # print(intercept)
            spread = basket_mid_price - nav
            # print('spread', spread)
            self.spreads_basket.append(spread)
            mean = statistics.mean(self.spreads_basket)
            std = statistics.stdev(self.spreads_basket)

            z_score = (spread - mean) / std

            # print('Z-SCORE: ',z_score)

            # print(intercept_std*1.2)

            # if z_score > -0.4 and z_score < 0.4 and current_position_all != 0:
            if spread < intercept_std * 0.1 and spread > intercept_std * (-0.1) and position_roses != 0:

                if position_roses > 0:
                    roses_orders.append(Order('ROSES', roses_bid, -max(min(position_roses, roses_bid_vol), 0)))
                    # roses_orders.append(Order('ROSES', roses_bid_2, -max(min(position_roses-roses_bid_vol, roses_bid_vol_2),0)))
                    # roses_orders.append(Order('ROSES', roses_bid_3, -max(min(position_roses-roses_bid_vol-roses_bid_vol_2, roses_bid_vol_3),0)))
                elif position_roses < 0:
                    roses_orders.append(Order('ROSES', roses_ask, max(min(-position_roses, -roses_ask_vol), 0)))
                    # roses_orders.append(Order('ROSES', roses_ask_2, max(min(-position_roses+roses_ask_vol, -roses_ask_vol_2),0)))
                    # roses_orders.append(Order('ROSES', roses_ask_3, max(min(-position_roses+roses_ask_vol+roses_ask_vol_2, -roses_ask_vol_3),0)))

            # if z_score > 2: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
            if spread > intercept_std * 1.2:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                qt_basket = min(abs(sell_volume_basket), basket_bid_vol)
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol))
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol))
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol))

                # n_tradable = min(qt_basket, math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries), math.floor(qt_roses/beta_roses ))
                n_tradable = 1

                if n_tradable > 0:
                    # basket_orders.append(Order('GIFT_BASKET', basket_bid, sell_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_bid, - min(abs(n_tradable), basket_bid_vol)))
                    # basket_orders.append(Order('GIFT_BASKET', basket_bid_2, - max(min(abs(n_tradable)-basket_bid, basket_bid_vol_2), 0)))
                    # basket_orders.append(Order('GIFT_BASKET', basket_bid_3, - max(min(abs(n_tradable)-basket_bid_vol-basket_bid_vol_2, basket_bid_vol_3), 0)))

                    roses_orders.append(Order('ROSES', roses_ask, buy_volume_roses))

            # elif z_score < -2: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            if spread < intercept_std * (-1.2):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol))
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol)
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol)
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol)

                # n_tradable = min(qt_basket, math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries), math.floor(qt_roses / beta_roses ))
                n_tradable = 1
                print(n_tradable)
                if n_tradable > 0:
                    # basket_orders.append(Order('GIFT_BASKET', basket_ask, buy_volume_basket))
                    ##basket_orders.append(Order('GIFT_BASKET', basket_ask, min(n_tradable, -basket_ask_vol)))
                    # basket_orders.append(Order('GIFT_BASKET', basket_ask_2, max(min(n_tradable+basket_ask_vol, -basket_ask_vol_2), 0)))
                    # basket_orders.append(Order('GIFT_BASKET', basket_ask_3, max(min(n_tradable+basket_ask_vol+basket_ask_vol_2, -basket_ask_vol_3), 0)))

                    roses_orders.append(Order('ROSES', roses_bid, sell_volume_roses))

        if len(self.mid_prices_history['STRAWBERRIES']) > 1:
            # rolling_spreads = self.spreads_chocolates[-10:]
            # rolling_spread = statistics.mean(rolling_spreads)
            # print('regression!')
            # print(beta_chocolates)
            # print(beta_strawberries)
            # print(beta_roses)
            # print(intercept)
            # mean = statistics.mean(self.spreads_chocolates)
            # std = statistics.stdev(self.spreads_chocolates)

            # z_score = (rolling_spread - mean) / std

            # print('Z-SCORE: ',z_score)

            # print(intercept_std*1.2)

            intercept = 26800
            intercept_std = 441.474
            # beta_chocolates = 5.2957
            beta_strawberries = 10.9044

            # intercept= -433.5429
            # intercept_std = 267.348
            # beta_strawberries = 17.6479

            nav = beta_strawberries * strawberries_mid_price + intercept
            spread = basket_mid_price - nav

            # if z_score > -0.4 and z_score < 0.4 and current_position_all != 0:
            if spread < intercept_std * 0.1 and spread > intercept_std * (-0.1) and (position_strawberries != 0):

                if position_strawberries > 0:
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid,
                                                     -max(min(position_strawberries, strawberries_bid_vol), 0)))
                    # strawberries_orders.append(Order('strawberries', strawberries_bid_2, -max(min(position_strawberries-strawberries_bid_vol, strawberries_bid_vol_2),0)))
                    # strawberries_orders.append(Order('strawberries', strawberries_bid_3, -max(min(position_strawberries-strawberries_bid_vol-strawberries_bid_vol_2, strawberries_bid_vol_3),0)))
                elif position_strawberries < 0:
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask,
                                                     max(min(-position_strawberries, -strawberries_ask_vol), 0)))
                    # strawberries_orders.append(Order('strawberries', strawberries_ask_2, max(min(-position_strawberries+strawberries_ask_vol, -strawberries_ask_vol_2),0)))
                    # strawberries_orders.append(Order('strawberries', strawberries_ask_3, max(min(-position_strawberries+strawberries_ask_vol+strawberries_ask_vol_2, -strawberries_ask_vol_3),0)))

            # if z_score > 1: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
            if spread > intercept_std * 1.2:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                # qt_basket = min(abs(sell_volume_basket), basket_bid_vol)
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol))
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol))
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol))

                # n_tradable = min(math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries))
                n_tradable = 1

                if n_tradable > 0:
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_ask, buy_volume_strawberries))


            # elif z_score < -1: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            elif spread < intercept_std * (-1.2):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol))
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol)
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol)
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol)

                # n_tradable = min(math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries))
                n_tradable = 1
                if n_tradable > 0:
                    strawberries_orders.append(Order('STRAWBERRIES', strawberries_bid, sell_volume_strawberries))

        if len(self.spreads_chocolates) > 1:

            intercept = 3878.8146
            intercept_std = 41.821
            # beta_chocolates = 5.2957
            beta_chocolates = 1.3427

            # intercept = -1400.8233
            # intercept_std = 74.048
            # beta_chocolates = 5.2957
            # beta_chocolates = 2.0010

            nav = beta_chocolates * chocolates_mid_price + intercept
            spread = roses_mid_price - nav

            # if z_score > -0.4 and z_score < 0.4 and current_position_all != 0:
            if spread < intercept_std * 0.1 and spread > intercept_std * (-0.1) and (position_chocolates != 0):

                if position_chocolates > 0:
                    chocolates_orders.append(
                        Order('CHOCOLATE', chocolates_bid, -max(min(position_chocolates, chocolates_bid_vol), 0)))
                    # chocolates_orders.append(Order('chocolates', chocolates_bid_2, -max(min(position_chocolates-chocolates_bid_vol, chocolates_bid_vol_2),0)))
                    # chocolates_orders.append(Order('chocolates', chocolates_bid_3, -max(min(position_chocolates-chocolates_bid_vol-chocolates_bid_vol_2, chocolates_bid_vol_3),0)))
                elif position_chocolates < 0:
                    chocolates_orders.append(
                        Order('CHOCOLATE', chocolates_ask, max(min(-position_chocolates, -chocolates_ask_vol), 0)))
                    # chocolates_orders.append(Order('chocolates', chocolates_ask_2, max(min(-position_chocolates+chocolates_ask_vol, -chocolates_ask_vol_2),0)))
                    # chocolates_orders.append(Order('chocolates', chocolates_ask_3, max(min(-position_chocolates+chocolates_ask_vol+chocolates_ask_vol_2, -chocolates_ask_vol_3),0)))

            # if z_score > 1: #BASKET OVER VALUED, SELL BASKET AND BUY INDIVIDUAL GOODS
            if spread > intercept_std * 1.2:
                #### SELLING BASKET AND BUYING INDIVIDUAL GOODS ####
                # qt_basket = min(abs(sell_volume_basket), basket_bid_vol)
                qt_chocolates = min(buy_volume_chocolates, abs(chocolates_ask_vol))
                qt_strawberries = min(buy_volume_strawberries, abs(strawberries_ask_vol))
                qt_roses = min(buy_volume_roses, abs(roses_ask_vol))

                # n_tradable = min(math.floor(qt_chocolates/beta_chocolates), math.floor(qt_strawberries/beta_strawberries))
                n_tradable = 1

                if n_tradable > 0:
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_ask, buy_volume_chocolates))


            # elif z_score < -1: #BASKET UNDER VALUED, BUY BASKET AND SELL INDIVIDUAL GOODS
            elif spread < intercept_std * (-1.2):
                #### BUYING BASKET AND SELLING INDIVIDUAL GOODS ####
                qt_basket = min(buy_volume_basket, abs(basket_ask_vol))
                qt_chocolates = min(abs(sell_volume_chocolates), chocolates_bid_vol)
                qt_strawberries = min(abs(sell_volume_strawberries), strawberries_bid_vol)
                qt_roses = min(abs(sell_volume_roses), roses_bid_vol)

                # n_tradable = min(math.floor(qt_chocolates / beta_chocolates), math.floor(qt_strawberries / beta_strawberries))
                n_tradable = 1
                if n_tradable > 0:
                    chocolates_orders.append(Order('CHOCOLATE', chocolates_bid, sell_volume_chocolates))

        while len(self.spreads_basket) > 100:
            self.spreads_basket.pop(0)
        while len(self.spreads_roses) > 100:
            self.spreads_roses.pop(0)
        while len(self.spreads_chocolates) > 100:
            self.spreads_chocolates.pop(0)
        while len(self.ratios_basket) > 100:
            self.ratios_basket.pop(0)

        return basket_orders, chocolates_orders, strawberries_orders, roses_orders

    def black_scholes(self, S, K, T, r, sigma, mean):
        def N(x):
            # return sp.stats.norm.cdf(x, mean, std**2)
            return 0.5 * (1 + math.erf((x - mean) / (sigma ** 2 * math.sqrt(2))))

        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * N(d1) - K * np.exp(-r * T) * N(d2)

    def get_z_score(self, state: TradingState):
        basket_items = ['GIFT_BASKET', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES']
        mid_price = {}

        for item in basket_items:
            _, best_sell_price = self.get_volume_and_best_price(state.order_depths[item].sell_orders,
                                                                buy_order=False)
            _, best_buy_price = self.get_volume_and_best_price(state.order_depths[item].buy_orders,
                                                               buy_order=True)

            mid_price[item] = (best_sell_price + best_buy_price) / 2

        mpb = mid_price['GIFT_BASKET']
        diy_basket = 4 * mid_price['CHOCOLATE'] + 6 * mid_price['STRAWBERRIES'] + mid_price['ROSES']

        spread = np.log(mpb) - np.log(diy_basket)

        z_score = spread / 0.001085120118193605

        return z_score

    def star_yhat(self, cache):

        coef = [0.12268419, 0.15664734, 0.23657205, 0.17495758, 0.30940913]
        intercept = -1.4125385825054764
        nxt_price = intercept
        for i, val in enumerate(cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))

    def get_volume_and_best_price(self, orders, buy_order):
        volume = 0
        best = 0 if buy_order else self.INF

        for price, vol in orders.items():
            if buy_order:
                volume += vol
                best = max(best, price)
            else:
                volume -= vol
                best = min(best, price)

        return volume, best

    def compute_orders(self, product, order_depth, our_bid, our_ask, orchids=False, gift_basket=False):
        orders: list[Order] = []

        basket_mult = {'CHOCOLATE': 4, 'ROSES': 1, 'STRAWBERRIES': 6}

        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        buy_vol, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

        print(f'Product: {product} - best sell: {best_sell_price}, best buy: {best_buy_price}')

        position = self.POSITION[product] if not orchids else 0
        limit = self.LIMIT[product]

        mm_buy = best_buy_price + 1
        mm_sell = best_sell_price - 1

        bid_price = min(mm_buy, our_bid)
        ask_price = max(mm_sell, our_ask)

        if orchids:
            ask_price = max(best_sell_price - self.ORCHID_MM_RANGE, our_ask)
        if gift_basket:
            bid_price = our_bid
            ask_price = our_ask

        for ask, vol in sell_orders.items():
            if position < limit and (ask <= our_bid or (position < 0 and ask == our_bid + 1)):
                num_orders = min(-vol, limit - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        if position < limit:
            num_orders = limit - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        position = self.POSITION[product] if not orchids else 0

        for bid, vol in buy_orders.items():
            if position > -limit and (bid >= our_ask or (position > 0 and bid + 1 == our_ask)):
                num_orders = max(-vol, -limit - position)
                position += num_orders
                orders.append(Order(product, bid, num_orders))

        if position > -limit:
            num_orders = -limit - position
            orders.append(Order(product, ask_price, num_orders))
            position += num_orders

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': [], 'GIFT_BASKET': [], 'CHOCOLATE': [],
                  'STRAWBERRIES': [], 'ROSES': [], 'COCONUT': [], 'COCONUT_COUPON': []}

        conversions = 0

        self.update_ema_prices(state)

        # PRICE HISTORY
        self.update_prices_history(state.own_trades, state.market_trades)
        self.update_mid_prices_history(state)
        # self.update_diff_history(self.mid_prices_history)
        self.update_diff_history(self.mid_p_diff_history, self.mid_prices_history)
        self.update_diff_history(self.p_diff_history, self.prices_history)
        # print(self.prices_history)


        for product in self.pnl_tracker.keys():
            while len(self.pnl_tracker[product]) > 10:
                self.pnl_tracker[product].pop(0)
            while len(self.forecasted_diff_history[product]) > 10:
                self.forecasted_diff_history[product].pop(0)
            while len(self.errors_history[product]) > 10:
                self.errors_history[product].pop(0)
            while len(self.last_variances) > 10:
                self.last_variances.pop(0)
            while len(self.coconuts_returns) > 10:
                self.coconuts_returns.pop(0)

        # BASKET STRATEGY
        try:
            result['GIFT_BASKET'], result['CHOCOLATE'], result['STRAWBERRIES'], result['ROSES'] = self.basket_strategy(
                state)

        except Exception as e:
            print(e)

        for product in state.order_depths:
            self.POSITION[product] = state.position[product] if product in state.position else 0

        try:
            order_depth: OrderDepth = state.order_depths['AMETHYSTS']
            result['AMETHYSTS'] = self.compute_orders('AMETHYSTS', order_depth, 10000 - self.AME_RANGE, 10000 + self.AME_RANGE)

        except Exception as e:
            print(e)

        try:
            order_depth: OrderDepth = state.order_depths['AMETHYSTS']
            if len(self.starfruit_cache) == self.STARFRUIT_CACHE_SIZE:
                self.starfruit_cache.pop(0)

            _, best_sell_price = self.get_volume_and_best_price(order_depth.sell_orders, buy_order=False)
            _, best_buy_price = self.get_volume_and_best_price(order_depth.buy_orders, buy_order=True)

            self.starfruit_cache.append((best_sell_price + best_buy_price) / 2)

            lower_bound = -self.INF
            upper_bound = self.INF

            if len(self.starfruit_cache) == self.STARFRUIT_CACHE_SIZE:
                lower_bound = self.star_yhat(self.starfruit_cache) - 2
                upper_bound = self.star_yhat(self.starfruit_cache) + 2

            result['STARFRUIT'] = self.compute_orders('STARFRUIT', order_depth, lower_bound, upper_bound)
        except Exception as e:
            print(e)

        try:
            order_depth: OrderDepth = state.order_depths['ORCHIDS']
            shipping_cost = state.observations.conversionObservations['ORCHIDS'].transportFees
            import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
            export_tariff = state.observations.conversionObservations['ORCHIDS'].exportTariff
            ducks_ask = state.observations.conversionObservations['ORCHIDS'].askPrice
            ducks_bid = state.observations.conversionObservations['ORCHIDS'].bidPrice

            buy_from_ducks_prices = ducks_ask + shipping_cost + import_tariff
            sell_to_ducks_prices = ducks_bid + shipping_cost + export_tariff

            lower_bound = int(round(buy_from_ducks_prices)) - 1
            upper_bound = int(round(buy_from_ducks_prices)) + 1

            result['ORCHIDS'] = self.compute_orders('ORCHIDS', order_depth, lower_bound, upper_bound, orchids=True)
            conversions = -self.POSITION['ORCHIDS']

            print(f'buying from ducks for: {buy_from_ducks_prices}')
            print(f'selling to ducks for: {sell_to_ducks_prices}')


        except Exception as e:

            print(e)

        try:
            order_depth: OrderDepth = state.order_depths['COCONUT_COUPON']
            items = ['COCONUT', 'COCONUT_COUPON']
            mid_price, best_bid_price, best_ask_price = {}, {}, {}

            for item in items:
                _, best_sell_price = self.get_volume_and_best_price(state.order_depths[item].sell_orders,
                                                                    buy_order=False)
                _, best_buy_price = self.get_volume_and_best_price(state.order_depths[item].buy_orders,
                                                                   buy_order=True)

                mid_price[item] = (best_sell_price + best_buy_price) / 2
                best_bid_price[item] = best_buy_price
                best_ask_price[item] = best_sell_price

            self.COCONUT_SUM += mid_price['COCONUT']
            self.COUPON_SUM += mid_price['COCONUT_COUPON']
            self.ITERS += 1
            coconut_mean = self.COCONUT_SUM / self.ITERS
            coupon_mean = self.COUPON_SUM / self.ITERS

            self.COCONUT_STORE.append(mid_price['COCONUT'])

            store = np.array(self.COCONUT_STORE)
            mean, std = np.mean(store), np.std(store)
            curr_bs_est = self.black_scholes(S=mid_price['COCONUT'], K=10_000, T=250, r=0.00026,
                                             sigma=88.75266514702373, mean=mean)

            # bs_mean = np.mean(data.COCONUT_BS_STORE) if len(data.COCONUT_BS_STORE) > 0 else 0

            # modified_bs = (curr_bs_est - bs_mean) * 3.229 + curr_bs_est

            self.COCONUT_BS_STORE.append(curr_bs_est)

            bs_std = np.std(self.COCONUT_BS_STORE)

            if len(self.COCONUT_STORE) >= self.COCONUT_STORE_SIZE:
                coco_price = mid_price['COCONUT']
                print(f'coconut price: {coco_price}, mean: {mean}, std: {std}')
                print(f'predicted coupon price: {curr_bs_est}')
                difference = mid_price['COCONUT_COUPON'] - curr_bs_est

                if difference > self.COUPON_Z_SCORE * bs_std:
                    # coupons overvalued, sell
                    result['COCONUT_COUPON'] = self.compute_orders('COCONUT_COUPON', order_depth, -self.INF,
                                                          best_bid_price['COCONUT_COUPON'])

                elif difference < -self.COUPON_Z_SCORE * bs_std:
                    # coupons undervalued, buy
                    result['COCONUT_COUPON'] = self.compute_orders('COCONUT_COUPON', order_depth, best_ask_price['COCONUT_COUPON'],
                                                          self.INF)

                self.COCONUT_BS_STORE.pop(0)
                self.COCONUT_STORE.pop(0)

            self.PREV_COCONUT_PRICE = mid_price['COCONUT']
            self.PREV_COUPON_PRICE = mid_price['COCONUT_COUPON']


        except Exception as e:
            print(e)


            print(f'placed orders: {result}')

        traderData = "SAMPLE"

        return result, conversions, traderData
