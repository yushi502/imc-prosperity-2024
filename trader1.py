from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
from collections import OrderedDict
import json
import statistics

import numpy as np


class Trader:
    POSITION_LIMITS = {
        "AMETHYSTS": 20,
        "STARFRUIT": 20,
        "ORCHIDS": 100,
        "CHOCOLATE": 250,
        "STRAWBERRIES": 350,
        "ROSES": 60,
        "GIFT_BASKET": 60,
        "COCONUT": 300,
        "COCONUT_COUPON": 600
    }

    positions = {
        "AMETHYSTS": 0,
        "STARFRUIT": 0,
        "ORCHIDS": 0,
        "CHOCOLATE": 0,
        "STRAWBERRIES": 0,
        "ROSES": 0,
        "GIFT_BASKET": 0,
        "COCONUT": 0,
        "COCONUT_COUPON": 0
    }

    # Linear regression parameters trained on data from days -2, -1 and 0
    starfruit_coef = [0.19276398, 0.22111366, 0.24350053, 0.34038018]
    starfruit_intercept = 11.302935408693884

    # Cache latest 4 midprice of best_ask and best_bid every iteration
    starfruit_cache = []
    starfruit_spread_cache = []

    gift_basket_std = 75

    etf_returns = []
    assets_returns = []
    chocolate_returns = []
    chocolate_estimated_returns = []
    strawberries_returns = []
    strawberries_estimated_returns = []
    roses_returns = []
    roses_estimated_returns = []

    coconut_coupon_returns = []
    coconut_coupon_bsm_returns = []
    coconut_returns = []
    coconut_estimated_returns = []

    rhianna_buy = False
    rhianna_trade_before = False

    ORCHID_MM_RANGE = 5

    INF = 10^8

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

    N = statistics.NormalDist(mu=0, sigma=1)

    def BS_CALL(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * self.N.cdf(d1) - K * np.exp(-r * T) * self.N.cdf(d2)

    def compute_vwap(self, order_depth):
        total_ask, total_bid = 0, 0
        ask_vol, bid_vol = 0, 0

        for ask, vol in order_depth.sell_orders.items():
            total_ask += ask * abs(vol)
            ask_vol += abs(vol)

        for bid, vol in order_depth.buy_orders.items():
            total_bid += bid * vol
            bid_vol += vol

        vwap_ask = total_ask / ask_vol
        vwap_bid = total_bid / bid_vol

        return (vwap_ask + vwap_bid) / 2

    def compute_starfruit_price(self):
        price = self.starfruit_intercept

        for idx, cached_price in enumerate(self.starfruit_cache):
            price += self.starfruit_coef[idx] * cached_price

        return int(round(price))

    def compute_orders(self, product, order_depth, our_bid, our_ask, orchids=False):
        orders: list[Order] = []

        basket_mult = {'CHOCOLATE': 4, 'ROSES': 1, 'STRAWBERRIES': 6}

        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        buy_vol, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

        print(f'Product: {product} - best sell: {best_sell_price}, best buy: {best_buy_price}')

        position = self.positions[product] if not orchids else 0
        limit = self.POSITION_LIMITS[product]

        mm_buy = best_buy_price + 1
        mm_sell = best_sell_price - 1

        bid_price = min(mm_buy, our_bid)
        ask_price = max(mm_sell, our_ask)

        if orchids:
            ask_price = max(best_sell_price - self.ORCHID_MM_RANGE, our_ask)

        for ask, vol in sell_orders.items():
            if position < limit and (ask <= our_bid or (position < 0 and ask == our_bid + 1)):
                num_orders = min(-vol, limit - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        if position < limit:
            num_orders = limit - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        position = self.positions[product] if not orchids else 0

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

    def compute_basket_orders3(self, state: TradingState):
        products = ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]
        positions, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": [], "GIFT_BASKET": []}

        for product in products:
            positions[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        estimated_price = 4.0 * prices["CHOCOLATE"] + 6.0 * prices["STRAWBERRIES"] + prices["ROSES"]

        price_diff = prices["GIFT_BASKET"] - estimated_price

        self.etf_returns.append(prices["GIFT_BASKET"])
        # Using assets cache to store price difference instead of total component value
        self.assets_returns.append(price_diff)

        if len(self.etf_returns) < 100 or len(self.assets_returns) < 100:
            return orders

        # Slow moving average
        assets_rolling_mean = statistics.fmean(self.assets_returns[-200:])
        # Fast moving average
        assets_rolling_mean_fast = statistics.fmean(self.assets_returns[-100:])

        # Empirically tuned to avoid noisy buy and sell signals - do nothing if sideways market
        if assets_rolling_mean_fast > assets_rolling_mean + 4:

            # Fixed entry every timestep that criteria is met, max-ing out early
            limit_mult = 3

            limit_mult = min(limit_mult, self.POSITION_LIMITS["GIFT_BASKET"] - positions["GIFT_BASKET"],
                             self.POSITION_LIMITS["GIFT_BASKET"])

            print("GIFT_BASKET positions:", positions["GIFT_BASKET"])
            print("BUY", "GIFT_BASKET", str(limit_mult) + "x", best_asks["GIFT_BASKET"])
            orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_asks["GIFT_BASKET"], limit_mult))

        elif assets_rolling_mean_fast < assets_rolling_mean - 4:

            # Fixed entry every timestep, max-ing out early
            limit_mult = -3

            limit_mult = max(limit_mult, -self.POSITION_LIMITS["GIFT_BASKET"] - positions["GIFT_BASKET"],
                             -self.POSITION_LIMITS["GIFT_BASKET"])

            print("GIFT_BASKET positions:", positions["GIFT_BASKET"])
            print("SELL", "GIFT_BASKET", str(limit_mult) + "x", best_bids["GIFT_BASKET"])
            orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_bids["GIFT_BASKET"], limit_mult))

        return orders

    def compute_coconut_coupon_orders(self, state: TradingState):
        products = ["COCONUT_COUPON", "COCONUT"]
        positions, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "COCONUT_COUPON": [], "COCONUT": []}

        for product in products:
            positions[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        # Use BSM
        S = prices["COCONUT"]
        K = 10000
        T = 250
        r = 0
        sigma = 0.01011932923
        bsm_price = self.BS_CALL(S, K, T, r, sigma)

        self.coconut_coupon_returns.append(prices["COCONUT_COUPON"])
        self.coconut_coupon_bsm_returns.append(bsm_price)

        # Dummy for now
        self.coconut_returns.append(prices["COCONUT"])
        self.coconut_estimated_returns.append(prices["COCONUT"])

        if len(self.coconut_coupon_returns) < 2 or len(self.coconut_coupon_bsm_returns) < 2:
            return orders

        coconut_coupon_rolling_mean = statistics.fmean(self.coconut_coupon_returns[-200:])
        coconut_coupon_rolling_std = statistics.stdev(self.coconut_coupon_returns[-200:])

        coconut_coupon_bsm_rolling_mean = statistics.fmean(self.coconut_coupon_bsm_returns[-200:])
        coconut_coupon_bsm_rolling_std = statistics.stdev(self.coconut_coupon_bsm_returns[-200:])

        if coconut_coupon_rolling_std != 0:
            coconut_coupon_z_score = (self.coconut_coupon_returns[
                                          -1] - coconut_coupon_rolling_mean) / coconut_coupon_rolling_std
        else:
            coconut_coupon_z_score = 0

        if coconut_coupon_bsm_rolling_std != 0:
            coconut_coupon_bsm_z_score = (self.coconut_coupon_bsm_returns[
                                              -1] - coconut_coupon_bsm_rolling_mean) / coconut_coupon_bsm_rolling_std
        else:
            coconut_coupon_bsm_z_score = 0

        # May need a catch here to set both == 0 if one or the other is 0, to avoid errorneous z scores

        coconut_coupon_z_score_diff = coconut_coupon_z_score - coconut_coupon_bsm_z_score

        # Option is underpriced
        if coconut_coupon_z_score_diff < -1.2:
            coconut_coupon_best_ask_vol = sell_orders["COCONUT_COUPON"][best_asks["COCONUT_COUPON"]]

            limit_mult = -coconut_coupon_best_ask_vol

            limit_mult = round(limit_mult * abs(coconut_coupon_z_score_diff) / 2)

            limit_mult = min(limit_mult, self.POSITION_LIMITS["COCONUT_COUPON"] - positions["COCONUT_COUPON"],
                             self.POSITION_LIMITS["COCONUT_COUPON"])

            print("COCONUT_COUPON positions:", positions["COCONUT_COUPON"])
            print("BUY", "COCONUT_COUPON", str(limit_mult) + "x", best_asks["COCONUT_COUPON"])
            orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_asks["COCONUT_COUPON"], limit_mult))

        # Option is overpriced
        elif coconut_coupon_z_score_diff > 1.2:
            coconut_coupon_best_bid_vol = buy_orders["COCONUT_COUPON"][best_bids["COCONUT_COUPON"]]

            limit_mult = coconut_coupon_best_bid_vol

            limit_mult = round(-limit_mult * abs(coconut_coupon_z_score_diff) / 2)

            limit_mult = max(limit_mult, -self.POSITION_LIMITS["COCONUT_COUPON"] - positions["COCONUT_COUPON"],
                             -self.POSITION_LIMITS["COCONUT_COUPON"])

            print("COCONUT_COUPON positions:", positions["COCONUT_COUPON"])
            print("SELL", "COCONUT_COUPON", str(limit_mult) + "x", best_bids["COCONUT_COUPON"])
            orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_bids["COCONUT_COUPON"], limit_mult))

        return orders

    def compute_roses_orders(self, state: TradingState):
        orders = []

        roses_pos = state.position["ROSES"] if "ROSES" in state.position else 0
        best_bid = max(state.order_depths["ROSES"].buy_orders.keys())
        bid_vol = state.order_depths["ROSES"].buy_orders[best_bid]
        best_ask = min(state.order_depths["ROSES"].sell_orders.keys())
        ask_vol = state.order_depths["ROSES"].sell_orders[best_ask]

        if "ROSES" not in state.market_trades:
            return orders

        for trade in state.market_trades["ROSES"]:
            if trade.buyer == "Rhianna":
                self.rhianna_buy = True
                self.rhianna_trade_before = True
            elif trade.seller == "Rhianna":
                self.rhianna_buy = False
                self.rhianna_trade_before = True

            # Buy signal
            if self.rhianna_buy:
                vol = max(-bid_vol, -self.POSITION_LIMITS["ROSES"] - min(0, roses_pos))
                print("SELL", "ROSES", str(vol) + "x", best_bid)
                orders.append(Order("ROSES", best_bid, vol))
                self.rhianna_buy = False
            # Sell signal
            elif self.rhianna_trade_before:
                vol = min(-ask_vol, self.POSITION_LIMITS["ROSES"] - max(0, roses_pos))
                print("BUY", "ROSES", str(vol) + "x", best_bid)
                orders.append(Order("ROSES", best_ask, vol))
                self.rhianna_buy = True

        return orders

    def compute_chocolate_orders(self, state: TradingState):
        products = ["CHOCOLATE"]
        positions, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "CHOCOLATE": []}

        for product in products:
            positions[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        self.chocolate_returns.append(prices["CHOCOLATE"])

        if len(self.chocolate_returns) < 100:
            return orders

        # Slow moving average
        chocolate_rolling_mean = statistics.fmean(self.chocolate_returns[-200:])
        # Fast moving average
        chocolate_rolling_mean_fast = statistics.fmean(self.chocolate_returns[-100:])

        # Empirically tuned to avoid noisy buy and sell signals - do nothing if sideways market
        if chocolate_rolling_mean_fast > chocolate_rolling_mean + 1.5:

            # Fixed entry every timestep that criteria is met, max-ing out early
            limit_mult = 12

            limit_mult = min(limit_mult, self.POSITION_LIMITS["CHOCOLATE"] - positions["CHOCOLATE"],
                             self.POSITION_LIMITS["CHOCOLATE"])

            print("CHOCOLATE positions:", positions["CHOCOLATE"])
            print("BUY", "CHOCOLATE", str(limit_mult) + "x", best_asks["CHOCOLATE"])
            orders["CHOCOLATE"].append(Order("CHOCOLATE", best_asks["CHOCOLATE"], limit_mult))

        elif chocolate_rolling_mean_fast < chocolate_rolling_mean - 1.5:

            # Fixed entry every timestep, max-ing out early
            limit_mult = -12

            limit_mult = max(limit_mult, -self.POSITION_LIMITS["CHOCOLATE"] - positions["CHOCOLATE"],
                             -self.POSITION_LIMITS["CHOCOLATE"])

            print("CHOCOLATE positions:", positions["CHOCOLATE"])
            print("SELL", "CHOCOLATE", str(limit_mult) + "x", best_bids["CHOCOLATE"])
            orders["CHOCOLATE"].append(Order("CHOCOLATE", best_bids["CHOCOLATE"], limit_mult))

        return orders

    def compute_strawberries_orders(self, state: TradingState):
        products = ["STRAWBERRIES"]
        positions, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "STRAWBERRIES": []}

        for product in products:
            positions[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        self.strawberries_returns.append(prices["STRAWBERRIES"])

        if len(self.strawberries_returns) < 100:
            return orders

        # Slow moving average
        strawberries_rolling_mean = statistics.fmean(self.strawberries_returns[-200:])
        # Fast moving average
        strawberries_rolling_mean_fast = statistics.fmean(self.strawberries_returns[-100:])

        # Empirically tuned to avoid noisy buy and sell signals - do nothing if sideways market
        if strawberries_rolling_mean_fast > strawberries_rolling_mean + 1.5:

            # Fixed entry every timestep that criteria is met, max-ing out early
            limit_mult = 18

            limit_mult = min(limit_mult, self.POSITION_LIMITS["STRAWBERRIES"] - positions["STRAWBERRIES"],
                             self.POSITION_LIMITS["STRAWBERRIES"])

            print("STRAWBERRIES positions:", positions["STRAWBERRIES"])
            print("BUY", "STRAWBERRIES", str(limit_mult) + "x", best_asks["STRAWBERRIES"])
            orders["STRAWBERRIES"].append(Order("STRAWBERRIES", best_asks["STRAWBERRIES"], limit_mult))

        elif strawberries_rolling_mean_fast < strawberries_rolling_mean - 1.5:

            # Fixed entry every timestep, max-ing out early
            limit_mult = -18

            limit_mult = max(limit_mult, -self.POSITION_LIMITS["STRAWBERRIES"] - positions["STRAWBERRIES"],
                             -self.POSITION_LIMITS["STRAWBERRIES"])

            print("STRAWBERRIES positions:", positions["STRAWBERRIES"])
            print("SELL", "STRAWBERRIES", str(limit_mult) + "x", best_bids["STRAWBERRIES"])
            orders["STRAWBERRIES"].append(Order("STRAWBERRIES", best_bids["STRAWBERRIES"], limit_mult))

        return orders

    def compute_coconut_orders(self, state: TradingState):
        products = ["COCONUT"]
        positions, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "COCONUT": []}

        for product in products:
            positions[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        self.coconut_returns.append(prices["COCONUT"])

        if len(self.coconut_returns) < 100:
            return orders

        # Slow moving average
        coconut_rolling_mean = statistics.fmean(self.coconut_returns[-200:])
        # Fast moving average
        coconut_rolling_mean_fast = statistics.fmean(self.coconut_returns[-100:])

        # Empirically tuned to avoid noisy buy and sell signals - do nothing if sideways market
        if coconut_rolling_mean_fast > coconut_rolling_mean + 4:

            # Fixed entry every timestep that criteria is met, max-ing out early
            limit_mult = 30

            limit_mult = min(limit_mult, self.POSITION_LIMITS["COCONUT"] - positions["COCONUT"],
                             self.POSITION_LIMITS["COCONUT"])

            print("COCONUT positions:", positions["COCONUT"])
            print("BUY", "COCONUT", str(limit_mult) + "x", best_asks["COCONUT"])
            orders["COCONUT"].append(Order("COCONUT", best_asks["COCONUT"], limit_mult))

        elif coconut_rolling_mean_fast < coconut_rolling_mean - 4:

            # Fixed entry every timestep, max-ing out early
            limit_mult = -30

            limit_mult = max(limit_mult, -self.POSITION_LIMITS["COCONUT"] - positions["COCONUT"],
                             -self.POSITION_LIMITS["COCONUT"])

            print("COCONUT positions:", positions["COCONUT"])
            print("SELL", "COCONUT", str(limit_mult) + "x", best_bids["COCONUT"])
            orders["COCONUT"].append(Order("COCONUT", best_bids["COCONUT"], limit_mult))

        return orders

    def run(self, state: TradingState):
        # Update positions
        for product, position in state.position.items():
            self.positions[product] = position
        conversions = 0
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            best_market_ask = min(order_depth.sell_orders.keys())
            best_market_bid = max(order_depth.buy_orders.keys())
            conversions = 0
            # market_price = (best_market_ask + best_market_bid) / 2
            market_price = self.compute_vwap(order_depth)

            if product == "AMETHYSTS":
                orders += self.compute_orders(product, order_depth, 10000 - 2, 10000 + 2)

            elif product == "STARFRUIT":
                # Pop oldest value from starfruit_cache if full
                if len(self.starfruit_cache) == 4:
                    self.starfruit_cache.pop(0)

                # Cache STARFRUIT prices
                self.starfruit_cache.append(market_price)
                # print(self.starfruit_cache)

                if len(self.starfruit_spread_cache) == 4:
                    self.starfruit_spread_cache.pop(0)

                # Cache spread of STARFRUIT orders
                self.starfruit_spread_cache.append(best_market_ask - best_market_bid)

                # Estimate price via linear regression
                if len(self.starfruit_cache) == 4:
                    # acceptable_price = self.compute_starfruit_price()
                    # print(acceptable_price)
                    acceptable_price = round(statistics.fmean(self.starfruit_cache[-5:]))
                    spread = round(statistics.fmean(self.starfruit_spread_cache[-5:]))

                    lower_bound = acceptable_price - (spread // 2)
                    upper_bound = acceptable_price + (spread // 2)
                    # lower_bound = acceptable_price - 1
                    # upper_bound = acceptable_price + 1
                else:
                    # spread = int(1e9)
                    lower_bound = -int(1e9)
                    upper_bound = int(1e9)

                cur_position = self.positions[product]

                # Construct buy orders
                for ask, vol in order_depth.sell_orders.items():
                    if ((ask <= lower_bound) or (
                            (self.positions[product] < 0) and (ask <= lower_bound + (spread // 2)))) and cur_position < \
                            self.POSITION_LIMITS[product]:
                        # if ((ask <= lower_bound) or ((self.positions[product] < 0) and (ask == lower_bound + 1))) and cur_position < self.POSITION_LIMITS[product]:
                        order_vol = min(-vol, self.POSITION_LIMITS[product] - cur_position)
                        cur_position += order_vol
                        # print("BUY", product, str(order_vol) + "x", ask)
                        orders.append(Order(product, ask, order_vol))

                undercut_market_ask = best_market_ask - 1
                undercut_market_bid = best_market_bid + 1

                # Spread = 1
                own_ask = max(undercut_market_ask, upper_bound)
                own_bid = min(undercut_market_bid, lower_bound)

                # Market take
                if cur_position < self.POSITION_LIMITS[product]:
                    order_vol = self.POSITION_LIMITS[product] - cur_position
                    cur_position += order_vol
                    # print("BUY", product, str(order_vol) + "x", own_bid)
                    orders.append(Order(product, own_bid, order_vol))

                cur_position = self.positions[product]

                # Construct sell orders
                for bid, vol in order_depth.buy_orders.items():
                    if ((bid >= upper_bound) or ((self.positions[product] > 0) and (
                            bid >= upper_bound - (spread // 2)))) and cur_position > -self.POSITION_LIMITS[product]:
                        # if ((bid >= upper_bound) or ((self.positions[product] > 0) and (bid == upper_bound - 1))) and cur_position > -self.POSITION_LIMITS[product]:
                        order_vol = max(-vol, -self.POSITION_LIMITS[product] - cur_position)
                        cur_position += order_vol
                        # print("SELL", product, str(order_vol) + "x", bid)
                        orders.append(Order(product, bid, order_vol))

                if cur_position > -self.POSITION_LIMITS[product]:
                    order_vol = max(-self.POSITION_LIMITS[product], -self.POSITION_LIMITS[product] - cur_position)
                    cur_position += order_vol
                    # print("SELL", product, str(order_vol) + "x", own_ask)
                    orders.append(Order(product, own_ask, order_vol))

            elif product == 'ORCHIDS':
                try:
                    shipping_cost = state.observations.conversionObservations['ORCHIDS'].transportFees
                    import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
                    export_tariff = state.observations.conversionObservations['ORCHIDS'].exportTariff
                    ducks_ask = state.observations.conversionObservations['ORCHIDS'].askPrice
                    ducks_bid = state.observations.conversionObservations['ORCHIDS'].bidPrice

                    buy_from_ducks_prices = ducks_ask + shipping_cost + import_tariff
                    sell_to_ducks_prices = ducks_bid + shipping_cost + export_tariff

                    lower_bound = int(round(buy_from_ducks_prices)) - 1
                    upper_bound = int(round(buy_from_ducks_prices)) + 1

                    orders += self.compute_orders('ORCHIDS', state.order_depths['ORCHIDS'], lower_bound,
                                                        upper_bound,
                                                        orchids=True)
                    conversions = -self.positions['ORCHIDS']
                except Exception as e:
                    print(f"ORCHIDS ERROR: {e}")

            result[product] = orders

        # 6 strawberry, 4 choc and 1 rose in 1 basket
        # treasure chest 7500 seashells each
        # basket_orders = self.compute_basket_orders(state.order_depths)
        basket_orders = self.compute_basket_orders3(state)

        for product, orders in basket_orders.items():
            result[product] = orders

        coconut_coupon_orders = self.compute_coconut_coupon_orders(state)

        for product, orders in coconut_coupon_orders.items():
            result[product] = orders

        result["ROSES"] = self.compute_roses_orders(state)

        chocolate_orders = self.compute_chocolate_orders(state)

        for product, orders in chocolate_orders.items():
            result[product] = orders

        strawberries_orders = self.compute_strawberries_orders(state)

        for product, orders in strawberries_orders.items():
            result[product] = orders

        coconut_orders = self.compute_coconut_orders(state)

        for product, orders in coconut_orders.items():
            result[product] = orders

        return result, conversions, 'SAMPLE'

