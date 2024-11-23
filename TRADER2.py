import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, OrderedDict
import math
import numpy as np
import statistics

class Trader:

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

    def compute_orders(self, product, order_depth, our_bid, our_ask, orchids=False, gift_basket=False, gb_item=False):
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
        if gb_item:
            limit = self.gift_basket_quantity * basket_mult[product]

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

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0

        for product in state.order_depths:
            self.POSITION[product] = state.position[product] if product in state.position else 0

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            if product == "AMETHYSTS":
                orders += self.compute_orders(product, order_depth, 10000 - self.AME_RANGE, 10000 + self.AME_RANGE)

            elif product == "STARFRUIT":
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

                orders += self.compute_orders(product, order_depth, lower_bound, upper_bound)

            elif product == 'ORCHIDS':
                shipping_cost = state.observations.conversionObservations['ORCHIDS'].transportFees
                import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
                export_tariff = state.observations.conversionObservations['ORCHIDS'].exportTariff
                ducks_ask = state.observations.conversionObservations['ORCHIDS'].askPrice
                ducks_bid = state.observations.conversionObservations['ORCHIDS'].bidPrice

                buy_from_ducks_prices = ducks_ask + shipping_cost + import_tariff
                sell_to_ducks_prices = ducks_bid + shipping_cost + export_tariff

                lower_bound = int(round(buy_from_ducks_prices)) - 1
                upper_bound = int(round(buy_from_ducks_prices)) + 1

                orders += self.compute_orders(product, order_depth, lower_bound, upper_bound, orchids=True)
                conversions = -self.POSITION[product]

                print(f'buying from ducks for: {buy_from_ducks_prices}')
                print(f'selling to ducks for: {sell_to_ducks_prices}')

            elif product == 'GIFT_BASKET':

                z_score = self.get_z_score(state)

                worst_bid_price = min(order_depth.buy_orders.keys())
                worst_ask_price = max(order_depth.sell_orders.keys())
                print(f'z_score = {z_score}')
                if z_score > 3:
                    orders += self.compute_orders(product, order_depth, -self.INF, worst_bid_price, gift_basket=True)

                elif z_score < -3:
                    orders += self.compute_orders(product, order_depth, worst_ask_price, self.INF, gift_basket=True)

                else:
                    continue

            elif product == 'ROSES':
                worst_bid_rose = min(state.order_depths['ROSES'].buy_orders.keys())
                worst_ask_rose = max(state.order_depths['ROSES'].sell_orders.keys())
                z_score = self.get_z_score(state)
                if z_score > 3:
                    orders += self.compute_orders('ROSES', order_depth, worst_ask_rose, self.INF, gift_basket=True, gb_item=True)
                elif z_score < -3:
                    orders += self.compute_orders('ROSES', order_depth, -self.INF, worst_bid_rose, gift_basket=True, gb_item=True)
                else:
                    continue

            elif product == 'STRAWBERRIES':
                worst_bid_strawb = min(state.order_depths['STRAWBERRIES'].buy_orders.keys())
                worst_ask_strawb = max(state.order_depths['STRAWBERRIES'].sell_orders.keys())
                z_score = self.get_z_score(state)
                if z_score > 3:
                    orders += self.compute_orders('STRAWBERRIES', order_depth, worst_ask_strawb, self.INF, gift_basket=True, gb_item=True)
                elif z_score < -3:
                    orders += self.compute_orders('STRAWBERRIES', order_depth, -self.INF, worst_bid_strawb, gift_basket=True, gb_item=True)
                else:
                    continue

            elif product == 'CHOCOLATE':
                worst_bid_choc = min(state.order_depths['CHOCOLATE'].buy_orders.keys())
                worst_ask_choc = max(state.order_depths['CHOCOLATE'].sell_orders.keys())
                z_score = self.get_z_score(state)
                if z_score > 3:
                    orders += self.compute_orders('CHOCOLATE', order_depth, worst_ask_choc, self.INF, gift_basket=True, gb_item=True)
                elif z_score < -3:
                    orders += self.compute_orders('CHOCOLATE', order_depth, -self.INF, worst_bid_choc, gift_basket=True, gb_item=True)
                else:
                    continue

            print(f'placed orders: {orders}')
            result[product] = orders

        return result, conversions, 'SAMPLE'
