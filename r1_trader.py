import jsonpickle
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Any, List, Dict
import numpy as np
import math
import collections


class Trader:

    position = {'AMETHYSTS': 0, 'STARFRUIT': 0}
    pos_lim = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    star_lag = []

    def star_yhat(self, state: TradingState):

        star_coeff = [4.39205423, 0.33042703, 0.22349804, 0.25166263, 0.19351935]

        lowest_sell_price = sorted(state.order_depths["STARFRUIT"].sell_orders.keys())[0]
        highest_buy_price = sorted(state.order_depths["STARFRUIT"].buy_orders.keys(), reverse=True)[0]

        mid_price = (lowest_sell_price + highest_buy_price) / 2

        self.star_lag.append(mid_price)

        if len(self.star_lag) > 4:
            self.star_lag.pop(0)

        if len(self.star_lag) < 4:
            return 'None'
        else:
            expected_price = star_coeff[0] + sum([star_coeff[i + 1] * self.star_lag[i] for i in range(4)])

        return expected_price

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):
        orders: List[Order] = []

        sell_ord = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_ord = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        cur_pos = self.position[product]

        for ask, vol in sell_ord.items():
            if (ask < acc_bid or (cur_pos < 0 and ask == acc_bid)) and cur_pos < 20:
                order_vol = min(-vol, 20 - cur_pos)
                cur_pos += order_vol
                orders.append(Order(product, ask, order_vol))

        for bid, vol in buy_ord.items():
            if (bid > acc_ask or (cur_pos > 0 and bid == acc_ask)) and cur_pos > -20:
                order_vol = max(-vol, -20 - cur_pos)
                cur_pos += order_vol
                orders.append(Order(product, bid, order_vol))

        self.position[product] = cur_pos

        return orders

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            if product == 'AMETHYSTS':
                acc_bid, acc_ask = 10000, 10000

                orders = self.compute_orders(product, order_depth, acc_bid, acc_ask)

                result[product] = orders

            if product == 'STARFRUIT':
                acc_bid = self.star_yhat(state)
                acc_ask = self.star_yhat(state)

                orders = self.compute_orders(product, order_depth, acc_bid, acc_ask)

                result[product] = orders

        traderData = "SAMPLE"
        conversions = 1
        return result, conversions, traderData
