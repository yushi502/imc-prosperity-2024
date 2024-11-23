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

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if (buy == 0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask

        return tot_vol, best_val

    def compute_orders_am(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product] < 0) and (ask == acc_bid))) and cpos < self.pos_lim[
                    'AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.pos_lim['AMETHYSTS'] - cpos)
                cpos += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr) / 2
        mprice_ours = (acc_bid + acc_ask) / 2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid - 1)  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 1)

        if (cpos < self.pos_lim['AMETHYSTS']) and (self.position[product] < 0):
            num = min(40, self.pos_lim['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid - 1), num))
            cpos += num

        if (cpos < self.pos_lim['AMETHYSTS']) and (self.position[product] > 15):
            num = min(40, self.pos_lim['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 1), num))
            cpos += num

        if cpos < self.pos_lim['AMETHYSTS']:
            num = min(40, self.pos_lim['AMETHYSTS'] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product] > 0) and (bid == acc_ask))) and cpos > -self.pos_lim[
                    'AMETHYSTS']:
                order_for = max(-vol, -self.pos_lim['AMETHYSTS'] - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.pos_lim['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-40, -self.pos_lim['AMETHYSTS'] - cpos)
            orders.append(Order(product, max(undercut_sell - 1, acc_ask + 1), num))
            cpos += num

        if (cpos > -self.pos_lim['AMETHYSTS']) and (self.position[product] < -15):
            num = max(-40, -self.pos_lim['AMETHYSTS'] - cpos)
            orders.append(Order(product, max(undercut_sell + 1, acc_ask + 1), num))
            cpos += num

        if cpos > -self.pos_lim['AMETHYSTS']:
            num = max(-40, -self.pos_lim['AMETHYSTS'] - cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def star_yhat(self, state: TradingState):

        star_coeff = [4.39205423, 0.33042703, 0.22349804, 0.25166263, 0.19351935]

        lowest_sell_price = sorted(state.order_depths["STARFRUIT"].sell_orders.keys())[0]
        highest_buy_price = sorted(state.order_depths["STARFRUIT"].buy_orders.keys(), reverse=True)[0]

        mid_price = (lowest_sell_price + highest_buy_price) / 2

        self.star_lag.append(mid_price)

        if len(self.star_lag) > 5:
            self.star_lag.pop(0)

        if len(self.star_lag) < 5:
            return None
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

                orders = self.compute_orders_am(product, order_depth, acc_bid, acc_ask)

                result[product] = orders

            if product == 'STARFRUIT':
                acc_bid = self.star_yhat(state)
                acc_ask = self.star_yhat(state)

                orders = self.compute_orders(product, order_depth, acc_bid, acc_ask)

                result[product] = orders

        traderData = "SAMPLE"
        conversions = 1
        return result, conversions, traderData
