from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import collections
from collections import defaultdict
import string

empty_dict = {
    'AMETHYSTS': 0,
    'STARFRUIT': 0
}

def get_value():
    return copy.deepcopy(empty_dict)

INF = int(1e9)

class Trader:
    pos = copy.deepcopy(empty_dict)
    pos_limit = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    traded_volume = copy.deepcopy(empty_dict)

    trader_position = defaultdict(get_value)

    amethyst_cache = []
    starfruit_cache = []

    def extract_values(self, order_dict, buy=0):
        total_volume = 0
        best_ask_value = -1
        max_volume = -1

        for ask, volume in order_dict.items():
            if buy == 0:
                volume *= -1
            
            total_volume += volume
            if total_volume > max_volume:
                max_volume = volume
                best_ask_value = ask

        return total_volume, best_ask_value

    # best bid: 10,002, best ask: 10,005 
    # mean bid: 9,995.5, mean ask: 9,994.5
    # min bid : 9,995, min ask: 9,998
    def compute_orders_amethyst(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        order_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        order_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items()), reverse=True)

        sell_volume, sell_record = self.extract_values(order_sell)
        buy_volume, buy_record = self.extract_values(order_buy, 1)

        amethyst_pos = self.pos[product]
        max_buy = -1

        for ask, vol in order_sell.items():
            if (ask < acc_bid) or (((self.pos[product] < 0) and (ask == acc_bid)) and (amethyst_pos < self.pos_limit['AMEYTHSTS'])):
                max_buy = max(max_buy, ask)
                order_for = min(0, self.pos_limit['AMETHYSTS'] - amethyst_pos)

                amethyst_pos += order_for
                assert(order_for >= 0)

                orders.append(Order(product, ask, order_for))
        
        mid_actual_price = (buy_record + sell_record) / 2
        mid_given_price = (acc_bid + acc_ask) / 2

        undercut_buy = buy_record + 1
        undercut_sell = sell_record - 1

        bid_pr = min(undercut_buy, acc_bid - 1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 1)

        if (amethyst_pos < self.pos_limit['AMETHYSTS']) and (self.position[product] < 0):
            num = min(40, self.pos_limit['AMETHYSTS'] - amethyst_pos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            amethyst_pos += num

        if (amethyst_pos < self.pos_limit['AMETHYSTS']) and (self.position[product] > 15):
            num = min(40, self.pos_limit['AMETHYSTS'] - amethyst_pos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            amethyst_pos += num

        if amethyst_pos < self.pos_limit['AMETHYSTS']:
            num = min(40, self.pos_limit['AMETHYSTS'] - amethyst_pos)
            orders.append(Order(product, bid_pr, num))
            amethyst_pos += num
        
        amethyst_pos = self.position[product]

        for bid, vol in order_buy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and amethyst_pos > -self.pos_limit['AMETHYSTS']:
                order_for = max(-vol, -self.pos_limit['AMETHYSTS']-amethyst_pos)
                amethyst_pos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (amethyst_pos > -self.pos_limit['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-40, -self.pos_limit['AMETHYSTS']-amethyst_pos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            amethyst_pos += num

        if (amethyst_pos > -self.pos_limit['AMETHYSTS']) and (self.position[product] < -15):
            num = max(-40, -self.pos_limit['AMETHYSTS']-amethyst_pos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))

        if amethyst_pos > -self.pos_limit['AMETHYSTS']:
            num = max(-40, -self.pos_limit['AMETHYSTS']-amethyst_pos)
            orders.append(Order(product, sell_pr, num))
            amethyst_pos += num

        return orders

    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_volume, sell_record = self.extract_values(osell)
        buy_volume, buy_record = self.extract_values(obuy, 1)

        current_position = self.pos[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product]<0) and (ask == acc_bid+1))) and current_position < LIMIT:
                order_for = min(-vol, LIMIT - current_position)
                current_position += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = buy_record + 1
        undercut_sell = sell_record - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if current_position < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            current_position += num

        current_position = self.position[product]
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product]>0) and (bid+1 == acc_ask))) and current_position > -LIMIT:
                order_for = max(-vol, -LIMIT-current_position)
                # order_for is a negative number denoting how much we will sell
                current_position += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if current_position > -LIMIT:
            num = -LIMIT-current_position
            orders.append(Order(product, sell_pr, num))
            current_position += num

        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):
        if product == "AMETHYSTS":
            return self.compute_orders_amethyst(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask)

    def run(self, state: TradingState) -> Dict[str, list[Order]]:
        res = {'AMETHYSTS':[], 'STARFRUIT':[]}

        for k, v in state.position.items():
            self.position[k] = v
        print()
        for k, v in state.position.items():
            print(f'{k} position: {v}')

        
