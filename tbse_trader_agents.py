"""Module containing all trader algos"""
# pylint: disable=too-many-lines
import math
import random
import sys
import os
import csv
import time

from tbse_msg_classes import Order
from tbse_sys_consts import TBSE_SYS_MAX_PRICE, TBSE_SYS_MIN_PRICE
import time as time1
import joblib
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime


# pylint: disable=too-many-instance-attributes
class Trader:
    """Trader superclass - mostly unchanged from original BSE code by Dave Cliff
    all Traders have a trader id, bank balance, blotter, and list of orders to execute"""

    def __init__(self, ttype, tid, balance, time):
        self.ttype = ttype  # what type / strategy this trader is
        self.tid = tid  # trader unique ID code
        self.balance = balance  # money in the bank
        self.blotter = []  # record of trades executed
        self.orders = {}  # customer orders currently being worked (fixed at 1)
        self.n_quotes = 0  # number of quotes live on LOB
        self.willing = 1  # used in ZIP etc
        self.able = 1  # used in ZIP etc
        self.birth_time = time  # used when calculating age of a trader/strategy
        self.profit_per_time = 0  # profit per unit t
        self.n_trades = 0  # how many trades has this trader done?
        self.last_quote = None  # record of what its last quote was
        self.times = [0, 0, 0, 0]  # values used to calculate timing elements
        self.quote_log = []

    def __str__(self):
        return f'[TID {self.tid} type {self.ttype} balance {self.balance} blotter {self.blotter} ' \
               f'orders {self.orders} n_trades {self.n_trades} profit_per_time {self.profit_per_time}]'
    
    def save_quote_log(self, lob, time, p_eq, q_eq, quote_price, limit_price, order_type):
        '''
        Recording the data used an input for each quote along with the quote price and limit price
        tracking it on a list assigned to each trader
        '''
        # tracking quote price and limit price data
        order_type_val = 0

        if order_type == "Ask":
            order_type_val = 1
        
        order_data = [self.ttype, order_type_val, limit_price,  quote_price]

        # tracking lob data
        if len(lob) > 0: 
            best_bid = lob['bids']['best']
            best_ask = lob['asks']['best']
            if best_bid == None:
                best_bid = 0
            if best_ask == None:
                best_ask = 0
            
            if best_bid + best_ask > 0:
                bid_ask_spread = abs(best_ask - best_bid)
                midprice = (best_bid + best_ask)/2
            else:
                bid_ask_spread = midprice = 0

            num_bids = lob['bids']['n']
            num_asks = lob['asks']['n']

            if (num_bids + num_asks != 0):
                micro_price = ((num_bids * best_ask) + (num_asks * best_bid)) / (num_bids + num_asks)
            else:
                micro_price = 0

        else:
            best_bid = 0
            best_ask = 0
            bid_ask_spread = 0
            bid_ask_spread = 0
            midprice = 0
            micro_price = 0

        if p_eq <= 0:
            p_eq = 0

        lob_data = [time, bid_ask_spread, midprice, micro_price, best_bid, best_ask, p_eq, q_eq]
        
        quote_log_entry = lob_data + order_data

        self.quote_log.append(quote_log_entry)
        return
    
    def quote_log_to_csv(self):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'quote_log_{self.ttype}_{self.tid}_{timestamp}.csv'
        folder_path = 'C:/Users/camer/Documents/Masters Thesis/Data/Training data'
        file_path = os.path.join(folder_path, filename)

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['time', 'bid_ask_spread', 'midprice', 'micro_price', 'best_bid', 'best_ask', 'prev_batch_price', 'prev_batch_qty', 'trader_id', 'limit_price', 'quote_price']
            writer.writerow(headers)
            writer.writerows(self.quote_log)
            return

    def add_order(self, order, verbose):
        """
        Adds an order to the traders list of orders
        in this version, trader has at most one order,
        if allow more than one, this needs to be self.orders.append(order)
        :param order: the order to be added
        :param verbose: should verbose logging be printed to console
        :return: Response: "Proceed" if no current offer on LOB, "LOB_Cancel" if there is an order on the LOB needing
                 cancelled.\
        """

        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders[order.coid] = order

        if verbose:
            print(f'add_order < response={response}')
        return response

    def del_order(self, coid):
        """
        Removes current order from traders list of orders
        :param coid: Customer order ID of order to be deleted
        """
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        # CHANGE TO DELETE THE HEAD OF THE LIST AND KEEP THE TAIL
        self.orders.pop(coid)

    def bookkeep(self, trade, order, verbose, time):
        """
        Updates trader's internal stats with trade and order
        :param trade: Trade that has been executed
        :param order: Order trade was in response to
        :param verbose: Should verbose logging be printed to console
        :param time: Current time
        """
        output_string = ""

        if trade['coid'] in self.orders:
            coid = trade['coid']
            order_price = self.orders[coid].price
        elif trade['counter'] in self.orders:
            coid = trade['counter']
            order_price = self.orders[coid].price
        else:
            print("COID not found")
            sys.exit("This is non ideal ngl.")

        self.blotter.append(trade)  # add trade record to trader's blotter
        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transaction_price = trade['price']
        if self.orders[coid].otype == 'Bid':
            profit = order_price - transaction_price
        else:
            profit = transaction_price - order_price
        self.balance += profit
        self.n_trades += 1
        self.profit_per_time = self.balance / (time - self.birth_time)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            print(str(trade['coid']) + " " + str(trade['counter']) + " " + str(order.coid) + " " + str(
                self.orders[0].coid))
            sys.exit()

        if verbose:
            print(f'{output_string} profit={profit} balance={self.balance} profit/t={self.profit_per_time}')
        self.del_order(coid)  # delete the order

    # pylint: disable=unused-argument,no-self-use
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):
        """
        specify how trader responds to events in the market
        this is a null action, expect it to be overloaded by specific algos
        :param time: Current time
        :param lob: Limit order book
        :param trade: Trade being responded to
        :param verbose: Should verbose logging be printed to console
        :return: Unused
        """
        return None

    # pylint: disable=unused-argument,no-self-use
   
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Get's the traders order based on the current state of the market
        :param time: Current time
        :param countdown: Time to end of session
        :param lob: Limit order book
        :return: The order
        """
        return None
    

class DeepFBATrader(Trader):
    '''
    Deep learning neural network based trader
    Loads a previously trained DLNN and allows it to function in the market
    '''
    model = None
    predict = None

    def __init__(self, ttype, tid, balance, time, model_filepath, scaler_path='scaler.joblib'):
        
        Trader.__init__(self, ttype, tid, balance, time)

        if DeepFBATrader.model is None:
            DeepFBATrader.model = tf.keras.models.load_model(model_filepath)

        if DeepFBATrader.predict is None:
                DeepFBATrader.predict = tf.function(DeepFBATrader.model, reduce_retracing=True)

        self.model = DeepFBATrader.model
        self.scaler = joblib.load(scaler_path)


        self.input_data = []

        # self.interpreter = tf.lite.Interpreter(model_path=model_filepath)
        # self.interpreter.allocate_tensors()
        # self.input_details = self.interpreter.get_input_details()
        # self.output_details = self.interpreter.get_output_details()

    def get_input_data(self,time,p_eq, q_eq,lob):
        '''
        Gets data from the LOB that was used to train the model

        Puts this data into the correct format for input into the model
        '''
        
        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if best_bid == None:
            best_bid = 0
        if best_ask == None:
            best_ask = 0
        
        if best_bid + best_ask > 0:
            bid_ask_spread = abs(best_ask - best_bid)
            midprice = (best_bid + best_ask)/2
        else:
            bid_ask_spread = midprice = 0
        
        num_bids = lob['bids']['n']
        num_asks = lob['asks']['n']

        if (num_bids + num_asks != 0):
            micro_price = ((num_bids * best_ask) + (num_asks * best_bid)) / (num_bids + num_asks)
        else:
            micro_price = 0
        
        if p_eq > 0:
            prev_batch_price = p_eq
            prev_batch_qty = q_eq
        else:
            prev_batch_price = 0
            prev_batch_qty = 0
        
        input_data = [time, bid_ask_spread, midprice, micro_price, best_bid, best_ask, prev_batch_price, prev_batch_qty]   
        
        return input_data

    
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):

        if len(self.orders) < 1:
            order = None
        else:
            coid = max(self.orders.keys())
            limit = self.orders[coid].price
            otype = self.orders[coid].otype

            input_data = self.get_input_data(time, p_eq, q_eq, lob)

            # input_data = self.input_data
            input_data = np.array(input_data).reshape(1, -1)
            input_data_scaled = self.scaler.transform(input_data)
            input_data_scaled = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))
            # input_data_scaled = input_data_scaled.astype(np.float16)

            # model_price = self.model(input_data_scaled)
            model_price = DeepFBATrader.predict(input_data_scaled).numpy()[0][0]
            model_price = int(model_price)
            model_price = round(model_price) # Ensuring that the quote price is of a valid tick size (1 in this)

            if otype == "Ask":
                if model_price < limit:
                    model_price = limit
            else:
                if model_price > limit:
                    model_price = limit

            quote_price = model_price

            order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order
            # print(f"Trader {self.tid} of {self.orders[coid].otype} has orders with limit prices {[o[1].price for o in self.orders.items()]} at time {time} \n")
        return order

class TraderGiveaway(Trader):
    """
    Trader subclass Giveaway
    even dumber than a ZI-U: just give the deal away
    (but never makes a loss)
    """
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Get's giveaway traders order - in this case the price is just the limit price from the customer order
        :param time: Current time
        :param countdown: Time until end of session
        :param lob: Limit order book
        :return: Order to be sent to the exchange
        """

        if len(self.orders) < 1:
            order = None
        else:
            coid = max(self.orders.keys())
            quote_price = self.orders[coid].price
            order = Order(self.tid,
                          self.orders[coid].otype,
                          quote_price,
                          self.orders[coid].qty,
                          time, self.orders[coid].coid, self.orders[coid].toid)
            
            limit_price = self.orders[coid].price
            order_type = self.orders[coid].otype
            # This code excludes duplicated quote prices from the quote log save
            # if self.last_quote is None:
            #     last_quote_price = 0
            # else:
            #     last_quote_price = self.last_quote.price
            # if quote_price != last_quote_price:
            #     self.save_quote_log(lob, time, p_eq, q_eq, quote_price, limit_price)
            self.save_quote_log(lob, time, p_eq, q_eq, quote_price, limit_price, order_type)

            self.last_quote = order
            # print(f"Trader {self.tid} of {self.orders[coid].otype} has orders with limit prices {[o[1].price for o in self.orders.items()]} at time {time} \n")
       
        return order
class TraderZic(Trader):
    """ Trader subclass ZI-C
    After Gode & Sunder 1993"""
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Gets ZIC trader, limit price is randomly selected
        :param time: Current time
        :param countdown: Time until end of current market session
        :param lob: Limit order book
        :return: The trader order to be sent to the exchange
        """

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            
            coid = max(self.orders.keys())
            
            min_price_lob = lob['bids']['worst']
            max_price_lob = lob['asks']['worst']
            limit = self.orders[coid].price
            otype = self.orders[coid].otype

            min_price = min_price_lob
            max_price = max_price_lob

            if otype == 'Bid':
                if min_price>limit:
                    min_price=min_price_lob
                quote_price = random.randint(min_price, limit)
            else:
                if max_price<limit:
                    max_price=max_price_lob
                quote_price = random.randint(limit, max_price)
                # NB should check it == 'Ask' and barf if not
            order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order    
        
        return order


class TraderShaver(Trader):
    """Trader subclass Shaver
    shaves a penny off the best price
    if there is no best price, creates "stub quote" at system max/min"""
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Get's Shaver trader order by shaving/adding a penny to current best bid
        :param time: Current time
        :param countdown: Countdown to end of market session
        :param lob: Limit order book
        :return: The trader order to be sent to the exchange
        """
        if len(self.orders) < 1:
            order = None
        else:

            coid = max(self.orders.keys())
            limit_price = self.orders[coid].price
            otype = self.orders[coid].otype

            best_bid = 500
            best_ask = 0
            
            if demand_curve!=[]:
                best_bid = max(demand_curve, key=lambda x: x[0])[0]+1

            if supply_curve!=[]:
                best_ask = min(supply_curve, key=lambda x: x[0])[0]-1    

            if otype == 'Bid':
                quote_price= best_bid
                quote_price = min(quote_price, limit_price)
            else:
                quote_price = best_ask
                quote_price = max(quote_price, limit_price)

            #quote_price = min(quote_price, limit_price)
            order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order

        return order


class TraderSniper(Trader):
    """
    Trader subclass Sniper
    Based on Shaver,
    "lurks" until t remaining < threshold% of the trading session
    then gets increasing aggressive, increasing "shave thickness" as t runs out"""
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        :param time: Current time
        :param countdown: Time until end of market session
        :param lob: Limit order book
        :return: Trader order to be sent to exchange
        """
        lurk_threshold = 0.2
        shave_growth_rate = 3
        shave = int(1.0 / (0.01 + countdown / (shave_growth_rate * lurk_threshold)))
        if (len(self.orders) < 1) or (countdown > lurk_threshold):
            order = None
        else:
            coid = max(self.orders.keys())
            limit_price = self.orders[coid].price
            otype = self.orders[coid].otype

            if demand_curve!=None and supply_curve!=None:

                best_bid = min(demand_curve, key=lambda x: x[0])[0]
                best_ask = max(supply_curve, key=lambda x: x[0])[0]
            else:
                best_bid = lob['bids']['worst'] - shave
                best_ask = lob['asks']['worst'] + shave


            if otype == 'Bid':
                    quote_price = best_bid+shave
                    quote_price = min(quote_price, limit_price)    

            else:
                    quote_price = best_ask-shave
                    quote_price = max(quote_price, limit_price)  
            
            quote_price = min(quote_price, limit_price)    
            order = Order(self.tid, otype, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order
        return order


# Trader subclass ZIP
# After Cliff 1997
# pylint: disable=too-many-instance-attributes
class TraderZip(Trader):
    """ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
    NB this implementation keeps separate margin values for buying & selling,
       so a single trader can both buy AND sell
       -- in the original, traders were either buyers OR sellers"""

    def __init__(self, ttype, tid, balance, time):

        Trader.__init__(self, ttype, tid, balance, time)
        m_fix = 0.05
        m_var = 0.3
        self.job = None  # this is 'Bid' or 'Ask' depending on customer order
        self.active = False  # gets switched to True while actively working an order
        self.prev_change = 0  # this was called last_d in Cliff'97
        self.beta = 0.2 + 0.2 * random.random()  # learning rate #0.1 + 0.2 * random.random()
        self.momentum = 0.3 * random.random()  # momentum #0.3 * random.random()
        self.ca = 0.10  # self.ca & .cr were hard-coded in '97 but parameterised later
        self.cr = 0.10
        self.margin = None  # this was called profit in Cliff'97
        self.margin_buy = -1.0 * (m_fix + m_var * random.random())
        self.margin_sell = m_fix + m_var * random.random()
        self.price = None
        self.limit = None
        self.times = [0, 0, 0, 0]
        # memory of best price & quantity of best bid and ask, on LOB on previous update
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None
        self.last_batch = None

    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        :param time: Current time
        :param countdown: Time until end of current market session
        :param lob: Limit order book
        :return: Trader order to be sent to exchange
        """
        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            coid = max(self.orders.keys())
            self.active = True
            self.limit = self.orders[coid].price
            self.job = self.orders[coid].otype
            if self.job == 'Bid':
                # currently a buyer (working a bid order)
                self.margin = self.margin_buy
            else:
                # currently a seller (working a sell order)
                self.margin = self.margin_sell
            quote_price = int(self.limit * (1 + self.margin))
            self.price = quote_price

            order = Order(self.tid, self.job, quote_price, self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            limit_price = self.limit
            order_type = self.orders[coid].otype
            self.save_quote_log(lob, time, p_eq, q_eq, quote_price, limit_price, order_type)
            self.last_quote = order
        return order

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):
        """
        update margin on basis of what happened in marke
        ZIP trader responds to market events, altering its margin
        does this whether it currently has an order to work or not
        :param time: Current time
        :param lob: Limit order book
        :param trade: Trade being responded to
        :param verbose: Should verbose logging be printed to console
        """
        
        if self.last_batch==(demand_curve,supply_curve):
            return
        else:
            self.last_batch = (demand_curve,supply_curve)
     
        trade = trades[0] if trades else None
    
        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if demand_curve!=[]:
            best_bid = max(demand_curve, key=lambda x: x[0])[0]
        if supply_curve!=[]:
            best_ask = min(supply_curve, key=lambda x: x[0])[0]

        def target_up(price):
            """
            generate a higher target price by randomly perturbing given price
            :param price: Current price
            :return: New price target
            """
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel + ptrb_abs, 0))

            return target

        def target_down(price):
            """
            generate a lower target price by randomly perturbing given price
            :param price: Current price
            :return: New price target
            """
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel - ptrb_abs, 0))

            return target

        def willing_to_trade(price):
            """
            am I willing to trade at this price?
            :param price: Price to be traded out
            :return: Is the trader willing to trade
            """
            willing = False
            if self.job == 'Bid' and self.active and self.price >= price:
                willing = True
            if self.job == 'Ask' and self.active and self.price <= price:
                willing = True
            return willing

        def profit_alter(price):
            """
            Update target profit margin
            :param price: New target profit margin
            """
            old_price = self.price
            diff = price - old_price
            change = ((1.0 - self.momentum) * (self.beta * diff)) + (self.momentum * self.prev_change)
            self.prev_change = change
            new_margin = ((self.price + change) / self.limit) - 1.0

            if self.job == 'Bid':
                if new_margin < 0.0:
                    self.margin_buy = new_margin
                    self.margin = new_margin
            else:
                if new_margin > 0.0:
                    self.margin_sell = new_margin
                    self.margin = new_margin

            # set the price from limit and profit-margin
            self.price = int(round(self.limit * (1.0 + self.margin), 0))

        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False

        #lob_best_bid_p = lob['bids']['best']
        lob_best_bid_p = best_bid #CHANGE HERE
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = 1
            if self.prev_best_bid_p is None:
                self.prev_best_bid_p = lob_best_bid_p
            elif self.prev_best_bid_p < lob_best_bid_p:
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1] #might have to check if has been cancelled at some point during batch 
            #for item in lob['tape'] check if cancel happened with price of
            if last_tape_item['type'] == 'Cancel': 
                #print("Last bid was cancelled") #test
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        #lob_best_ask_p = lob['asks']['best']
        lob_best_ask_p = best_ask #CHANGE HERE
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = 1
            if self.prev_best_ask_p is None:
                self.prev_best_ask_p = lob_best_ask_p
            elif self.prev_best_ask_p > lob_best_ask_p:
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                #print("Last bid was cancelled") # test
                ask_lifted = False
            else:
                ask_lifted = True

        if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
            print('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)

        deal = bid_hit or ask_lifted

        if trade is None:
            deal = False

        if self.job == 'Ask':
            # seller
            if deal:
                trade_price = trade['price']
                if self.price <= trade_price:
                    # could sell for more? raise margin
                    target_price = target_up(trade_price)
                    profit_alter(target_price)
                elif ask_lifted and self.active and not willing_to_trade(trade_price):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_down(trade_price)
                    profit_alter(target_price)
            else:
                # no deal: aim for a target price higher than best bid
                if ask_improved and self.price > lob_best_ask_p:
                    if lob_best_bid_p is not None:
                        target_price = target_up(lob_best_bid_p)
                    else:
                        target_price = lob['asks']['worst']  # stub quote
                    profit_alter(target_price)

        if self.job == 'Bid':
            # buyer
            if deal:
                trade_price = trade['price']
                if self.price >= trade_price:
                    # could buy for less? raise margin (i.e. cut the price)
                    target_price = target_down(trade_price)
                    profit_alter(target_price)
                elif bid_hit and self.active and not willing_to_trade(trade_price):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_up(trade_price)
                    profit_alter(target_price)
            else:
                # no deal: aim for target price lower than best ask
                if bid_improved and self.price < lob_best_bid_p:
                    if lob_best_ask_p is not None:
                        target_price = target_down(lob_best_ask_p)
                    else:
                        target_price = lob['bids']['worst']  # stub quote
                    profit_alter(target_price)

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q


# pylint: disable=too-many-instance-attributes
class TraderAa(Trader):
    """
    Daniel Snashall's implementation of Vytelingum's AA trader, first described in his 2006 PhD Thesis.
    For more details see: Vytelingum, P., 2006. The Structure and Behaviour of the Continuous Double
    Auction. PhD Thesis, University of Southampton
    """

    def __init__(self, ttype, tid, balance, time):
        # Stuff about trader
        super().__init__(ttype, tid, balance, time)
        self.active = False

        self.limit = None
        self.job = None

        # learning variables
        self.r_shout_change_relative = 0.05
        self.r_shout_change_absolute = 0.05
        self.short_term_learning_rate = random.uniform(0.1, 0.5)
        self.long_term_learning_rate = random.uniform(0.1, 0.5)
        self.moving_average_weight_decay = 0.95  # how fast weight decays with t, lower is quicker, 0.9 in vytelingum
        self.moving_average_window_size = 5
        self.offer_change_rate = 3.0
        self.theta = -2.0
        self.theta_max = 2.0
        self.theta_min = -8.0
        self.market_max = TBSE_SYS_MAX_PRICE

        # Variables to describe the market
        self.previous_transactions = []
        self.moving_average_weights = []
        for i in range(self.moving_average_window_size):
            self.moving_average_weights.append(self.moving_average_weight_decay ** i)
        self.estimated_equilibrium = []
        self.smiths_alpha = []
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

        # Trading Variables
        self.r_shout = None
        self.buy_target = None
        self.sell_target = None
        self.buy_r = -1.0 * (0.3 * random.random())
        self.sell_r = -1.0 * (0.3 * random.random())

        #define last batch so that internal values are only updated upon new batch matching
        self.last_batch = None

    def calc_eq(self):
        """
        Calculates the estimated 'eq' or estimated equilibrium price.
        Slightly modified from paper, it is unclear in paper
        N previous transactions * weights / N in Vytelingum, swap N denominator for sum of weights to be correct?
        :return: Estimated equilibrium price
        """
        if len(self.previous_transactions) == 0:
            return
        if len(self.previous_transactions) < self.moving_average_window_size:
            # Not enough transactions
            self.estimated_equilibrium.append(
                float(sum(self.previous_transactions)) / max(len(self.previous_transactions), 1))
        else:
            n_previous_transactions = self.previous_transactions[-self.moving_average_window_size:]
            thing = [n_previous_transactions[i] * self.moving_average_weights[i] for i in
                     range(self.moving_average_window_size)]
            eq = sum(thing) / sum(self.moving_average_weights)
            self.estimated_equilibrium.append(eq)

    def calc_alpha(self):
        """
        Calculates trader's alpha value - see AA paper for details.
        """
        alpha = 0.0
        for p in self.estimated_equilibrium:
            alpha += (p - self.estimated_equilibrium[-1]) ** 2
        alpha = math.sqrt(alpha / len(self.estimated_equilibrium))
        self.smiths_alpha.append(alpha / self.estimated_equilibrium[-1])

    def calc_theta(self):
        """
        Calculates trader's theta value - see AA paper for details.
        """
        gamma = 2.0  # not sensitive apparently so choose to be whatever
        # necessary for initialisation, div by 0
        if min(self.smiths_alpha) == max(self.smiths_alpha):
            alpha_range = 0.4  # starting value i guess
        else:
            alpha_range = (self.smiths_alpha[-1] - min(self.smiths_alpha)) / (
                    max(self.smiths_alpha) - min(self.smiths_alpha))
        theta_range = self.theta_max - self.theta_min
        desired_theta = self.theta_min + theta_range * (1 - (alpha_range * math.exp(gamma * (alpha_range - 1))))
        self.theta = self.theta + self.long_term_learning_rate * (desired_theta - self.theta)

    def calc_r_shout(self):
        """
        Calculates trader's r shout value - see AA paper for details.
        """
        p = self.estimated_equilibrium[-1]
        lim = self.limit
        theta = self.theta
        if self.job == 'Bid':
            # Currently a buyer
            if lim <= p:  # extra-marginal!
                self.r_shout = 0.0
            else:  # intra-marginal :(
                if self.buy_target > self.estimated_equilibrium[-1]:
                    # r[0,1]
                    self.r_shout = math.log(((self.buy_target - p) * (math.exp(theta) - 1) / (lim - p)) + 1) / theta
                else:
                    # r[-1,0]
                    self.r_shout = math.log((1 - (self.buy_target / p)) * (math.exp(theta) - 1) + 1) / theta

        if self.job == 'Ask':
            # Currently a seller
            if lim >= p:  # extra-marginal!
                self.r_shout = 0
            else:  # intra-marginal :(
                if self.sell_target > self.estimated_equilibrium[-1]:
                    # r[-1,0]
                    self.r_shout = math.log(
                        (self.sell_target - p) * (math.exp(theta) - 1) / (self.market_max - p) + 1) / theta
                else:
                    # r[0,1]
                    a = (self.sell_target - lim) / (p - lim)
                    self.r_shout = (math.log((1 - a) * (math.exp(theta) - 1) + 1)) / theta

    def calc_agg(self):
        """
        Calculates Trader's aggressiveness parameter - see AA paper for details.
        """
        if self.job == 'Bid':
            # BUYER
            if self.buy_target >= self.previous_transactions[-1]:
                # must be more aggressive
                delta = (1 + self.r_shout_change_relative) * self.r_shout + self.r_shout_change_absolute
            else:
                delta = (1 - self.r_shout_change_relative) * self.r_shout - self.r_shout_change_absolute

            self.buy_r = self.buy_r + self.short_term_learning_rate * (delta - self.buy_r)

        if self.job == 'Ask':
            # SELLER
            if self.sell_target > self.previous_transactions[-1]:
                delta = (1 + self.r_shout_change_relative) * self.r_shout + self.r_shout_change_absolute
            else:
                delta = (1 - self.r_shout_change_relative) * self.r_shout - self.r_shout_change_absolute

            self.sell_r = self.sell_r + self.short_term_learning_rate * (delta - self.sell_r)

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def calc_target(self):
        """
        Calculates trader's target price - see AA paper for details.
        """
        p = 1
        if len(self.estimated_equilibrium) > 0:
            p = self.estimated_equilibrium[-1]
            if self.limit == p:
                p = p * 1.000001  # to prevent theta_bar = 0
        elif self.job == 'Bid':
            p = self.limit - self.limit * 0.2  # Initial guess for eq if no deals yet!!....
        elif self.job == 'Ask':
            p = self.limit + self.limit * 0.2
        lim = self.limit
        theta = self.theta
        if self.job == 'Bid':
            # BUYER
            minus_thing = (math.exp(-self.buy_r * theta) - 1) / (math.exp(theta) - 1)
            plus_thing = (math.exp(self.buy_r * theta) - 1) / (math.exp(theta) - 1)
            theta_bar = (theta * lim - theta * p) / p
            if theta_bar == 0:
                theta_bar = 0.0001
            if math.exp(theta_bar) - 1 == 0:
                theta_bar = 0.0001
            bar_thing = (math.exp(-self.buy_r * theta_bar) - 1) / (math.exp(theta_bar) - 1)
            if lim <= p:  # Extra-marginal
                if self.buy_r >= 0:
                    self.buy_target = lim
                else:
                    self.buy_target = lim * (1 - minus_thing)
            else:  # intra-marginal
                if self.buy_r >= 0:
                    self.buy_target = p + (lim - p) * plus_thing
                else:
                    self.buy_target = p * (1 - bar_thing)
            self.buy_target = min(self.buy_target, lim)

        if self.job == 'Ask':
            # SELLER
            minus_thing = (math.exp(-self.sell_r * theta) - 1) / (math.exp(theta) - 1)
            plus_thing = (math.exp(self.sell_r * theta) - 1) / (math.exp(theta) - 1)
            theta_bar = (theta * lim - theta * p) / p
            if theta_bar == 0:
                theta_bar = 0.0001
            if math.exp(theta_bar) - 1 == 0:
                theta_bar = 0.0001
            bar_thing = (math.exp(-self.sell_r * theta_bar) - 1) / (math.exp(theta_bar) - 1)  # div 0 sometimes what!?
            if lim <= p:  # Extra-marginal
                if self.buy_r >= 0:
                    self.buy_target = lim
                else:
                    self.buy_target = lim + (self.market_max - lim) * minus_thing
            else:  # intra-marginal
                if self.buy_r >= 0:
                    self.buy_target = lim + (p - lim) * (1 - plus_thing)
                else:
                    self.buy_target = p + (self.market_max - p) * bar_thing
            if self.sell_target is None:
                self.sell_target = lim
            elif self.sell_target < lim:
                self.sell_target = lim

    # pylint: disable=too-many-branches
    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Creates an AA trader's order
        :param time: Current time
        :param countdown: Time left in the current trading period
        :param lob: Current state of the limit order book
        :return: Order to be sent to the exchange
        """
        if len(self.orders) < 1:
            self.active = False
            return None
        coid = max(self.orders.keys())
        self.active = True
        self.limit = self.orders[coid].price
        self.job = self.orders[coid].otype
        self.calc_target()

        if self.prev_best_bid_p is None:
            o_bid = 0
        else:
            o_bid = self.prev_best_bid_p
        if self.prev_best_ask_p is None:
            o_ask = self.market_max
        else:
            o_ask = self.prev_best_ask_p

        quote_price = TBSE_SYS_MIN_PRICE
        if self.job == 'Bid':  # BUYER
            if self.limit <= o_bid:
                return None
            if len(self.previous_transactions) > 0:  # has been at least one transaction
                o_ask_plus = (1 + self.r_shout_change_relative) * o_ask + self.r_shout_change_absolute
                quote_price = o_bid + ((min(self.limit, o_ask_plus) - o_bid) / self.offer_change_rate)
            else:
                if o_ask <= self.buy_target:
                    quote_price = o_ask
                else:
                    quote_price = o_bid + ((self.buy_target - o_bid) / self.offer_change_rate)
        elif self.job == 'Ask':
            if self.limit >= o_ask:
                return None
            if len(self.previous_transactions) > 0:  # has been at least one transaction
                o_bid_minus = (1 - self.r_shout_change_relative) * o_bid - self.r_shout_change_absolute
                quote_price = o_ask - ((o_ask - max(self.limit, o_bid_minus)) / self.offer_change_rate)
            else:
                if o_bid >= self.sell_target:
                    quote_price = o_bid
                else:
                    quote_price = o_ask - ((o_ask - self.sell_target) / self.offer_change_rate)
        limit_price = self.limit
        order_type = self.orders[coid].otype
        self.save_quote_log(lob, time, p_eq, q_eq, quote_price, limit_price)
        order = Order(self.tid, self.job, int(quote_price), self.orders[coid].qty, time, self.orders[coid].coid,
                      self.orders[coid].toid)
        self.last_quote = order
        return order

    # pylint: disable=too-many-branches
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):
        """
        Updates AA trader's internal variables based on activities on the LOB
        Beginning nicked from ZIP
        what, if anything, has happened on the bid LOB? Nicked from ZIP.
        :param time: current time
        :param lob: current state of the limit order book
        :param trade: trade which occurred to trigger this response
        :param verbose: should verbose logging be printed to the console
        """

        if self.last_batch==(demand_curve,supply_curve):
            return
        else:
            self.last_batch = (demand_curve,supply_curve)
     
        trade = trades[0] if trades else None
    
        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if demand_curve!=[]:
            best_bid = max(demand_curve, key=lambda x: x[0])[0]
        if supply_curve!=[]:
            best_ask = min(supply_curve, key=lambda x: x[0])[0]

        bid_hit = False
        #lob_best_bid_p = lob['bids']['best'] #CHANGED
        lob_best_bid_p = best_bid
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = 1
            if self.prev_best_bid_p is None:
                self.prev_best_bid_p = lob_best_bid_p
            # elif self.prev_best_bid_p < lob_best_bid_p :
            #     # best bid has improved
            #     # NB doesn't check if the improvement was by self
            #     bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        # ask_improved = False
        ask_lifted = False

        #lob_best_ask_p = lob['asks']['best'] #CHANGED THIS
        lob_best_ask_p = best_ask
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = 1
            if self.prev_best_ask_p is None:
                self.prev_best_ask_p = lob_best_ask_p
            # elif self.prev_best_ask_p > lob_best_ask_p :
            #     # best ask has improved -- NB doesn't check if the improvement was by self
            #     ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                ask_lifted = False
            else:
                ask_lifted = True

        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

        deal = bid_hit or ask_lifted
        if (trades==[]):
            deal = False

        # End nicked from ZIP

        if deal:
            # if trade is not None:
            self.previous_transactions.append(trade['price'])
            if self.sell_target is None:
                self.sell_target = trade['price'] #CHANGED THIS
                #self.sell_target = best_ask
            if self.buy_target is None:
                self.buy_target = trade['price'] #CHANGED THIS
                #self.sell_target = best_bid
            self.calc_eq()
            self.calc_alpha()
            self.calc_theta()
            self.calc_r_shout()
            self.calc_agg()
            self.calc_target()


# pylint: disable=too-many-instance-attributes
class TraderGdx(Trader):
    """
    Daniel Snashall's implementation of Tesauro & Bredin's GDX Trader algorithm. For more details see:
    Tesauro, G., Bredin, J., 2002. Sequential Strategic Bidding in Auctions using Dynamic Programming.
    Proceedings AAMAS2002.
    """
    def __init__(self, ttype, tid, balance, time):
        super().__init__(ttype, tid, balance, time)
        self.prev_orders = []
        self.job = None  # this gets switched to 'Bid' or 'Ask' depending on order-type
        self.active = False  # gets switched to True while actively working an order
        self.limit = None

        # memory of all bids and asks and accepted bids and asks
        self.outstanding_bids = []
        self.outstanding_asks = []
        self.accepted_asks = []
        self.accepted_bids = []

        self.price = -1

        # memory of best price & quantity of best bid and ask, on LOB on previous update
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

        self.first_turn = True

        self.gamma = 0.9

        self.holdings = 25
        self.remaining_offer_ops = 25
        self.values = [[0 for _ in range(self.remaining_offer_ops)] for _ in range(self.holdings)]

        #define last batch so that internal values are only updated upon new batch matching
        self.last_batch = None

    def get_order(self,time,p_eq ,q_eq, demand_curve,supply_curve,countdown,lob):
        """
        Creates a GDX trader's order
        :param time: Current time
        :param countdown: Time left in the current trading period
        :param lob: Current state of the limit order book
        :return: Order to be sent to the exchange
        """
        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            coid = max(self.orders.keys())
            self.active = True
            self.limit = self.orders[coid].price
            self.job = self.orders[coid].otype

            # calculate price
            if self.job == 'Bid':
                self.price = self.calc_p_bid(self.holdings - 1, self.remaining_offer_ops - 1)
            if self.job == 'Ask':
                self.price = self.calc_p_ask(self.holdings - 1, self.remaining_offer_ops - 1)

            order = Order(self.tid, self.job, int(self.price), self.orders[coid].qty, time, self.orders[coid].coid,
                          self.orders[coid].toid)
            self.last_quote = order

        if self.first_turn or self.price == -1:
            return None
        return order

    def calc_p_bid(self, m, n):
        """
        Calculates the price the GDX trader should bid at. See GDX paper for more details.
        :param m: Table of expected values
        :param n: Remaining opportunities to make an offer
        :return: Price to bid at
        """
        best_return = 0
        best_bid = 0
        # second_best_return = 0
        second_best_bid = 0

        # first step size of 1 get best and 2nd best
        for i in [x * 2 for x in range(int(self.limit / 2))]:
            thing = self.belief_buy(i) * ((self.limit - i) + self.gamma * self.values[m - 1][n - 1]) + (
                    1 - self.belief_buy(i) * self.gamma * self.values[m][n - 1])
            if thing > best_return:
                second_best_bid = best_bid
                # second_best_return = best_return
                best_return = thing
                best_bid = i

        # always best bid largest one
        if second_best_bid > best_bid:
            a = second_best_bid
            second_best_bid, best_bid = best_bid, a
            # second_best_bid = best_bid
            # best_bid = a

        # then step size 0.05
        for i in [x * 0.05 for x in range(int(second_best_bid), int(best_bid))]:
            thing = self.belief_buy(i + second_best_bid) * (
                    (self.limit - (i + second_best_bid)) + self.gamma * self.values[m - 1][n - 1]) + (
                            1 - self.belief_buy(i + second_best_bid) * self.gamma * self.values[m][n - 1])
            if thing > best_return:
                best_return = thing
                best_bid = i + second_best_bid

        return best_bid

    def calc_p_ask(self, m, n):
        """
        Calculates the price the GDX trader should sell at. See GDX paper for more details.
        :param m: Table of expected values
        :param n: Remaining opportunities to make an offer
        :return: Price to sell at
        :return: Price to sell at
        """
        best_return = 0
        best_ask = self.limit
        # second_best_return = 0
        second_best_ask = self.limit

        # first step size of 1 get best and 2nd best
        for i in [x * 2 for x in range(int(self.limit / 2))]:
            j = i + self.limit
            thing = self.belief_sell(j) * ((j - self.limit) + self.gamma * self.values[m - 1][n - 1]) + (
                    1 - self.belief_sell(j) * self.gamma * self.values[m][n - 1])
            if thing > best_return:
                second_best_ask = best_ask
                # second_best_return = best_return
                best_return = thing
                best_ask = j
        # always best ask largest one
        if second_best_ask > best_ask:
            a = second_best_ask
            second_best_ask, best_ask = best_ask, a
            # second_best_ask = best_ask
            # best_ask = a

        # then step size 0.05
        for i in [x * 0.05 for x in range(int(second_best_ask), int(best_ask))]:
            thing = self.belief_sell(i + second_best_ask) * (
                    ((i + second_best_ask) - self.limit) + self.gamma * self.values[m - 1][n - 1]) + (
                            1 - self.belief_sell(i + second_best_ask) * self.gamma * self.values[m][n - 1])
            if thing > best_return:
                best_return = thing
                best_ask = i + second_best_ask

        return best_ask

    def belief_sell(self, price):
        """
        Calculates the 'belief' that a certain price will be accepted and traded on the exchange.
        :param price: The price for which we want to calculate the belief.
        :return: The belief value (decimal).
        """
        accepted_asks_greater = 0
        bids_greater = 0
        unaccepted_asks_lower = 0
        for p in self.accepted_asks:
            if p >= price:
                accepted_asks_greater += 1
        for p in [thing[0] for thing in self.outstanding_bids]:
            if p >= price:
                bids_greater += 1
        for p in [thing[0] for thing in self.outstanding_asks]:
            if p <= price:
                unaccepted_asks_lower += 1

        if accepted_asks_greater + bids_greater + unaccepted_asks_lower == 0:
            return 0
        return (accepted_asks_greater + bids_greater) / (accepted_asks_greater + bids_greater + unaccepted_asks_lower)

    def belief_buy(self, price):
        """
        Calculates the 'belief' that a certain price will be accepted and traded on the exchange.
        :param price: The price for which we want to calculate the belief.
        :return: The belief value (decimal).
        """
        accepted_bids_lower = 0
        asks_lower = 0
        unaccepted_bids_greater = 0
        for p in self.accepted_bids:
            if p <= price:
                accepted_bids_lower += 1
        for p in [thing[0] for thing in self.outstanding_asks]:
            if p <= price:
                asks_lower += 1
        for p in [thing[0] for thing in self.outstanding_bids]:
            if p >= price:
                unaccepted_bids_greater += 1
        if accepted_bids_lower + asks_lower + unaccepted_bids_greater == 0:
            return 0
        return (accepted_bids_lower + asks_lower) / (accepted_bids_lower + asks_lower + unaccepted_bids_greater)

    def get_best_n_bids(self,demand_curve, n):
        bids = []
        last_item_count = 0
        for price, quantity in demand_curve:
            num_bids = quantity-last_item_count
            last_item_count = quantity
            bids += [price] * num_bids
            if len(bids) >= n:
                return bids[:n]
        return bids

    def get_best_n_asks(self,supply_curve, n):
        asks = []
        last_item_count = 0
        for price, quantity in reversed(supply_curve):
            num_asks = quantity-last_item_count
            last_item_count = quantity
            asks += [price] * num_asks
            if len(asks) >= n:
                return asks[:n]
        return asks
    
    def respond(self,time,p_eq ,q_eq, demand_curve,supply_curve,lob,trades,verbose):
        """
        Updates GDX trader's internal variables based on activities on the LOB
        :param time: current time
        :param lob: current state of the limit order book
        :param trade: trade which occurred to trigger this response
        :param verbose: should verbose logging be printed to the console
        """
        if self.last_batch==(demand_curve,supply_curve):
            return
        else:
            self.last_batch = (demand_curve,supply_curve)
            # print(f"demand_curve {demand_curve}")
            # print(f"supply curve {supply_curve}")
     
        trade = trades[0] if trades else None
    
        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if demand_curve!=[]:
            best_bid = max(demand_curve, key=lambda x: x[0])[0]
        if supply_curve!=[]:
            best_ask = min(supply_curve, key=lambda x: x[0])[0]
        
        # what, if anything, has happened on the bid LOB?
        self.outstanding_bids = lob['bids']['lob']
        # bid_improved = False
        # bid_hit = False
        # lob_best_bid_p = lob['bids']['best']
        lob_best_bid_p = best_bid
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = 1
            if self.prev_best_bid_p is None:
                self.prev_best_bid_p = lob_best_bid_p
            # elif self.prev_best_bid_p < lob_best_bid_p :
            #     # best bid has improved
            #     # NB doesn't check if the improvement was by self
            #     bid_improved = True

            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit

                #self.accepted_bids.append(self.prev_best_bid_p) #CHANGED HERE
                self.accepted_bids.extend(self.get_best_n_bids(demand_curve,q_eq))
                #self.accepted_bids.extend([p for p,q in demand_curve[:q_eq]])
                #print(f"adding {[p for p,q in demand_curve[:q_eq]]}")
                # bid_hit = True
        # elif self.prev_best_bid_p is not None:
        #     # the bid LOB has been emptied: was it cancelled or hit?
        #     last_tape_item = lob['tape'][-1]
        # if last_tape_item['type'] == 'Cancel' :
        #     bid_hit = False
        # else:
        #     bid_hit = True

        # what, if anything, has happened on the ask LOB?
        self.outstanding_asks = lob['asks']['lob']
        # ask_improved = False
        # ask_lifted = False
        #lob_best_ask_p = lob['asks']['best']
        lob_best_ask_p = best_ask
        lob_best_ask_q = None

        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = 1
            if self.prev_best_ask_p is None:
                self.prev_best_ask_p = lob_best_ask_p
            # elif self.prev_best_ask_p > lob_best_ask_p :
            # best ask has improved -- NB doesn't check if the improvement was by self
            # ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # assume previous best ask was lifted
                #self.accepted_asks.append(self.prev_best_ask_p) #CHANGED THIS
                self.accepted_asks.extend(self.get_best_n_asks(supply_curve,q_eq))
                #self.accepted_asks.extend([p for p,q in supply_curve[-q_eq:]])
                #print(f"adding {[p for p,q in supply_curve[-q_eq:]]}")

                # ask_lifted = True
        # elif self.prev_best_ask_p is not None:
        # the ask LOB is empty now but was not previously: canceled or lifted?
        # last_tape_item = lob['tape'][-1]
        # if last_tape_item['type'] == 'Cancel' :
        #     ask_lifted = False
        # else:
        #     ask_lifted = True

        # populate expected values
        if self.first_turn:
            self.first_turn = False
            for n in range(1, self.remaining_offer_ops):
                for m in range(1, self.holdings):
                    if self.job == 'Bid':
                        # BUYER
                        self.values[m][n] = self.calc_p_bid(m, n)

                    if self.job == 'Ask':
                        # BUYER
                        self.values[m][n] = self.calc_p_ask(m, n)

        # deal = bid_hit or ask_lifted

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

    # ----------------trader-types have all been defined now-------------
