'''
This file extracts training data for training a deep learning trader

The features that are extracted are as follows:

Inputs:
1. The time the lob was release (previous batch)
2. The best bid in the previous batch
3. The best ask in the previous batch
4. The bid-ask spread in the previous batch
5. The midprice in the previous batch
6. The microprice in the previous batch
7. The final trade price (equilibrium price) in the previous batch
8. The quantity of trades completed in the previous batch

Output:
The final trade price of the current batch (equilibrium price)
'''
import csv
import os
import time
from datetime import datetime

def export_quote_logs(traders, tracked_trader_types):
    print('Saving quote logs...')
    folder_path = 'C:/Users/camer/Documents/Masters Thesis/Data/Training data'

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f'training_quote_logs_{timestamp}.csv'
    output_path = os.path.join(folder_path, output_file)

    quote_logs = []

    for trader in traders.values():
        if trader.ttype in tracked_trader_types:
            for log_entry in trader.quote_log:
                quote_logs.append(log_entry)

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        headers = ['time', 'bid_ask_spread', 'midprice', 'micro_price', 'best_bid', 'best_ask', 'p_eq', 'q_eq', 'tid', 'limit_price', 'quote_price']
        writer.writerow(headers)
        
        writer.writerows(quote_logs)

    print(f"Quote logs saved")
    return

def get_trade_data(lob, time, trades_prev_batch, prev_eq_price):
    '''
    Getting data from the LOB and transaction records

    Creating inputs for training deep learning algorithm
    
    '''
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
    
    if len(trades_prev_batch) > 0:
        prev_batch_price = prev_eq_price
        prev_batch_qty = len(trades_prev_batch)
    else:
        prev_batch_price = 0
        prev_batch_qty = 0
    
    published_trade_data = {
        "time_of_publish": time,
        "bid_ask_spread": bid_ask_spread,
        "midprice": midprice,
        "microprice": micro_price,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "prev_batch_price": prev_batch_price,
        "prev_batch_trade_qty": prev_batch_qty
    }    

    return published_trade_data

def get_trade_price(transactions, time):
    '''
    Method that gets trade information: price and time of trade
    '''
    if len(transactions) > 0:
        trade_price = transactions[0]['price']
    else:
        trade_price = 0

    final_trade_price = {
         "time_of_trade": time,
         "final_trade_price": trade_price,
    }
     
    return final_trade_price

def get_order_data(order):

    tid = order.tid
    quote_price = order.price

    order_data = {
        "trader_id": tid,
        "quote_price": quote_price,
        "customer_order_id": order.coid,
    }

    return order_data

def make_csv(folder_path='C:/Users/camer/Documents/Masters Thesis/Data/Training data'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_data_{timestamp}.csv"
    full_file_path = os.path.join(folder_path, filename)
        
    return full_file_path
def write_to_csv(visible_trade_data, trade_price, order_data, file_path):
    '''
    Writes the current trade data to a csv file
    '''

    training_data = {**visible_trade_data, **order_data, **trade_price}
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=training_data.keys())
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(training_data)



