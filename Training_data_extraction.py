'''
This file extracts training data for training a deep learning trader

The features that are extracted are as follows (from  Wray et al.):
Inputs:
1. – The time t the trade took place.
4. + The LOB's bid-ask spread at time t.
5. + The LOB's midprice at time t.
6. + The LOB's microprice at time t.
7. + The best (highest) bid-price on the LOB at time t.
8. – The best (lowest) ask-price on the LOB at time t.
10. – The LOB imbalance at time t.
11. – The total quantity of all quotes on the LOB at time t.
12. – An estimate P* of the competitive equilibrium price
at time t, using the method reported in [25][26].
13. – Smith's a metric [19], calculated from P* at time t.

Output:
The price of the trade.

'''
import csv
import os


def get_trade_data(lob, time):
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

        #microprice = 
    else:
        best_bid = 0
        best_ask = 0
        bid_ask_spread = 0
        bid_ask_spread = 0
        midprice = 0
    
    published_trade_data = {
        "time_of_publish": time,
        "bid_ask_spread": bid_ask_spread,
        "midprice": midprice,
        #"microprice": microprice,
        "best_bid": best_bid,
        "best_ask": best_ask,
    }    

    return published_trade_data

def get_trade_price(transactions, time):
    '''
    Method that gets trade information: price and time of trade
    '''
    trade_price = transactions[0]['price']

    final_trade_price = {
         "time_of_trade": time,
         "final_trade_price": trade_price,
    }
     
    return final_trade_price

def write_to_csv(visible_trade_data, trade_price, folder_path='C:/Users/camer/Documents/Masters Thesis/Data/Training data', filename='training_data.csv'):
        '''
        Writes the current trade data to a csv file
        '''
        training_data = {**visible_trade_data, **trade_price}

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        full_file_path = os.path.join(folder_path, filename)
        file_exists = os.path.isfile(full_file_path)
        
        with open(full_file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=training_data.keys())
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(training_data)



