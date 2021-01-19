import numpy as np
import pandas as pd
#import statsmodels
#import statsmodels.formula.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from System import *
from System.Collections.Generic import List
from QuantConnect import *
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Data.UniverseSelection import *

class PairStrat(QCAlgorithm): 

    def Initialize(self):
        # Initialise Algo date and funding for backtesting
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2018, 10, 10)
        self.SetCash(100000)
        
        self.AddUniverse(self.Coarse)#, self.Fine)
        self.UniverseSettings.Resolution = Resolution.Daily
        # Algorithm component init
        self.LookbackPeriod = 200
        # https://github.com/QuantConnect/Lean/blob/master/Algorithm.CSharp/EmaCrossUniverseSelectionAlgorithm.cs

    # test for git push

    def Coarse(self, coarse):
        
        # Universe DataFrame
        columns = ['Price', 'HasFundamentalData']
        data_df = pd.DataFrame.from_records(
            [[getattr(s, name) for name in columns] for s in coarse],
            index   = [s.Symbol for s in coarse],
            columns = columns,
            coerce_float=True)
        self.universe = data_df.index.tolist()
        return self.universe
        context.Log("course size is: {}".format(universe.volume))
        return self.universe.index.tolist()
        data = pd.DataFrame(self.universe)

        # DataFrames for ADF
        history = history = self.History(data.Keys, 50, Resolution.Daily)
        returns = history.unstack(level = 1).close.transpose().pct_change().dropna()
        
        # ADF Test - return true for stationarity
        def adfTest(returns, siglevel = 0.05):
            return pd.Series([adfuller(values)[1] < siglevel for columns, values\
                                in returns.iteritems()], index = returns.columns)
        
        adf = adfTest(returns)
        # List for confimed symbols to be added to
        adfTested = []
        # Compute ADF test for coarse assets
        for symbol, value in adf.iteritems():
            if value == True:
                adfTested.append(symbol)
        
        # Get historical data for ADF tested assets
        adfTestedHistory = self.History(adfTested, 100, Resolution.Daily)
        # Create Correlation Matrix using historical data - Kendalls Tau
        correlationMatrix = adfTestedHistory.unstack(level=1).close.transpose()\
                            .corr(method='kendall').dropna()
        
        # Convert to simpler structure to remove low correlation pairs and retrieve listed() pairs
        coarseSelection = correlationMatrix.unstack().drop_duplicates()
        # Convert into a DataFrame to use with pandas
        coarseSelection = pd.DataFrame(coarseSelection)
        # Rename 1st column to correlation for future specification
        coarseSelection = correlation.rename({0: 'correlation'}, axis=1)

        # Remove perfectly correlated, t(x,x) correlations and those lower than 0.9
        for correlation, row in coarseSelection.iterrows():
            if row['correlation'] == 1 or row['correlation'] < 0.9:
                coarseSelection.drop(coarseSelection[(coarseSelection['correlation'] == 1)| \
                    (coarseSelection['correlation'] < 0.9)].index, inplace=True)
        
        # Remove the correlation column
        coarseSelection = coarseSelection.drop(['correlation'], axis = 1)
        
        # Return a list of assets (the asset names are the index of the DataFrame)
        pairs =  list(coarseSelection.index)
        return pairs

'''
    def Fine(self, fine):

        def cointTest(pairs, siglevel = 0.05):
            for pair in fine:
                h0 = self.History(pair[0], self.LookbackPeriod, Resolution.Daily)
                h1 = self.History(pair[1], self.LookbackPeriod, Resolution.Daily)
                r0 = h0.unstack(level = 1).close.transpose().pct_change().dropna()
                r1 = h1.unstack(level = 1).close.transpose().pct_change().dropna()

                return pd.Series([coint(r0, r1)[1] < siglevel for columns, values \
                                    in fine], index = fine)
    
            # for i, j in fine:
            #     ih = self.History(self.Symbol(str(i)), 1000, Resolution.Daily)
            #     jh = self.History(self.Symbol(str(j)), 1000, Resolution.Daily)
            #     ir = ih.unstack(level = 1).close.transpose().pct_change().dropna()
            #     jr = jh.unstack(level = 1).close.transpose().pct_change().dropna()

            #     return pd.Series([coint(ir, ij)[1] < siglevel for columns, values \
            #                         in fine], index = fine)
        
        cointTested = cointTest(fine)
        coPairs = []
        for symbols, value in cointTested.iteritems():
            if valeue == True:
                coPairs.append(symbols)

        pairs = coPairs
        return pairs

        
    def OnSecuritiesChhanged(self, changes):
        self.changes = changes
        self.Log(f"OnSecuritiesChanged({self.UtcTime}):: {changes}")
        for security in changes.RemoveSecurities:
            self.Liquidate(security.Symbol)
        for security in changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 0.1)
    
    
    def OnData(self, pairs):
            for pair in self.pairs:
                # Calculate the spread of two price series.
                history0 = self.History(self.pairs[0], self.LookbackPeriod, Resolution.Daily)
                history1 = self.History(self.pairs[1], self.LookbackPeriod, Resolution.Daily)
                spread = np.array(history0) - np.array(history1)
                mean = np.mean(spread)
                std = np.std(spread)
                ratio = self.Portfolio[pair[0]].Price / self.Portfolio[pair[1]].Price

            # Long-short position is opened when pair prices have diverged by two standard deviations.
            weight = 1 / self.max_traded_pairs
            if spread[-1] > mean + 2*std:
                if not self.Portfolio[pair[0]].Invested and not self.Portfolio[pair[1]].Invested:
                    if len(self.traded_pairs) < self.max_traded_pairs:
                        self.SetHoldings(pair[0], -weight)
                        self.SetHoldings(pair[1], weight)
                        
                        if pair not in self.traded_pairs:
                            self.traded_pairs.append(pair)
                            
                elif self.Portfolio[pair[0]].Invested and self.Portfolio[pair[1]].Invested:
                    self.SetHoldings(pair[0], -weight)
                    self.SetHoldings(pair[1], weight)
            
            elif spread[-1] < mean - 2*std:
                if not self.Portfolio[pair[0]].Invested and not self.Portfolio[pair[1]].Invested:
                    if len(self.traded_pairs) < self.max_traded_pairs:
                        self.SetHoldings(pair[0], weight)
                        self.SetHoldings(pair[1], -weight)
    
                        if pair not in self.traded_pairs:
                            self.traded_pairs.append(pair)
                            
                elif self.Portfolio[pair[0]].Invested and self.Portfolio[pair[1]].Invested:
                    self.SetHoldings(pair[0], weight)
                    self.SetHoldings(pair[1], -weight)

            # The position is closed when prices revert back.
            else:
                if self.Portfolio[pair[0]].Invested and self.Portfolio[pair[1]].Invested:
                    self.Liquidate(pair[0]) 
                    self.Liquidate(pair[1])
                    
                    if pair in self.traded_pairs:
                        pairs_to_remove.append(pair)
            
            for pair in pairs_to_remove:
                self.traded_pairs.remove(pair)
            pairs_to_remove.clear()
            
            # self.Log(len(self.traded_pairs))
                
    def Distance(self, price_a, price_b):
        # Calculate the sum of squared deviations between two normalized price series.
        norm_a = np.array(price_a) / price_a[0]
        norm_b = np.array(price_b) / price_b[0]
        return sum((norm_a - norm_b)**2)
        
    def Selection(self):
        if self.month == 6:
            self.selection_flag = True
            
        self.month += 1
        if self.month > 12:
            self.month = 1
            
    def OnOrderEvent(self, orderEvent):
            order = self.Transactions.GetOrderById(orderEvent.OrderId)
            self.Debug(str(orderEvent.Symbol) + "{0}: {1}: {2}".format(self.Time, order.Type, orderEvent))
# Johansen test for >2 cointegrating levels
    # from statsmodels.tsa.vector_ar.vecm import coint_johansen, JohansenTestResult
        #endog = pd.concat([returnsi, returnsj], axis=1)
        #return 'trace stat:', coint_johansen(endog, -1, 1).lr1, 'crit val (90, 95, 99', coint_johansen(endog, -1, 1).cvt
        #return coint(returnsi, returnsj, trend='c', method='aeg', maxlag=None, \
                     #autolag='aic', return_results=True)
    '''