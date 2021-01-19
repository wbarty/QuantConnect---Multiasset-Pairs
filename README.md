# QuantConnect---Multiasset-Pairs

Pairs Trading algorithm designed to trade any variety of assets, 
the algorithm uses the ADF test for stationarity and Kendalls Tau
correlation coefficient for a broad spectrum filter. For the filtered
asset pairs, the Engle-Granger test for cointegration is used, in future
I would like to test for cointegration between assets in groups
larger than 2 using the Johansen Test. Once the asset pairs are
loaded they are tracked and traded when one of the assets moves
2 or more sd from its long-run mean, excecuting a basic long/
short mean reversion strategy and ideally catpuring statisical 
arbritage.