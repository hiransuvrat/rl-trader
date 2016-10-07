__author__ = 'suvrat'

from ModelBuilder import Model
from DataFetcher import DataFetcher

dataFetcher = DataFetcher('data/nifty_1500.csv').initProcessing('data/train.pkl', 'data/test/pkl')

