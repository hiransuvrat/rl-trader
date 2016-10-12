__author__ = 'suvrat'

from ModelBuilder import Model
from DataFetcher import DataFetcher

inputFile = 'data/nifty_1500.csv'
dataFetcher = DataFetcher('data/nifty_1500.csv', 'data/train.pkl', 'data/test.pkl', splitPercentage=.8)
modelBuilder = Model(inputFile)
modelBuilder.setModel()
modelBuilder.trainModel()
