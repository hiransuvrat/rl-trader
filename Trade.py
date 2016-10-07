__author__ = 'suvrat'

from ModelBuilder import Model
from DataFetcher import DataFetcher

dataFetcher = DataFetcher('data/nifty_1500.csv', 'data/train.pkl', 'data/test.pkl', splitPercentage=.8)
modelBuilder = Model(dataFetcher)
modelBuilder.setModel()
modelBuilder.trainModel()
