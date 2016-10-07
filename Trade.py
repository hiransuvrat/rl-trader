__author__ = 'suvrat'

from ModelBuilder import Model
from DataFetcher import DataFetcher

dataFetcher = DataFetcher('data/nifty_1500.csv')
dataFetcher.initProcessing('data/train.pkl', 'data/test.pkl')
modelBuilder = Model(dataFetcher)
modelBuilder.setModel()
modelBuilder.trainModel()
