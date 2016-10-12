__author__ = 'suvrat'

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing

class DataFetcher:
    dataFile = 0
    trainData = 0
    testData = 0
    features = 0
    scaler = 0
    totalData = 0
    trainPrice = 0
    testPrice = 0
    CLOSE = 'close'
    def __init__(self, filename, trainFile, testFile, splitPercentage = .8):
        self.dataFile = filename
        self.initProcessing(trainFile, testFile, splitPercentage)

    def readInit(self, trainFile, testFile):
        self.trainData = pd.read_pickle(trainFile)
        self.testData = pd.read_pickle(testFile)

    def initProcessing(self, trainFile, testFile, splitPercentage):
        self.readData()
        self.splitTrainTest(splitPercentage = splitPercentage)
        self.trainPrice = trainFile + 'price'
        self.testPrice = testFile + 'price'
        self.trainPrice.to_csv(self.trainPrice, columns=[self.CLOSE])
        self.testPrice.to_csv(self.testPrice, columns=[self.CLOSE])
        self.buildFeatures()
        self.saveFinalTrainAndTest(trainFile, testFile)

    def readData(self):
        self.totalData = pd.read_csv(self.dataFile, sep=",", skiprows=0, header=0, index_col=0, parse_dates=True,
                                names=['date', 'open', 'high', 'low', self.CLOSE, 'vol', 'total'])


    def splitTrainTest(self, splitPercentage = .8):
        '''
        Will split dataFile into train and test file
        :param splitPercentage:
        :return:
        '''
        trainSize = int(len(self.totalData) * (1-splitPercentage))
        self.trainData = self.totalData.iloc[:trainSize,]
        self.testData = self.totalData.iloc[trainSize:,]

    def fetchCompleteTrainData(self):
        '''
        Returns complete training data in pandas format
        :return:
        '''

        return self.trainData

    def fetchCompleteTestData(self):
        '''
        Return complete test data in pandas format
        :return:
        '''
        return self.testData

    def buildFeatures(self):
        '''
        Build features for the data set
        :return:
        '''
        self.trainData = self.trainData[[self.CLOSE]]
        self.trainData['diff'] = self.trainData[[self.CLOSE]].diff(periods=1, axis=0)
        self.trainData['diff'] = self.trainData['diff'].fillna(0)
        self.scaler = preprocessing.StandardScaler()
        self.trainData = self.scaler.fit_transform(self.trainData)
        self.testData = self.testData[[self.CLOSE]]
        self.testData['diff'] = self.testData[[self.CLOSE]].diff(periods=1, axis=0)
        self.testData['diff'] = self.testData['diff'].fillna(0)
        self.testData = self.scaler.transform(self.testData)
        return

    def saveFinalTrainAndTest(self, trainFile, testFile):
        '''
        Save final files for reusing
        :return:
        '''
        pd.DataFrame(self.trainData).to_pickle(trainFile)
        pd.DataFrame(self.testData).to_pickle(testFile)
        return