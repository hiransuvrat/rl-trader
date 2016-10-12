__author__ = 'suvrat'

from DataFetcher import DataFetcher
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam
import random
import pandas as pd
import numpy as np
from backtest import Backtest

class Model:
    CLOSE = 'close'
    priceData = 0
    initialStateTrain = 0
    initialStateTest = 0
    trainData = 0
    testData = 0
    tSteps = 1
    batchSize = 20
    numFeatures = 2
    model = 0
    epochs = 30 #100
    gamma = 0.95 #since the reward can be several time steps away, make gamma high
    epsilon = 1
    buffer = 100
    replay = []
    learningProgress = []
    signal = 0
    noActions = 4


    def __init__(self, dataFetcher, filename):
        self.setTrainData(dataFetcher)
        self.setTestData(dataFetcher)
        self.priceData = pd.read_csv(filename, sep=",", skiprows=0, header=0, index_col=0, parse_dates=True,
                                names=['date', 'open', 'high', 'low', self.CLOSE, 'vol', 'total'], usecols=[self.CLOSE])
        self.signal = pd.Series(index=np.arange(len(self.trainData)))
        self.signal.fillna(value=0, inplace=True)
        self.initialStateTrain = self.getState(self.trainData[0,:])

    def setTrainData(self, dataFetcher):
        self.trainData = dataFetcher.fetchCompleteTrainData()

    def setTestData(self, dataFetcher):
        self.testData = dataFetcher.fetchCompleteTestData()

    def preProrcessData(self):
        '''
        Run all preprocessing on data required
        :return:
        '''

    def getState(self, record):
        return np.array([[record.tolist()]])

    def setModel(self):
        self.model = Sequential()
        self.model.add(LSTM(4,
               input_shape=(1, self.numFeatures),
               return_sequences=False,
               stateful=False))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(4, init='lecun_uniform'))
        self.model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        rms = RMSprop()
        adam = Adam()
        self.model.compile(loss='mse', optimizer=adam)

    def trainModel(self):
        for i in range(self.epochs):
            status = 1
            terminalState = 0
            timeStep = 1
            state = self.initialStateTrain
            while (status == 1):
                qVal = self.model.predict(state, batch_size=1)

                if (self.chooseExplore()):
                    action = np.random.randint(0, self.noActions)
                else:
                    action = np.argmax(qVal)

                #Take action move to next state
                newState, timeStep, terminalState = self.performAction(state, action, timeStep)

                #Observe reward
                reward = self.getReward(newState, timeStep, action, terminalState)

                state = newState
                if terminalState == 1:
                    status = 0

        if self.epsilon > 0.1: #decrement epsilon over time
            self.epsilon -= (1.0/self.epochs)

    def testModel(self):
        for i in range(len(self.testData)):
            state = self.getState(self.testData[i, :])
            qVal = self.model.predict(state, batch_size=1)
            action = np.argmax(qVal)
            print(qVal, action)

    def getReward(self, newState, timeStep, action, terminalState, eval=False, epoch=0):
        reward = 0

        if eval == False:
            reward = (self.signal[timeStep-1:timeStep] - self.signal[timeStep-2:timeStep])
            #bt = Backtest(pd.Series(data=[x for x in self.trainData[timeStep-2:timeStep]],
            #                        index=self.signal[timeStep-2:timeStep].index.values),
            #              self.signal[timeStep-2:timeStep], initialCash = 100000, signalType='capital')
            #reward = ((bt.data['price'].iloc[-1] - bt.data['price'].iloc[-2])*bt.data['shares'].iloc[-1])

        if terminalState == 1 and eval:
            reward = (self.signal[timeStep-1:timeStep] - self.signal[timeStep-2:timeStep])
            #bt = Backtest(pd.Series(data=[x for x in self.trainData], index=self.signal.index.values),
            #                  self.signal, signalType='shares')
            #reward = bt.pnl.iloc[-1]

        return reward

    def performAction(self, state, action, timeStep):
        '''

        :param state:
        :param action:
        :param timeStep:
        :return:
        '''

        timeStep += 1
        terminalState = 0
        state = self.getState(self.trainData[timeStep, :])
        if timeStep + 1 == len(self.trainData):
            terminalState = 1
            self.signal.loc[timeStep] = 0
            return state, timeStep, terminalState

        if action == 1:
            self.signal.loc[timeStep] = 1
        elif action == 2:
            self.signal.loc[timeStep] = -1
        else:
            self.signal.loc[timeStep] = 0

        return state, timeStep, terminalState

    def chooseExplore(self, algo='epsilonGreedy'):
        if (algo == 'epsilonGreedy'):
            if (random.random() < self.epsilon):
                return True
            else:
                return False

