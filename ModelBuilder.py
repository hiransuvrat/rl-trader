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
    noActions = 2
    testPrice = 0
    result = 0
    dataFetcher = 0

    def __init__(self, dataFetcher, filename):
        self.dataFetcher = dataFetcher
        self.setTrainData(dataFetcher)
        self.setTestData(dataFetcher)
        self.priceData = pd.read_csv(dataFetcher.trainPriceFile)
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
        #self.model.add(Dropout(0.5))

        self.model.add(Dense(self.noActions, init='lecun_uniform'))
        self.model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        rms = RMSprop()
        adam = Adam()
        self.model.compile(loss='mse', optimizer=adam)

    def trainModel(self):
        replay = []
        buffer = 40
        h = 0
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



                                #Experience replay storage
                if (len(replay) < buffer): #if buffer not filled, add to it
                    replay.append((state, action, reward, newState))
                    #print(time_step, reward, terminal_state)
                else: #if buffer full, overwrite old values
                    if (h < (buffer-1)):
                        h += 1
                    else:
                        h = 0
                    replay[h] = (state, action, reward, newState)
                    #randomly sample our experience replay memory
                    minibatch = random.sample(replay, self.batchSize)
                    X_train = []
                    y_train = []
                    for memory in minibatch:
                        #Get max_Q(S',a)
                        old_state, action, reward, new_state = memory
                        old_qval = self.model.predict(old_state, batch_size=1)
                        newQ = self.model.predict(new_state, batch_size=1)
                        maxQ = np.max(newQ)
                        y = np.zeros((1,self.noActions))
                        y[:] = old_qval[:]
                        if terminalState == 0: #non-terminal state
                            update = (reward + (self.gamma * maxQ))
                        else: #terminal state
                            update = reward
                        y[0][action] = update
                        #print(time_step, reward, terminal_state)
                        X_train.append(old_state)
                        y_train.append(y.reshape(self.noActions,))

                    X_train = np.squeeze(np.array(X_train), axis=(1))
                    y_train = np.array(y_train)
                    self.model.fit(X_train, y_train, batch_size=self.batchSize, nb_epoch=1, verbose=0)

                    state = newState
                if terminalState == 1: #if reached terminal state, update epoch status
                    status = 0

        if self.epsilon > 0.1: #decrement epsilon over time
            self.epsilon -= (1.0/self.epochs)

    def testModel(self):
        testSize = len(self.testData)
        qValRes0 = []
        qValRes1 = []
        for i in range(len(self.testData)):
            state = self.getState(self.testData[i, :])
            qVal = self.model.predict(state, batch_size=1)
            action = np.argmax(qVal)
            qValRes0.append(qVal[0, 0])
            qValRes1.append(qVal[0, 1])
        self.result = pd.DataFrame(data={'0':qValRes0, '1':qValRes1})

    def doTestCompare(self):
        self.priceTestData = pd.read_csv(self.dataFetcher.testPriceFile)
        self.priceTestData = pd.concat([self.priceTestData, self.result], axis=1)



    def getReward(self, newState, timeStep, action, terminalState, eval=False, epoch=0):
        reward = 0

        if eval == False:
            print self.priceData
            reward = self.priceData.ix[timeStep-1, self.CLOSE] - self.priceData[timeStep-2, self.CLOSE]
            print 'reawrd', reward
            #bt = Backtest(pd.Series(data=[x for x in self.trainData[timeStep-2:timeStep]],
            #                        index=self.signal[timeStep-2:timeStep].index.values),
            #              self.signal[timeStep-2:timeStep], initialCash = 100000, signalType='capital')
            #reward = ((bt.data['price'].iloc[-1] - bt.data['price'].iloc[-2])*bt.data['shares'].iloc[-1])

        if terminalState == 1 and eval:
            reward = self.priceData[timeStep-1, self.CLOSE] - self.priceData[timeStep-2, self.CLOSE]
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
            self.signal.loc[timeStep] = 1
            return state, timeStep, terminalState

        if action == 1:
            self.signal.loc[timeStep] = 1
        else:
            self.signal.loc[timeStep] = -1


        return state, timeStep, terminalState

    def chooseExplore(self, algo='epsilonGreedy'):
        if (algo == 'epsilonGreedy'):
            if (random.random() < self.epsilon):
                return True
            else:
                return False

