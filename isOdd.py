###### check sanidade, determinar se um número é par ou ímpar com redes neurais ######

#'/home/jaco/Projetos/numpyPureNNMNIST/data/test_iseven.csv'

import numpy as np

class simpleNN:
    def __init__(self,inputSize,nLayers,outputSize,nodesPerLayer,data,learning_rate,actfun):
        self.inputSize = inputSize
        self.nLayers = nLayers
        self.outputSize = outputSize
        self.nodesPerLayer = nodesPerLayer
        self.data = data
        self.learning_rate = learning_rate
        self.actfun = actfun

        weightList = []
        weight0List = []

        for idx,layer in enumerate(self.nodesPerLayer):
            if idx == 0:
                continue
            weights = np.random.normal(loc=0, scale=0.01, size=(self.nodesPerLayer[idx-1],layer))
            weightList.append(weights)

        for idx,layer in enumerate(self.nodesPerLayer):
            if idx == 0:
                continue
            weights = np.random.normal(loc=0, scale=0.01, size=(1,self.nodesPerLayer[idx]))
            weight0List.append(weights)

        self.weightList = weightList
        self.weight0List = weight0List


    def fit(self,epoch):

        for indice in range(0,len(self.data['X_train'])):

            ###### foward propag:

            for e in range(0,epoch):

                hiddenLayerResults = []

                for idx,layer in enumerate(self.nodesPerLayer):
                    if idx == 0:
                        continue
                
                    row = np.array([])
                    if idx == 1:
                        row = self.data['X_train'][indice]
                        row = row.reshape(1,row.size)
                        row_y = self.data['Y_train'][indice]
                    else:
                        row = hiddenLayerResults[idx-2]
                    hiddenLayerResults.append(actFun(self.weight0List[idx-1] + np.matmul(row,self.weightList[idx-1]),self.actfun))

                ####### back propag:

                errorLayerResults = []

                for idx,layer in reversed(list(enumerate(self.nodesPerLayer))):
                    if idx == 0:
                        continue

                    if idx == len(self.nodesPerLayer)-1:
                        error = np.multiply((hiddenLayerResults[idx-1] - row_y) , actFunDeriv(hiddenLayerResults[idx-1],self.actfun))
                        errorLayerResults.append(error)
                    else:
                        deriv = actFunDeriv(hiddenLayerResults[idx-1],self.actfun)
                        errorLayerResults.append(np.multiply(np.matmul(errorLayerResults[-1],self.weightList[idx].T),deriv))

                errorLayerResults.reverse()

                ######## update weights :

                for idx,layer in enumerate(self.nodesPerLayer):
                    if idx == 0:
                        continue

                    row = np.array([])
                    if idx == 1:
                        row = self.data['X_train'][indice]
                        row = row.reshape(1,row.size)
                        row_y = self.data['Y_train'][indice]
                    else:
                        row = hiddenLayerResults[idx-2]

                    self.weightList[idx-1] = self.weightList[idx-1] - (self.learning_rate * np.dot(row.T,errorLayerResults[idx-1]))

                print(f'current error: {errorLayerResults[-1].sum()} at epoch:{e} and sample: {indice}')

    def predict(self,input):

        hiddenLayerResults=[]

        for idx,layer in enumerate(self.nodesPerLayer):
            if idx == 0:
                continue
        
            row = np.array([])
            if idx == 1:
                row = self.data['labelmap_x'][input]
                row = row.reshape(1,row.size)
            else:
                row = hiddenLayerResults[idx-2]
            hiddenLayerResults.append(actFun(self.weight0List[idx-1] + np.matmul(row,self.weightList[idx-1]),self.actfun))

        arr=hiddenLayerResults[-1].tolist()[0]
        maxPost = arr.index(max(arr))

        return maxPost

    def evaluate(self):

        accu_erro = []
        hiddenLayerResults=[]

        for indice in range(0,len(self.data['X_test'])):

            for idx,layer in enumerate(self.nodesPerLayer):
                if idx == 0:
                    continue
            
                row = np.array([])
                if idx == 1:
                    row = self.data['X_test'][indice]
                    row = row.reshape(1,row.size)
                    row_y = self.data['Y_train'][indice]
                else:
                    row = hiddenLayerResults[idx-2]
                hiddenLayerResults.append(actFun(self.weight0List[idx-1] + np.matmul(row,self.weightList[idx-1]),self.actfun))

            errorVec = hiddenLayerResults[-1] - row_y
            accu_erro.append(errorVec.sum())

        return sum(accu_erro)

def getData(fileStr,split):

    '''
    Function that takes bla bla:
    outputs bla bla
    '''

    #read data from txt
    data = np.genfromtxt(fileStr, delimiter=',',skip_header=1)

    numberOfRows = data[0:,0].size
    numberOfColumns = data[0].size
    numberOfFeatColumns = numberOfColumns-1
    
    #encode data, only numerical features assumed
    #passar as colunas separadas para ca:
    data_x,labelmap_x = dtEncodOneHot(data=data[:,0]) #só passar 1 coluna
    data_y,labelmap_y = dtEncodOneHot(data=data[:,-1]) #só passar 1 coluna

    #get test size
    testsize = int(round(numberOfRows*split,0))

    #split randomly data, train and test
    indicesTest = np.random.choice(numberOfRows, testsize, replace=False)

    indicesNTest = []

    for n in range(0,numberOfRows):
        if n not in indicesTest:
            indicesNTest.append(n)

    indicesNTest = np.array(indicesNTest)

    X_train = data_x[indicesNTest]
    Y_train = data_y[indicesNTest]

    X_test = data_x[indicesTest]
    Y_test = data_y[indicesTest]

    inputSize = X_train[0].size
    outputSize = Y_train[0].size

    dataDic = {'X_train':X_train,'X_test':X_test,'Y_train':Y_train,'Y_test':Y_test,'data':data,'labelmap_x':labelmap_x,'labelmap_y':labelmap_y,'nFeatCols':numberOfFeatColumns
    ,'inputSize':inputSize,'outputSize':outputSize}

    return dataDic

def dtEncodOneHot(data):

    '''
    Function that takes bla bla:
    outputs bla bla
    '''

    #encode a column

    uniqueFeats = np.sort(np.unique(data))

    encodingDic = {}
    zeros = np.zeros((uniqueFeats.size, uniqueFeats.size))

    for idx,n in enumerate(zeros):
        n[idx] = 1.0        
        key = uniqueFeats[idx]
        encodingDic[key] = n

    dataEnc = np.empty((1,uniqueFeats.size), np.float64)

    for row in data:
        vec = encodingDic[row]
        dataEnc = np.append(dataEnc,vec.reshape(1,vec.size), axis=0)

    dataEnc = np.delete(dataEnc, obj=0,axis=0)

    return dataEnc,encodingDic

def softMax(arr):

    e = 2.718281828459045
    sum = 0
    maxList = []
    for m in arr:
        for n in arr:
            sum = sum + e**n
        
        result = np.array(m/sum)
        result = np.log(result)

        maxList.append(result)

    return max(maxList),maxList.index(max(maxList)),maxList

def actFunDeriv(transf,type_):

    e = 2.718281828459045
    returnVec = []
    returnVec2D = []

    if type(transf).__module__=='numpy':
        for n in transf:
            for cell in n:
                if type_ == 'ReLU':
                    if cell < 0:
                        returnVec.append(0)
                    else:
                        returnVec.append(1)

                if type_ == 'sigmoid':
                    returnVec.append(cell * (1.0 - cell))

            returnVec2D.append(returnVec)
            returnVec = []

        return np.array(returnVec2D)

def actFun(transf,type_):

    e = 2.718281828459045
    returnVec = []
    returnVec2D = []

    if type(transf).__module__=='numpy':
        for n in transf:
            for cell in n:
                if type_ == 'ReLU':
                    if cell < 0:
                        returnVec.append(0)
                    else:
                        returnVec.append(cell)

                if type_ == 'sigmoid':
                    returnVec.append(1/(1+e**(-cell)))

            returnVec2D.append(returnVec)
            returnVec = []

        return np.array(returnVec2D)

def main():

    FILEPATH = '/home/jaco/Projetos/numpyPureNNMNIST/data/test_iseven.csv'

    dataDic = getData(fileStr=FILEPATH,split=0.2)

    nLayers = 1
    nodesPerLayer = np.array([dataDic['inputSize'],16,dataDic['outputSize']])

    neuralNet = simpleNN(inputSize=dataDic['inputSize'],nLayers=nLayers,outputSize=dataDic['outputSize'],nodesPerLayer=nodesPerLayer,data=dataDic,learning_rate=0.05,actfun='ReLU')

    neuralNet.fit(epoch=10)

    print(neuralNet.predict(1))
    print(neuralNet.evaluate())
    print('pred')

    #unitActivation(neuralNet,dataDic,cLayer=2)

    return None

if __name__ == '__main__':
    main()