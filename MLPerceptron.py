import numpy as np

class MLP:
    __eta = 0.005
    __a = 1
    __b = 1

    def __init__(self,
                 inp: list,
                 out: list,
                 neuronNum: tuple = (4, 3)
                 ):
        # Количество слоев
        self.__layers = 2 + len(neuronNum)
        # Количество нейронов на каждом слое
        nN = [len(inp[0]), len(out[0])]
        self.__nN = np.insert(nN, 1, neuronNum)
        self.__inp = np.array(inp)
        self.__out = np.array(out)
        self.__w = [np.random.rand(
            self.__nN[i] + 1,
            self.__nN[i + 1] +
            (0 if i == self.__layers - 2 else 1)
        ) for i in range(self.__layers - 1)]


    def nonLinAct(self, x):
        return np.array(self.__a * np.tanh(self.__b * x))


    def nonLinActDer(self, x):
        return np.array(self.__b / self.__a * \
                        (self.__a - self.nonLinAct(x)) * \
                        (self.__a + self.nonLinAct(x)))


    def linAct(self, x):
        return np.array(x)


    def linActDer(self, x):
        return np.array(1)


    def learn(self,
              epoches = 1000,
              epsilon = 0.002):
        #Слои
        l = np.array([None for i in range(self.__layers)])
        #Ошибки
        l_err = np.array([None for i in range(1, self.__layers)])
        #deltas
        l_delta = np.array([None for i in range(1, self.__layers)])

        #Переобозначение входов и выходов для простоты записи
        inp = self.__inp
        out = self.__out
        #Счетчик эпох
        k = 0
        #Полная ошибка сети
        err_n = epsilon+1
        #Начало процесса обучения
        while k < epoches and err_n > epsilon:
            err_n = 0
            k += 1
            #Проход по обучающей выборке
            for i in range(len(inp)):
                l[0] = np.array([np.insert(inp[i], 0, 1)]) #Попробовать вынести за цикл while
                #Прямой проход по сети
                for j in range(1, self.__layers-1):
                    #проход по скрытым слоям
                    l[j] = self.nonLinAct(
                        np.dot(l[j-1], self.__w[j-1])
                    )
                #вычисление значения на выходном слое
                l[self.__layers-1] = self.linAct(
                    np.dot(l[self.__layers-2], self.__w[self.__layers-2])
                )
                #Обратный проход по сети
                l_err[self.__layers-2] = out - l[self.__layers-1]