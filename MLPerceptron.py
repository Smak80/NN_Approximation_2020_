import numpy as np

import data_loader as DataLoader


class MLP:
    __eta = 0.005
    __a = 1
    __b = 1

    def __init__(self,
                 ld: DataLoader.loader,
                 neuronNum: tuple = (4, 3)
                 ):
        # Количество слоев
        self.__layers = 2 + len(neuronNum)
        # Количество нейронов на каждом слое
        inp = ld.getTrainInp()
        out = ld.getTrainOut()
        nN = [len(inp[0]), len(out[0])]
        self.__nN = np.insert(nN, 1, neuronNum)
        self.__inp = np.array(inp)
        self.__out = np.array(out)
        self.__tst_inp = np.array(ld.getTestInp())
        self.__tst_out = np.array(ld.getTestOut())
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
              epoches=1000,
              epsilon=0.002):
        e_full_tr = []
        e_full_ts = []
        # Индуцированное локальное поле
        v = np.array([None for i in range(self.__layers)])
        # Слои (выходы из слоёв)
        l = np.array([None for i in range(self.__layers)])
        # Ошибки
        l_err = np.array([None for i in range(1, self.__layers)])
        # deltas
        l_delta = np.array([None for i in range(1, self.__layers)])

        # Переобозначение входов и выходов для простоты записи
        inp = self.__inp
        out = self.__out
        # Счетчик эпох
        k = 0
        # Полная ошибка сети на тестовой выборке
        err_n = epsilon + 1
        # Полная ошибка сети на обучающей выборке
        tr_err_n = epsilon + 1
        # Начало процесса обучения
        while k < epoches and err_n > epsilon:
            err_n = 0
            tr_err_n = 0
            k += 1
            # Проход по обучающей выборке
            for i in range(len(inp)):
                l[0] = np.array([np.insert(inp[i], 0, 1)])  # Попробовать вынести за цикл while
                # Прямой проход по сети
                for j in range(1, self.__layers - 1):
                    # индуцированное локальное поле
                    v[j] = np.dot(l[j - 1], self.__w[j - 1])
                    # проход по скрытым слоям
                    l[j] = self.nonLinAct(v[j])
                # вычисление индуцированного локального поля на выходном слое
                v[self.__layers - 1] = np.dot(l[self.__layers - 2], self.__w[self.__layers - 2])
                # вычисление значения на выходном слое
                l[self.__layers - 1] = self.linAct(
                    v[self.__layers - 1]
                )
                # Обратный проход по сети
                l_err[self.__layers - 2] = out[i] - l[self.__layers - 1]
                # Накопление общей ошибки сети на данной эпохе
                tr_err_n += 0.5 * (out[i] - l[self.__layers - 1]) ** 2
                # Нахождение \delta_k
                l_delta[self.__layers - 2] = \
                    np.array([l_err[self.__layers - 2][0] * (
                        self.linActDer(v[self.__layers - 1])
                    )])
                # Нахождение \delta_j
                for j in range(self.__layers - 2, 0, -1):
                    l_err[j - 1] = np.dot(l_delta[j], self.__w[j].T)
                    l_delta[j - 1] = l_err[j - 1] * self.nonLinActDer(v[j])
                # Определение изменения весовых коэффициентов \Delta w
                deltaW = [self.__eta * np.dot(l_delta[j].T, l[j])
                          for j in range(self.__layers - 1)]
                for j in range(0, self.__layers - 1):
                    self.__w[j] += deltaW[j].T
            # Вычисление общей ошибки сети для данной эпохи
            tr_err_n /= len(inp)
            outt = self.calc(self.__tst_inp)
            ln = len(outt)
            soutt = np.array([self.__tst_out[i][0] for i in range(len(self.__tst_out))])
            err_n = np.sum(0.5 * (soutt - outt) ** 2) / ln
            e_full_tr.append(tr_err_n)
            e_full_ts.append(err_n)
            print("Epoche", k, "Error (train)=", tr_err_n, "Error test=", err_n)
        return e_full_tr, e_full_ts

    #Вычисление выходов по входом обученной сетью
    def calc(self, inps):
        outs = np.array([])
        # Для каждого входного значения
        for i in range(len(inps)):
            inp = np.array([np.insert(inps[i], 0, 1)])
            # Прямой проход по сети (все слои, кроме последнего)
            for lr in range(self.__layers - 2):
                inp = self.nonLinAct(np.dot(inp, self.__w[lr]))
            # Получение результата на последнем слое
            # и добавлени его в массив выходов
            outs = np.append(outs,
                             self.linAct(np.dot(inp, self.__w[self.__layers-2]))
                            )
        return outs