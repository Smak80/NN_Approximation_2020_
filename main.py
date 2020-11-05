from matplotlib import pyplot as plt

import data_loader as dl
from MLPerceptron import MLP


def plot2d():
    ld = dl.loader(trainPercent=75)
    tri = ld.getTrainInp()
    tro = ld.getTrainOut()
    tsi = ld.getTestInp()
    tso = ld.getTestOut()
    mlp = MLP(ld, (12, 30))
    e_tr, e_ts = mlp.learn(2500, epsilon=0.002)
    e_tr = [e_tr[i][0][0] for i in range(len(e_tr))]
    e_tr_x = [i for i in range(1, len(e_tr) + 1)]
    e_ts_x = [i for i in range(1, len(e_ts) + 1)]
    f = plt.figure()
    fa1 = f.add_subplot(3, 1, 1)
    fa1.plot(tri, tro, "r+")
    out = mlp.calc(tri)
    fa1.plot(tri, out, "bo")
    out = mlp.calc(tsi)
    fa1.plot(tsi, tso, "yo")
    fa1.plot(tsi, out, "go")
    fa2 = f.add_subplot(3, 1, 2)
    fa2.plot(e_tr_x, e_tr, "r-")
    fa3 = f.add_subplot(3, 1, 3)
    fa3.plot(e_ts_x, e_ts, "b-")
    plt.show()


plot2d()