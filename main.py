import data_loader as dl
from MLPerceptron import MLP
from matplotlib import pyplot as plt

def plot2d():
    ld = dl.loader()
    tri = ld.getTrainInp()
    tro = ld.getTrainOut()
    tsi = ld.getTestInp()
    tso = ld.getTestOut()
    mlp = MLP(tri, tro, (50, 25))
    mlp.learn(2500, 0.005)
    plt.plot(tri, tro, "r+")
    out = mlp.calc(tri)
    plt.plot(tri, out, "b-")
    out = mlp.calc(tsi)
    plt.plot(tsi, tso, "yo")
    plt.plot(tsi, out, "go")
    plt.show()


plot2d()