from Code.tpdroneutils import *
from kppv import *
from bayes import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix

def learn():
    path1 = "Data/Mer/"
    path2 = "Data/Ailleurs/"

    print("Chargement des vecteurs images")
    nV, dataV, targetV, sizeV = chargementVecteursImages(path1, path2, 1, -1, 64)
    print("Chargement des histogrammes")
    nH, dataH, targetH, sizeH = chargementHistogrammesImages(path1, path2, 1, -1)
    print("Conversion en np array")
    #.............Vecteurs
    dataV = np.array(dataV)
    dataV = dataV.reshape((nV, sizeV))
    targetV = np.array(targetV)
    #kppvLearn(dataV, targetV)
    #kppvBestParam(dataV, targetV)

    #...........Histogrammes

    dataH = np.array(dataH)
    dataH = dataH.reshape((nH, sizeH))
    targetH = np.array(targetH)

    #GaussianNBLearn(dataV, targetV)
    #GaussianNBLearn(dataH, targetH)

    #decisionTreeLearn(dataV, targetV)
    #decisionTreeLearn(dataH, targetH)

    kppvLearn(dataV, targetV)
    kppvLearn(dataH, targetH)

    #SVCLearn(dataV, targetV)
    #SVCLearn(dataH, targetH)

    #PerceptronLearn(dataV, targetV)
    #PerceptronLearn(dataH, targetH)

def predict():
    path1 = "Data/Mer/"
    path2 = "Data/Ailleurs/"
    data = np.load("BinTest/tPixel-60.npy")
    nV, dataV, targetV, sizeV = chargementVecteursImages(path1, path2, 1, -1, 60)
    dataV = np.array(dataV)
    dataV = dataV.reshape((nV, sizeV))
    targetV = np.array(targetV)

    dataV_sparse = coo_matrix(dataV)
    dataV, dataV_sparse, targetV = shuffle(dataV, dataV_sparse, targetV)

    results = KPPVPredict(dataV, targetV, data)
    print(results)

#learn()
predict()
