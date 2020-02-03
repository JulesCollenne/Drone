from Code.tpdroneutils import *
from decisiontrees import decisionTreeLearn
from kppv import *
from bayes import *
from svc import SVCLearn


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

    #kppvLearn(dataV, targetV)
    #kppvLearn(dataH, targetH)

    SVCLearn(dataV, targetV)
    SVCLearn(dataH, targetH)

def predict():
    path1 = "Data/Mer/"
    path2 = "Data/Ailleurs/"
    data = np.load("BinTest/tHisto.npy")
    nH, dataH, targetH, sizeH = chargementHistogrammesImages(path1, path2, 1, -1)
    dataH = np.array(dataH)
    dataH = dataH.reshape((nH, sizeH))
    targetH = np.array(targetH)
    results = GaussianNBPredict(dataH, targetH, data)
    print(results)

learn()
#predict()
