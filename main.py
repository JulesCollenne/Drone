from Code.tpdroneutils import *
from kppv import *
from bayes import *

path1 = "Data/Mer/"
path2 = "Data/Ailleurs/"

print("Chargement des vecteurs images")
nV, dataV, targetV, sizeV = chargementVecteursImages(path1, path2, 1, -1, 50)
print("Chargement des histogrammes")
nH, dataH, targetH, sizeH = chargementHistogrammesImages(path1, path2, 1, -1)
print("Conversion en np array")
dataV = np.array(dataV)
dataV = dataV.reshape((nV, sizeV))
targetV = np.array(targetV)
#targetV = targetV.reshape((sizeV))

#kppvLearn(dataV, targetV)
#kppvBestParam(dataV, targetV)

dataH = np.array(dataV)
dataH = dataV.reshape((nV, sizeV))
targetH = np.array(targetV)
print(dataH.shape)

GaussianNBLearn(dataV, targetV)
GaussianNBLearn(dataH, targetH)