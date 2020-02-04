from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Perceptron

def PerceptronLearn(X, Y):
    print("GaussianNB en cours...")
    parameters = {'alpha' : [0.0001, 0.01, 0.0001],
                  'penalty' : ('l2', 'l1', 'elasticnet')}

    perceptron = Perceptron()
    clf = GridSearchCV(perceptron, parameters, cv=10)
    clf.fit(X, Y)

    print("Les meilleurs parametres de GaussianNB sont : ")
    print(clf.best_params_)
    print("Score : ")
    print(clf.best_score_)
    print("Résultats totaux : ")
    print(1-clf.cv_results_['mean_test_score'])
    print(clf.cv_results_['std_test_score'])

    #fichier = open("gaussian.txt", "w")
    #fichier.write(clf.best_params_)
    #fichier.close()
