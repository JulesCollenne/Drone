from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB

def GaussianNBLearn(X, Y):
    print("GaussianNB en cours...")
    parameters = {'priors': ((0.6,0.4), (0.7, 0.3), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.01, 0.99), (0.001, 0.999)),
                  'var_smoothing': [0.0, 10, 0.2]}

    bayes = GaussianNB()
    clf = GridSearchCV(bayes, parameters, cv=10)
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

#Les meilleurs parametres de GaussianNB sont :
#{'priors': (0.6, 0.4), 'var_smoothing': 2.5}
#Score :
#0.7850830677098426

def GaussianNBPredict(X_train, y_train, X_test):
    fichier = open("gaussian.txt", "r")
    params = fichier.read()
    bayes = GaussianNB(params)
    bayes.fit(X_train, y_train)
    return bayes.predict(X_test)

def kppvBestParam(X, y):
    print("KNN en cours...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    bayes = GaussianNB(priors=20, var_smoothing=2.5)
    bayes.fit(X_train, y_train)
    print("Score : ")
    print(bayes.score(X_test, y_test))
    print("Résultats totaux : ")
    print()
