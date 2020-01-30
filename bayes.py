from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB

def GaussianNBLearn(X, Y):
    print("GaussianNB en cours...")
    parameters = {'priors': ((0.6,0.4), (0.7, 0.3)),
                  'var_smoothing': [0.5, 2.5, 0.2]}

    bayes = GaussianNB()
    clf = GridSearchCV(bayes, parameters, scoring='f1', cv=10)
    clf.fit(X, Y)

    print("Les meilleurs parametres de GaussianNB sont : ")
    print(clf.best_params_)
    print("Score : ")
    print(clf.best_score_)

#Les meilleurs parametres de GaussianNB sont :
#{'priors': (0.6, 0.4), 'var_smoothing': 2.5}
#Score :
#0.7850830677098426

def kppvBestParam(X, y):
    print("KNN en cours...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    bayes = GaussianNB(priors=20, var_smoothing=2.5)
    bayes.fit(X_train, y_train)
    print("Score : ")
    print(bayes.score(X_test, y_test))