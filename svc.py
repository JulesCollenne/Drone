from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def SVCLearn(X, Y):
    print("SVC en cours...")
    parameters = { 'C' : [0.1,10,0.5],
                   'kernel' : ('rbf', 'linear', 'poly', 'sigmoid')}
    svc = SVC()
    clf = GridSearchCV(svc, parameters, scoring='f1', cv=10)
    clf.fit(X, Y)

    print("Les meilleurs parametres de GaussianNB sont : ")
    print(clf.best_params_)
    print("Score : ")
    print(1-clf.best_score_)
    print("Résultats totaux : ")
    print(1-clf.cv_results_['mean_test_score'])
    print(clf.cv_results_['std_test_score'])
