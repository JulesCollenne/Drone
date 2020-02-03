from sklearn import tree
from sklearn.model_selection import GridSearchCV

def decisionTreeLearn(X, Y):
    print("DT en cours...")
    parameters = { 'criterion' : ('gini', 'entropy'),
                   'max_depth' : [1,1000,10],
                   'splitter' : ('best', 'random')}

    dt = tree.DecisionTreeClassifier()
    clf = GridSearchCV(dt, parameters, scoring='f1', cv=10)
    clf.fit(X, Y)

    print("Les meilleurs parametres de GaussianNB sont : ")
    print(clf.best_params_)
    print("Score : ")
    print(1-clf.best_score_)
    print("Résultats totaux : ")
    print(1-clf.cv_results_['mean_test_score'])
    print(clf.cv_results_['std_test_score'])
