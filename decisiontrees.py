from sklearn import tree
from sklearn.model_selection import GridSearchCV


def decisionTreeLearn(X, Y):
    print("DT en cours...")
    parameters = { 'criterion' : ('gini', 'entropy'),
                   'max_depth' : [10,100,10] }
    dt = tree.DecisionTreeClassifier()
    clf = GridSearchCV(dt, parameters, scoring='f1', cv=10)
    clf.fit(X, Y)

    print("Les meilleurs parametres du DT sont : ")
    print(sorted(clf.cv_results_.keys()))
    print(sorted(clf.cv_results_))
