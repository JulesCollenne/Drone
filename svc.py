from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def SVCLearn(X, Y):
    print("SVC en cours...")
    parameters = { 'C' : (),
                   'max_depth' : [10,100,10] }
    svc = SVC()
    clf = GridSearchCV(svc, parameters, scoring='f1', cv=10)
    clf.fit(X, Y)

    print("Les meilleurs parametres du SVC sont : ")
    print(clf.cv_results_)