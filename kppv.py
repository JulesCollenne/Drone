from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split


def kppvLearn(X, Y):
    print("KNN en cours...")
    parameters = {'n_neighbors': [1, 20, 2],
                  'weights': ('uniform', 'distance')}

    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters, scoring='f1', cv=10)
    clf.fit(X, Y)

    print("Les meilleurs parametres de KNN sont : ")
    print(clf.best_params_)
    print("Score : ")
    print(clf.best_score_)

# Les meilleurs parametres de KNN sont :
# {'n_neighbors': 20, 'weights': 'distance'}
# Score :
# 0.7897093952241011
#
def kppvBestParam(X, y):
    print("KNN en cours...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=20, weights='distance')
    knn.fit(X_train, y_train)
    print("Score : ")
    print(knn.score(X_test, y_test))

def kppvPredict():
    print("lol")
