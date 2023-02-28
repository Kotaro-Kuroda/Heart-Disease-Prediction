import read_csv
from sklearn.svm import LinearSVC


def train(x_train, y_train):
    clf = LinearSVC(penalty='l2', loss='squared_hinge', verbose=True, dual=True, max_iter=10000)
    clf.fit(x_train, y_train)
    return clf


def main():
    csv_path = './heart.csv'
    x_train, x_test, y_train, y_test = read_csv.read_csv(csv_path)
    clf = train(x_train, y_train)
    score = clf.score(x_test, y_test)
    print(score)


if __name__ == '__main__':
    main()
