# Logistic Regression

from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

models ={
    'logistic_regression': LogisticRegression,
    'kn_classifier': KNeighborsClassifier,
    'decision_tree_classifier': DecisionTreeClassifier
}

def load_model(ml_model):
    # fit a logistic regression model to the data
    # model = LogisticRegression()

    # fit a k-nearest neighbor model to the data
    model = ml_model()
    return model


def run_model(model):

    # load the iris datasets
    dataset = datasets.load_iris()

    model.fit(dataset.data, dataset.target)
    print(model)
    # make predictions
    expected = dataset.target
    predicted = model.predict(dataset.data)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


if __name__ == "__main__":
    model = load_model(models['kn_classifier'])
    run_model(model)
