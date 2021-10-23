"""

    A personalized classifier must implement the following 
    scikit-learn methods:

    1   .fit(X, y)
    2   .predict_proba(X)
    3   .predict(X)

    Its class name must be Classifier.

    The constructor method must work without any argumets.
    If you want to work with unimplemented sklearn classifiers,
    you'll probably just need to wirite a similar code to 
    this one.

"""

from sklearn.ensemble import RandomForestClassifier as RFC
class Classifier(RFC):

    def __init__(self):
        super().__init__()