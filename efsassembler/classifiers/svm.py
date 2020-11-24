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

from sklearn.svm import SVC
class Classifier(SVC):

    def __init__(self):
        super().__init__(gamma='auto', probability=True)