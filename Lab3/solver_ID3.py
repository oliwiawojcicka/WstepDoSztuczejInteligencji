from solver import Solver
from model import DecisionTreeClassifier

class solver_ID3(Solver):

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def get_parameters(self):
        return {
            "max_depth": self.max_depth,
            "tree": self.model.tree
        }

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
