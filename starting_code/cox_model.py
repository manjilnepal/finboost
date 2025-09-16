class Model:
    """
    A wrapper class for the CoxPHFitter to ensure a consistent `.predict()`
    interface that returns survival predictions (higher is better).
    """
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """
        Predicts partial hazard (a risk score) and negates it to return a
        survival prediction.
        """
        return -self.model.predict_partial_hazard(X)
