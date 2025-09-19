class Model:
    """
    A wrapper class for the WeibullAFTFitter to ensure a consistent `.predict()`
    interface for the CodaBench competition.
    """
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """Predicts the median survival time, which serves as a survival prediction."""
        return self.model.predict_median(X)
