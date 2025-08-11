import logging
import pytest
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator

from vortexclust.models.validation import *

logger = logging.getLogger(__name__)

def test_check_unfitted_model():
    model = KMeans(n_clusters=3)
    check_unfitted_model(model)

# Missing fit_predict method
class NoFitPredict(BaseEstimator):
    def get_params(self, deep=True):
        return {'n_clusters': 3}

class NotCallableFitPredict(BaseEstimator):
    fit_predict = "not a function"
    def get_params(self, deep=True):
        return {'n_clusters': 3}

class NoClustersParam(BaseEstimator):
    def fit_predict(self, X):
        return [0] * len(X)

    def get_params(self, deep=True):
        return {}

def test_check_fitted_model_invalid():
    model = KMeans(n_clusters=3)
    model.labels_ = [0, 1, 0] # simulate fitted state
    with pytest.raises(TypeError, match="appears to be fitted"):
        check_unfitted_model(model)

    # model without fit_predict
    model = NoFitPredict()
    with pytest.raises(AttributeError, match="Provided model must implement"):
        check_unfitted_model(model)
    # model without callable fit_predict
    model = NotCallableFitPredict()
    with pytest.raises(AttributeError, match="Provided model must implement"):
        check_unfitted_model(model)
    # model without n_clusters parameter
    model = NoClustersParam()
    with pytest.raises(ValueError, match="support the 'n_clusters' parameter"):
        check_unfitted_model(model)


