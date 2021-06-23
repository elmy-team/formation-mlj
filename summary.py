import pandas, os, time
import sklearn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import make_scorer

import scipy.stats as stats
from joblib import dump, load

from numpy.random import seed
import random
SEED = 90125
seed(SEED)

path = os.path.join(os.environ["VOLTAIRE"], "data", "datasets", "SILFIAC")
data = pandas.read_csv(os.path.join(path, "train.csv"))

labels = [f"power_{x}" for x in range(24)]
y = data.loc[:, labels].mean(axis=1)
X = data.drop(columns = np.concatenate((np.array(labels), np.array(["date_col"]))))

class FeatureSelector(SelectorMixin, BaseEstimator):
    def __init__(self, filter_):
        SelectorMixin.__init__(self)
        self.filter_ = filter_

    def _get_support_mask(self):
        return np.array([self.filter_(name) for name in self.names])

    def fit(self, X, y=None):
        self.names = X.keys()
        return self

pipe = make_pipeline(
    FeatureSelector(lambda x: "48.3_-3.0" not in x),
    StandardScaler(),
    PCA(),
    SVR(max_iter=500000))

regr = TransformedTargetRegressor(
    regressor=pipe,
    func = lambda x: np.log(x+1),
    inverse_func = lambda x: np.exp(x)-1)
regr.fit(X, y)
regr.predict(X)

def smape(y_true, y_pred):
    return 200 * np.mean(np.mean(
        np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + np.finfo(np.float64).eps)))
smape(y, regr.predict(X))

mask = -np.ones(X.shape[0])
mask[np.array([d[:4] == "2019" for d in data.date_col])] = 0
cv = PredefinedSplit(mask)

distributions = {"regressor__svr__C" : stats.loguniform(10e-9, 10e9),
                 "regressor__svr__gamma" : stats.loguniform(10e-9, 10e9),
                 "regressor__pca__n_components" : [0.75, 0.9, 0.95, 0.99],
                 "regressor__featureselector__filter_" : [
                     lambda x : True,
                     lambda x : "48.3_-3.0" not in x,
                     lambda x : "48.2_-3.0" not in x,
                     lambda x : "48.3_-3.1" not in x,
                     lambda x : "48.1_-3.0" not in x,
                     lambda x : "48.2_-3.1" not in x,
                     lambda x : "48.1_-3.1" not in x,
                     lambda x : "48.1_-3.2" not in x,
                     lambda x : "48.2_-3.2" not in x,
                     lambda x : "48.2_-3.3" not in x]}
grid = RandomizedSearchCV(
    regr, distributions,
    scoring=make_scorer(smape, greater_is_better=False),
    n_jobs=-1, cv=cv, n_iter=250)
start = time.time()
grid.fit(X, y)
stop = time.time()

print("BEST SMAPE IS : ", -grid.cv_results_["mean_test_score"][np.logical_not(np.isnan(grid.cv_results_["mean_test_score"]))].max())
print("TOTAL TIME IS : ", stop-start, "s")
# 33.44%, 182s

dump(grid, 'my_grid.joblib')
