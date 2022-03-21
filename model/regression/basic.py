import numpy as np
from sklearn.linear_model import LinearRegression as Linear, Ridge, Lasso
from tqdm.autonotebook import tqdm

from ..features import Features
from ..targets import Targets
from .base import Model


class NaiveRegression(Model):

    def __init__(self):
        super().__init__()
        pass

    def set_data(self, features: Features, targets: Targets):
        super().set_data(features, targets)
        return self.fit()

    def fit(self):

        self.F = self.features.X.reindex(columns = self.targets.sites.index).T
        self.F = self.F.fillna(self.F.mean())

        return self

class SKLearnModel(Model):

    def __init__(self, **params):
        super().__init__(**params)
        self.model = None

    def set_data(self, features: Features, targets: Targets):
        super().set_data(features, targets)
        return self.fit()

    def fit(self):
        self.model.fit(self.features.X.loc[self.targets.train_dates, :], self.targets.Y.T)
        self.F = self.targets.Y.copy()
        self.F.loc[:, self.targets.test_dates] = self.model.predict(self.features.X.loc[self.targets.test_dates, :]).T
        self.F.loc[:, self.targets.train_dates] = self.model.predict(self.features.X.loc[self.targets.train_dates, :]).T
        self.F = self.F.reindex(self.targets.sites.index)
        self.F = self.F.fillna(self.F.mean())
        return self


class LinearRegression(SKLearnModel):

    def __init__(self, fit_intercept: bool = True):
        super().__init__(fit_intercept=fit_intercept)
        self.model = Linear(fit_intercept=fit_intercept)

    def optimize(self):

        r0 = self.RMSE_unlabelled_full()
        self.model = Linear(fit_intercept=not self.params['fit_intercept'])
        r1 = self.fit().RMSE_unlabelled_full()

        if r0 < r1:
            self.model = Linear(fit_intercept=self.params['fit_intercept'])
            return self.params, r0
        else:
            self.params['fit_intercept'] = not self.params['fit_intercept']
            return self.params, r1


class RidgeRegression(SKLearnModel):

    def __init__(self, alpha: float, fit_intercept: bool = True):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept)
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)

    def update_alpha(self, alpha: float):
        self.params['alpha'] = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=self.params['fit_intercept'])
        return self.fit()

    def update_intercept(self, fit_intercept: bool):
        self.params['fit_intercept'] = fit_intercept
        self.model = Ridge(alpha=self.params['alpha'], fit_intercept=fit_intercept)
        return self.fit()

    def optimize_alpha(self):
        alphas = np.logspace(1, 4)
        error = [self.update_alpha(alpha).RMSE_unlabelled_full() for alpha in tqdm(alphas, leave=False)]
        return alphas[np.argmin(error)]

    def optimize_intercept(self):
        r0 = self.RMSE_unlabelled_full()
        r1 = self.update_intercept(not self.params['fit_intercept']).RMSE_unlabelled_full()
        if r0 > r1:
            return self.params['fit_intercept']
        else:
            return not self.params['fit_intercept']

    def optimize(self):

        alphas = [self.update_intercept(fit_intercept).optimize_alpha() for fit_intercept in [True, False]]

        r0 = self.update_alpha(alphas[1]).RMSE_unlabelled_full()
        self.model = Ridge(alpha=alphas[0], fit_intercept=True)
        r1 = self.RMSE_unlabelled_full()

        if r0 < r1:
            self.params = {'fit_intercept': False, 'alpha': alphas[1]}
            self.model = Ridge(**self.params)
            return self.params, r0
        else:
            self.params = {'fit_intercept': True, 'alpha': alphas[0]}
            return self.params, r1


class LassoRegression(SKLearnModel):

    def __init__(self, alpha: float, fit_intercept: bool = True):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept)
        self.model = Lasso(alpha=alpha, fit_intercept=fit_intercept)

    def update_alpha(self, alpha: float):
        self.params['alpha'] = alpha
        self.model = Lasso(alpha=alpha, fit_intercept=self.params['fit_intercept'])
        return self.fit()

    def update_intercept(self, fit_intercept: bool):
        self.params['fit_intercept'] = fit_intercept
        self.model = Lasso(alpha=self.params['alpha'], fit_intercept=fit_intercept)
        return self.fit()

    def optimize_alpha(self):
        alphas = np.logspace(-3, 1)
        error = [self.update_alpha(alpha).RMSE_unlabelled_full() for alpha in tqdm(alphas, leave=False)]
        return alphas[np.argmin(error)]

    def optimize_intercept(self):
        r0 = self.RMSE_unlabelled_full()
        r1 = self.update_intercept(not self.params['fit_intercept']).RMSE_unlabelled_full()
        if r0 > r1:
            return self.params['fit_intercept']
        else:
            return not self.params['fit_intercept']

    def optimize(self):

        alphas = [self.update_intercept(fit_intercept).optimize_alpha() for fit_intercept in [True, False]]

        r0 = self.update_alpha(alphas[1]).RMSE_unlabelled_full()
        self.model = Lasso(alpha=alphas[0], fit_intercept=True)
        r1 = self.RMSE_unlabelled_full()

        if r0 < r1:
            self.params = {'fit_intercept': False, 'alpha': alphas[1]}
            self.model = Lasso(**self.params)
            return self.params, r0
        else:
            self.params = {'fit_intercept': True, 'alpha': alphas[0]}
            return self.params, r1