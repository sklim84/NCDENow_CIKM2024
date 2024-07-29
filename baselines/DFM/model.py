import statsmodels.api as sm
import numpy as np
import pandas as pd

class DynamicFactorModel:
    def __init__(self, data, idx_target, num_monthly=None, factors=1, factor_orders=1):
        self.idx_target = idx_target
        self.num_monthly = num_monthly
        self.factors = factors
        self.factor_orders = factor_orders
        self.dfm_model = sm.tsa.DynamicFactorMQ(endog=data,
                                                k_endog_monthly=num_monthly,
                                                factors=self.factors,
                                                factor_orders=self.factor_orders)
        self.dfm_fitted = None

    def fit(self):
        self.dfm_fitted = self.dfm_model.fit()  # default method = em

        print(self.dfm_fitted.summary())
        # print(self.dfm_fitted.params)

        return self.dfm_fitted

    def apply_and_forecast(self, data, forcast_steps):
        dfm_applied = self.dfm_fitted.apply(endog=data, k_endog_monthly=self.num_monthly)

        # dfm_applied.states.smoothed.to_csv('dfm_states_moothed.csv')
        # print(dfm_applied.summary())

        dfm_forecast = dfm_applied.forecast(forcast_steps)

        if isinstance(dfm_forecast, pd.DataFrame):
            y_pred = dfm_forecast[self.idx_target].values
        elif isinstance(dfm_forecast, np.ndarray):
            y_pred = dfm_forecast[:, self.idx_target]
        return y_pred
