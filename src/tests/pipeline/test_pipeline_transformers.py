import numpy as np
import pandas as pd
import unittest

from sklearn.preprocessing import MinMaxScaler

from src.pipeline.pipeline_transformers import TargetOneHotTransformer, TargetScalerTransformer


class TestModelPipeline(unittest.TestCase):
    def setUp(self) -> None:
        return

    def test_target_scaler_transformer(self):
        X = pd.DataFrame({
            'dataset': [1, 2, 1, 2, 1, 2, 1, 2],
            'model': np.full(shape=8, fill_value='lgbm'),
            'tuning': np.full(shape=8, fill_value='none'),
            'scoring': np.full(shape=8, fill_value='f1_score'),
            'encoder': ['onehot', 'onehot', 'target', 'target', 'ordinal', 'ordinal', 'binary', 'binary'],
        })
        y = pd.Series([100, 0, 150, 0.5, 250, 1.5, 300, 2], name='ranking')

        target_transformer = TargetScalerTransformer(
            scaler=MinMaxScaler(feature_range=(0, 1)),
            group_by=['dataset', 'model', 'tuning', 'scoring']
        )
        _, y_transformed = target_transformer.fit_transform(X, y)
        _, y_inverse_transformed = target_transformer.inverse_transform(X, y_transformed)

        self.assertTrue(y_transformed.equals(pd.Series([0.0, 0.0, 0.25, 0.25, 0.75, 0.75, 1.0, 1.0], name='ranking')))
        self.assertTrue(y.equals(y_inverse_transformed))

    def test_target_onehot_transformer(self):
        X = pd.DataFrame({
            'dataset': [1, 2, 1, 2, 1, 2, 1, 2],
            'model': np.full(shape=8, fill_value='lgbm'),
            'tuning': np.full(shape=8, fill_value='none'),
            'scoring': np.full(shape=8, fill_value='f1_score'),
            'encoder': ['onehot', 'onehot', 'target', 'target', 'ordinal', 'ordinal', 'binary', 'binary'],
        })
        y = pd.Series([0, 3, 1, 2, 2, 1, 3, 0], name='ranking')

        target_transformer = TargetOneHotTransformer()
        _, y_transformed = target_transformer.fit_transform(X, y)
        _, y_inverse_transformed = target_transformer.inverse_transform(X, y_transformed)

        self.assertTrue(y_transformed.equals(pd.DataFrame({
            'ranking_1': [1, 0, 0, 0, 0, 0, 0, 1],
            'ranking_2': [0, 1, 0, 0, 0, 0, 1, 0],
            'ranking_3': [0, 0, 1, 0, 0, 1, 0, 0],
            'ranking_4': [0, 0, 0, 1, 1, 0, 0, 0],
        })))
        self.assertTrue(np.array_equal(np.array(y).flatten(), np.array(y_inverse_transformed).flatten()))


if __name__ == '__main__':
    unittest.main()
