import numpy as np
import pandas as pd
import unittest

from category_encoders.one_hot import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from src.pipeline.pipeline_transformers import GroupwiseTargetTransformer, RankingBinarizerTransformer, \
    TargetPivoterTransformer


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

        target_transformer = GroupwiseTargetTransformer(
            transformer=MinMaxScaler(feature_range=(0, 1)),
            group_by=['dataset', 'model', 'tuning', 'scoring']
        )
        _, y_transformed = target_transformer.fit_transform(X, y)
        _, y_inverse_transformed = target_transformer.inverse_transform(X, y_transformed)

        self.assertTrue(y_transformed.equals(pd.Series([0.0, 0.0, 0.25, 0.25, 0.75, 0.75, 1.0, 1.0])))
        self.assertTrue(np.array_equal(np.array(y).flatten(), np.array(y_inverse_transformed).flatten()))

    def test_target_onehot_transformer(self):
        X = pd.DataFrame({
            'dataset': [1, 2, 1, 2, 1, 2, 1, 2],
            'model': np.full(shape=8, fill_value='lgbm'),
            'tuning': np.full(shape=8, fill_value='none'),
            'scoring': np.full(shape=8, fill_value='f1_score'),
            'encoder': ['onehot', 'onehot', 'target', 'target', 'ordinal', 'ordinal', 'binary', 'binary'],
        })
        y = pd.Series([0, 3, 1, 2, 2, 1, 3, 0], name='ranking')

        target_transformer = GroupwiseTargetTransformer(
            transformer=OneHotEncoder(cols=['ranking'])
        )
        _, y_transformed = target_transformer.fit_transform(X, y)
        _, y_inverse_transformed = target_transformer.inverse_transform(X, y_transformed)

        self.assertTrue(y_transformed.equals(pd.DataFrame({
            'ranking_1': [1, 0, 0, 0, 0, 0, 0, 1],
            'ranking_2': [0, 1, 0, 0, 0, 0, 1, 0],
            'ranking_3': [0, 0, 1, 0, 0, 1, 0, 0],
            'ranking_4': [0, 0, 0, 1, 1, 0, 0, 0],
        })))
        self.assertTrue(np.array_equal(np.array(y).flatten(), np.array(y_inverse_transformed).flatten()))

    def test_ranking_binarizer_transformer(self):
        X = pd.DataFrame({
            'ranking': [0, 1, 2, 3, 3, 2, 1, 0]
        })

        target_transformer = RankingBinarizerTransformer()
        X_transformed = target_transformer.fit_transform(X)
        X_inverse_transformed = target_transformer.inverse_transform(X_transformed)

        self.assertTrue(X_transformed.equals(pd.DataFrame({
            '0': [0, 1, 1, 1, 1, 1, 1, 0],
            '1': [0, 0, 1, 1, 1, 1, 0, 0],
            '2': [0, 0, 0, 1, 1, 0, 0, 0],
        })))
        self.assertTrue(np.array_equal(np.array(X).flatten(), np.array(X_inverse_transformed).flatten()))

    def test_target_pivoter_transformer(self):
        X = pd.DataFrame({
            'dataset': [1, 2, 1, 2, 1, 2, 1, 2],
            'model': np.full(shape=8, fill_value='lgbm'),
            'tuning': np.full(shape=8, fill_value='none'),
            'scoring': np.full(shape=8, fill_value='f1_score'),
            'encoder': ['binary', 'binary', 'onehot', 'onehot', 'ordinal', 'ordinal', 'target', 'target'],
        })
        y = pd.Series([1, 0, 0, 1, 3, 2, 2, 3], name='ranking')

        target_transformer = TargetPivoterTransformer(
            factors=['dataset', 'model', 'tuning', 'scoring'],
            columns='encoder',
            target='ranking'
        )
        X_transformed, y_transformed = target_transformer.fit_transform(X, y)
        X_inverse_transformed, y_inverse_transformed = target_transformer.inverse_transform(X_transformed, y_transformed)

        self.assertTrue(y_transformed.equals(pd.DataFrame({
            'binary': [1, 0],
            'onehot': [0, 1],
            'ordinal': [3, 2],
            'target': [2, 3]
        }, index=pd.Index(name='encoder', data=[0, 1]))))
        self.assertTrue(
            pd.concat([X, y], axis=1).equals(
                pd.concat([X_inverse_transformed, y_inverse_transformed], axis=1)
            )
        )


if __name__ == '__main__':
    unittest.main()
