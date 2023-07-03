import category_encoders.target_encoder
import pandas as pd

from category_encoders.binary import BinaryEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from enum import Enum
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skopt.space import Categorical, Integer, Real

from src import configuration as config
from src.features.encoder_utils import load_graph
from src.pipeline.evaluation.evaluation_utils import PointwiseSpearmanScorer
from src.pipeline.model_pipeline import ModelPipeline, EvaluationType
from src.pipeline.pipeline_transformers import PoincareEmbedding, OpenMLMetaFeatureTransformer, \
    GeneralPurposeEncoderTransformer, ColumnKeeper, TargetScalerTransformer, TargetOneHotTransformer


class ModelType(Enum):
    REGRE_BASELINE = "regre_baseline"
    CLASS_BASELINE = "class_baseline"
    LINEAR_REGRESSION = "linear_regression"
    REGRE_PREPROCESSED = "regre_preprocessed"
    REGRE_TEST = "regre_test"
    REGRE_NO_SEARCH = "regre_no_search"
    REGRE_BAYES_SEARCH = "regre_bayes_search"
    POINTWISE_REGRESSION_NO_SEARCH = "pointwise_regression_no_search"
    POINTWISE_CLASSIFICATION_NO_SEARCH = "pointwise_classification_no_search"
    POINTWISE_ORDINAL_REGRESSION_NO_SEARCH = "pointwise_ordinal_regression_no_search"


class PipelineFactory:
    """
    Class that creates ModelPipelines
    """

    def create_pipeline(
            self,
            X_train: pd.DataFrame,
            model_type: ModelType,
            evaluation: EvaluationType = EvaluationType.BASIC,
            y_train=None,
            X_test=None,
            target: str = "cv_score",
            split_factors=["dataset", "model", "tuning", "scoring"],
            param_grid=[],
            scorer=None,
            n_folds=5,
            verbose_level=0,
            **kwargs
    ):
        """
        Create a ModelPipeline of the specified type.

        Args:
            model_type (ModelType): The type of ModelPipeline to create
            x_train_df (pd.DataFrame): The training data
            y_train_df (pd.DataFrame): The training labels
            verbose (int, optional): The verbosity level. Defaults to 0.
            evaluation (str, optional): The type of evaluation to perform.
                Can be one of "basic", "cross_validation", "grid_search". Defaults to "basic".

        Raises:
            ValueError: If the model_type is not recognized

        Returns:
            ModelPipeline: The created ModelPipeline
        """
        if y_train is not None:
            # merge the X and y dataframes
            X_train = pd.concat([X_train, y_train], axis=1)

        # depending on the model type create the appropriate pipeline
        if model_type == "regre_baseline" or model_type == ModelType.REGRE_BASELINE:
            pipeline_steps = [
                ("estimator", DummyRegressor(strategy="mean"))
            ]

        elif model_type == "class_baseline" or model_type == ModelType.CLASS_BASELINE:
            pipeline_steps = [
                ("estimator", DummyClassifier(strategy="most_frequent"))
            ]

        elif model_type == "linear_regression" or model_type == ModelType.LINEAR_REGRESSION:
            graph = load_graph(config.DATA_DIR / "external/graphs/encodings_graph.adjlist")
            poincare_embedddings_transformer = PoincareEmbedding(graph=graph, epochs=100)

            pipeline_steps = [
                ("poincare_embedding", poincare_embedddings_transformer),
                ("estimator", LinearRegression())
            ]

        elif model_type == "regre_preprocessed" or model_type == ModelType.REGRE_PREPROCESSED:
            pipeline_steps = [
                ("encoder_transformer", PoincareEmbedding(
                    load_graph(config.ROOT_DIR / "data/external/graphs/encodings_graph.adjlist"),
                    epochs=500,
                    batch_size=50,
                    size=3
                )),
                ("dataset_transformer", OpenMLMetaFeatureTransformer(
                    nan_ratio_feature_drop_threshold=0.4,
                    imputer=SimpleImputer(strategy='mean'),
                    scaler=StandardScaler(),
                    expected_pca_variance=1,
                    encoder=TargetEncoder()
                )),
                ("general_transformer", GeneralPurposeEncoderTransformer(
                    OrdinalEncoder(),
                    BinaryEncoder(),
                    BinaryEncoder()
                )),
                ("estimator", LGBMRegressor())
            ]

        elif model_type == "regre_test" or model_type == ModelType.REGRE_TEST:
            pipeline_steps = [
                ("encoder_transformer", PoincareEmbedding(
                    load_graph(config.ROOT_DIR / "data/external/graphs/encodings_graph.adjlist"),
                    epochs=500,
                    batch_size=50,
                    size=3,
                    encoder=category_encoders.one_hot.OneHotEncoder()

                )),
                ("dataset_transformer", OpenMLMetaFeatureTransformer(
                    nan_ratio_feature_drop_threshold=0.25,
                    imputer=SimpleImputer(strategy='mean'),
                    scaler=StandardScaler(),
                    expected_pca_variance=0.6,
                    encoder=None
                )),
                ("general_transformer", GeneralPurposeEncoderTransformer(
                    OrdinalEncoder(),
                    OrdinalEncoder(),
                    OrdinalEncoder()
                )),
                ("estimator", DecisionTreeRegressor())
            ]

        elif model_type == "regre_no_search" or model_type == ModelType.REGRE_NO_SEARCH:
            pipeline_steps = [
                ("encoder_transformer", PoincareEmbedding(
                    load_graph(config.ROOT_DIR / "data/external/graphs/encodings_graph.adjlist"),
                    epochs=500,
                    batch_size=50,
                    size=3,
                    encoder=category_encoders.one_hot.OneHotEncoder()

                )),
                ("dataset_transformer", OpenMLMetaFeatureTransformer(
                    nan_ratio_feature_drop_threshold=0.25,
                    imputer=SimpleImputer(strategy='mean'),
                    scaler=StandardScaler(),
                    expected_pca_variance=0.6,
                    encoder=None
                )),
                ("general_transformer", GeneralPurposeEncoderTransformer(
                    OneHotEncoder(),
                    TargetEncoder(),
                    TargetEncoder()
                )),
                ("estimator", DecisionTreeRegressor())
            ]

        elif model_type == "regre_bayes_search" or model_type == ModelType.REGRE_BAYES_SEARCH:
            pipeline_steps = [
                ("encoder_transformer", PoincareEmbedding(
                    load_graph(config.ROOT_DIR / "data/external/graphs/encodings_graph.adjlist"),
                    epochs=500,
                    batch_size=50,
                    size=3,
                    encoder=category_encoders.one_hot.OneHotEncoder()

                )),
                ("dataset_transformer", OpenMLMetaFeatureTransformer(
                    nan_ratio_feature_drop_threshold=0.25,
                    imputer=SimpleImputer(strategy='mean'),
                    scaler=StandardScaler(),
                    expected_pca_variance=0.6,
                    encoder=None
                )),
                ("general_transformer", GeneralPurposeEncoderTransformer(
                    OneHotEncoder(),
                    TargetEncoder(),
                    TargetEncoder()
                )),
                ("estimator", DecisionTreeRegressor())
            ]

            evaluation = EvaluationType.BAYES_SEARCH

            param_grid = {
                'encoder_transformer__encoder': Categorical([None, OneHotEncoder()]),
                'dataset_transformer__nan_ratio_feature_drop_threshold': Categorical([0.25, 0.4, 0.45, 0.5]),
                'dataset_transformer__expected_pca_variance': Real(0.25, 1.0),
                'dataset_transformer__encoder': Categorical([None, OneHotEncoder()]),
                'general_transformer__model_encoder': Categorical([OneHotEncoder(), OrdinalEncoder(), TargetEncoder()]),
                'general_transformer__tuning_encoder': Categorical([OneHotEncoder(), OrdinalEncoder(), TargetEncoder()]),
                'general_transformer__scoring_encoder': Categorical([OneHotEncoder(), OrdinalEncoder(), TargetEncoder()]),
                'estimator__max_depth': Categorical([1, 10, 50, 100, 250, 500, None]),  # default=None
                'estimator__min_samples_split': Integer(2, 5),  # default=2
                'estimator__min_samples_leaf': Integer(1, 5),  # default=1
                'estimator__max_features': Categorical([None, 'sqrt', 'log2']),  # default=None
            }

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # create the new pipeline and return it
        return ModelPipeline(
            X_train,
            steps=pipeline_steps,
            evaluation=evaluation,
            X_test=X_test,
            verbose_level=verbose_level,
            target=target,
            split_factors=split_factors,
            param_grid=param_grid,
            scorer=scorer,
            n_folds=n_folds,
            **kwargs
        )
