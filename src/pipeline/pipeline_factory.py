import category_encoders.target_encoder
import pandas as pd

from category_encoders.binary import BinaryEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from enum import Enum
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skopt.space import Categorical, Integer, Real
from sklearn.multioutput import MultiOutputClassifier

from src import configuration as config
from src.features.encoder_utils import load_graph
from src.pipeline.evaluation.evaluation_utils import SpearmanScorer
from src.pipeline.model_pipeline import ModelPipeline, EvaluationType
from src.pipeline.pipeline_transformers import PoincareEmbedding, OpenMLMetaFeatureTransformer, \
    GeneralPurposeEncoderTransformer, ColumnKeeper, GroupwiseTargetTransformer, RankingBinarizerTransformer, \
    get_column_names


class ModelType(Enum):
    REGRE_BASELINE = "regre_baseline"
    CLASS_BASELINE = "class_baseline"
    LINEAR_REGRESSION = "linear_regression"
    REGRE_PREPROCESSED = "regre_preprocessed"
    REGRE_TEST = "regre_test"
    REGRE_NO_SEARCH = "regre_no_search"
    REGRE_BAYES_SEARCH = "regre_bayes_search"
    POINTWISE_REGRESSION_NO_SEARCH = "pointwise_regression_no_search"
    POINTWISE_NORMALIZED_REGRESSION_NO_SEARCH = "pointwise_normalized_regression_no_search"
    POINTWISE_CLASSIFICATION_NO_SEARCH = "pointwise_classification_no_search"
    POINTWISE_ORDINAL_REGRESSION_NO_SEARCH = "pointwise_ordinal_regression_no_search"
    POINTWISE_REGRESSION_GRID_SEARCH = "pointwise_regression_grid_search"
    POINTWISE_NORMALIZED_REGRESSION_BAYES_SEARCH = "pointwise_normalized_regression_bayes_search"
    POINTWISE_CLASSIFICATION_BAYES_SEARCH = "pointwise_classification_bayes_search"
    POINTWISE_ORDINAL_REGRESSION_BAYES_SEARCH = "pointwise_ordinal_regression_bayes_search"
    PAIRWISE_CLASSIFICATION_NO_SEARCH = "pairwise_classification_no_search"
    PAIRWISE_CLASSIFICATION_OPTUNA_SEARCH = "pairwise_classification_optuna_search"


class PipelineFactory:
    """
    Class that creates ModelPipelines
    """

    def create_pipeline(
            self,
            train_df: pd.DataFrame,
            model_type: ModelType,
            evaluation: EvaluationType = EvaluationType.BASIC,
            y_train=None,
            X_test=None,
            target: str = "cv_score",
            split_factors=["dataset", "model", "tuning", "scoring"],
            param_grid=[],
            target_transformer=None,
            original_target=None,
            scorer=None,
            n_folds=5,
            bayes_n_iter=None,
            bayes_n_points=None,
            bayes_cv=None,
            bayes_n_jobs=None,
            verbose_level=0,
            opt_iterations=0,
            as_pairwise=False,
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
        print('Creating pipeline ...')

        if y_train is not None:
            # merge the X and y dataframes
            train_df = pd.concat([train_df, y_train], axis=1)

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
            scorer = SpearmanScorer(factors=split_factors)

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

            bayes_n_iter = 200 if bayes_n_iter is None else bayes_n_iter
            bayes_n_points = 4 if bayes_n_points is None else bayes_n_points
            bayes_cv = 4 if bayes_cv is None else bayes_cv
            bayes_n_jobs = -1 if bayes_n_jobs is None else bayes_n_jobs

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

        elif model_type == "pointwise_regression_no_search" or model_type == ModelType.POINTWISE_REGRESSION_NO_SEARCH:
            pipeline_steps = [
                ("keeper", ColumnKeeper(columns=[
                    'dataset',
                    'model',
                    'tuning',
                    'scoring',
                    'encoder'
                ])),
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
                ("estimator", RandomForestRegressor())
            ]

        elif model_type == "pointwise_normalized_regression_no_search" \
                or model_type == ModelType.POINTWISE_NORMALIZED_REGRESSION_NO_SEARCH:
            target_transformer = GroupwiseTargetTransformer(
                transformer=MinMaxScaler(feature_range=(0, 1)),
                group_by=split_factors
            )

            _, transformed_target = target_transformer.fit_transform(
                train_df.drop(columns=[target]),
                train_df[target]
            )
            train_df = pd.concat([train_df.drop(columns=[target]), transformed_target], axis=1)

            original_target = target
            target = get_column_names(transformed_target)

            scorer = SpearmanScorer(factors=split_factors, transformer=target_transformer)

            pipeline_steps = [
                ("keeper", ColumnKeeper(columns=[
                    'dataset',
                    'model',
                    'tuning',
                    'scoring',
                    'encoder'
                ])),
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
                ("estimator", RandomForestRegressor())
            ]

        elif model_type == "pointwise_classification_no_search" or model_type == ModelType.POINTWISE_CLASSIFICATION_NO_SEARCH:
            pipeline_steps = [
                ("keeper", ColumnKeeper(columns=[
                    'dataset',
                    'model',
                    'tuning',
                    'scoring',
                    'encoder'
                ])),
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
                ("estimator", DecisionTreeClassifier())
            ]

        elif model_type == "pointwise_ordinal_regression_no_search" or model_type == ModelType.POINTWISE_ORDINAL_REGRESSION_NO_SEARCH:
            target_transformer = GroupwiseTargetTransformer(
                transformer=RankingBinarizerTransformer()
            )

            _, transformed_target = target_transformer.fit_transform(
                train_df.drop(columns=target, axis=1),
                train_df[target]
            )
            train_df = pd.concat([train_df.drop(columns=target, axis=1), transformed_target], axis=1)

            original_target = target
            target = get_column_names(transformed_target)

            scorer = SpearmanScorer(factors=split_factors, transformer=target_transformer)

            pipeline_steps = [
                ("keeper", ColumnKeeper(columns=[
                    'dataset',
                    'model',
                    'tuning',
                    'scoring',
                    'encoder'
                ])),
                ("encoder_transformer", PoincareEmbedding(
                    load_graph(config.ROOT_DIR / "data/external/graphs/encodings_graph.adjlist"),
                    epochs=500,
                    batch_size=50,
                    size=3,
                    encoder=OneHotEncoder()
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
                    OneHotEncoder(),
                    OneHotEncoder()
                )),
                ("estimator", DecisionTreeClassifier())
            ]

        elif model_type == "pointwise_normalized_regression_bayes_search" \
                or model_type == ModelType.POINTWISE_NORMALIZED_REGRESSION_BAYES_SEARCH:
            target_transformer = GroupwiseTargetTransformer(
                transformer=MinMaxScaler(feature_range=(0, 1)),
                group_by=split_factors
            )

            _, transformed_target = target_transformer.fit_transform(
                train_df.drop(columns=[target]),
                train_df[target]
            )
            train_df = pd.concat([train_df.drop(columns=[target]), transformed_target], axis=1)

            original_target = target
            target = get_column_names(transformed_target)

            scorer = SpearmanScorer(factors=split_factors, transformer=target_transformer)

            pipeline_steps = [
                ("keeper", ColumnKeeper(columns=[
                    'dataset',
                    'model',
                    'tuning',
                    'scoring',
                    'encoder'
                ])),
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
                ("estimator", RandomForestRegressor())
            ]

            evaluation = EvaluationType.BAYES_SEARCH

            bayes_n_iter = 200 if bayes_n_iter is None else bayes_n_iter
            bayes_n_points = 4 if bayes_n_points is None else bayes_n_points
            bayes_cv = 4 if bayes_cv is None else bayes_cv
            bayes_n_jobs = -1 if bayes_n_jobs is None else bayes_n_jobs

            param_grid = {
                'encoder_transformer__encoder': Categorical([None, OneHotEncoder()]),
                'dataset_transformer__nan_ratio_feature_drop_threshold': Categorical([0.25, 0.4, 0.45, 0.5]),
                'dataset_transformer__expected_pca_variance': Real(0.25, 1.0),
                'dataset_transformer__encoder': Categorical([None, OneHotEncoder()]),
                'general_transformer__model_encoder': Categorical([OneHotEncoder(), OrdinalEncoder(), TargetEncoder()]),
                'general_transformer__tuning_encoder': Categorical([OneHotEncoder(), OrdinalEncoder(), TargetEncoder()]),
                'general_transformer__scoring_encoder': Categorical([OneHotEncoder(), OrdinalEncoder(), TargetEncoder()]),
                'estimator__max_depth': Categorical([1, 10, 50, 100, 250, 500, None]),  # default=None
                'estimator__n_estimators': Integer(50, 200),  # default=100
                'estimator__min_samples_leaf': Integer(1, 5),  # default=1
                'estimator__max_features': Categorical([None, 'sqrt', 'log2']),  # default=None
            }

        elif model_type == "pointwise_classification_bayes_search" \
                or model_type == ModelType.POINTWISE_CLASSIFICATION_BAYES_SEARCH:
            pipeline_steps = [
                ("keeper", ColumnKeeper(columns=[
                    'dataset',
                    'model',
                    'tuning',
                    'scoring',
                    'encoder'
                ])),
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
                ("estimator", DecisionTreeClassifier())
            ]

            evaluation = EvaluationType.BAYES_SEARCH

            bayes_n_iter = 200 if bayes_n_iter is None else bayes_n_iter
            bayes_n_points = 4 if bayes_n_points is None else bayes_n_points
            bayes_cv = 4 if bayes_cv is None else bayes_cv
            bayes_n_jobs = -1 if bayes_n_jobs is None else bayes_n_jobs

            param_grid = {
                'encoder_transformer__encoder': Categorical([None, OneHotEncoder()]),
                'dataset_transformer__nan_ratio_feature_drop_threshold': Categorical([0.25, 0.4, 0.45, 0.5]),
                'dataset_transformer__expected_pca_variance': Real(0.25, 1.0),
                'dataset_transformer__encoder': Categorical([None, OneHotEncoder()]),
                'general_transformer__model_encoder': Categorical([OneHotEncoder(), OrdinalEncoder()]),
                'general_transformer__tuning_encoder': Categorical([OneHotEncoder(), OrdinalEncoder()]),
                'general_transformer__scoring_encoder': Categorical([OneHotEncoder(), OrdinalEncoder()]),
                'estimator__max_depth': Categorical([1, 10, 50, 100, 250, 500, None]),  # default=None
                'estimator__min_samples_split': Integer(2, 5),  # default=2
                'estimator__min_samples_leaf': Integer(1, 5),  # default=1
                'estimator__max_features': Categorical([None, 'sqrt', 'log2']),  # default=None
            }

        elif model_type == "pointwise_ordinal_regression_bayes_search" \
                or model_type == ModelType.POINTWISE_ORDINAL_REGRESSION_BAYES_SEARCH:
            target_transformer = GroupwiseTargetTransformer(
                transformer=RankingBinarizerTransformer()
            )

            _, transformed_target = target_transformer.fit_transform(
                train_df.drop(columns=target, axis=1),
                train_df[target]
            )
            train_df = pd.concat([train_df.drop(columns=target, axis=1), transformed_target], axis=1)

            original_target = target
            target = get_column_names(transformed_target)

            scorer = SpearmanScorer(factors=split_factors, transformer=target_transformer)

            pipeline_steps = [
                ("keeper", ColumnKeeper(columns=[
                    'dataset',
                    'model',
                    'tuning',
                    'scoring',
                    'encoder'
                ])),
                ("encoder_transformer", PoincareEmbedding(
                    load_graph(config.ROOT_DIR / "data/external/graphs/encodings_graph.adjlist"),
                    epochs=500,
                    batch_size=50,
                    size=3,
                    encoder=OneHotEncoder()

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
                    OneHotEncoder(),
                    OneHotEncoder()
                )),
                ("estimator", DecisionTreeClassifier())
            ]

            evaluation = EvaluationType.BAYES_SEARCH

            bayes_n_iter = 200 if bayes_n_iter is None else bayes_n_iter
            bayes_n_points = 4 if bayes_n_points is None else bayes_n_points
            bayes_cv = 4 if bayes_cv is None else bayes_cv
            bayes_n_jobs = -1 if bayes_n_jobs is None else bayes_n_jobs

            param_grid = {
                'encoder_transformer__encoder': Categorical([None, OneHotEncoder()]),
                'dataset_transformer__nan_ratio_feature_drop_threshold': Categorical([0.25, 0.4, 0.45, 0.5]),
                'dataset_transformer__expected_pca_variance': Real(0.25, 1.0),
                'dataset_transformer__encoder': Categorical([None, OneHotEncoder()]),
                'general_transformer__model_encoder': Categorical([OneHotEncoder(), OrdinalEncoder()]),
                'general_transformer__tuning_encoder': Categorical([OneHotEncoder(), OrdinalEncoder()]),
                'general_transformer__scoring_encoder': Categorical([OneHotEncoder(), OrdinalEncoder()]),
                'estimator__max_depth': Categorical([1, 10, 50, 100, 250, 500, None]),  # default=None
                'estimator__min_samples_split': Integer(2, 5),  # default=2
                'estimator__min_samples_leaf': Integer(1, 5),  # default=1
                'estimator__max_features': Categorical([None, 'sqrt', 'log2']),  # default=None
            }
            
        elif model_type == "pairwise_classification_no_search" or model_type == ModelType.PAIRWISE_CLASSIFICATION_NO_SEARCH:
            pipeline_steps = [
                ("dataset_transformer", OpenMLMetaFeatureTransformer(
                    nan_ratio_feature_drop_threshold=0.25,
                    imputer=SimpleImputer(strategy='mean'),
                    scaler=StandardScaler(),
                    expected_pca_variance=0.6,
                    encoder=None
                )),
                ("general_transformer", GeneralPurposeEncoderTransformer(
                    OneHotEncoder(),
                    OneHotEncoder(),
                    OneHotEncoder()
                )),
                ("estimator", MultiOutputClassifier(DecisionTreeClassifier()))
            ]
            
            as_pairwise = True
            
        elif model_type == "pairwise_classification_optuna_search" or model_type == ModelType.PAIRWISE_CLASSIFICATION_OPTUNA_SEARCH:
            pipeline_steps = [
                ("dataset_transformer", OpenMLMetaFeatureTransformer(
                    nan_ratio_feature_drop_threshold=0.25,
                    imputer=SimpleImputer(strategy='mean'),
                    scaler=StandardScaler(),
                    expected_pca_variance=0.6,
                    encoder=None
                )),
                ("general_transformer", GeneralPurposeEncoderTransformer(
                    OneHotEncoder(),
                    OneHotEncoder(),
                    OneHotEncoder()
                )),
                ("estimator", MultiOutputClassifier(DecisionTreeClassifier()))
            ]

            param_grid = {
                'dataset_transformer__nan_ratio_feature_drop_threshold': [0.25, 0.5],
                'dataset_transformer__expected_pca_variance': [0.25, 1.0],
                'dataset_transformer__encoder': [OneHotEncoder(), BinaryEncoder()],
                'general_transformer__model_encoder': [OneHotEncoder(), BinaryEncoder(), OrdinalEncoder()],
                'general_transformer__tuning_encoder': [OneHotEncoder(), BinaryEncoder(), OrdinalEncoder()],
                'general_transformer__scoring_encoder': [OneHotEncoder(), BinaryEncoder(), OrdinalEncoder()],
                # 'estimator__estimator__max_depth': [1, 500],  # default=None
                'estimator__estimator__min_samples_split': [2, 5],  # default=2
                'estimator__estimator__min_samples_leaf': [1, 5],  # default=1
                # 'estimator__estimator__max_features': [None, 'sqrt', 'log2'],  # default=None
            }
            
            evaluation = EvaluationType.OPTUNA
            as_pairwise = True
            opt_iterations = 200

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # create the new pipeline and return it
        return ModelPipeline(
            train_df,
            steps=pipeline_steps,
            evaluation=evaluation,
            X_test=X_test,
            verbose_level=verbose_level,
            target=target,
            split_factors=split_factors,
            param_grid=param_grid,
            target_transformer=target_transformer,
            original_target=original_target,
            scorer=scorer,
            n_folds=n_folds,
            bayes_n_iter=bayes_n_iter,
            bayes_n_points=bayes_n_points,
            bayes_cv=bayes_cv,
            bayes_n_jobs=bayes_n_jobs,
            as_pairwise=as_pairwise,
            opt_iterations=opt_iterations,
            **kwargs
        )
