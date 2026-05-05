import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt
import mlflow
import json
import pandas as pd
import numpy as np
import wandb
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def delta_date_feature(dates):
    date_sanitized = pd.DataFrame(dates).apply(
        lambda d: pd.to_datetime(d, errors="coerce").fillna(pd.Timestamp("2010-01-01"))
    )
    return date_sanitized.apply(
        lambda d: (pd.Timestamp.today() - d).dt.days
    ).values


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.trainval_artifact).file()
    X = pd.read_csv(artifact_local_path)
    y = X.pop("price")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.val_size,
        stratify=X[args.stratify_by] if args.stratify_by != "none" else None,
        random_state=args.random_seed,
    )

    logger.info("Preparing pipeline")
    sk_pipe, processed_features = get_inference_pipeline(args)

    logger.info("Fitting pipeline")
    sk_pipe.fit(X_train, y_train)

    logger.info("Scoring on validation set")
    r_squared = sk_pipe.score(X_val, y_val)
    y_pred = sk_pipe.predict(X_val)
    mae = np.mean(np.abs(y_val - y_pred))

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    run.summary["r2"] = r_squared
    run.summary["mae"] = mae

    logger.info("Exporting model")
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    signature = infer_signature(X_val[processed_features], y_pred)
    mlflow.sklearn.save_model(
        sk_pipe,
        "random_forest_dir",
        signature=signature,
        input_example=X_val[processed_features].iloc[:5],
    )

    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Random forest pipeline export",
    )
    artifact.add_dir("random_forest_dir")
    run.log_artifact(artifact)

    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)
    run.log({"feature_importance": wandb.Image(fig_feat_imp)})


def plot_feature_importance(pipe, feat_names):
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names)]
    nlim = min(len(feat_names), feat_imp.shape[0])
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.barh(
        range(nlim),
        feat_imp[:nlim],
        align="center",
    )
    sub_feat_imp.set_yticks(range(nlim))
    sub_feat_imp.set_yticklabels(feat_names[:nlim])
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(args):
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    ordinal_categorical_preproc = OrdinalEncoder()

    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )

    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude",
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    date_imputer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="2010-01-01"),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    reshape_to_1d = FunctionTransformer(lambda x: x.reshape(-1), validate=False)
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=args.max_tfidf_features,
            sublinear_tf=True,
        ),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"]),
        ],
        remainder="drop",
    )

    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review"] + ["name"]

    with open(args.rf_config) as fp:
        rf_config = json.load(fp)

    random_forest = RandomForestRegressor(**rf_config)

    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_forest),
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest")

    parser.add_argument("--trainval_artifact", type=str, help="Trainval artifact", required=True)
    parser.add_argument("--val_size", type=float, help="Validation size", required=True)
    parser.add_argument("--random_seed", type=int, help="Random seed", default=42)
    parser.add_argument("--stratify_by", type=str, help="Stratify by column", default="none")
    parser.add_argument("--rf_config", type=str, help="RF config JSON", required=True)
    parser.add_argument("--max_tfidf_features", type=int, help="Max tfidf features", required=True)
    parser.add_argument("--output_artifact", type=str, help="Output artifact name", required=True)

    args = parser.parse_args()
    go(args)
