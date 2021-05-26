import shutil
import typing as T
from dataclasses import dataclass
import os
import glob

import dask_cudf
import cudf
import nvtabular as nvt
from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
from nvtabular.utils import download_file
from sklearn.model_selection import train_test_split
import tensorflow as tf


@dataclass
class Dataset(object):
    train_path: str
    eval_path: str

    categorical_features: T.List[str]
    continuous_features: T.List[str]
    targets: T.List[str]

    @property
    def train_files(self):
        return sorted(glob.glob(os.path.join(self.train_path, "*.parquet")))

    def train_df(self, sample=0.1):
        return dask_cudf.read_parquet(self.train_files).sample(frac=sample).compute()

    def train_tf_dataset(self, batch_size, shuffle=True, buffer_size=0.06, parts_per_chunk=1):
        return KerasSequenceLoader(
            self.train_files,
            batch_size=batch_size,
            label_names=self.targets,
            cat_names=self.categorical_features,
            cont_names=self.continuous_features,
            engine="parquet",
            shuffle=shuffle,
            buffer_size=buffer_size,  # how many batches to load at once
            parts_per_chunk=parts_per_chunk,
        )

    @property
    def eval_files(self):
        return sorted(glob.glob(os.path.join(self.eval_path, "*.parquet")))

    def eval_df(self, sample=0.1):
        return dask_cudf.read_parquet(self.eval_files).sample(frac=sample).compute()

    def eval_tf_dataset(self, batch_size, shuffle=True, buffer_size=0.06, parts_per_chunk=1):
        return KerasSequenceLoader(
            self.eval_files,
            batch_size=batch_size,
            label_names=self.targets,
            cat_names=self.categorical_features,
            cont_names=self.continuous_features,
            engine="parquet",
            shuffle=shuffle,
            buffer_size=buffer_size,  # how many batches to load at once
            parts_per_chunk=parts_per_chunk,
        )

    def eval_tf_callback(self, batch_size, **kwargs):
        return KerasSequenceValidater(self.eval_tf_dataset(batch_size, **kwargs))

    def create_default_embedding_columns(self, workflow: nvt.Workflow):
        embedding_size = nvt.ops.get_embedding_sizes(workflow)
        embedding_cols = []
        for col in self.categorical_features:
            embedding_cols.append(
                tf.feature_column.embedding_column(
                    tf.feature_column.categorical_column_with_identity(
                        col, embedding_size[col][0]
                    ),  # Input dimension (vocab size)
                    embedding_size[col][1],  # Embedding output dimension
                )
            )

        return embedding_cols

    def create_keras_inputs(self, multi_hot_columns=None):
        if multi_hot_columns is None:
            multi_hot_columns = []
        inputs = {}

        for col in self.continuous_features:
            inputs[col] = tf.keras.Input(name=col, dtype=tf.float32, shape=(1,))

        for col in self.categorical_features:
            if col in multi_hot_columns:
                inputs[col + "__values"] = tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,))
                inputs[col + "__nnzs"] = tf.keras.Input(name=f"{col}__nnzs", dtype=tf.int64, shape=(1,))
            else:
                inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(1,))

        return inputs


@dataclass
class MovieLensData(object):
    train_path: str
    eval_path: str
    movies_path: str

    @classmethod
    def load(cls, output_path):
        return load(output_path)

    @property
    def movies_df(self):
        return cudf.read_parquet(self.movies_path)

    def transform(self, output_path, workflow: nvt.Workflow, continuous_features, categorical_features, targets,
                  part_size="100MB",  shuffle=nvt.io.Shuffle.PER_PARTITION, **kwargs):
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        train_dataset = nvt.Dataset([self.train_path], part_size=part_size)
        valid_dataset = nvt.Dataset([self.eval_path], part_size=part_size)

        print("Fitting dataset...")

        workflow.fit(train_dataset)

        train = os.path.join(output_path, "train")
        print("Transforming train-dataset...")
        workflow.transform(train_dataset).to_parquet(
            output_path=train,
            shuffle=shuffle,
            **kwargs
        )

        eval = os.path.join(output_path, "valid")
        print("Transforming eval-dataset...")
        workflow.transform(valid_dataset).to_parquet(
            output_path=eval,
            shuffle=shuffle,
            **kwargs
        )

        return Dataset(train, eval,
                       continuous_features=continuous_features,
                       categorical_features=categorical_features,
                       targets=targets)


def load(output_path):
    zip_path = os.path.join(output_path, "ml-25m.zip")
    if not os.path.exists(zip_path):
        download_file("http://files.grouplens.org/datasets/movielens/ml-25m.zip", zip_path)

    # Convert movie dataset
    converted_movies_path = os.path.join(output_path, "movies_converted.parquet")
    if not os.path.exists(converted_movies_path):
        movies = cudf.read_csv(os.path.join(output_path, "ml-25m/movies.csv"))
        movies["genres"] = movies["genres"].str.split("|")
        movies = movies.drop("title", axis=1)
        movies.to_parquet(converted_movies_path)

    # Split ratings into train/test
    train_path = os.path.join(output_path, "train.parquet")
    valid_path = os.path.join(output_path, "valid.parquet")
    if not os.path.exists(train_path) and not os.path.exists(valid_path):
        ratings = cudf.read_csv(os.path.join(output_path, "ml-25m", "ratings.csv"))
        ratings = ratings.drop("timestamp", axis=1)
        train, valid = train_test_split(ratings, test_size=0.2, random_state=42)
        train.to_parquet(train_path)
        valid.to_parquet(valid_path)

    return MovieLensData(train_path, valid_path, converted_movies_path)
