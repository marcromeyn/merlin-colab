import shutil
from dataclasses import dataclass
import os

import cudf
import nvtabular as nvt
from nvtabular.utils import download_file
from sklearn.model_selection import train_test_split

from merlin_colab.nvt_extended import Dataset


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
