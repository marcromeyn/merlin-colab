import glob
import math
import os
import typing as T
from dataclasses import dataclass

import dask_cudf
import nvtabular as nvt
import tensorflow as tf
from nvtabular.framework_utils.tensorflow.layers import DenseFeatures
from nvtabular.loader.tensorflow import KerasSequenceLoader as _KerasSequenceLoader, KerasSequenceValidater
from tensorflow.python.tpu.tpu_embedding_v2_utils import (
    FeatureConfig,
    TableConfig,
)
from tensorflow.python.ops import init_ops_v2


class EmbeddingsLayer(tf.keras.layers.Layer):
    """Mimics the API of [TPUEmbedding-layer](https://github.com/tensorflow/recommenders/blob/main/tensorflow_recommenders/layers/embedding/tpu_embedding_layer.py#L221)
    from TF-recommenders, use this for efficient embeddings on CPU or GPU."""

    def __init__(self, feature_config: T.Dict[str, FeatureConfig], **kwargs):
        super().__init__(**kwargs)
        self.feature_config = feature_config

    def build(self, input_shapes):
        self.embedding_tables = {}
        tables: T.Dict[str, TableConfig] = {}
        for name, feature in self.feature_config.items():
            table: TableConfig = feature.table
            if table.name not in tables:
                tables[table.name] = table

        for name, table in tables.items():
            shape = (table.vocabulary_size, table.dim)
            self.embedding_tables[name] = self.add_weight(
                name="{}/embedding_weights".format(name),
                trainable=True,
                initializer=table.initializer,
                shape=shape,
            )
        super().build(input_shapes)

    def call(self, inputs, **kwargs):
        embedded_outputs = {}
        for name, val in inputs.items():
            if name in self.feature_config:
                table: TableConfig = self.feature_config[name].table
                table_var = self.embedding_tables[table.name]
                if isinstance(val, tf.SparseTensor):
                    embeddings = tf.nn.safe_embedding_lookup_sparse(
                        table_var, tf.cast(val, tf.int32), None, combiner=table.combiner
                    )
                else:
                    # embeddings = tf.gather(table_var, tf.cast(val, tf.int32))
                    embeddings = tf.gather(table_var, val[:, 0])
                embedded_outputs[name] = embeddings

        return embedded_outputs

    @classmethod
    def from_nvt_workflow(cls, workflow: nvt.Workflow, combiner="mean") -> "EmbeddingsLayer":
        embedding_size = nvt.ops.get_embedding_sizes(workflow)
        feature_config: T.Dict[str, FeatureConfig] = {}
        for name, (vocab_size, dim) in embedding_size.items():
            feature_config[name] = FeatureConfig(
                TableConfig(
                    vocabulary_size=vocab_size,
                    dim=dim,
                    combiner=combiner,
                    name=name,
                    initializer=init_ops_v2.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(dim)),
                )
            )

        return cls(feature_config)


class InputLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        continuous_feature_names: T.List[str],
        embeddings_layer: T.Optional[EmbeddingsLayer] = None,
        aggregation="concat",
        sequential=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sequential = sequential
        self.continuous_feature_names = continuous_feature_names
        self.embeddings_layer = embeddings_layer
        self.aggregation = aggregation

    def transform_categorical_features(self, inputs):
        if not self.embeddings_layer:
            return []

        embedded = self.embeddings_layer(inputs)
        embedded_tensors = [embedded[f] for f in self.categorical_feature_names]

        return embedded_tensors

    @property
    def categorical_feature_names(self):
        if not getattr(self, "_sorted_cat_feature_names", None):
            self._sorted_cat_feature_names = sorted(self.embeddings_layer.feature_config.keys())

        return self._sorted_cat_feature_names

    def transform_continuous_features(self, inputs):
        return [inputs[f] for f in self.continuous_feature_names]

    def post_process(self, continuous_features, categorical_features):
        if self.aggregation == "stack":
            return tf.stack([*continuous_features, *categorical_features], axis=-1)

        continuous_features = [tf.expand_dims(tf.cast(f, tf.float32), -1) for f in continuous_features]

        return tf.concat([*continuous_features, *categorical_features], axis=-1)

    def call(self, inputs, **kwargs):
        continuous = self.transform_continuous_features(inputs)
        categorical = self.transform_categorical_features(inputs)

        return self.post_process(continuous, categorical)


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
        return KerasSparseSequenceLoader(
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
        return KerasSparseSequenceLoader(
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

    def create_default_input_layer(self, workflow: nvt.Workflow):
        return InputLayer(self.continuous_features, EmbeddingsLayer.from_nvt_workflow(workflow))

    def create_keras_inputs(self):
        inputs = {}

        for col in self.continuous_features:
            inputs[col] = tf.keras.Input(name=col, dtype=tf.float32, shape=(None, 1))

        for col in self.categorical_features:
            inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(None, 1))

        return inputs


class KerasSparseSequenceLoader(_KerasSequenceLoader):
    def __init__(self, paths_or_dataset, batch_size, label_names, feature_columns=None, cat_names=None, cont_names=None,
                 engine=None, shuffle=True, seed_fn=None, buffer_size=0.1, device=None, parts_per_chunk=1,
                 reader_kwargs=None, global_size=None, global_rank=None, drop_last=False):
        super().__init__(paths_or_dataset, batch_size, label_names, feature_columns, cat_names, cont_names, engine,
                         shuffle, seed_fn, buffer_size, device, parts_per_chunk, reader_kwargs, global_size,
                         global_rank, drop_last)
        self._map_fns = []

    def map(self, fn):
        self._map_fns.append(fn)

        return self

    def _handle_tensors(self, cats, conts, labels):
        X = {}
        for tensor, names in zip([cats, conts], [self.cat_names, self.cont_names]):
            lists = {}
            if isinstance(tensor, tuple):
                tensor, lists = tensor
            names = [i for i in names if i not in lists]

            # break list tuples into two keys, with postfixes
            # TODO: better choices for naming?
            list_columns = list(lists.keys())
            for column in list_columns:
                values, nnzs = lists.pop(column)
                values = values[:, 0]
                row_lengths = tf.cast(nnzs[:, 0], tf.int32)
                lists[column] = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_sparse()

            # now add in any scalar tensors
            if len(names) > 1:
                tensors = tf.split(tensor, len(names), axis=1)
                lists.update(zip(names, tensors))
            elif len(names) == 1:
                lists[names[0]] = tensor
            X.update(lists)

        # TODO: use dict for labels as well?
        # would require output layers to match naming
        if len(self.label_names) > 1:
            labels = tf.split(labels, len(self.label_names), axis=1)

        output = X, labels
        for map_fn in self._map_fns:
            output = map_fn(*output)

        return output
