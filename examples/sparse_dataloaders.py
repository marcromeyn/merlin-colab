import os
import shutil
from tqdm import tqdm

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
os.environ["TF_MEMORY_ALLOCATION"] = "0.7"  # fraction of free memory

import pandas as pd
import tensorflow as tf
from tensorflow.experimental.dlpack import to_dlpack as tf_to_dlpack

import cudf
import dask_cudf
import nvtabular as nvt
from nvtabular.loader.tensorflow import KerasSequenceLoader
from merlin_colab.movielens import MovieLensData
from merlin_colab import nvt_extended


from tensorflow.python.keras.engine.data_adapter import GeneratorDataAdapter, KerasSequenceAdapter, is_none_or_empty, unpack_x_y_sample_weight, _process_tensorlike
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape


def better_init(self,
            x,
            y=None,
            sample_weights=None,
            shuffle=False,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10,
            model=None,
            **kwargs):
    if not is_none_or_empty(y):
        raise ValueError("`y` argument is not supported when using "
                        "`keras.utils.Sequence` as input.")
    if not is_none_or_empty(sample_weights):
        raise ValueError("`sample_weight` argument is not supported when using "
                        "`keras.utils.Sequence` as input.")

    self._size = len(x)
    self._shuffle_sequence = shuffle
    self._keras_sequence = x
    self._enqueuer = None
    
    # Generators should never shuffle as exhausting the generator in order to
    # shuffle the batches is inefficient.
    kwargs.pop("shuffle", None)

    if not is_none_or_empty(y):
        raise ValueError("`y` argument is not supported when using "
                        "python generator as input.")
    if not is_none_or_empty(sample_weights):
        raise ValueError("`sample_weight` argument is not supported when using "
                        "python generator as input.")

    super(GeneratorDataAdapter, self).__init__(x, y, **kwargs)

    # Since we have to know the dtype of the python generator when we build the
    # dataset, we have to look at a batch to infer the structure.
    peek, x = self._peek_and_restore(x)
    peek = self._standardize_batch(peek)
    peek = _process_tensorlike(peek)

    # Need to build the Model on concrete input shapes.
    if model is not None and not model.built:
        concrete_x, _, _ = unpack_x_y_sample_weight(peek)
        model.distribute_strategy.run(
            lambda x: model(x, training=False), args=(concrete_x,))

    self._first_batch_size = int(nest.flatten(peek)[0].shape[0])

    def _get_dynamic_shape(t):
        shape = t.shape
        # Unknown number of dimensions, `as_list` cannot be called.
        if shape.rank is None:
            return shape
        return tensor_shape.TensorShape([None for _ in shape.as_list()])

    output_shapes = nest.map_structure(_get_dynamic_shape, peek)
    output_types = nest.map_structure(lambda t: t.dtype, peek)

    # Note that dataset API takes a callable that creates a generator object,
    # rather than generator itself, which is why we define a function here.
    generator_fn = self._handle_multiprocessing(x, workers, use_multiprocessing,
                                                max_queue_size)

    def wrapped_generator():
        for data in generator_fn():
            yield self._standardize_batch(data)

    # dataset = dataset_ops.DatasetV2.from_generator(
    #     wrapped_generator, output_types, output_shapes=output_shapes)

    dataset = dataset_ops.DatasetV2.from_generator(
        wrapped_generator,
        output_signature=({
            "userId": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
            "movieId": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
            "genres": tf.SparseTensorSpec(dtype=tf.int64)
        }, tf.TensorSpec(shape=(None, 1), dtype=tf.int8)))

    if workers == 1 and not use_multiprocessing:
        dataset = dataset.prefetch(1)

    self._dataset = dataset


KerasSequenceAdapter.__init__ = better_init


SPARSE = True


DATA_DIR = "/romeyn/notebooks/movielens/data/"
movielens = MovieLensData.load(DATA_DIR)

CATEGORICAL_COLUMNS = ["userId", "movieId"]
LABEL_COLUMNS = ["rating"]
TRANSFORMED_DIR = "movielens/transformed/"

cat_features = (["userId", "movieId"] 
          >> nvt.ops.JoinExternal(movielens.movies_df, on=["movieId"])
          >> nvt.ops.Categorify()
          )

# Make rating a binary target
rating = nvt.ColumnGroup(["rating"]) >> (lambda col: (col > 3).astype("int8"))

output = cat_features + rating
workflow = nvt.Workflow(output)
transformed_data = movielens.transform(TRANSFORMED_DIR, workflow,
                                       continuous_features=[],
                                       categorical_features=CATEGORICAL_COLUMNS + ["genres"],
                                       targets=LABEL_COLUMNS)


layer_dims = [512, 256, 128]
dropout = 0.5

inputs = transformed_data.create_keras_inputs(for_prediction=SPARSE, sparse_columns=["genres"])
x = transformed_data.create_default_input_layer(workflow)(inputs)
for dim in layer_dims:
    x = tf.keras.layers.Dense(dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

model = tf.keras.Model(inputs=inputs, outputs=x)
model.compile("sgd", "binary_crossentropy", 
              run_eagerly=False,
              metrics=[tf.keras.metrics.AUC(), 
                       tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.BinaryCrossentropy()])



BATCH_SIZE = 10000


# KerasSequenceAdapter.__init__ = new_init



train_dataset_tf = transformed_data.train_tf_dataset(BATCH_SIZE, sparse=SPARSE)
# train_dataset_tf = tf.data.Dataset.from_generator(
#     lambda: transformed_data.train_tf_dataset(BATCH_SIZE, sparse=True),
#     output_signature=({
#         "userId": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
#         "movieId": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
#         "genres": tf.SparseTensorSpec(dtype=tf.int64)
#     }, tf.TensorSpec(shape=(None, 1), dtype=tf.int8))
# )


model.fit(train_dataset_tf, epochs=10)