import tensorflow as tf


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _bytes_feature(values):
    if isinstance(values[0], str):
        values = values.astype(bytes)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


class TFExample(object):
    def __init__(self, input_features):
        self.convert_funcs = {}
        self.feature_specs = {}
        self.input_features = input_features.copy()
        for k, fea in input_features.items():
            if fea.dtype in [tf.int8, tf.int16, tf.int32, tf.int32, tf.int64]:
                self.convert_funcs[fea.name] = _int64_feature
                dtype = tf.int64
            elif fea.dtype in [tf.float32, tf.float64]:
                self.convert_funcs[fea.name] = _float_feature
                dtype = tf.float32
            elif fea.dtype in [tf.string]:
                self.convert_funcs[fea.name] = _bytes_feature
                dtype = tf.string
            self.feature_specs[fea.name] = tf.io.FixedLenFeature(fea.shape, dtype)

    def serialize(self, example):
        for k, v in example.items():
            example[k] = self.convert_funcs[k](v.flatten())
        example_proto = tf.train.Example(features=tf.train.Features(feature=example))
        return example_proto.SerializeToString()

    def deserialize(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.feature_specs)


def gather_seqs(seqs, gather_inds):
    dtype = gather_inds.dtype
    n = len(gather_inds.get_shape())
    shape = tf.shape(gather_inds)
    r_shape = [shape[0]]
    inds=tf.range(shape[0]); tile=[1]
    inds = tf.expand_dims(inds, -1)
    for i in range(1, n-1):
        inds=tf.expand_dims(inds, i)
        inds = tf.tile(inds, tile+[shape[i]] +[1])
        r_shape.append(shape[i])
        r_inds = tf.range(shape[i])
        r_inds = tf.expand_dims(r_inds, -1)
        for j in tile[::-1]:
            r_inds = tf.expand_dims(r_inds, 0)
        r_inds = r_inds * tf.ones(r_shape+[1], tf.int32)
        inds = tf.concat([inds, r_inds], -1)
        tile += [1]
    inds = tf.tile(tf.expand_dims(inds, -2), tile + [shape[-1]] + [1])
    gather_inds = tf.expand_dims(gather_inds, -1)
    gather_inds = tf.concat([inds, gather_inds], -1)
    rst = tf.gather_nd(seqs, gather_inds)
    return rst

def flat_gradients(grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
    '''Convert gradients if it's tf.IndexedSlices.
    When computing gradients for operation concerning `tf.gather`, the type of gradients
    '''
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            grads_or_idx_slices.dense_shape
        )
    return grads_or_idx_slices
