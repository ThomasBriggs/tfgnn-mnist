import tensorflow as tf
import tensorflow_gnn as tfgnn

TYPE_SPEC = {
    "mnist_graph": tfgnn.GraphTensorSpec(
        {
            'context': tfgnn.ContextSpec(
                {
                    # 'label': tf.TensorSpec(
                    #     shape=(1,), dtype=tf.uint8, name=None
                    # )
                },
                tf.TensorShape([]),
                tf.int32,
                None),

            'node_sets': {
                'pixel': tfgnn.NodeSetSpec(
                    {
                        'features': {
                            'hidden_state': tf.TensorSpec(
                                shape=(784, 1), dtype=tf.float64, name=None
                            )
                        },
                        'sizes': tf.TensorSpec(
                            shape=(1,), dtype=tf.int32, name=None
                        )
                    },
                    tf.TensorShape([]),
                    tf.int32,
                    None
                )
            },

            'edge_sets': {
                'connected': tfgnn.EdgeSetSpec(
                    {
                        'features': {},
                        'sizes': tf.TensorSpec(
                            shape=(1,), dtype=tf.int32, name=None
                        ),
                        'adjacency': tfgnn.AdjacencySpec(
                            {
                                '#index.0': tf.TensorSpec(
                                    shape=(1512,), dtype=tf.int64, name=None
                                ),
                                '#index.1': tf.TensorSpec(
                                    shape=(1512,), dtype=tf.int64, name=None
                                )
                            },
                            tf.TensorShape([]),
                            tf.int32,
                            {'#index.0': 'pixel', '#index.1': 'pixel'}
                        )
                    },
                    tf.TensorShape([]),
                    tf.int32,
                    None)
            }
        },
        tf.TensorShape([]),
        tf.int32,
        None
    )
}
