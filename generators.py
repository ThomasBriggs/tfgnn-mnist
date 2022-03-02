import tensorflow_gnn as tfgnn
import tensorflow as tf
import numpy as np
import networkx as nx


def graph_generator(data):
    G = nx.grid_2d_graph(data.shape[1], data.shape[2])
    nx.set_node_attributes(G, 0, "value")
    for k in range(data.shape[0]):
        for i, iv in enumerate(data[k]):
            for j, jv in enumerate(iv):
                G.nodes[i, j]["value"] = jv
        yield G


def graph_tensor_generator(data, lbl):
    graph_gen = graph_generator(data)
    for i, graph in enumerate(graph_gen):
        graph = nx.convert_node_labels_to_integers(
            graph, label_attribute="graph_index")
        num_edges = graph.size()
        edge_list = np.asarray(graph.edges)
        edges = tfgnn.EdgeSet.from_fields(
            sizes=[num_edges],
            adjacency=tfgnn.Adjacency.from_indices(
                source=("pixel", edge_list[:, 0]),
                target=("pixel", edge_list[:, 1])
            )
        )

        num_nodes = graph.number_of_nodes()
        features = [x[1]["value"] for x in graph.nodes(data=True)]
        nodes = tfgnn.NodeSet.from_fields(
            features= {
                "hidden_state": np.asarray(features).reshape((1, 784))
            },
            sizes=[num_nodes]
        )

        context = tfgnn.Context.from_fields(features=None)

        # context = tfgnn.Context.from_fields(
        #     features={
        #         "label": np.asarray([lbl[i]])
        #     },
        #     shape = ()
        # )

        graph_tensor = tfgnn.GraphTensor.from_pieces(
            edge_sets={"connected": edges},
            node_sets={"pixel": nodes},
            context=context
        )

        yield graph_tensor, lbl[i]


def load_dataset_from_data(data, lbl, batch_size, graph_type_spec):
    def generator():
        graphs = graph_tensor_generator(data, lbl)
        yield from graphs

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=(
            graph_type_spec, tf.TensorSpec(shape=(), dtype=tf.int64))
    )

    return (
        dataset
        .cache()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .map(lambda x, y: (x.merge_batch_to_components(), y))
    )
