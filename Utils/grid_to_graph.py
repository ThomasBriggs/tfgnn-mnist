import networkx as nx

def graph_generator(data):
    G = nx.grid_2d_graph(data.shape[1],data.shape[2])
    nx.set_node_attributes(G, 0, "value")
    for k in range(data.shape[0]):
        for i, iv in enumerate(data[k]):
            for j, jv in enumerate(iv):
                G.nodes[i, j]["value"] = jv
        yield G