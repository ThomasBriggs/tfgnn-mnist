def print_history(history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy',
                'Validation Accuracy'])
    plt.title('Accuracy Curves')

    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training Loss',
                'Validation Loss'])
    plt.title('Loss Curves')
    plt.show()


def print_2d_graph(G):
    import matplotlib.pyplot as plt
    import networkx as nx
    pos = {(x, y):(y, -x) for x, y in G.nodes()}
    plt.figure(figsize=(plt.rcParams["figure.figsize"][0]*2, plt.rcParams["figure.figsize"][1]*2))
    colors = [x["value"]/255 for x in G.nodes().values()]
    nx.draw(G, pos, node_size=70, node_color=colors)