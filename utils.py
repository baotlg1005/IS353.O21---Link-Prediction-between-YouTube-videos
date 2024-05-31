import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import csv

# Hashes string to create unique integer id
def convertStrToUniqueInt(token):
    return int(str(hash(token))[:7])

# Gets the id of the graph from a dictionary id
def getGraphId(video_id, dict_to_graph):
    return dict_to_graph[video_id]

# Gets the id of the graph from a dictionary id
def getVideoId(node_id, graph_to_dict):
    return graph_to_dict[node_id]

# Adds node to the graph and gives node a graph id
def addNodeToGraph(G, video_dict_id, dict_to_graph, graph_to_dict, current_graph_counter):
    if video_dict_id not in dict_to_graph:
        dict_to_graph[video_dict_id] = current_graph_counter
        graph_to_dict[current_graph_counter] = video_dict_id
        G.add_node(current_graph_counter)
        current_graph_counter += 1
    return current_graph_counter

# Loads undirected graph based on video id dictionary
def load_graph_undirected(video_dict_list):
    G = nx.Graph()
    dict_to_graph = {}
    graph_to_dict = {}
    current_graph_counter = 0

    for video_node in video_dict_list.values():
        video_dict_id = video_node['video_id']
        current_graph_counter = addNodeToGraph(G, video_dict_id, dict_to_graph, graph_to_dict, current_graph_counter)
        chosen_graph_id = dict_to_graph[video_dict_id]

        related_dict_ids = video_node['related_ids']
        for related_dict_id in related_dict_ids:
            current_graph_counter = addNodeToGraph(G, related_dict_id, dict_to_graph, graph_to_dict, current_graph_counter)
            related_graph_id = dict_to_graph[related_dict_id]
            if not G.has_edge(chosen_graph_id, related_graph_id):
                G.add_edge(chosen_graph_id, related_graph_id)
    nx.write_edgelist(G, 'youtube_graph.txt', data=False)
    return G, dict_to_graph, graph_to_dict

# Loads the youtube dataset file and constructs a dictionary
def load_file(fname):
    fieldnames = ['video_id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments', 'related_ids']
    video_dict_list = {}
    with open(fname) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            new_video_node = {}
            if len(row) >= len(fieldnames):
                for i in range(len(fieldnames) - 1):
                    new_video_node[fieldnames[i]] = row[i]
                related_video_list = row[len(fieldnames)-1 : ]
                new_video_node['related_ids'] = related_video_list
                video_dict_list[row[0]] = new_video_node
    return video_dict_list

# Plots a bar graph
def plot_barGraph(data, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    names = data[0]
    values = data[1]
    plt.bar(range(len(names)), values, align='center')
    plt.xticks(range(len(names)), names)
    plt.show()

# Plots a histogram
def plot_hist(data, x_label, y_label):
    n, bins, patches = plt.hist(x=data, bins=100)
    plt.grid(axis='y')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    return n, bins, patches

# Loads networkX graph
def load_graph_networkX(fname_edgelist):
    G = nx.read_edgelist(fname_edgelist, nodetype=int)
    return G

# Gets all the categories in the current dictionary
def get_all_categories(video_dict_list):
    categories = set()
    for video_id in video_dict_list:
        categories.add(video_dict_list[video_id]['category'])
    return list(categories)

# Plots networkX graph with color and node_sizes
def plot_graph_networkX(G, graph_to_dict, video_dict_list):
    node_sizes = [10.0 if graph_to_dict[int(node)] in video_dict_list else 0.01 for node in G]
    categories = get_all_categories(video_dict_list)
    node_colors = []
    for node in G:
        if graph_to_dict[int(node)] in video_dict_list:
            node_colors.append((categories.index(video_dict_list[getVideoId(int(node), graph_to_dict)]['category']) + 1.0) / float(len(categories)))
        else:
            node_colors.append(0.0)
    nx.draw_kamada_kawai(G, with_labels=False, node_size=node_sizes, node_color=node_colors, cmap=plt.get_cmap('viridis'))
    plt.show()
    plt.savefig('youtube_graph.png')

# Example usage (uncomment to run):
# video_dict_list = load_file('youtube_data.txt')
# G, dict_to_graph, graph_to_dict = load_graph_undirected(video_dict_list)
# plot_graph_networkX(G, graph_to_dict, video_dict_list)
