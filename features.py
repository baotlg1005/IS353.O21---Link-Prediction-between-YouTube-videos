import utils
import networkx as nx
from collections import defaultdict

# Using this video dict list, extract proximity features from the descriptions of nodes
def proximity_features(video_dict_list):
    pass

def aggregate_features(video_dict_list, vid1, vid2, fields):
    feature_size = len(fields) * 2
    # If both exist in the video dict list, calculate the aggregates
    if vid1 in video_dict_list and vid2 in video_dict_list:
        features = []
        # For each field, aggregate by taking sum and difference
        for field in fields:
            num1 = video_dict_list[vid1][field]
            num2 = video_dict_list[vid2][field]
            features.append(float(num1) + float(num2))
            features.append(abs(float(num1) - float(num2)))

        genre1 = video_dict_list[vid1]['category']
        genre2 = video_dict_list[vid2]['category']
        if genre1 == genre2:
            features.append(1)
        else:
            features.append(0)
        return features
    else:
        return None

def common_neighbors_similarity(n1, n2, neighbors):
    return len(neighbors[n1] & neighbors[n2])

def jaccard_similarity(n1, n2, neighbors):
    intersection = neighbors[n1] & neighbors[n2]
    union = neighbors[n1] | neighbors[n2]
    return float(len(intersection)) / len(union)

def get_neighbors(G):
    neighbors = defaultdict(set)
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            neighbors[node].add(neighbor)
    return neighbors

def topological_features(n1, n2, neighbors):
    c = common_neighbors_similarity(n1, n2, neighbors)
    j = jaccard_similarity(n1, n2, neighbors)
    features = [c, j]
    return features

def get_feature_dict(G, video_dict_list, graph_to_dict, neighbors, fields):
    # Loop through pairs of nodes in the graph
    feature_dict = {} 

    for node1 in graph_to_dict:
        for node2 in graph_to_dict:
            vid1 = graph_to_dict[node1]
            vid2 = graph_to_dict[node2]

            agg_features = aggregate_features(video_dict_list, vid1, vid2, fields)
            topo_features = topological_features(node1, node2, neighbors)

            # Combine all the features    
            if agg_features:
                features = agg_features + topo_features
                feature_dict[(node1, node2)] = features

    return feature_dict

def extract_features(video_dict_list, graph_to_dict, neighbors, fields, node1, node2):
    vid1 = graph_to_dict[node1]
    vid2 = graph_to_dict[node2]
    agg_features = aggregate_features(video_dict_list, vid1, vid2, fields)

    return agg_features

    # FLAG: for now ignore the topological features
    # topo_features = topological_features(node1, node2, neighbors)

    # if agg_features:
    #     return agg_features + topo_features
    # return None

def get_vars(fname, fname_extended):
    video_dict_list = utils.load_file(fname)
    video_dict_list_extended = utils.load_file(fname_extended)
    G, dict_to_graph, graph_to_dict = utils.load_graph_undirected(video_dict_list)
    neighbors = get_neighbors(G)
    fields = ['views', 'ratings', 'comments']
    
    video_dict_list.update(video_dict_list_extended)

    return G, video_dict_list, graph_to_dict, neighbors, fields

    # # Use the extended video dict list as it will contain more information on the videos from the crawl
    # feature_dict = extract_features(G, video_dict_list_extended, graph_to_dict)
    # return feature_dict

# Test to see if everything is working
def main():
    fname = './dataset/0222/0.txt'
    fname_extended = './dataset/0222/1.txt'
    G, video_dict_list, graph
