import utils
from collections import defaultdict
import numpy as np
import networkx as nx

# Partitions the dictionary list based on the value to be partitioned on (such as category)
def partition_dict(video_dict_list, partition_key, numeric_bins=None):
    partitioned_dict = defaultdict(list)
    for dict_id in video_dict_list:
        partition_value = video_dict_list[dict_id][partition_key]
        if numeric_bins is not None:
            for i in range(len(numeric_bins)):
                if float(partition_value) <= numeric_bins[i]:
                    partition_value = numeric_bins[i]
                    break
        partitioned_dict[partition_value].append(dict_id)
    return partitioned_dict

# Gets the distribution of views of all videos and plots histogram
# Returns a dictionary {views: [list of video_id]}
def get_views_distribution(fname):
    video_dict_list = utils.load_file(fname)
    views = [int(video_dict_list[video_id]['views']) for video_id in video_dict_list]
    sorted_views = sorted(views)
    n, bins, patches = utils.plot_hist(sorted_views[:100], 'View count', 'Frequency')
    n2, bins2, patches2 = utils.plot_hist(sorted_views[100:200], 'View count', 'Frequency')
    return partition_dict(video_dict_list, 'views', bins)

# Gets the distribution of categories of all videos
# Returns a dictionary {video_category: [list of video_id]}
def get_categories_distribution(fname):
    video_dict_list = utils.load_file(fname)
    categories = partition_dict(video_dict_list, 'category')
    X = []
    Y = []
    for key in categories:
        X.append(key)
        Y.append(len(categories[key]))
    utils.plot_barGraph([X, Y], "Categories", "Frequency")
    return categories

# Gets the distribution of number of ratings of all videos and plots histogram
# Returns a dictionary {ratings: [list of video_id]}
def get_ratings_distribution(fname):
    video_dict_list = utils.load_file(fname)
    ratings = [float(video_dict_list[video_id]['ratings']) for video_id in video_dict_list]
    sorted_ratings = sorted(ratings)
    n, bins, patches = utils.plot_hist(sorted_ratings, 'Number of ratings', 'Frequency')
    return partition_dict(video_dict_list, 'ratings', bins)

# Gets the distribution of actual ratings (rate) of all videos and plots histogram
# Returns a dictionary {rate: [list of video_id]}
def get_rate_distribution(fname):
    video_dict_list = utils.load_file(fname)
    rate = [float(video_dict_list[video_id]['rate']) for video_id in video_dict_list]
    sorted_ratings = sorted(rate)
    n, bins, patches = utils.plot_hist(sorted_ratings, 'Rate', 'Frequency')
    return partition_dict(video_dict_list, 'rate', bins)

# Gets the probability that same genre nodes are linked
def genre_probability(fname, fname_extended):
    video_dict_list = utils.load_file(fname)
    video_dict_list_extended = utils.load_file(fname_extended)
    G, dict_to_graph, graph_to_dict = utils.load_graph_undirected(video_dict_list)
    video_dict_list_extended.update(video_dict_list)

    equals_edges = 0.0
    total_edges = 0.0

    for edge in G.edges():
        try:
            srcGraphId, dstGraphId = edge
            total_edges += 1
            if video_dict_list_extended[graph_to_dict[srcGraphId]]['category'] == video_dict_list_extended[graph_to_dict[dstGraphId]]['category']:
                equals_edges += 1
        except:
            continue
    print(equals_edges / float(total_edges))
    return equals_edges / float(total_edges)

# Plots the different distributions based on 0 and 1 crawl
fname = './dataset/0222/0.txt'
fname_extended = './dataset/0222/1.txt'

views_dict = get_views_distribution(fname)
categories_dict = get_categories_distribution(fname)
ratings_dict = get_ratings_distribution(fname)
rates_dict = get_rate_distribution(fname)
