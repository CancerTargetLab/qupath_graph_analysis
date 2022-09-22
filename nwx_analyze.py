# Path: libpysal_approach.py
#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libpysal import weights
import networkx as nx
import re
import seaborn as sns
from scipy.spatial import distance
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', type=str, help='path to the csv file')
parser.add_argument('--results_dir', type=str, help='path to the results directory')
parser.add_argument('--critical_distance', type=float, help='critical distance')

args = parser.parse_args()


def network_plot(df, image, critical_distance, results_dir, prepend=''):
    """
    This function subsets images from a dataframe and plots the network graph as well as the corresponding aac
    """
    out_file = f'{prepend}image_{image}_distance_{critical_distance}.png'
    # extract the image
    one_pic = df.groupby('Image').get_group(image).reset_index(drop=True)
    #print(one_pic.head())
    # extract the spatial coordinates
    coordinates = one_pic.iloc[:, [5, 6]]
    # extract the cell types
    cell_types = one_pic.Class
    #print(cell_types)

    # Creating a graph from coordinates
    positions = coordinates.to_numpy()
    # create a weights object   
    w = weights.DistanceBand.from_array(positions, threshold=critical_distance)
    # create a networkx graph from the weights object
    G = nx.from_numpy_matrix(w.full()[0])

    # calculate euclidean distance
    # dist = distance.pdist(positions)
    # create a networkx graph from the distance matrix
    # G = nx.from_numpy_matrix(distance.squareform(dist))
    # add the coordinates to the graph  
    for i, (x, y) in enumerate(positions):
        G.nodes[i]['pos'] = (x, y)
    # add the cell types to the graph   
    for i, cell_type in enumerate(cell_types):
        G.nodes[i]['cell_type'] = cell_type
    # extract the cell types from the graph
    # cell_types = [G.nodes[i]['cell_type'] for i in range(len(G.nodes))]
    # extract the coordinates from the graph
    # positions = [G.nodes[i]['pos'] for i in range(len(G.nodes))]
    # Plotting the graph
    aac = nx.attribute_assortativity_coefficient(G, 'cell_type')
    fig, ax = plt.subplots(figsize=(10, 10))
    # automatize cell type name and colors
    color_map = ["red" if cell_type == "PANCK" else "blue" for cell_type in
                 cell_types]
    nx.draw(G, positions, node_size=10, node_color=color_map, with_labels=False,
            ax=ax)
    plt.title('attribute assortativity coefficient ' + str(aac))
    plt.savefig(f'{results_dir}/{out_file}')
    # construct dict with image name, centrality measures, ratio, aac, number of cells in classes and islands

if __name__ == "__main__":
    if args.file:
        df = pd.read_csv(args.file, sep='\t', low_memory=False).dropna(axis=1)
        for image in df.Image.unique():
            network_plot(df, image, args.critical_distance, args.results_dir)
    else:
        tma1_data = pd.read_csv('./ML_TMA1/obj_class_TMA1.csv',
                                low_memory=False, decimal=',').dropna(axis=1)
        tma2_data = pd.read_csv('./ML_TMA2/obj_class_TMA2.csv',
                                low_memory=False, decimal=',').dropna(axis=1)
        for image in tma1_data.Image.unique():
            network_plot(tma1_data, image, 50, results_dir='Results',
                         prepend='TMA1')
        for image in tma2_data.Image.unique():
            network_plot(tma2_data, image, 50, results_dir='Results',
                         prepend='TMA2')

