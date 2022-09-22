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
parser.add_argument('-i','--file', type=str, help='path to the csv file',
                    required=True)
parser.add_argument('-o', '--results_dir',
                    type=str,
                    help='path to the results directory'
                         ', default current directory',
                    default='.')
parser.add_argument('-c', '--critical_distance',
                    type=int, help='critical distance, default=30',
                    default=30)
parser.add_argument('-p', '--pair', type=str,
                    help='cell type pair', required=True)
parser.add_argument('-s', '--sep',help='csv separator, default tab',
                    default='\t')
parser.add_argument('-d', '--decimal',help='float decimal sign, default .',
                    default='.')

args = parser.parse_args()


def network_plot(df, image, critical_distance, results_dir, prepend='',
                 pair='CD45:PANCK'):
    """
    This function subsets images from a dataframe and plots the network graph as well as the corresponding aac
    """
    cell_type_filter = pair.split(':')
    #class_1 = df.groupby('Class').get_group(cell_type_filter[0])
    #class_2 = df.groupby('Class').get_group(cell_type_filter[1])
    filtered = df[(df['Class'] == cell_type_filter[0]) | (df['Class'] == cell_type_filter[1])].reset_index(drop=True)
    out_file = f'{pair}_{prepend}image_{image}_distance_{critical_distance}.png'
    # extract the image
    one_pic = filtered.groupby('Image').get_group(image).reset_index(drop=True)

    # extract the spatial coordinates
    coordinates = one_pic.iloc[:, [5, 6]]
    # extract the cell types
    cell_types = one_pic.Class


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

    # Plotting the graph
    aac = nx.attribute_assortativity_coefficient(G, 'cell_type')
    fig, ax = plt.subplots(figsize=(10, 10))
    # automatize cell type name and colors
    color_map = ["red" if cell_type == cell_type_filter[0] else "blue" for cell_type in
                 cell_types]
    nx.draw(G, positions, node_size=10, node_color=color_map, with_labels=False,
            ax=ax)
    plt.title('attribute assortativity coefficient ' + str(aac))
    plt.savefig(f'{results_dir}/{out_file}')
    # construct dict with image name, centrality measures, ratio, aac, number of cells in classes and islands
    centrality_measures = nx.degree_centrality(G)
    n_cells_class_1 = one_pic.loc[one_pic['Class'] == cell_type_filter[0], 'Class'].count()
    n_cells_class_2 = one_pic.loc[one_pic['Class'] == cell_type_filter[1], 'Class'].count()
    n_islands = len(w.islands)
    # ratio is the proportion of class 1 cells in the network
    ratio = n_cells_class_1 / (n_cells_class_2 + n_cells_class_1)
    return pd.Series({'image': image,
            'class_1': cell_type_filter[0],
            'class_2': cell_type_filter[1],
            #'degree_centrality': centrality_measures,
            'ratio 1/1+2': ratio,
            'aac': aac,
            'n_cells_class_1': n_cells_class_1,
            'n_cells_class_2': n_cells_class_2,
            'n_islands': n_islands})

if __name__ == "__main__":
    # create the output dictionary
    out = pd.DataFrame({'image': [],
            'class_1': [],
            'class_2': [],
            # 'degree_centrality': [],
            'ratio 1/1+2': [],
            'aac': [],
            'n_cells_class_1': [],
            'n_cells_class_2': [],
            'n_islands': []})
    # read the csv file
    df = pd.read_csv(args.file,
                     sep=args.sep,
                     decimal=args.decimal,
                     low_memory=False).dropna(axis=1)
    # run main function for all images
    for image in df.Image.unique():
        out = pd.concat([network_plot(df,
                     image=image,
                     critical_distance=args.critical_distance,
                     results_dir=args.results_dir,
                     pair=args.pair).to_frame().T, out], ignore_index=True)

    # save the output
    out.to_csv(f'{args.results_dir}/results_{args.pair}_distance_{args.critical_distance}.csv', index=False)


