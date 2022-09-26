#!/usr/bin/env python
# coding: utf-8
# author: Mattis
# email: mattisknulst@gmail.com
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from libpysal import weights

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--file', type=str, help='path to the csv file',
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
parser.add_argument('-s', '--sep', help='csv separator, default tab',
                    default='\t')
parser.add_argument('-d', '--decimal', help='float decimal sign, default .',
                    default='.')
parser.add_argument('-t', '--tiff_dir', type=str,
                    help='path to the tiff directory', required=True)

args = parser.parse_args()


def make_graph(df, critical_distance):
    """
    Make a graph from a dataframe with x, y coordinates and cell type
    :param df: dataframe with x, y coordinates and cell type (qupath output)
    :param critical_distance: distance in pixels
    :return: graph and weights
    """
    # extract the spatial coordinates
    coordinates = df.loc[:, ["Centroid X px", "Centroid Y px"]]
    # extract the cell types
    cell_types = df.Class
    # Creating a graph from coordinates
    positions = (coordinates.to_numpy())

    # create a weights object
    w = weights.DistanceBand.from_array(positions, threshold=critical_distance)

    # create a networkx graph from the weights object
    G = nx.from_numpy_matrix(w.full()[0])

    # add the coordinates to the graph
    for i, (x, y) in enumerate(positions):
        G.nodes[i]['pos'] = (x, y)

    # add the cell types to the graph
    for i, cell_type in enumerate(cell_types):
        G.nodes[i]['cell_type'] = cell_type

    return G, w


def plot_graph(G, pair, results_dir, cell_type_filter, image_file,
               critical_distance, prepend=''):
    """
    This function plots the graph
    :param G: graph
    :param pair: cell type pair
    :param results_dir: path to the results directory
    :param cell_type_filter: cell type filter
    :param image_file: path to the image file
    :param critical_distance: critical distance
    :param prepend: string to prepend to the filename
    :return: None
    """
    # get image
    my_img = plt.imread(image_file)
    # we don't want colons in file names
    nice = pair.replace(':', '_')
    out_file = f'{nice}_{prepend}image_{image}_distance_{critical_distance}.png'
    cell_types = [cell_type for node, cell_type in G.nodes(data='cell_type')]
    # plot the graph
    fig = plt.figure(figsize=(10, 10))
    y_lim, x_lim = my_img.shape[0], my_img.shape[1]
    extent = 0, x_lim, 0, y_lim
    # transpose image

    my_img = np.flipud(my_img)
    # draw the image
    plt.imshow(my_img, cmap='gray', extent=extent, interpolation='nearest')
    # cell type name and colors
    c1 = f"{cell_type_filter[0]} (lime)"
    c2 = f"{cell_type_filter[1]} (magenta)"
    color_map = ["lime" if cell_type == cell_type_filter[0] else "magenta" for
                 cell_type in
                 cell_types]
    # draw graph on image
    nx.draw_networkx(G, pos=nx.get_node_attributes(G, 'pos'),
                     node_color=color_map, node_size=10, alpha=0.5,
                     edge_color='yellow', width=0.5, with_labels=False)

    plt.title(f'{c1} {c2} image {image} distance {critical_distance}px')

    plt.savefig(f'{results_dir}/{out_file}')


def calculate_statistics(df, G, w, cell_type_filter):
    """
    This function calculates the statistics
    :param df: dataframe with x, y coordinates and cell type (qupath output)
    :param G: graph
    :param w: weights object
    :param cell_type_filter: cell type filter
    :return: tuple of statistics
    """
    aac = nx.attribute_assortativity_coefficient(G, 'cell_type')
    # count nr of cells in each class
    n_cells_class_1 = df.loc[
        df['Class'] == cell_type_filter[0], 'Class'].count()
    n_cells_class_2 = df.loc[
        df['Class'] == cell_type_filter[1], 'Class'].count()
    # count number of islands
    n_islands = len(w.islands)
    # get all nodes belonging to class 1
    class_1_nodes = [node for node, cell_type in G.nodes(data='cell_type') if
                     cell_type == cell_type_filter[0]]
    # calculate group degree centrality for class 1
    try:
        centrality_measures = nx.group_degree_centrality(G, class_1_nodes)
    except Exception as e:
        print(e)
        print('happens when there is only one class!')
        centrality_measures = 'NA'
    # ratio is the proportion of class 1 cells in the network
    ratio = n_cells_class_1 / (n_cells_class_2 + n_cells_class_1)

    return aac, n_cells_class_1, n_cells_class_2, n_islands, centrality_measures, ratio


def network_plot(df, image, tiff_dir, critical_distance, results_dir,
                 prepend='',
                 pair='CD45:PANCK'):
    """
    main function for iterating image by image
    """
    cell_type_filter = pair.split(':')
    # class_1 = df.groupby('Class').get_group(cell_type_filter[0])
    # class_2 = df.groupby('Class').get_group(cell_type_filter[1])
    # get image file path
    image_file = tiff_dir + '/' + image
    # filter on cell type included in analysis
    filtered = df[(df['Class'] == cell_type_filter[0]) | (
                df['Class'] == cell_type_filter[1])].reset_index(drop=True)
    # extract the image
    one_pic = filtered.groupby('Image').get_group(image).reset_index(drop=True)
    # make graph
    G, w = make_graph(one_pic, critical_distance)
    # Plotting the graph
    plot_graph(G, pair, results_dir, cell_type_filter, image_file,
               critical_distance, prepend=prepend)
    # calculate statistics
    aac, n_cells_class_1, n_cells_class_2, n_islands, centrality_measures, ratio = calculate_statistics(
        one_pic, G, w, cell_type_filter)

    return pd.Series({'image': image,
                      'class_1': cell_type_filter[0],
                      'class_2': cell_type_filter[1],
                      'degree_centrality': centrality_measures,
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
                        'degree_centrality': [],
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
        try:
            out = pd.concat([network_plot(df,
                                          image=image,
                                          tiff_dir=args.tiff_dir,
                                          critical_distance=args.critical_distance,
                                          results_dir=args.results_dir,
                                          pair=args.pair).to_frame().T, out],
                            ignore_index=True)
        except Exception as e:
            print(e)
            continue

    # save the output
    nice = args.pair.replace(':', '_')
    out.to_csv(
        f'{args.results_dir}/results_{nice}_distance_{args.critical_distance}.csv',
        index=False)
