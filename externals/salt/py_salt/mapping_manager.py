import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib

from matplotlib.patheffects import withStroke
from matplotlib.lines import Line2D
from networkx.drawing.nx_agraph import graphviz_layout

from . import general_utils as utils


class MappingExplorer():
  """MappingExplorer class with tools to explore the mapping of the
  taxonomy
  """
  def __init__(self, map_md_file_path, roots_md_file_path, std_col):
    if map_md_file_path is not None:
      assert os.path.isfile(map_md_file_path), f"Could not find file: {map_md_file_path}"

    self._map_md_file_path = map_md_file_path
    self._roots_md_file_path = roots_md_file_path
    self._std_col = std_col
    self._map_df = None

    self._necessary_cols = [self._std_col, 'dataset_label', 'dataset']

    self._leaf_nodes = None
    self._roots = None

    self._init_map_df()
    self._init_roots_dict()
    self._init_leaf()

    assert set(self._necessary_cols).issubset(set(self._map_df.columns))

  @property
  def leaf_nodes(self):
    return self._leaf_nodes

  @property
  def roots(self):
    return self._roots

  @property
  def map_df(self):
    return self._map_df

  @property
  def all_labels(self):
    return self._map_df[self._std_col].unique().tolist()

  ### --- Init functions

  def _init_map_df(self):
    """Load mapping from csv file in a dataframe
    """
    map_df = pd.read_csv(self._map_md_file_path, sep='\t')

    # Discard all empty rows
    map_df = map_df.dropna(how='all').reset_index(drop=True)

    # Replace nan values with empty strings
    map_df.replace(pd.NA, '', inplace=True)

    self._map_df = map_df


  def _init_roots_dict(self):
    """Read taxonomy's root dictionaries from json file

    Raises
    ------
    ValueError
        If a standardized root label is not found in the mapping dataframe
    """
    if os.path.isfile(self._roots_md_file_path):
      with open(self._roots_md_file_path, 'r', encoding='utf-8') as json_file:
        self._roots = json.load(json_file)
    else:
      self.generate_taxonomy_roots()

    for root in self._roots.keys():
      if root not in self._map_df[self._std_col].unique():
        raise ValueError('Not all root labels were found in the mapping file. '
                         'Perhaps the roots_md_file_path is incorect...')

  def _init_leaf(self):
    def get_empty_list_keys(d):
      empty_keys = []
      for k, v in d.items():
        if isinstance(v, list) and not v:  # If the value is an empty list
          empty_keys.append(k)
        elif isinstance(v, dict):  # If the value is a dictionary, recurse
          nested_empty_keys = get_empty_list_keys(v)
          if not nested_empty_keys and not v:  # If nested dictionary is empty
            empty_keys.append(k)
          else:
            empty_keys.extend(nested_empty_keys)
      return list(set(empty_keys))

    self._leaf_nodes = get_empty_list_keys(self._roots)


  def _compute_file_hash(self, file_path):
    """Compute hash for a file to detect changes."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
      buf = f.read()
      hasher.update(buf)
    return hasher.hexdigest()

  def generate_taxonomy_roots(self):
    """
    Generates a JSON file containing a hierarchical tree structure
    (dictionary) for each standardized root label. This method only
    regenerates the file if it does not exist or if the content has changed
    ensuring efficient updates.

    This method performs the following steps:
    1. Checks for an existing JSON file with tree data.
    2. Computes hashes to determine if there are changes to the mapping file.
    3. If changes are detected, prompts the user to overwrite the file.
    4. Generates a new tree structure for each root label in the mapping data.
    5. Saves the updated tree structure in JSON file and caches the data.
    """

    # Define the path to the JSON file that stores the tree structure
    root_file = Path(self._roots_md_file_path)

    # Check if JSON file already exists and if there are content changes
    if root_file.exists():
      # Compute hash for the existing roots file
      file_hash = self._compute_file_hash(self._roots_md_file_path)

      # Compute hash for the mapping file to check if data has changed
      current_map_hash = self._compute_file_hash(self._map_md_file_path)

      # Check if cached hash exists and matches the current file hash
      if hasattr(self, '_cached_map_hash') and file_hash == self._cached_map_hash:
        # If cached hash matches, no updates are needed
        print('Taxonomy roots are up-to-date. No regeneration required.')
        return

      # Update the cached map hash with the latest hash of the mapping file
      self._cached_map_hash = current_map_hash

    # Ask the user for permission to overwrite the existing roots file
    overwrite = utils.confirm_overwriting(self._roots_md_file_path,
                                          exit_python=False)

    # If user declines to overwrite return
    if not overwrite:
      return

    print('--- Generating taxonomy roots dictionary')

    # Init the DataFrame containing the mapping information
    self._init_map_df()

    # Init an empty dictionary to hold the root tree for each std label
    roots = {}

    # Loop through each unique std label in the mapping DataFrame
    for std_label in self._map_df[self._std_col].unique():
      # Retrieve the list of coarse labels for the current std label
      coarse_labels = self.get_coarse_labels_for_std_label(std_label)

      # Only create a new root tree if the std label has no coarse labels (root)
      if not coarse_labels:
        print(f'\tFound root label: "{std_label}". Calculating label tree...')

        # Generate the hierarchical tree structure for the current root label
        root_tree = self.get_dictionary_tree_for_std_label(std_label)

        # Basic structure validation: Ensure root tree has valid nested structure
        if isinstance(root_tree, dict) and std_label in root_tree:
          # Add the valid root tree to the roots dictionary
          roots[std_label] = root_tree
        else:
          # Warn if the structure of the generated root tree is invalid
          print(f'\tWarning: Structure of root "{std_label}" is invalid.')

    # Save the updated roots dictionary to a JSON file for future use
    with open(self._roots_md_file_path, 'w', encoding='utf-8') as json_file:
      json.dump(roots, json_file)

    # Cache the roots dictionary for later use in the current session
    self._roots = roots


  def get_dictionary_tree_for_std_label(self, std_label, debug=True):
    """
    Constructs a hierarchical dictionary (tree) representing the structure
    of labels for a given standard label. This function uses a breadth-first
    approach to traverse and build the tree for the specified root label.

    Parameters
    ----------
    std_label : str
      The root standard label for which the dictionary tree is constructed.
    debug : bool, optional
      Flag to enable debug output for each step in tree construction,
      by default True.

    Returns
    -------
    dict
      A dictionary representing the hierarchical structure of labels
      for the specified standard label, where each label key has nested
      child labels as values.
    """
    if debug:
      print(f'Starting tree generation for "{std_label}"')

    # Init tree with root label as the main key, starting an empty dictionary
    tree = {std_label: {}}

    # Init a queue for breadth-first traversal, starting with the root label and its dictionary
    queue = [(std_label, tree[std_label])]

    # Perform breadth-first traversal to build the hierarchical tree
    while queue:
      # Extract the current label and its dictionary reference from the queue
      current_label, current_dict = queue.pop(0)

      # Retrieve all fine labels associated with the current label (child nodes)
      fine_labels = self.get_fine_labels_for_std_label(current_label)

      # Loop through each fine label, adding it as a child to the current dictionary
      for fine_label in fine_labels:
        # Add fine label as a nested key with an empty dictionary for further children
        current_dict[fine_label] = {}

        # Append the fine label to the queue for further exploration of its children
        queue.append((fine_label, current_dict[fine_label]))

        if debug:
          print(f'  - Adding fine label: "{fine_label}" under "{current_label}"')

    # Return the fully constructed dictionary tree for the specified standard label
    return tree



  ### - taxonomy utils

  def get_mapped_datasets(self):
    if self._map_df is None:
      self._init_map_df()
    return self._map_df['dataset'].unique().tolist()

  def get_all_dataset_labels(self):
    dataset_labels = {}
    for dataset in self._map_df['dataset'].unique():
      dataset_labels[dataset] = self._map_df.loc[
        self._map_df['dataset'] == dataset]['dataset_label'].unique().tolist()

    return dataset_labels

  def is_mapped(self, dataset_label : str, dataset : str=None):
    """Check if a dataset label is mapped to the taxonomy. Additionally,
    the dataset's name can be specified to avoid collisions with similarly
    named labels

    Parameters
    ----------
    dataset_label : str
        The dataset label
    dataset : str, optional
        The dataset name, by default None

    Returns
    -------
    bool
        True if the dataset label is found in taxonomy, else False
    """
    if dataset:
      return not self._map_df.loc[
        (self._map_df['dataset_label'] == dataset_label) &
        (self._map_df['dataset'] == dataset)
      ].empty

    return not self._map_df.loc[
      (self._map_df['dataset_label'] == dataset_label)
    ].empty

  def get_default_labels_for_dataset(self, dataset: str):
    """Get all the (default) labels for a mapped dataset.

    Parameters
    ----------
    dataset : str
        The dataset's name as it is mapped in SALT

    Returns
    -------
    list
        List with the default dataset labels.

    Raises
    ------
    ValueError
        Raises ValueError if the dataset is not found in SALT
    """
    if dataset not in self._map_df['dataset'].unique():
      raise ValueError(f'Dataset "{dataset}" not present in the taxonomy')

    return self._map_df.loc[self._map_df['dataset'] == dataset][
      'dataset_label'].unique().tolist()


  def get_mapping_for_std_label(self, std_label):
    """Get the datasets-dataset labels mapping in a dictionary with
    datasets as keys and dataset labels as values for a given standard
    label.

    Parameters
    ----------
    std_label : str
        The standard label to get the mapping for.

    Returns
    -------
    dict
        Dictionary with dataset names as keys and the corresponding
        dataset labels as values
    """
    if self._map_df is None:
      self._init_map_df()

    filtered_df = self._map_df.loc[
      self._map_df[self._std_col] == std_label].copy()

    filtered_df.drop_duplicates(subset=['dataset_label', 'dataset'],
                                inplace=True)

    mapping_dict = {}

    for dataset in filtered_df['dataset'].unique():
      mapping_dict[dataset] = filtered_df.loc[
        filtered_df['dataset'] == dataset]['dataset_label'].unique().tolist()

    return mapping_dict


  def get_std_label_from_dataset_label(self, dataset_label):
    """Get standard label given a dataset label

    Parameters
    ----------
    dataset_label : str
        A mapped dataset's label to find its corresponding standard label.

    Returns
    -------
    std
        The standard label corresponding to the input dataset label.

    Raises
    ------
    ValueError
        In case dataset label is not found in the mapping.
    """
    standardized_string = utils.standardize_string(dataset_label)
    if standardized_string in self._map_df[self._std_col].unique():
      return standardized_string

    # Check for filtered datasets
    if not dataset_label in self._map_df['dataset_label'].unique():
      raise ValueError(f'Dataset label "{dataset_label}" does not exist in '
                       'current mapping...')

    filtered_dbs = self._map_df['dataset'].unique().tolist()
    self.reset_map_df()

    # Get standard labels associated with the given dataset label (coarse/fine)
    std_labels =  self._map_df.loc[self._map_df[
      'dataset_label'] == dataset_label][self._std_col].unique().tolist()

    mapped_label = std_labels[0]
    mapped_label_lvl = len(
      self._map_df.loc[self._map_df[self._std_col] == std_labels[0]][
        'dataset_label'].unique().tolist()
    )
    for std_label in std_labels[1:]:
      current_lvl = len(
        self._map_df.loc[self._map_df[self._std_col] == std_label][
          'dataset_label'].unique().tolist()
      )
      if current_lvl < mapped_label_lvl:
        mapped_label = std_label
        mapped_label_lvl = current_lvl

    # Apply filtering again
    self.filter_by_datasets(filtered_dbs)

    return mapped_label

  def get_mapping_for_dataset_label(self,
                                    dataset_label: str,
                                    return_std_label=False):
    # Find associated standard label
    std_label = self.get_std_label_from_dataset_label(dataset_label)
    # print(std_label)

    # Get the dataset mapping of the standard label
    mapping_dict = self.get_mapping_for_std_label(std_label)

    if return_std_label:
      return mapping_dict, std_label
    else:
      return mapping_dict


  # Function to find all subsets of a specific label
  def get_coarse_labels_for_std_label(self, std_label):
    """Find all the coarser labels (supersets) of the "std_label" label.

    Parameters
    ----------
    std_label : str
        The standard label for which to find its supersets

    Returns
    -------
    list
        list with supersets of "std_label" label
    """
    subsets = []
    label_indices = self._map_df[self._std_col] == std_label
    label_data = self._map_df.loc[label_indices]
    unique_labels = self._map_df[self._std_col].unique()
    for other_label in unique_labels:
      if other_label != std_label:
        other_subset_indices = self._map_df[self._std_col] == other_label
        other_subset_data = self._map_df.loc[other_subset_indices]
        if utils.is_subset(label_data['dataset_label'],
                      other_subset_data['dataset_label']):
          subsets.append(other_label)

    return subsets


  def get_fine_labels_for_std_label(self, std_label):
    """Find all the fine labels (subsets) of the "std_label" label.

    Parameters
    ----------
    std_label : str
        The standard label for which to find its subsets

    Returns
    -------
    list
        list with subsets of "std_label" label
    """
    subsets = []
    label_indices = self._map_df[self._std_col] == std_label
    label_data = self._map_df.loc[label_indices]
    unique_labels = self._map_df[self._std_col].unique()
    for other_label in unique_labels:
      if other_label != std_label:
        other_subset_indices = self._map_df[self._std_col] == other_label
        other_subset_data = self._map_df.loc[other_subset_indices]
        if utils.is_subset(other_subset_data['dataset_label'],
                      label_data['dataset_label']):
          subsets.append(other_label)

    return subsets


  def get_paths_to_label(self, target_label : str):
    """Find path/paths (from coarase to fine) to a standard label.

    Parameters
    ----------
    target_label : str
        The standard label to find paths to.

    Returns
    -------
    list
        list with paths to the target label. Each path is a separate
        list from coarse to fine labels.

    Raises
    ------
    ValueError
        If the given label does not exist in the taxonomy
    """
    if not target_label in self._map_df[self._std_col].unique():
      raise ValueError(f'Label "{target_label}" not present in the taxonomy')

    if target_label in self._roots:
      return [[target_label]]

    return utils.find_all_paths_to_key(dictionary=self._roots,
                                        target_key=target_label)


  def filter_by_datasets(self, dataset_list : list):
    """Filter map_df through a list of one or more datasets.

    Parameters
    ----------
    dataset_list : list
        List with datasets to filter.

    Raises
    ------
    ValueError
        Raise ValueError if a dataset that is not mapped exists in the
        input list.
    """
    if self._map_df is None:
      self._init_map_df()

    for dataset in dataset_list:
      if dataset not in self._map_df['dataset'].unique():
        raise ValueError(f'Dataset {dataset} not found in mapping...')

    # Filter by input datasets
    filtered_df = self._map_df.loc[self._map_df['dataset'].isin(dataset_list)]
    filtered_df = filtered_df.reset_index(drop=True)

    # Replace NaN with empty strings
    #filtered_df.replace(pd.NA, '', inplace=True)

    self._map_df = filtered_df


  def reset_map_df(self):
    self._init_map_df()


  def find_datasets_intersection(self, datasets: list):
    """Find the intersection of standard labels for the given datasets.

    Parameters
    ----------
    datasets : list
        List with dataset names.

    Returns
    -------
    list
        List with the intersection of standard labels for the given
        datasets.

    Raises
    ------
    ValueError
        If a given dataset does not exist in the mapping.
    """
    for dataset in datasets:
      if not dataset in self._map_df['dataset'].unique():
        raise ValueError(f'Dataset {dataset} not found in the mapping')
    labels_list = []
    for dataset in datasets:
      dataset_labels = self._map_df.loc[self._map_df['dataset'] == dataset][
        self._std_col].unique().tolist()

      labels_list.append(dataset_labels)

    intersection_set = set(labels_list[0]).intersection(*labels_list)

    return list(intersection_set)


  def get_parent_label_for_std_label(self, std_label : str):
    """Get the parent standard label for a given standard label.

    Parameters
    ----------
    std_label : str
        The standard label to get its parent.

    Returns
    -------
    list or str
        The parent(s) of the the given standard label

    Raises
    ------
    ValueError
        If the given label does not exist or if it is a root.
    """
    if std_label in self._roots:
      raise ValueError(f'"{std_label}" is a root label.')

    if not std_label in self._map_df[self._std_col].unique():
      raise ValueError(f'Label "{std_label}" not present in the taxonomy.')

    # Get coarse-to-fine paths
    paths = self.get_paths_to_label(std_label)

    # For each path, get the parent label (second to last)
    parents = []
    for path in paths:
      parents.append(path[-2])

    parents = list(set(parents)) # Ensure no duplicates

    # If there's a single parent return str, else list
    if len(parents) == 1:
      return parents[0]
    else:
      return parents


  def get_children_labels_for_std_label(self, std_label : str):
    """Get list with children label(s) for a given standard label.

    Parameters
    ----------
    std_label : str
        The standard label to get its children labels

    Returns
    -------
    list
        list with children label(s)

    Raises
    ------
    ValueError
        If the given standard label does not exist in the taxonomy.
    """
    if not std_label in self._map_df[self._std_col].unique():
      raise ValueError(f'Label "{std_label}" not present in the taxonomy.')
    # Get tree dictionary for the standard label
    dict_tree = utils.find_dict_with_value(self._roots, std_label)

    # Trancate the dictionary to get only the 1st level of nodes
    dict_tree = utils.truncate_dict_at_depth(dict_tree, max_depth=2)

    # If std_label already a leaf, return empty list
    if dict_tree[std_label] == []:
      return []

    # Get children labels
    return list(dict_tree[std_label].keys())


  def get_siblings_labels_for_std_label(self, std_label : str):
    """Get sibling labels for a given standard label.

    Parameters
    ----------
    std_label : str
        The standard label to look for its siblings.

    Returns
    -------
    dict
        Dict with parent labels as keys and sibling standard labels of
        the given label as values.

    Raises
    ------
    ValueError
        If the label is not present in the taxonomy.
    """
    if not std_label in self._map_df[self._std_col].unique():
      raise ValueError(f'Label "{std_label}" not present in the taxonomy.')

    # Get parent label/labels
    parent = self.get_parent_label_for_std_label(std_label)

    siblings = {}
    if isinstance(parent, str):
      parent = [parent]

    # For each parent label (if more than 1) get children labels
    # (except std label)
    for p in parent:
      p_siblings = self.get_children_labels_for_std_label(p)
      p_siblings.remove(std_label)
      siblings[p] = p_siblings

    return siblings


  def add_new_mapping(self, dataset_label, dataset, map_to, parent_labels=None):
    # Check if already mapped
    if self.is_mapped(dataset_label, dataset):
      raise ValueError(f'Label "{dataset_label}" from dataset "{dataset}" '
                        'is already mapped')
    
  def plot_std_label_mapping(self, std_label, figsize=(8, 8)):
    """Plot mapping tree for standard label. The standard label appears
    as a root node, intermediate nodes represent the mapping datasets
    and leaf nodes represent the dataset labels corresponding to the
    standard label.

    Parameters
    ----------
    std_label : str
        Standardized label
    figsize : tuple, optional
        Figsize, by default (8, 8)
    """
    # --- Create graph edges
    map_dict = {}
    map_dict[std_label] = self.get_mapping_for_std_label(std_label)

    edges = []
    for root, std_label_tree in map_dict.items():
      for dataset, dataset_labels in std_label_tree.items():
        edges.append((f'std: {root}', dataset))
        for dataset_label in dataset_labels:
          edges.append((dataset, f'"{dataset_label}"'))

    # --- Graph parameters
    # Text parameters
    text_linewidth = 1.5
    text_foreground = 'white'
    text_size = 10
    text_rotation = 20

    # Node parameters
    node_size = 1000
    node_color = 'lightblue'
    edge_color = 'lightgrey'

    plt.figure(figsize=figsize)

    # Create the graph and compute the layout
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    pos = graphviz_layout(graph, prog='dot', args='-Grankdir=LR')

    # Determine intermediate nodes
    intermediate_nodes = [node for node in graph.nodes()
                          if graph.in_degree(node) > 0
                          and graph.out_degree(node) > 0]

    # Determine root nodes (nodes with in_degree 0)
    root_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]

    # Create a color map based on the number of intermediate nodes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(intermediate_nodes)))
    intermediate_color_map = dict(zip(intermediate_nodes, colors))

    # Drawing the graph
    nx.draw_networkx(graph, pos,
                     with_labels=False,
                     node_size=node_size,
                     node_color=node_color)

    # Draw edges from root to intermediate nodes in light grey
    for root in root_nodes:
      root_edges = [(root, target) for target in graph.successors(root)]
      nx.draw_networkx_edges(graph,
                            pos,
                            edgelist=root_edges,
                            edge_color=edge_color,
                            arrows=True)

    # Draw edges with different colors based on the intermediate node
    for node in intermediate_nodes:
      edges = [(node, target) for target in graph.successors(node)]
      edge_color = intermediate_color_map[node]
      nx.draw_networkx_edges(graph,
                            pos,
                            edgelist=edges,
                            edge_color=[edge_color],
                            arrows=True)

    text = nx.draw_networkx_labels(graph, pos)

    # Text config
    for _, t in text.items():
      white_border = withStroke(linewidth=text_linewidth,
                                foreground=text_foreground)
      t.set_path_effects([white_border])
      t.set_size(text_size)
      t.set_rotation(text_rotation)

    # Find the x values of the most left and most right nodes in the graph
    x_values = [coord[0] for coord in pos.values()]

    # Set a safe offset to ensure no node is cropped by the graph
    (x,_) = figsize
    offset_horizontal = x*45
    lim_left = min(x_values) - offset_horizontal / 2
    lim_right = max(x_values) + offset_horizontal

    # Find the y values of the topmost and bottommost nodes in the whole graph
    y_values = [coord[1] for coord in pos.values()]

    # Set a safe offset to ensure no node is cropped by the graph
    offset_vertical = 70
    lim_top = min(y_values) - offset_vertical
    lim_bottom = max(y_values) + offset_vertical

    plt.xlim(lim_left, lim_right)
    plt.ylim(lim_top, lim_bottom)

    # Create a legend
    legend_elements = [Line2D([0], [0], color=color, lw=2, label=node)
                       for node, color in intermediate_color_map.items()]

    plt.legend(handles=legend_elements, loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()
