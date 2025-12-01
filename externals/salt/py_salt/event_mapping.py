"""Module for data exploration of the event mapping
"""
import os
import re
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from pathlib import Path
from typing import Union
from tqdm import tqdm
from ast import literal_eval
from typing import Union, List

from .mapping_manager import MappingExplorer
from . import general_utils as utils
from .scene_mapping import SceneExplorer

class EventExplorer(MappingExplorer):
  """EventExplorer class with tools to explore the event_mapping of the
  Audible taxonomy
  """

  def __init__(self,
               map_md_file_path=None,
               roots_md_file_path=None):

    assets_dir_path = os.path.join(Path(__file__).parent.parent, 'assets')
    assert os.path.isdir(assets_dir_path)

    if map_md_file_path is None:
      map_md_file_path = config.IMPULSES_MAP_FP

    if roots_md_file_path is None:
      roots_md_file_path = config.IMPULSES_ROOTS_JSON_FP

    std_col = 'standard_event'

    # Init from base class
    super().__init__(map_md_file_path, roots_md_file_path, std_col)

  @property
  def all_events(self):
    return self._all_events


  def _init_map_df(self):
    super()._init_map_df()

    # Get actions/activities
    self._all_events = self._map_df[self._std_col].unique().tolist()

    # Sort
    self._all_events = sorted(self._all_events)



  def get_map_for_activities(self, activities : Union[list, str]):
    """Get the map dataframe with all the standard events containing the
    action / list of activities passed as arguments.

    Parameters
    ----------
    activities : Union[list, str]
        activity or list of activities (standard events)

    Returns
    -------
    pd.DataFrame
        Subset of self.map_df including all events that contain the asked
        activity/activities.
    """
    if isinstance(activities, str):
      activities = [activities]

    df = self._map_df[self._map_df['standard_activity'].apply(
      lambda lst: any(action in item for item in lst for action in activities))]

    return df.reset_index(drop=True)


  def add_mapping(self,
                  label_to_map: str,
                  dataset: str,
                  map_to: Union[str, None],
                  action: Union[list, str, None] = None,
                  activity: Union[list, str, None] = None):
    """Map a new event to SALT.

    Parameters
    ----------
    label_to_map : str
        The new event's label
    dataset : str
        The dataset's label
    map_to : Union[str, None]
        The standard event that the event will be mapped to. When None,
        a new standard event will be created.
    action : Union[list, str, None], optional
        The action label(s) (standard events) associated with the event,
        by default None
    activity : Union[list, str, None], optional
        The activity label(s) (standard events) associated with the event,
        by default None

    Raises
    ------
    ValueError
        _description_TODO: Complete docstring
    ValueError
        _description_
    ValueError
        _description_
    """
    # Check if label_to_map already exists
    if label_to_map in self._all_events:
      if dataset in self.get_mapping_for_dataset_label(label_to_map):
        if not utils.confirm_overwrite_mapping(label_to_map, dataset):
          return

    if map_to is not None and map_to not in self.all_labels:
      raise ValueError(f'Label "{map_to}" not present in the taxonomy')

    if action is None:
      action=''
    else:
      if action not in self.actions:
        raise ValueError(f'Label "{action}" not present in the taxonomy')

    if activity is None:
      activity=''
    else:
      if activity not in self.activities:
        raise ValueError(f'Label "{activity}" not present in the taxonomy')

    map_to_parents = list({lbl for path in self.get_paths_to_label(map_to)
                           for lbl in path})


    debug_df = pd.DataFrame()
    for std_label in map_to_parents:
      index = self._map_df[self._map_df[self._std_col] == std_label].index[-1]
      new_row = pd.DataFrame({
        'standard_event': std_label,
        'dataset_label': label_to_map,
        'dataset': dataset
      }, index=[index])

      debug_df = pd.concat([debug_df, new_row], ignore_index=True)

      # # Insert the new row at the specified index
      # self._map_df = pd.concat(
      #   [self._map_df.iloc[:index+1],
      #    new_row, self._map_df.iloc[index+1:]
      #   ])

      # # Reset index
      # self._map_df = self._map_df.reset_index(drop=True)

    return debug_df
  
  def _add_new_action(self,
                      action: Union [list, str],
                      dataset: str):
    new_row = pd.DataFrame({
      'standard_event': action,
      'dataset_label': action,
      'dataset': dataset
    })

    self._map_df = pd.concat([self._map_df, new_row], ignore_index=True)






  def _generate_tree_graph(self, dict_tree=None, parent=None, graph=None):
    """Generates a tree graph from the provided dictionary tree.

    Parameters
    ----------
    dict_tree : dict, optional
        A hierarchical dictionary representing the tree structure,
        by default None
    parent : str, optional
        The parent node in the graph, by default None
    graph : networkx.DiGraph, optional
        The graph object to which nodes and edges are added, by default None

    Returns
    -------
    networkx.DiGraph
        A directed graph representing the tree structure.
    """
    # Initialize a new directed graph if not provided
    if graph is None:
      graph = nx.DiGraph()

    for key, value in dict_tree.items():
      if isinstance(value, dict):
        # If the value is a dictionary, recursively generate the graph
        graph.add_node(key)
        if parent is not None:
          graph.add_edge(parent, key)
        self._generate_tree_graph(value, key, graph)
      elif isinstance(value, list) and not value:
        # If the value is an empty list, consider it as a leaf node
        graph.add_node(key)
        if parent is not None:
          graph.add_edge(parent, key)

    return graph

  @staticmethod
  def _remove_extra_edges(graph):
    """Removes extra edges (e.g. from root to leaf nodes) from the graph.

    Parameters
    ----------
    graph : NetworkX graph
        The graph from which extra edges are to be removed.

    Returns
    -------
    NetworkX graph
        The graph with extra edges removed.
    """
    extra_edges = []
    for source in graph.nodes():
      for target in graph.nodes():
        if source != target and nx.has_path(graph, source, target):
          for intermediate_node in graph.nodes():
            if source != intermediate_node and target != intermediate_node and nx.has_path(
              graph, source, intermediate_node) and nx.has_path(graph, intermediate_node, target):
              edge_to_remove = (source, target)
              if edge_to_remove in graph.edges():
                extra_edges.append(edge_to_remove)
    graph.remove_edges_from(extra_edges)
    return graph


  def plot_hierarchical_tree_graph(self,
                                   std_event : str,
                                   max_depth=None,
                                   edge_color='lightgrey',
                                   bold=False,
                                   text_border=True,
                                   figsize=None,
                                   text_size=None,
                                   title=None,
                                   save_fig=None,
                                   ):
    """Plot hierarhical (from coarse to fine) tree graph with standard labels.
    Root of the graph: "std_event".

    Parameters
    ----------
    std_event : str
        root of the graph
    max_depth : int, optional
        If not None, will plot the graph until "max_depth" nodes,
        by default None
    """
    self._plot_std_label_graph_top_down(
      std_label=std_event,
      roots_dict=self._roots,
      max_depth=max_depth,
      edge_color=edge_color,
      text_border=text_border,
      bold=bold,
      figsize=figsize,
      text_size=text_size,
      title=title,
      save_fig=save_fig
    )

  def plot_dataset_tree_graph(self, dataset, **kargs):
    """_summary_

    Parameters
    ----------
    dataset : _type_
        _description_
    """
    def filter_dict(d, keys):
      """Filter roots dict throuth a dataset's std labels.
      """
      if not isinstance(d, dict):
        return d

      filtered_dict = {}
      for k, v in d.items():
        if k in keys:
          if isinstance(v, dict):
            filtered_subdict = filter_dict(v, keys)
            if filtered_subdict:
              filtered_dict[k] = filtered_subdict
          else:
            filtered_dict[k] = v

      return filtered_dict

    if not dataset in self._map_df['dataset'].unique():
      raise ValueError(f'Dataset {dataset} not found in the current mapping.')

    dataset_std_labels = dataset_std_labels = self._map_df.loc[
      self._map_df['dataset'] == dataset][self._std_col].unique().tolist()

    dataset_roots_dict = filter_dict(self._roots, dataset_std_labels)

    for root in dataset_roots_dict:
      self._plot_std_label_graph_top_down(std_label=root,
                                          roots_dict=dataset_roots_dict,
                                          **kargs)



  def _plot_std_label_graph_top_down(self,
                                     std_label : str,
                                     roots_dict: dict,
                                     max_depth=None,
                                     edge_color='lightgray',
                                     text_border=True,
                                     bold=False,
                                     figsize=None,
                                     text_size=None,
                                     title=None,
                                     save_fig=None
                                     ):
    """Plot a hierarchical standard label tree for a standard label.

    Parameters
    ----------
    std_label : str
        The standard label to plot its graph
    roots_dict : dict
        The roots dictionary of the label (could be filtered per dataset)
    max_depth : int, optional
        The max depth of the tree, by default None
    edge_color : str, optional
        The edge color, by default 'lightgray'
    text_border : bool, optional
        Text border, by default True
    bold : bool, optional
        Bold text, by default False
    """
    # Get dictionary label tree from roots dictionary
    dict_tree = utils.find_dict_with_value(roots_dict, std_label)

    # Truncate the tree to max depth
    if not max_depth is None:
      if max_depth <= 0:
        raise ValueError('Max depth must be positive.')
      else:
        dict_tree = utils.truncate_dict_at_depth(dict_tree, max_depth=max_depth)

    # Generate the tree graph
    gragh = self._generate_tree_graph(dict_tree)

    # Remove unecessary edges
    gragh = self._remove_extra_edges(gragh)
    pos = nx.drawing.nx_agraph.graphviz_layout(gragh, prog='dot')

    # Calculate the maximum number of consecutive horizontal nodes
    max_horizontal_nodes = max(
      len([node for node, (x, y) in pos.items() if y == level])
      for level in set(y for x, y in pos.values()))

    # --- Get max vertical nodes
    # Calculate the number of nodes at each depth level
    levels = {}
    for node, (x, y) in pos.items():
      if y not in levels:
        levels[y] = 0
      levels[y] += 1

    # Maximum number of horizontal nodes
    max_horizontal_nodes = max(levels.values())

    # Maximum depth of the tree (number of vertical levels)
    max_vertical_nodes = len(levels)

    # Adjust the figure size based on the maximum number of consecutive
    # horizontal nodes
    node_width = 1.8  # Adjust the scaling factor as needed
    node_height = 2.7  # Adjust the scaling factor as needed

    # For very wide plots, double the height
    if max_horizontal_nodes / max_vertical_nodes > 10:
      max_vertical_nodes *= 2

    if figsize is None:
      figsize = (max_horizontal_nodes * node_width,
                 max_vertical_nodes * node_height)

    plt.figure(figsize=figsize)

    # Define parameters
    node_color = 'lightblue'
    node_size = 2000

    if text_size is None:
      text_size = 10

    pos = nx.drawing.nx_agraph.graphviz_layout(gragh, prog='dot')
    nx.draw_networkx(gragh, pos, with_labels=False, node_size=node_size,
                     node_color=node_color, edge_color=edge_color)

    text = nx.draw_networkx_labels(gragh, pos)
    for _, t in text.items():
      t.set_size(text_size)
      t.set_rotation(20)

      if text_border:
        white_border = withStroke(linewidth=2.0, foreground='white')
        t.set_path_effects([white_border])

      if bold:
        t.set_fontweight('bold')

    # Find the x values of the most left and most right nodes in the graph
    x_values = [coord[0] for coord in pos.values()]

    # Set a safe offset to ensure no node is cropped by the graph
    offset_horizontal = 150
    lim_left = min(x_values) - offset_horizontal
    lim_right = max(x_values) + offset_horizontal

    # Find the y values of the topmost and bottommost nodes in the whole graph
    y_values = [coord[1] for coord in pos.values()]

    # Set a safe offset to ensure no node is cropped by the graph
    offset_vertical = 20
    lim_top = min(y_values) - offset_vertical
    lim_bottom = max(y_values) + offset_vertical

    plt.xlim(lim_left, lim_right)
    plt.ylim(lim_top, lim_bottom)

    if title is None:
      plt.title(f'Tree graph of standard label: "{std_label}"')
    
    if save_fig is not None:
      print('save')
      plt.savefig(save_fig)
    plt.show()



class EventToSceneExplorer():
  def __init__(self,
               raw_map_md_file_path : str=None,
               std_event_col: str = 'standard_event',
               std_scene_col: str = 'standard_scene',
               all_val: list = None):
    """TODO: Complete...

    Parameters
    ----------
    raw_map_md_file_path : str, optional
        _description_, by default None
    std_event_col : str, optional
        _description_, by default 'event'
    std_scene_col : str, optional
        _description_, by default 'scene'
    all_val : list, optional
        _description_, by default None
    """
    assets_dir_path = os.path.join(Path(__file__).parent.parent, 'assets')
    assert os.path.isdir(assets_dir_path)

    if raw_map_md_file_path is None:
      raw_map_md_file_path = os.path.join(
        assets_dir_path, 'raw_salt_event_to_scene_mapping.tsv')

    self._std_event_col = std_event_col
    self._std_scene_col = std_scene_col
    self._all_val_list = all_val

    self._scene_mapper = SceneExplorer()

    self._raw_map_md_file_path = raw_map_md_file_path
    self._final_map_md_file_path = os.path.join(
      assets_dir_path, 'full_salt_event_to_scene_mapping.tsv')

    self._event_mapper = EventExplorer()
    self._load_event_to_scene_mapping()

  @property
  def map_df(self):
    return self._map_df

  @staticmethod
  def _safe_literal_eval(val):
    try:
      return literal_eval(val)
    except Exception:
      val = re.sub(r',?\s*nan\]$', ']', val)
      return literal_eval(val)

  def _load_event_to_scene_mapping(self):
    if os.path.isfile(self._final_map_md_file_path):
      self._map_df = pd.read_csv(self._final_map_md_file_path, sep='\t')
      self._map_df[self._std_scene_col] = self._map_df[
        self._std_scene_col].apply(self._safe_literal_eval)
    else:
      self._generate_mapping_file()

  def _generate_mapping_file(self):
    print('Generating full mapping metadata file...')

    # Load initial mapping
    df = pd.read_csv(self._raw_map_md_file_path, sep='\t')
    assert self._std_event_col in df.columns
    assert self._std_scene_col in df.columns

    # Convert scenes from string to list and explode rows
    df[self._std_scene_col] = df[self._std_scene_col].str.split(', ')
    df = df.explode(self._std_scene_col)

    # Identify unmapped events
    mapped_events = df[self._std_event_col].unique().tolist()
    events_to_map = list(set(self._event_mapper.all_events) - set(mapped_events))

    # Map remaining events using fine labels
    new_mappings = []
    for event in tqdm(events_to_map, desc='Adding non-leaf events'):
      fine_events = set(self._event_mapper.get_fine_labels_for_std_label(event))
      scenes = set()
      for fine_event in fine_events:
        scenes.update(df.loc[df[self._std_event_col] == fine_event, self._std_scene_col])
      new_mappings.append({
        self._std_event_col: event,
        self._std_scene_col: list(scenes)
      })

    # Merge with original mapping
    new_df = pd.DataFrame(new_mappings).explode(self._std_scene_col)
    df = pd.concat([df, new_df], ignore_index=True)

    # Cache all possible scene labels, excluding 'all'
    if self._all_val_list is None:
      self._all_val_list = df[self._std_scene_col].unique().tolist()
    if 'all' in self._all_val_list:
      self._all_val_list.remove('all')

    # Handle 'all' as a wildcard scene
    df_expanded = df[df[self._std_scene_col] != 'all'].copy()
    df_all = df[df[self._std_scene_col] == 'all']
    repeated_rows = df_all.loc[df_all.index.repeat(len(self._all_val_list))].copy()
    repeated_rows[self._std_scene_col] = self._all_val_list * len(df_all)
    df = pd.concat([df_expanded, repeated_rows], ignore_index=True).dropna()

    # Expand coarse scene labels
    expanded_data = []
    for event in tqdm(df[self._std_event_col].unique(), desc='Expanding coarse scene labels'):
      scenes = df.loc[df[self._std_event_col] == event, self._std_scene_col].unique().tolist()
      all_scenes = set(scenes)
      for scene in scenes:
        all_scenes.update(self._scene_mapper.get_fine_labels_for_std_label(scene))
      expanded_data.append({
        self._std_event_col: event,
        self._std_scene_col: list(all_scenes)
      })

    df = pd.DataFrame(expanded_data).explode(self._std_scene_col)
    df = df.drop_duplicates(ignore_index=True)
    
    # Final aggregation and export
    df = df.groupby(self._std_event_col, as_index=False).agg({self._std_scene_col: list})
    df.to_csv(self._final_map_md_file_path, sep='\t', index=False)




  def get_scenes_for_standard_events(self, std_labels: Union[str, List[str]]):
    """Get the scenes intersection for 1 or more standard event labels.

    Parameters
    ----------
    std_labels : Union[str, List[str]]
        The standard event label(s)

    Returns
    -------
    list
        The list of possible scenes that the event occurs in.

    Raises
    ------
    ValueError
        If an event is not mapped to any scene.
    """
    if isinstance(std_labels, str):
      std_labels = [std_labels]

    possible_scenes = []
    for std_label in std_labels:
      scenes = self._map_df.loc[
        self._map_df[self._std_event_col] == std_label,
        self._std_scene_col
      ].values

      if len(scenes) == 0:
        raise ValueError(f'No scene mapping found for standard event: {std_label}')

      scene_val = scenes[0]
      if isinstance(scene_val, str):
        scene_set = {scene_val}
      else:
        scene_set = set(scene_val)

      possible_scenes.append(scene_set)

    return list(set.intersection(*possible_scenes)) if possible_scenes else []
