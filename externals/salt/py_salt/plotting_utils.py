from .event_mapping import EventExplorer
from .scene_mapping import SceneExplorer
from collections import defaultdict
from dash import dcc, html
from dash.dependencies import Input, Output

import plotly.graph_objects as go
import dash

class InteractivePlotter:
  """InteractivePlotter class
  """

  @property
  def explorer(self):
    return self._explorer_obj

  def __init__(self):
    self._explorer_obj = None
    self._std_label = None

  def get_explorer(self,
                   explorer_type,
                   event_map_md_file_path=None,
                   event_roots_md_file_path=None,
                   scene_map_md_file_path=None,
                   scene_roots_md_file_path=None):
    if explorer_type == 'EventExplorer':
      if not(event_map_md_file_path is None or event_roots_md_file_path is None):
        self._explorer_obj = EventExplorer(event_map_md_file_path,
                                            event_roots_md_file_path)
      else:
        self._explorer_obj = EventExplorer()
    elif explorer_type == 'SceneExplorer':
      if not(scene_map_md_file_path is None or scene_roots_md_file_path is None):
        self._explorer_obj = SceneExplorer(scene_map_md_file_path, scene_roots_md_file_path)
      else:
        self._explorer_obj = SceneExplorer()
    else:
      raise ValueError(f'Invalid explorer type: {explorer_type}')

  def _convert_labels(self, root_label, parent_name=None):
    """Recursively converts a label and its hierarchical structure into
    lists of labels and their corresponding parent labels.

    Parameters
    ----------
    root_label : str
        The starting label (root label) for the conversion process. This
        label may have child labels associated with it.
    parent_name : str, optional
        The parent label of the root label. If not provided, the root
        label has no parent, by default None.

    Returns
    -------
    tuple
        A tuple containing two lists:
        - A list of labels (str) corresponding to the root label and
        all its descendants.
        - A list of parent labels (str) corresponding to each label in
        the labels list.
    """
    # Initialize lists to store the labels and their corresponding parents
    labels = []
    parents = []

    # Get the children labels for the current root label
    children = self._explorer_obj.get_children_labels_for_std_label(root_label)

    # Add the current label and its parent to the lists
    labels.append(root_label)
    parents.append(parent_name if parent_name else '')  # Root has no parent

    # For each child, recursively extract its labels and parents
    for child in children:
      child_labels, child_parents = self._convert_labels(child, root_label)
      labels.extend(child_labels)
      parents.extend(child_parents)

    return labels, parents

  @staticmethod
  def _build_parent_label_tree(parent_label_pairs):
    """Build a tree-structure dictionary from a list of parent-label
    pairs. Each key is a parent and the associated value is a list of
    labels (children).

    Parameters
    ----------
    parent_label_pairs : list of tuples
        A list of tuples, where each tuple contains a parent and a
        corresponding label.
        The parent can be any hashable object, and the label is the child
        associated with that parent.

    Returns
    -------
    dict
        A dictionary where the keys are parent elements, and the values
        are lists of labels (children) associated with each parent.
    """
    tree = defaultdict(list)
    for parent, label in parent_label_pairs:
      tree[parent].append(label)
    return tree


  def _extract_all_parents_labels(self, tree, node, parent_id=''):
    """Recursively extracts all labels in a hierarchical tree along with
    their parent labels and unique IDs.

    Parameters
    ----------
    tree : dict
        A dictionary representing a hierarchical tree where each key is
        a node and the associated value is a list of its child nodes.
    node : str
        The current node in the tree whose label and parent information
        is being extracted.
    parent_id : str, optional
        The unique identifier of the parent node. By default, it is an
        empty string, indicating no parent for the root node.

    Returns
    -------
    tuple
        A tuple containing three lists:
        - A list of labels (str) for the current node and all its descendants.
        - A list of parent IDs (str) corresponding to each label in the
        labels list.
        - A list of unique IDs (str) for each label, formed by concatenating
        the parent IDs and node labels.
    """
    labels = []
    parents = []
    ids = []

    node_id = f'{parent_id}/{node}' if parent_id else node

    labels.append(node)
    parents.append(parent_id)
    ids.append(node_id)

    for child in tree[node]:
      child_labels, child_parents, child_ids = \
        self._extract_all_parents_labels(tree, child, node_id)
      labels.extend(child_labels)
      parents.extend(child_parents)
      ids.extend(child_ids)

    return labels, parents, ids


  def generate_sunburst_fig(self, ids, labels, parents, write_to_html=True):
    '''Generates and displays an interactive sunburst chart with adjustable 
    depth, text size, and figure size options in Plotly.

    Parameters
    ----------
    ids : list of str
        Unique identifiers for each node in the sunburst, defining the 
        node structure.
    labels : list of str
        Labels for each node in the sunburst, displayed as node names.
    parents : list of str
        Parent nodes for each item in ids, establishing node hierarchy 
        by mapping child nodes to parent nodes.
    renderer : str, optional
        Specifies the output rendering mode for the chart. Common options 
        include 'vscode' for VS Code’s plot viewer or 'browser' for web 
        display. Default is 'vscode'.

    Notes
    -----
    The sunburst chart includes interactive controls:
    - Depth Control: Switch between viewing limited (e.g., depth 2) and 
      fully expanded chart depths.
    - Text Size: Adjust the size of node labels.
    - Figure Size: Change the overall chart dimensions.
    
    Dropdown menus dynamically update the chart based on user selections, 
    making it easy to explore complex hierarchies in an interactive format.
    '''
    # Switch to interactive mode with an external window
    # pio.renderers.default = renderer

    # Define different depth levels for each trace to enable button toggling
    depth_levels = [1, 2, 3, None]  # None represents full expansion

    # Initialize the figure
    fig = go.Figure()

    t_offset = 70   # Default figure height
    y_labels = 1.05  # Position of labels relative to height
    y_menu = y_labels    # Position of menus relative to height
    font_familly = 'Courier New, monospace'

    # Create a trace for each depth level
    for depth in depth_levels:
      fig.add_trace(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        branchvalues='total',
        maxdepth=depth,
        textinfo='label',
        insidetextfont={
          'family': font_familly,
          'size': 16
        },
        outsidetextfont={
          'family': font_familly,
          'size': 16
        },
        visible=(depth == 2),
        # Increase the border width of each node segment
        marker={'line': {'width': 0.5, 'color': 'white'}}
      ))

    # Update layout with menus
    fig.update_layout(
      plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot area
      paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent chart background
      updatemenus=[
        {
          'type': 'buttons',
          'buttons': [
            {
              'label': 'Collapse',
              'method': 'update',
              'args': [
                {'visible': [depth == 2 for depth in depth_levels]},
                {'sunburstcolorway': ['#636efa', '#ef553b', '#00cc96'],
                'root': {'ids': self._std_label}}
              ]
            },
            {
              'label': 'Expand',
              'method': 'update',
              'args': [
                {'visible': [depth is None for depth in depth_levels]},
                {'root': {'ids': self._std_label}}
              ]
            }
          ],
          'showactive': True,  # Highlight the active button
          'pad': {'r': 10, 't': 10},
          'x': 0.0,
          'xanchor': 'center',
          'y': y_menu,
          'yanchor': 'top',
          'font': {'size': 14},
          'bgcolor': '#f0f0f0',
          'active': 0  # Set the initial active button index (0 or 1)
        },
        {
          'type': 'dropdown',
          'direction': 'down',
          'buttons': [
            {'label': 'Tiny (8)', 'method': 'update', 'args': [{'insidetextfont': {'size': 8, 'family': font_familly}, 'outsidetextfont': {'size': 10, 'family': font_familly}}]},
            {'label': 'Small (12)', 'method': 'update', 'args': [{'insidetextfont': {'size': 12, 'family': font_familly}, 'outsidetextfont': {'size': 14, 'family': font_familly}}]},
            {'label': 'Medium (16)', 'method': 'update', 'args': [{'insidetextfont': {'size': 16, 'family': font_familly}, 'outsidetextfont': {'size': 18, 'family': font_familly}}]},
            {'label': 'Large (20)', 'method': 'update', 'args': [{'insidetextfont': {'size': 20, 'family': font_familly}, 'outsidetextfont': {'size': 22, 'family': font_familly}}]},
            {'label': 'X-Large (24)', 'method': 'update', 'args': [{'insidetextfont': {'size': 24, 'family': font_familly}, 'outsidetextfont': {'size': 26, 'family': font_familly}}]},
            {'label': 'XX-Large (30)', 'method': 'update', 'args': [{'insidetextfont': {'size': 30, 'family': font_familly}, 'outsidetextfont': {'size': 32, 'family': font_familly}}]}
          ],
          'pad': {'r': 10, 't': 10},
          'showactive': True,
          'x': 0.5,
          'xanchor': 'center',
          'y': y_menu,
          'yanchor': 'top',
          'font': {'size': 14},
          'bgcolor': '#f0f0f0',
          'active': 0
        },
        {
          'type': 'dropdown',
          'direction': 'down',
          'buttons': [
            {'label': 'Tiny (600x600)', 'method': 'relayout', 'args': [{'width': 600, 'height': 600 + t_offset}]},
            {'label': 'Small (800x800)', 'method': 'relayout', 'args': [{'width': 800, 'height': 800 + t_offset}]},
            {'label': 'Medium (1000x1000)', 'method': 'relayout', 'args': [{'width': 1000, 'height': 1000 + t_offset}]},
            {'label': 'Large (1200x1200)', 'method': 'relayout', 'args': [{'width': 1200, 'height': 1200 + t_offset}]},
            {'label': 'X-Large (1400x1400)', 'method': 'relayout', 'args': [{'width': 1400, 'height': 1400 + t_offset}]},
            {'label': 'XX-Large (1600x1600)', 'method': 'relayout', 'args': [{'width': 1600, 'height': 1600 + t_offset}]}
          ],
          #'pad': {'r': 10, 't': 10},
          'showactive': True,
          'x': 0.9,
          'xanchor': 'center',
          'y': y_menu,
          'yanchor': 'top',
          'font': {'size': 14},
          'bgcolor': '#f0f0f0',
          'active': 1
        }
      ],
      #margin={'t': 100, 'l': 20, 'r': 20, 'b': 0},
      width=800,
      height=800 + 80,
      annotations=[
        {'x': 0.0, 'y': y_labels, 'xanchor': 'center', 'yanchor': 'bottom', 'text': 'Graph Type', 'showarrow': False, 'font': {'size': 12}},
        {'x': 0.5, 'y': y_labels, 'xanchor': 'center', 'yanchor': 'bottom', 'text': 'Text Size', 'showarrow': False, 'font': {'size': 12}},
        {'x': 0.9, 'y': y_labels, 'xanchor': 'center', 'yanchor': 'bottom', 'text': 'Fig Size', 'showarrow': False, 'font': {'size': 12}}
      ],
    )

    if write_to_html:
      fig.write_html('sunburst_chart.html')
    return fig


  def prepare_sunburst_data(self, label):
    """Prepares the data needed for generating a sunburst plot."""
    # Convert labels and their parent associations based on the root label
    labels, parents = self._convert_labels(label)
    parent_label_pairs = list((p, l) for p, l in set(zip(parents, labels)))
    tree = self._build_parent_label_tree(parent_label_pairs)
    labels, parents, ids = self._extract_all_parents_labels(tree, label)
    return ids, labels, parents


class SunburstApp:
  """_summary_
  """
  def __init__(self, map_md_file_path, roots_md_file_path):
    # Create the Dash app
    self.app = dash.Dash(__name__, assets_folder='../assets/')

    self._scene_map_md_file_path = map_md_file_path
    self._scene_roots_md_file_path = roots_md_file_path

    # Instantiate the plotter
    self.plotter = InteractivePlotter()

    # Set up the layout
    self.app.layout = self._create_layout()

    # Register callbacks
    self._register_callbacks()




  def _create_layout(self):
      """Generates the layout for the Dash web application with a relaxing animated background."""

      return html.Div([
          # Main container with relaxing animated background class
          html.Div([

              # Header with a bold, modern style
              html.Div([
                  html.H1(
                      'SALT',
                      style={
                          'font-family': 'Poppins, sans-serif',
                          'font-weight': 'bold',
                          'text-align': 'center',
                          'color': '#ffffff',
                          'font-size': '5rem',
                          'margin-top': '20px',
                          'margin-bottom': '0px'
                      }
                  ),
                  # Subtitle with a calming message
                  html.H2([
                      html.Em('A modern interface for label exploration & aggregation')
                  ], style={
                      'font-family': 'Poppins, sans-serif',
                      'font-weight': 'normal',
                      'text-align': 'center',
                      'color': '#ffffff',
                      'font-size': '1.5rem',
                      'margin-top': '5px',
                      'margin-bottom': '60px'
                  })
              ]),

              # Dropdown container with labels and centered layout
              html.Div([
                  # First dropdown: Taxonomy
                  html.Div([
                      html.Label(
                          'Taxonomy',
                          style={
                              'font-size': '18px',
                              'font-weight': 'bold',
                              'margin-bottom': '5px',
                              'color': '#ffffff'
                          }
                      ),
                      dcc.Dropdown(
                          id='explorer-dropdown',
                          options=[
                              {'label': 'Sound events', 'value': 'EventExplorer'},
                              {'label': 'Acoustic scenes', 'value': 'SceneExplorer'}
                          ],
                          value='EventExplorer',
                          style={
                              'width': 'auto',
                              'min-width': '250px',
                              'max-width': '600px',
                              'border-radius': '12px',
                              'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)',
                              'padding': '5px',
                              'background-color': '#f7f7f7',
                              'font-size': '16px'
                          }
                      )
                  ], style={'margin-bottom': '20px', 'text-align': 'center'}),

                  # Second dropdown: Root label
                  html.Div([
                      html.Label(
                          'Root label',
                          style={
                              'font-size': '18px',
                              'font-weight': 'bold',
                              'margin-bottom': '5px',
                              'color': '#ffffff'
                          }
                      ),
                      dcc.Dropdown(
                          id='root-label-dropdown',
                          style={
                              'width': 'auto',
                              'min-width': '350px',
                              'max-width': '1000px',
                              'border-radius': '12px',
                              'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)',
                              'padding': '5px',
                              'background-color': '#f7f7f7',
                              'font-size': '16px'
                          }
                      )
                  ], style={'margin-bottom': '20px', 'text-align': 'center'}),

              ], style={
                  'display': 'flex',
                  'flex-direction': 'row',  # Change to row for side-by-side layout
                  'align-items': 'center',
                  'justify-content': 'center',
                  'gap': '20px',  # Space between the dropdowns
                  'height': 'auto',
              }),

              # Graph container
              html.Div([
                  dcc.Graph(
                      id='sunburst-graph',
                      config={'displayModeBar': False},
                      style={
                          'background': 'transparent'  # Ensures the graph container background is transparent
                      }
                  )
              ], style={
                  'margin-top': '50px',
                  'border-radius': '12px',
                  'box-shadow': '0px 6px 12px rgba(0, 0, 0, 0.5)',  # Soft shadow for depth
                  'background': 'rgba(255, 255, 255, 0.15)',  # Slightly less transparent background                  'padding': '20px',
                  'max-width': '1600px',
                  'margin': '0 auto',
              }),

              # Footer section as defined in your original layout
              html.Div([
                  html.Div([
                      html.A('EMAIL', href='mailto:stamatiadis@telecom-paris.fr', style={
                          'color': '#ffffff', 'font-size': '16px', 'text-decoration': 'none', 'margin-right': '80px',
                          'transition': 'transform 0.3s ease-in-out'
                      }, className='zoom-on-hover'),
                      html.A('ADASP GitHub', href='https://github.com/tpt-adasp', target='_blank', style={
                          'color': '#ffffff', 'font-size': '16px', 'text-decoration': 'none', 'margin-right': '80px',
                          'transition': 'transform 0.3s ease-in-out'
                      }, className='zoom-on-hover'),
                      html.A('SALT Github', href='https://github.com/tpt-adasp/salt?tab=readme-ov-file', target='_blank', style={
                          'color': '#ffffff', 'font-size': '16px', 'text-decoration': 'none', 'margin-right': '80px',
                          'transition': 'transform 0.3s ease-in-out'
                      }, className='zoom-on-hover'),
                      html.A('SALT Paper', href='https://arxiv.org/abs/2409.11746', target='_blank', style={
                          'color': '#ffffff', 'font-size': '16px', 'text-decoration': 'none',
                          'transition': 'transform 0.3s ease-in-out'
                      }, className='zoom-on-hover'),
                  ], style={
                      'display': 'flex', 'justify-content': 'center', 'margin-bottom': '10px',
                      'flex-wrap': 'wrap', 'z-index': 1,
                  }),

                  html.Div(
                      '© 2024 LTCI, Telecom Paris, Institut Polytechnique de Paris, France',
                      style={
                          'font-size': '16px', 'color': '#ffffff', 'text-align': 'center', 'flex': '1',
                          'padding': '10px 0',
                      }
                  ),

                  html.Img(
                      src='https://adasp.telecom-paris.fr/assets/images/logos/telecomparis_cropped.jpg',
                      style={
                          'width': '100px', 'height': 'auto', 'position': 'absolute', 'bottom': '30px', 'right': '120px',
                      }
                  ),
                  html.Img(
                      src='https://adasp.telecom-paris.fr/assets/images/logos/adasp_reverse_blue_nobg_100px.png',
                      style={
                          'width': '200px', 'height': 'auto', 'position': 'absolute', 'bottom': '40px', 'right': '245px',
                      }
                  ),
              ], style={
                  'width': '100%', 'position': 'relative', 'text-align': 'center',
                  'padding-top': '50px', 'padding-bottom': '50px',
              }),

          ], className='relaxing-animated-background')  # Apply relaxing animated background class here
      ])




















  def _register_callbacks(self):
    """Register callbacks for the Dash app."""
    @self.app.callback(
      [Output('root-label-dropdown', 'options'),
       Output('root-label-dropdown', 'value'),
       Output('sunburst-graph', 'figure')],
      [Input('explorer-dropdown', 'value'),
       Input('root-label-dropdown', 'value')]
    )
    def update_graph(explorer_type, root_label):
      # Get the explorer object and available root labels
      self.plotter.get_explorer(explorer_type,
                                scene_map_md_file_path=self._scene_map_md_file_path,
                                scene_roots_md_file_path=self._scene_roots_md_file_path)
      available_labels = list(self.plotter.explorer._roots.keys())

      # Set default root label if not already selected
      if root_label not in available_labels:
        root_label = available_labels[0]

      # Generate the sunburst figure
      ids, labels, parents = self.plotter.prepare_sunburst_data(root_label)
      sunburst_fig = self.plotter.generate_sunburst_fig(ids, labels, parents)

      # Prepare the options for the root label dropdown
      root_options = [{'label': label, 'value': label}
                      for label in available_labels]

      return root_options, root_label, sunburst_fig

  def run(self, host='0.0.0.0', port=8050, debug=True):
    """Run the Dash server."""
    self.app.run(debug=debug, host=host, port=port)
