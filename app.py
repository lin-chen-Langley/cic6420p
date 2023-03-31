__author__ = "Lin"
__version__ = "0.1"
__email__ = "lin.chen@ieee.org"
__status__ = "Prototype"

import os
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import dash_uploader as du
from dash import html
from dash import dcc
from dash import dash_table
import json
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor

# create app
app = dash.Dash(__name__)
server = app.server

# setup folder saving uploaded files
UPLOAD_FOLDER_ROOT = r"Uploads"
du.configure_upload(app, UPLOAD_FOLDER_ROOT)

# numerical features
num_features = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'num_file_creations', 'num_access_files', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

# categorical features
cat_features = ['protocol_type', 'service', 'flag', 'land', 'wrong_fragment', 'urgent', 'num_failed_logins', 'logged_in', 'root_shell', 'su_attempted', 'num_shells', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

initial_features = ['protocol_type', 'service', 'flag', 'duration', 'count']
features = num_features + cat_features

training_data = None
test_data = None

def get_label(label):
    if label == 'normal':
        return 1
    else:
        return -1

class Get_top_categories(BaseEstimator, TransformerMixin):
    """Create a class to keep the top categories, the rest categories are labeled as 'other'
    """

    def __init__(self, top_num = 10): # no *args or **kargs
        """Create a class

        Arg:
            top_num (int), the number of top categories kept, default number is 10
        """
        self.top_num = top_num

    def fit(self, X, y = None):
        """Fit the class

        Arg:
            X (Pandas.Series), a column of a Pandas.DataFrame
            y (None), not used
        """
        temp = X.value_counts()
        self.columns = list(temp[:self.top_num].index)
        return self

    def containe(self, s):
        """Process record

        Arg:
            s (str), a recod in the categorical column

        Return:
            str, return the same string is a recod in the top category list; otherwise, return 'other'
        """
        if s in self.columns:
            return s
        else:
            return 'other_category'

    def transform(self, X):
        """Convert a specific categorical column

        Arg:
            X (Pandas.Series), a column of a Pandas.DataFrame

        Return:
            Pandas.Series, processed column
        """
        temp = X.apply(self.containe)
        return temp

class DoNothing(BaseEstimator, TransformerMixin):
    """Do not change anything"""
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        temp = X.copy()
        return temp

# process numerical features
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

num_pipeline_gaussian = Pipeline([
    ('quantile', QuantileTransformer(output_distribution='normal', random_state=0)),
    #('std_scaler', StandardScaler()),
])

# process categorical features with bag of words
cat_pipeline = Pipeline([
    ('bag_of_words', CountVectorizer()),
])

cat_pipeline_five = Pipeline([
    ('more_than_five', Get_top_categories(top_num=5)),
    ('bag_of_words', CountVectorizer()),
])

cat_pipeline_ten = Pipeline([
    ('more_than_ten', Get_top_categories()),
    ('bag_of_words', CountVectorizer()),
])

# do not change features
do_nothing_pipeline = Pipeline([
    ('do_nothing', DoNothing())
])

preprocess_pipeline = ColumnTransformer([
        ("num_pipeline_guassion", num_pipeline_gaussian, ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'num_file_creations', 'num_access_files', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']), # 3, pass a DataFrame to num_pipeline
        ("cat_pipeline_protocol_type", cat_pipeline, 'protocol_type'), # 3, pass a Series to cat_pipeline
        ("cat_pipeline_service", cat_pipeline_ten, 'service'), # 11, pass a Series to cat_pipeline_ten
        ("cat_pipeline_flag", cat_pipeline_five, 'flag'), # 6, pass a Series to cat_pipeline_ten
        ("do_nothing", do_nothing_pipeline, ['land', 'wrong_fragment', 'urgent', 'num_failed_logins', 'logged_in', 'root_shell', 'su_attempted', 'num_shells', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']) # 1, pass a DataFrame to num_pipeline
    ])

def get_layout():
     """Get the initial layout

        Contains title, stores, and project id input component

     Return:
         html.Div, a Dash component
     """
     return html.Div([
         get_title_container(),
         html.Div(get_interaction_tabs(), id = 'container'),
     ])

def get_title_container():
    """ Get the title container

    Return:
        html.Div: a Dash component
    """
    return html.Div([
        dbc.Container([
            html.Div([
                html.Img(src='assets/ndu-logo.png', className='header-icon-l'),
                html.Div(children="CIC 6420 Project", className="header-title"),
                html.Img(src='assets/College_of_Information_and_Cyberspace_Seal.png', className='header-icon-r')
            ], className = 'title-container')
        ])
    ])

def get_test_upload():
    """Get a training data upload page

    Arg:
        project_id (str), a unique project id

    Return:
        html.Div, a Dash component
    """
    return html.Div([
            html.Div('Upload Test Data', className = 'text-body sub-title'),
            du.Upload(id='upload-test', max_file_size=1800, filetypes=['csv'], upload_id='project')
    ])

def get_training_upload():
    """Get a training data upload page

    Arg:
        project_id (str), a unique project id

    Return:
        html.Div, a Dash component
    """
    return html.Div([
            html.Div('Upload Training Data', className = 'text-body sub-title'),
            du.Upload(id='upload-training', max_file_size=1800, filetypes=['csv'], upload_id='project')
    ])

def get_ip_record_table(project_id):
    """Get a record table

    Return:
        dash_table.DataTable, a Dash component
    """
    features = initial_features

    #if len(features) > 10:
        #features = features[:10]+features[-1:]

    return dash_table.DataTable(
        id='ip-record-table',
        columns=[{"name": i, "id": i, 'selectable':True} for i in features],
        page_current=0,
        page_size=10,
        page_action='custom',
        style_table = {'color': 'silver'},
        style_cell = {'background-color': 'rgba(51, 51, 51, 1)', 'text-align':'center', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0},
        cell_selectable=False,
        sort_action='custom',
        sort_mode='multi',
        sort_by=[],
    )

def get_test_content():
    return [html.Div(get_test_upload(), id = 'upload-container-test'),
            html.Div(id = 'category-container-test'),
            html.Div(id = 'visualization-container-test')]

def get_features_test():
    return [html.Div('Select Features', className = 'text-body sub-title'),
            dcc.Dropdown(options = [{'label':feature, 'value':feature} for feature in features], value = initial_features, multi=True, id = 'test-menu')]

def get_features_training():
    return [html.Div('Select Features', className = 'text-body sub-title'),
            dcc.Dropdown(options = [{'label':feature, 'value':feature} for feature in features], value = initial_features, multi=True, id = 'training-menu')]

def get_training_content():
    return [html.Div(get_training_upload(), id = 'upload-container-training'),
            html.Div(id = 'category-container-training'),
            html.Div(id = 'visualization-container-training')]

def get_interaction_tabs():
    """Get tabs

    Return:
        html.Div, a Dash component
    """
    return html.Div([
        dbc.Container([
            html.Br(),
            html.Div(get_tabs(), id = 'tab-list'),
            dcc.Loading(html.Div('Tab Container', id = 'tab-container'), type='dot')
    ])])

def get_tabs():
    """Get tabs component

    Return:
        dbc.Tabs, a DBC component
    """
    return dbc.Tabs(
        [
            dbc.Tab(label="Training", tab_id = 'tab-1',
                    label_style = {'color':'silver', 'text-align': 'center', 'font-weight':'bold'},
                    tab_style = {'width': '20%', 'background-color': '#1C2833', 'margin-left':'5px'},
                    active_label_style = {'color': 'white'}),
            dbc.Tab(label="Test", tab_id = 'tab-2',
                    label_style = {'color':'silver', 'text-align': 'center', 'font-weight':'bold'},
                    tab_style = {'width': '20%', 'background-color': '#1C2833', 'margin-left':'5px'},
                    active_label_style = {'color': 'white'}),
        ], id = 'tabs', active_tab = 'tab-1')

def get_test_graphs(selected_features):
    """Add graphs in tab 3

    Args:
        dataset_id (str), dataset id
        project_id (str), project id

    Return:
        dbc.Container, a DBC component
    """
    graph_containers = []
    graph_containers.append(get_graph_container_protocol_attack(test_data))
    for feature in selected_features:
        graph_containers.append(get_graph_container_test(test_data, feature))

    rows = []
    for graph_container in graph_containers:
        rows.append(html.Div(graph_container, className = 'col-lg-4'))

    content = html.Div([dbc.Row(rows)])
    return dbc.Container(content)

def get_training_graphs(selected_features):
    """Add graphs in tab 3

    Args:
        dataset_id (str), dataset id
        project_id (str), project id

    Return:
        dbc.Container, a DBC component
    """
    graph_containers = []
    graph_containers.append(get_graph_container_protocol_attack(training_data))
    for feature in selected_features:
        graph_containers.append(get_graph_container(training_data, feature))

    rows = []
    for graph_container in graph_containers:
        rows.append(html.Div(graph_container, className = 'col-lg-4'))

    content = html.Div([dbc.Row(rows)])
    return dbc.Container(content)

def get_graph_container_protocol_attack(data):
    """Get a graph container in tab 1 and tab 2

        Each graph container contains a graph title, a graph, a store containing graph parameters, a models for adding interactive labeling datatable

    Args:
        dataset_id (str), dataset id
        project_id (str), project id
        tab_id (str), tab id
        figure_id (str), graph id

    Return:
        html.Div, a Dash component
    """
    graph_title = get_graph_title('Protocol & Attack', className='option-label-2')
    graph = dcc.Graph(figure=get_protocol_attack(data), className='plot-figure')

    return html.Div([graph_title, graph], className='plot-board')

def get_graph_container_test(data, feature):
    """Get a graph container in tab 1 and tab 2

        Each graph container contains a graph title, a graph, a store containing graph parameters, a models for adding interactive labeling datatable

    Args:
        dataset_id (str), dataset id
        project_id (str), project id
        tab_id (str), tab id
        figure_id (str), graph id

    Return:
        html.Div, a Dash component
    """
    graph_title = get_graph_title(feature, className='option-label-2')
    graph = get_graph_test(data, feature)

    return html.Div([graph_title, graph], className='plot-board')

def get_graph_container(data, feature):
    """Get a graph container in tab 1 and tab 2

        Each graph container contains a graph title, a graph, a store containing graph parameters, a models for adding interactive labeling datatable

    Args:
        dataset_id (str), dataset id
        project_id (str), project id
        tab_id (str), tab id
        figure_id (str), graph id

    Return:
        html.Div, a Dash component
    """
    graph_title = get_graph_title(feature, className='option-label-2')
    graph = get_graph(data, feature)

    return html.Div([graph_title, graph], className='plot-board')

def get_graph_title(children, className=None):
    """Get the title of a graph

    Args:
        children (str), title cotent
        className (str), names of class

    Return:
        html.Label, a Dash component
    """
    return html.Label(children=children, className=className)

def get_graph_test(data, feature):
    """Get an interactive graph

    Args:
        dataset_id (str), dataset id
        project_id (str), project id
        graph_id (str), graph id
        args (dict), graph parameters, read from configuration file

    Return:
        dcc.Graph, a Graph component
    """
    return dcc.Graph(figure=get_plot_test(data, feature), className='plot-figure')

def get_graph(data, feature):
    """Get an interactive graph

    Args:
        dataset_id (str), dataset id
        project_id (str), project id
        graph_id (str), graph id
        args (dict), graph parameters, read from configuration file

    Return:
        dcc.Graph, a Graph component
    """
    return dcc.Graph(figure=get_plot(data, feature), className='plot-figure')

def get_plot_test(df, feature):
    """Generate according to the parameters read from configuration file

    Args:
        df (pandas.DataFrame), data
        args (dict), parameters

    Return:
        fig
    """
    if feature in ['protocol_type', 'service', 'flag']:
        return get_plot_cat_bar_test(df, feature) # bar chart
    return get_plot_num_hist_test(df, feature) # histogram

def get_plot(df, feature):
    """Generate according to the parameters read from configuration file

    Args:
        df (pandas.DataFrame), data
        args (dict), parameters

    Return:
        fig
    """
    if feature in ['service', 'flag']:
        return get_plot_cat_bar(df, feature) # bar chart
    elif feature in ['protocol_type']:
        return get_plot_cat_pie(df, feature) # pie chart
    return get_plot_num_hist(df, feature) # histogram

def get_protocol_attack(df):
    temp = df.value_counts(['protocol_type', 'attack'])
    temp = temp.rename('count')
    temp = temp.reset_index()
    fig = px.treemap(temp, path=['protocol_type', 'attack'], values='count')
    fig.update_layout(
                font_color = 'white',
                showlegend=False,
                margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
            paper_bgcolor='rgba(51, 51, 51, 1)',
            plot_bgcolor='rgba(51, 51, 51, 1)',
                )

    return fig

def get_plot_num_hist_test(df, feature):
    """Get a bar chart

    Args:
        df (Pandas DataFrame), data
        args (dict), parameters
            field (str), field used in figure
            display (str), display field name
            type (int), 3 for bar chart of domain length in tab 1
    """
    temp = df.copy()

    fig = go.Figure()

    fig.add_trace(go.Histogram(x = temp[temp['pred_label']==1][feature], nbinsx = 20, name = 'Normal'))
    fig.add_trace(go.Histogram(x = temp[temp['pred_label'] != 2][feature], nbinsx = 20, name = 'Attack'))
    fig.update_layout(
                font_color = 'white',
                showlegend=False,
                xaxis_title=feature,
                yaxis={'title':''},
                margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
            paper_bgcolor='rgba(51, 51, 51, 1)',
            plot_bgcolor='rgba(51, 51, 51, 1)',
                )

    return fig

def get_plot_num_hist(df, feature):
    """Get a bar chart

    Args:
        df (Pandas DataFrame), data
        args (dict), parameters
            field (str), field used in figure
            display (str), display field name
            type (int), 3 for bar chart of domain length in tab 1
    """
    temp = df.copy()
    fig = px.histogram(temp, x=feature, nbins = 20)
    fig.update_layout(
                font_color = 'white',
                showlegend=False,
                xaxis_title=feature,
                yaxis={'title':''},
                margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
            paper_bgcolor='rgba(51, 51, 51, 1)',
            plot_bgcolor='rgba(51, 51, 51, 1)',
                )

    return fig

def get_plot_cat_pie(df, feature):
    """Get a pie chart

    Args:
        df (Pandas DataFrame), data
        args (dict), parameters
            field (str), field used in figure
            display (str), display field name
            type (int), 2 for pie chart in tab 1
            n (int), top n categories
            dropna (boolean), True, drop na records; False, otherwise
            others (boolean), True, group other categories into 'Others'; False, otherwise

    Return:
        Plotly Express figure
    """
    field = feature
    temp = df[field].value_counts(dropna=True)
    temp = temp.to_frame()
    temp.columns = ['number']
    temp[field] = temp.index
    fig = px.pie(temp, values='number', names=field)
    fig.update_layout(
            font_color = 'white',
            showlegend=False,
            margin=dict(
        l=0,
        r=0,
        b=30,
        t=30,
        pad=0
        ),
            paper_bgcolor='rgba(51, 51, 51, 1)',
            )

    return fig

def get_plot_cat_bar_test(df, feature):
    """Get a bar chart

    Args:
        df (Pandas DataFrame), data
        args (dict), parameters
            field (str), field used in figure
            display (str), display field name
            type (int), 1 for bar chart in tab 1
            n (int), top n categories
            dropna (boolean), True, drop na records; False, otherwise
            others (boolean), True, group other categories into 'Others'; False, otherwise

    Return:
        Plotly Express figure
    """
    temp_1 = df[df['pred_label'] == 1][feature].value_counts(dropna=True)
    temp_2 = df[df['pred_label'] == -1][feature].value_counts(dropna=True)
    temp_1 = temp_1.to_frame()
    temp_2 = temp_2.to_frame()
    temp_1.columns = ['count']
    temp_2.columns = ['count']
    temp_1[feature] = temp_1.index
    temp_2[feature] = temp_2.index

    fig = go.Figure()
    fig.add_trace(go.Bar(x=temp_1[feature], y = temp_1['count'], name = 'Normal'))
    fig.add_trace(go.Bar(x=temp_2[feature], y = temp_2['count'], name = 'Attack'))

    fig.update_layout(
                font_color = 'white',
                showlegend=False,
                xaxis_title=feature,
                yaxis={'title':''},
                margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
            paper_bgcolor='rgba(51, 51, 51, 1)',
            plot_bgcolor='rgba(51, 51, 51, 1)',
                )

    return fig

def get_plot_cat_bar(df, feature):
    """Get a bar chart

    Args:
        df (Pandas DataFrame), data
        args (dict), parameters
            field (str), field used in figure
            display (str), display field name
            type (int), 1 for bar chart in tab 1
            n (int), top n categories
            dropna (boolean), True, drop na records; False, otherwise
            others (boolean), True, group other categories into 'Others'; False, otherwise

    Return:
        Plotly Express figure
    """
    temp = df[feature].value_counts(dropna=True)
    temp = temp.to_frame()
    temp.columns = ['count']
    temp[feature] = temp.index
    fig = px.bar(temp, x=feature, y='count')
    fig.update_layout(
                font_color = 'white',
                showlegend=False,
                xaxis_title=feature,
                yaxis={'title':''},
                margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
            paper_bgcolor='rgba(51, 51, 51, 1)',
            plot_bgcolor='rgba(51, 51, 51, 1)',
                )

    return fig

app.layout = get_layout()
app.config['suppress_callback_exceptions'] = True

@app.callback(Output("visualization-container-test", "children"),
              Input("test-menu", "value"))
def output_feature_menu(values):
    return [
            html.Div('Visualizing Features', className = 'text-body sub-title'),
            get_test_graphs(values)
    ]

@app.callback(Output("visualization-container-training", "children"),
              Input("training-menu", "value"))
def output_feature_menu(values):
    return [
            html.Div('Visualizing Features', className = 'text-body sub-title'),
            get_training_graphs(values)
    ]

@app.callback(Output("tab-container", "children"),
              Input("tabs", "active_tab"))
def output_text(at):
    if at == 'tab-1':
        return get_training_content()
    elif at == 'tab-2':
        return get_test_content()

    return 'Unknown ...'

@du.callback(
    output=Output('category-container-test', 'children'),
    id='upload-test',
)

def update_test_data_path(filenames):
    # upload test data
    path = filenames[0]
    base = os.path.basename(path)
    dirname = os.path.dirname(path)
    os.rename(path, os.path.join(dirname, 'test.csv'))
    global test_data
    test_data = pd.read_csv('Uploads/project/test.csv')

    preprocessing_pipeline = joblib.load('Uploads/project/preprocessing.pkl')
    lof = joblib.load('Uploads/project/lof.pkl')

    test_x = preprocess_pipeline.fit_transform(test_data)
    test_y = test_data['attack'].apply(get_label)

    test_pred = lof.predict(test_x)

    test_data['pred_label'] = test_pred
    test_data['true_label'] = test_y

    test_data.to_csv('Uploads/project/test_label.csv', index=False)

    return get_features_test()

@du.callback(
    output=Output('category-container-training', 'children'),
    id='upload-training',
)
def update_training_data_path(filenames):
    # upload test data
    path = filenames[0]
    base = os.path.basename(path)
    dirname = os.path.dirname(path)
    os.rename(path, os.path.join(dirname, 'training.csv'))
    global training_data
    training_data = pd.read_csv('Uploads/project/training.csv')
    train_x = preprocess_pipeline.fit_transform(training_data)
    train_y = training_data['attack'].apply(get_label)
    lof = LocalOutlierFactor(novelty=True, n_neighbors = 200, algorithm = 'auto', metric = 'manhattan')
    lof.fit(train_x[train_y==1])
    joblib.dump(preprocess_pipeline, 'Uploads/project/preprocessing.pkl')
    joblib.dump(lof, 'Uploads/project/lof.pkl')
    return get_features_training()

if __name__ == '__main__':
    app.run_server(debug = True)
    #app.run_server(host='0.0.0.0', port=8082)
