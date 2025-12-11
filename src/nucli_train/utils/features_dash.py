from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt


from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from dash import Dash, dcc, html, Input, Output, callback, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc



from dash.exceptions import PreventUpdate

import base64
import io
from PIL import Image, ImageDraw

import plotly.express as px

# Dummy image drawing function




def plot_magns_features(feature_names, magns_sc, magns_lc):
    assert len(feature_names) == len(magns_sc) == len(magns_lc), "All lists must be of equal length"

    n_features = len(feature_names)

    # Choose fixed consistent colors
    sc_color = 'blue'
    lc_color = 'red'

    fig = make_subplots(rows=1, cols=n_features, subplot_titles=feature_names)

    for i in range(n_features):
        x_vals = np.arange(len(magns_sc[i]))

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=magns_sc[i],
                mode='lines',
                name='SC' if i == 0 else None,
                line=dict(color=sc_color),
                showlegend=(i == 0)
            ),
            row=1,
            col=i + 1
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=magns_lc[i],
                mode='lines',
                name='LC' if i == 0 else None,
                line=dict(color=lc_color),
                showlegend=(i == 0)
            ),
            row=1,
            col=i + 1
        )

    fig.update_layout(
        height=300,
        width=600 * n_features,
        title_text="SC vs LC Magnitudes per Feature",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2)
    )
    return fig



def plot_diff_features(feature_names, diffs):
    assert len(feature_names) == len(diffs), "All lists must be of equal length"

    n_features = len(feature_names)

    sc_color = 'blue'

    fig = make_subplots(rows=1, cols=n_features, subplot_titles=feature_names)

    for i in range(n_features):
        x_vals = np.arange(len(diffs[i]))

        # SC Line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=diffs[i],
                mode='lines',
                line=dict(color=sc_color),
            ),
            row=1,
            col=i + 1
        )


    fig.update_layout(
        height=300,
        width=600 * n_features,
        title_text="Difference between SC and LC Feature Maps",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2)
    )
    return fig


class FeatureExtractor:
    def __init__(self):
        self.model = torch.hub.load("warvito/MedicalNet-models", model="medicalnet_resnet10_23datasets")
        self.FE = None
        self.nodes = get_graph_node_names(self.model)[0]
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        sc_im = np.load('/home/vicky/projects/nuclivision/norms/val.npy')[5][1:]
        lc_im = np.load('/home/vicky/projects/nuclivision/norms/val.npy')[5][:1]
        
        sc = torch.tensor(sc_im).unsqueeze(0)
        avg, std = torch.mean(sc), torch.std(sc)
        sc = (sc - avg) / std # change normalization once we want more general script beyond medicalnet

        lc = torch.tensor(lc_im).unsqueeze(0)
        lc = (lc - avg) / std

        self.sc = sc
        self.lc = lc





    def setup_extractor(self, feature_names):
        self.lc_features = None
        self.sc_features = None
        self.FE = create_feature_extractor(self.model, return_nodes=feature_names)
        self.feature_names = feature_names


    def set_features(self):
        self.lc_features = self.FE(self.lc)
        self.sc_features = self.FE(self.sc)

    def plot_magnitudes(self):
        if self.lc_features is None or self.sc_features is None:
            self.set_features()



        magns_sc, magns_lc = [], []
        for f_name in self.feature_names:
            feature_sc = self.sc_features[f_name].squeeze().numpy() 
            magns_sc.append(np.sqrt(np.sum(feature_sc**2, axis=(1, 2, 3))))

            feature_lc = self.lc_features[f_name].squeeze().numpy() 
            magns_lc.append(np.sqrt(np.sum(feature_lc**2, axis=(1, 2, 3))))
        
        magn_fig = plot_magns_features(self.feature_names, magns_sc, magns_lc)

        return magn_fig

    def plot_differences(self):
        if self.lc_features is None or self.sc_features is None:
            self.set_features()



        diffs = []
        for f_name in self.feature_names:
            diff = self.sc_features[f_name].squeeze().numpy()  - self.lc_features[f_name].squeeze().numpy()
            diffs.append(np.sqrt(np.sum(diff**2, axis=(1, 2, 3))))

        
        diff_fig = plot_diff_features(self.feature_names, diffs)

        return diff_fig

model = FeatureExtractor()





selection_header = dbc.Row([dcc.Dropdown(model.nodes, id='nodes', multi=True), html.Button('Submit', id='submit')])


content = html.Div(children=[dbc.Row([html.Div(id='magnitudes'), html.Div(children=[dcc.Graph(id='dif-fig')], id='differences')]),
        html.Div(children=[], id='feature-maps')], id='content')
import pandas as pd
app = Dash()
app.layout = html.Div([
    selection_header,
    content
])



def draw_feature(feature_name, x_val):
    sc = model.sc_features[feature_name].squeeze().numpy()[x_val]
    lc = model.lc_features[feature_name].squeeze().numpy()[x_val]
    
    image = np.stack([sc, lc])


    image = np.clip(image, a_min=np.min(sc), a_max=np.mean(sc)) # some baseline so we make sure we can see the image

    fig = px.imshow(image, facet_col=0, animation_frame=3, binary_string=True, labels=dict(animation_frame="slice"))

    return fig


@callback([Output('magnitudes', 'children'), Output('differences', 'children')],
    Input('submit', 'n_clicks'),
    State('nodes', 'value')
)
def update_output(n_clicks, nodes):
    if nodes is None or len(nodes) == 0:
        return [], []
    model.setup_extractor(nodes)
    magn = model.plot_magnitudes()
    diff = model.plot_differences()
    return [dcc.Graph(figure=magn)], [dcc.Graph(figure=diff, id='dif-fig')]


@callback(
    Output('feature-maps', 'children'),
    Input('dif-fig', 'clickData'), suppress_callback_exceptions=True, prevent_initial_call =True  # Assume your dcc.Graph has id='diff-feature-fig'
)
def update_image_from_click(clickData):
    if not clickData or 'points' not in clickData:
        raise PreventUpdate
    print(clickData)

    point = clickData['points'][0]
    x_val = point['x']
    curve_num = point['curveNumber']  # corresponds to trace index in figure

    # Map curveNumber to feature index (since each subplot has 1 trace per feature)
    feature_index = curve_num  # since 1 trace per subplot in your loop
    if feature_index >= len(model.feature_names):
        return PreventUpdate

    feature_name = model.feature_names[feature_index]
    fig = draw_feature(feature_name, x_val)

    return [dcc.Graph(figure=fig)]
if  __name__ == '__main__':
    app.run(debug=True)