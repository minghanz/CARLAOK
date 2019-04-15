import datetime
import threading
import collections
import numpy as np

import plotly
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash()

colors = {
        'bg':'#282726',
        'text':'#E1DCD9',
        'line-color': ['#E1DCD9', '#A7414A', '#A37C27'],
}
data = {
    'time': collections.deque(maxlen=200),
    'throttle': collections.deque(maxlen=200),
    'brake': collections.deque(maxlen=200),
    'steering': collections.deque(maxlen=200),
    'speed': collections.deque(maxlen=200),

    'enable_safeguard': False,
    'enable_perception': False,
    'enable_aggressive': False,

    'dist_inner_behind_true': collections.deque(maxlen=200),
    'dist_inner_behind_perc': collections.deque(maxlen=200),
    'dist_inner_front_true': collections.deque(maxlen=200),
    'dist_inner_front_perc': collections.deque(maxlen=200),
    'dist_outer_front_true': collections.deque(maxlen=200),
    'dist_outer_front_perc': collections.deque(maxlen=200),
    'dist_outer_behind_true': collections.deque(maxlen=200),
    'dist_outer_behind_perc': collections.deque(maxlen=200),

    'speed_inner_behind_true': collections.deque(maxlen=200),
    'speed_inner_behind_perc': collections.deque(maxlen=200),
    'speed_inner_front_true': collections.deque(maxlen=200),
    'speed_inner_front_perc': collections.deque(maxlen=200),
    'speed_outer_front_true': collections.deque(maxlen=200),
    'speed_outer_front_perc': collections.deque(maxlen=200),
    'speed_outer_behind_true': collections.deque(maxlen=200),
    'speed_outer_behind_perc': collections.deque(maxlen=200),
}

infotext_style = {
    'font-size': '10pt',
    'font-family': 'Calibri',
    'color': colors['text'],
    'margin-left': '16px',
    'margin-up': '0px',
    'margin-down': '0px',
}
infotext_time_style = {
    'font-size': '16pt',
    'font-family': 'Calibri',
    'color': colors['text'],
    'text-align': 'center'
}

app.layout = html.Div(
    children=[
        html.Table(
            [html.Tr([
                html.Td([
                    html.Div([
                        html.P(
                            '',
                            id='server_time',
                            style=infotext_time_style,
                        ),
                        html.P(
                            '',
                            id='rl_state',
                            style=infotext_style,
                        ),
                        html.P(
                            '',
                            id='percp_state',
                            style=infotext_style,
                        ),
                        html.P(
                            '',
                            id='aggr_state',
                            style=infotext_style,
                        ),
                        html.P(
                            '',
                            id='throttle',
                            style=infotext_style,
                        ),
                        html.P(
                            '',
                            id='brake',
                            style=infotext_style,
                        ),
                        html.P(
                            '',
                            id='steering',
                            style=infotext_style,
                        )
                    ]),
                ],
                rowSpan=2,
                style={
                    'width': '25%',
                }),
                html.Td([
                    dcc.Graph(
                        id = 'dist-graph'
                    )],
                    style={
                        'width': '37.5%'
                    }
                ),
                html.Td([
                    dcc.Graph(
                        id = 'speed-graph'
                    )],
                    style={
                        'width': '37.5%'
                    }
                )],
            )],
            # html.Tr([],
            # )],
            # style={
            #     'width': '100%'
            # }
        ),
        dcc.Interval( # fresher
            id='interval-component',
            interval=200, # in milliseconds
            n_intervals=0
        )
    ],
    style={
        'backgroundColor':colors['bg']
    }, 
)


truth_line = dict(
    width = 1.5,
)
percp_line = dict(
    width = 1.5,
    dash = 'dash'
)

@app.callback(Output('server_time', 'children'),
             [Input('interval-component', 'n_intervals')])
def fresh_time(niter):
    return ('Sim time: %.2fs' % data['time'][-1]) if len(data['time'])>0 else 'N/A'

@app.callback(Output('rl_state', 'children'),
             [Input('interval-component', 'n_intervals')])
def fresh_rl_state(niter):
    return "RL Kicking In: {0}".format(not data['enable_safeguard'])

@app.callback(Output('percp_state', 'children'),
             [Input('interval-component', 'n_intervals')])
def fresh_percp_state(niter):
    return "Enable Perception: {0}".format(data['enable_perception'])

@app.callback(Output('aggr_state', 'children'),
             [Input('interval-component', 'n_intervals')])
def fresh_aggr_state(niter):
    return "Is Aggressive: {0}".format(data['enable_aggressive'])

@app.callback(Output('throttle', 'children'),
             [Input('interval-component', 'n_intervals')])
def fresh_throttle(niter):
    return "Throttle: {0}%".format(int(data['throttle'][-1]*100) if len(data['throttle']) > 0 else 'N/A')

@app.callback(Output('brake', 'children'),
             [Input('interval-component', 'n_intervals')])
def fresh_brake(niter):
    return "Brake: {0}%".format(int(data['brake'][-1]*100) if len(data['brake']) > 0 else 'N/A')

@app.callback(Output('steering', 'children'),
             [Input('interval-component', 'n_intervals')])
def fresh_steering(niter):
    return "Steering: {0}".format(int(data['steering'][-1]*100) if len(data['steering']) > 0 else 'N/A')

@app.callback(Output('speed-graph', 'figure'),
             [Input('interval-component', 'n_intervals')])
def fresh_speed_graph(niter):
    return {
        'data':[
            # go.Scatter(
            #     x = list(data['time']),
            #     y = list(data['speed_outer_behind_true']),
            #     name='outer-behind-veh (Truth)',
            #     line=truth_line,
            # ),
            # go.Scatter(
            #     x = list(data['time']),
            #     y = list(data['speed_outer_behind_perc']),
            #     name='outer-behind-veh (Percp)',
            #     line=percp_line,
            # ),
            # go.Scatter(
            #     x = list(data['time']),
            #     y = list(data['speed_inner_behind_true']),
            #     name='inner-behind-veh (Truth)',
            #     line=truth_line,
            # ),
            # go.Scatter(
            #     x = list(data['time']),
            #     y = list(data['speed_inner_behind_perc']),
            #     name='inner-behind-veh (Percp)',
            #     line=percp_line,
            # ),
            go.Scatter(
                x = list(data['time']),
                y = list(data['speed']),
                name='ego-veh',
                line=dict(width = 1.5, color=colors['line-color'][2]),
            ),
            go.Scatter(
                x = list(data['time']),
                y = list(data['speed_outer_front_true']),
                name='outer-front-veh (Truth)',
                line=dict(width = 1.5, color=colors['line-color'][0]),
            ),
            go.Scatter(
                x = list(data['time']),
                y = list(data['speed_outer_front_perc']),
                name='outer-front-veh (Percp)',
                line=dict(width = 1.5, dash = 'dash', color=colors['line-color'][0]),
            ),
            go.Scatter(
                x = list(data['time']),
                y = list(data['speed_inner_front_true']),
                name='inner-front-veh (Truth)',
                line=dict(width = 1.5, color=colors['line-color'][1]),
            ),
            go.Scatter(
                x = list(data['time']),
                y = list(data['speed_inner_front_perc']),
                name='inner-front-veh (Percp)',
                line=dict(width = 1.5, dash = 'dash', color=colors['line-color'][1]),
            ),
        ],
        'layout':{
            'plot_bgcolor': colors['bg'],
            'paper_bgcolor': colors['bg'],
            'font':{
                'color': colors['text']
            },
            'title':'Surrounding Vehicle Speed Visualization',
            'legend':dict(orientation="h"),
            'yaxis':dict(range=[0,60]),
            'autosize': True,
            'width': 500,
            'height': 500,
        },
    }

@app.callback(Output('dist-graph', 'figure'),
             [Input('interval-component', 'n_intervals')])
def fresh_dist_graph(niter):
    return {
        'data':[
            go.Scatter(
                x = list(data['time']),
                y = list(data['dist_outer_front_true']),
                name='outer-front-veh (Truth)',
                line=dict(width = 1.5, color=colors['line-color'][0]),
            ),
            go.Scatter(
                x = list(data['time']),
                y = list(data['dist_outer_front_perc']),
                name='outer-front-veh (Percp)',
                line=dict(width = 1.5, dash = 'dash', color=colors['line-color'][0]),
            ),
            go.Scatter(
                x = list(data['time']),
                y = list(data['dist_inner_front_true']),
                name='inner-front-veh (Truth)',
                line=dict(width = 1.5, color=colors['line-color'][1]),
            ),
            go.Scatter(
                x = list(data['time']),
                y = list(data['dist_inner_front_perc']),
                name='inner-front-veh (Percp)',
                line=dict(width = 1.5, dash = 'dash', color=colors['line-color'][1]),
            ),
        ],
        'layout':{
            'plot_bgcolor': colors['bg'],
            'paper_bgcolor': colors['bg'],
            'font':{
                'color': colors['text']
            },
            'title':'Surrounding Vehicle Distance Visualization',
            'legend':dict(orientation="h"),
            'yaxis':dict(range=[0,80]),
            'autosize': True,
            'width': 500,
            'height': 500,
        },
    }
    
def run_thread():
    thread = threading.Thread(target=lambda: app.run_server(host='0.0.0.0'))
    thread.start()
    return thread

def data_append(**kvargs):
    for k, v in kvargs.items():
        if k == "enable_safeguard":
            data["enable_safeguard"] = v
        elif k == "enable_perception":
            data["enable_perception"] = v
        elif k == "enable_aggressive":
            data["enable_aggressive"] = v
        else:
            data[k].append(v)

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')
