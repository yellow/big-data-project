import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
from MachineLearning import load_data, transform_data,  plot_importantfeatures

from Clustering import get_occupdata, scale_data, kmeans

df = pd.read_csv("nooutliers.TXT")
# the style arguments for the sidebar.
geojson = px.data.election_geojson()

SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#fbfaff'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '23%',
    'margin-right': '2%',
    'padding': '20px 10p',

}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#55126b'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#007afc'
}
df1 = df.sort_values(by="LOC")
controls = dbc.FormGroup(
    [
        html.P('Select States', style={
            'textAlign': 'center'
        }),
        dcc.Dropdown(
            id='dropdown',
            options=[{'label': i, 'value': i} for i in df1['LOC'].unique()],
            value=['PENNSYLVANIA'],  # default value
            multi=True
        ),
        html.Br(),
        html.P('Length of Service', style={
            'textAlign': 'center'
        }),
        dcc.RangeSlider(
            id='range_slider',
            min=0,
            max=50,
            value=[0, 50],
            marks={
                0: '0',
                10: '10',
                20: '20',
                30: '30',
                40: '40',
                50: '50'
            },

        ),
        html.P('Education Level', style={
            'textAlign': 'center'
        }),
        dbc.Card([dbc.Checklist(
            id='check_list',
            options=[{'label': i, 'value': i} for i in df['EDLVL'].unique()],
            value=['BACHELORS'],
            inline=True
        )]),

        html.P('Industry', style={
            'textAlign': 'center'
        }),
        dbc.Card([dbc.RadioItems(
            id='radio_items',
            options=[{'label': 'STEM',
                      'value': ['SCIENCE OCCUPATIONS', 'TECHNOLOGY OCCUPATION', 'ENGINEERING OCCUPATIONS',
                                'MATHEMATICS OCCUPATIONS']
                      },
                     {
                         'label': 'Health',
                         'value': 'HEALTH OCCUPATIONS'
                     },
                     {
                         'label': 'Others',
                         'value': 'ALL OTHER OCCUPATIONS'
                     }
                     ],
            value='SCIENCE OCCUPATIONS',
            style={
                'margin': 'auto'
            }
        )]),
        html.Br(),
        dbc.Button(
            id='submit_button',
            n_clicks=0,
            children='Submit',
            color='primary',
            block=True
        ),
    ]
)

sidebar = html.Div(
    [
        html.H2('Constraints', style=TEXT_STYLE),
        html.Hr(),
        controls
    ],
    style=SIDEBAR_STYLE,
)

content_first_row = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4(id='card_title_1', children=['Card Title 1'], className='card-title',
                                style=CARD_TEXT_STYLE),
                        html.P(id='card_text_1', children=['Sample text.'], style=CARD_TEXT_STYLE),
                    ]
                )
            ]
        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4(id='card_title_2', children=['Card Title 2'], className='card-title',
                                style=CARD_TEXT_STYLE),
                        html.P(id='card_text_2', children=['Sample text.'], style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4(id='card_title_3', children=['Card Title 3'], className='card-title',
                                style=CARD_TEXT_STYLE),
                        html.P(id='card_text_3', children=['Sample text.'], style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4(id='card_title_4', children=['Card Title 4'], className='card-title',
                                style=CARD_TEXT_STYLE),
                        html.P(id='card_text_4', children=['Sample text.'], style=CARD_TEXT_STYLE),
                    ]
                ),
            ]
        ),
        md=3
    )
])

content_fifth_row = dbc.Row(

    [
        dbc.Col(
            dcc.Graph(id='graph_7'), md=12,
        )
    ]
)
content_second_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph_1'), md=4
        ),
        dbc.Col(
            dcc.Graph(id='graph_2'), md=4
        ),
        dbc.Col(
            dcc.Graph(id='graph_3'), md=4
        )
    ]
)

content_third_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph_4'), md=12,
        )
    ]
)

content_fourth_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph_5'), md=6
        ),
        dbc.Col(
            dcc.Graph(id='graph_6'), md=6
        ),

    ]
)

content = html.Div(
    [
        html.H2('EDA Federal Employees Dataset', style=TEXT_STYLE),
        html.Hr(),

        content_first_row,

        content_second_row,
        content_fifth_row,
        content_third_row,
        content_fourth_row
    ],
    style=CONTENT_STYLE
)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([sidebar, content])


@app.callback(
    Output('graph_1', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_graph_1(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    df_penn = df[df["LOC"] == "PENNSYLVANIA"]
    fig = px.histogram(df_penn, x='EDLVL', barmode='group')
    return fig


@app.callback(
    Output('graph_2', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_graph_2(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    loc = ["PENNSYLVANIA", "NEW JERSEY", "NEW YORK"]
    df_tristate = df[df["LOC"].isin(loc)]
    fig = px.strip(df_tristate, x="SALARY", y="EDLVL", color="is_STEM")

    return fig


@app.callback(
    Output('graph_3', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_graph_3(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    df_bach = df[df["EDLVL"] == "BACHELORS"]
    fig = px.violin(df_bach, y="SALARY")
    return fig


@app.callback(
    Output('graph_4', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_graph_4(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    df_stem = df[df["STEMOCC"] != "ALL OTHER OCCUPATIONS"]
    df_stem = df_stem[df_stem["STEMOCC"] != "UNSPECIFIED"]
    fig = px.box(df_stem, y="LOS", x="STEMOCC")
    return fig


@app.callback(
    Output('graph_5', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_graph_5(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    df_loc = df[df["LOC"] == "MICHIGAN"]
    fig = px.histogram(df_loc, x="GEN", barmode="group")
    return fig


@app.callback(
    Output('graph_6', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_graph_6(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    df_type = df[df["LOC"] == "PENNSYLVANIA"]
    fig = px.box(df_type, x="PATCO", y="SALARY", color="is_STEM")
    return fig


@app.callback(
    Output('graph_7', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_graph_7(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    df_range_filter = df[(df['LOS'] >= float(range_slider_value[0])) & (df['LOS'] <= float(range_slider_value[1]))]
    if len(dropdown_value) != 0:
        df_state = df_range_filter[df_range_filter["LOC"].isin(dropdown_value)]
    else:
        df_state = df_range_filter
    if len(check_list_value) != 0:
        df_edu = df_state[df_state["EDLVL"].isin(check_list_value)]
    else:
        df_edu = df_state
    if len(radio_items_value) != 0:
        df_industry = df_edu[df_edu["STEMOCC"] == radio_items_value]
    else:
        df_industry = df_edu
    fig = px.scatter(df_industry, x="SALARY", y="LOS")
    return fig


@app.callback(
    Output('card_title_1', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_card_title_1(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    return 'Median income of Federal Employees in the USA'


@app.callback(
    Output('card_text_1', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_card_text_1(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    amount = df["SALARY"].median()
    return "${:,.2f}".format(amount)


@app.callback(
    Output('card_title_2', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_card_title_2(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    return 'Average length of service of federal employees'


@app.callback(
    Output('card_text_2', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_card_text_2(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    length = df["LOS"].mean()
    return str(length.round(1)) + " years"


@app.callback(
    Output('card_title_3', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_card_title_3(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    return 'Number of Employees stationed in the USA'


@app.callback(
    Output('card_text_3', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_card_text_3(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    amount = df["SALARY"].count()
    return "{:,.2f}".format(amount)


@app.callback(
    Output('card_title_4', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_card_title_4(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    return 'Month of data collection'


@app.callback(
    Output('card_text_4', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'), State('range_slider', 'value'), State('check_list', 'value'),
     State('radio_items', 'value')
     ])
def update_card_text_4(n_clicks, dropdown_value, range_slider_value, check_list_value, radio_items_value):
    a = "June 2021"
    return a


if __name__ == '__main__':
    print("Choose an  option: - ")
    print("1. Launch Dashboard")
    print("2. Get ML Classification Report along with important features that affect salary prediction")
    print("3. Get clustering information for Computer Science jobs in the public sector")
    val = int(input("Enter your value: "))
    if val == 1:
        app.run_server(port=8085)
    elif val == 2:
        df = load_data()
        df = transform_data(df)
        plot_importantfeatures(df)
    elif val == 3:
        occ = "COMPUTER SCIENCE"
        df = get_occupdata(occ)
        X = scale_data(df)
        kmeans(X, 3, df)
