import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import requests

# constants
current_year = pd.Timestamp.now().year

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Bike Buddy", className="text-center mb-4"),
    
    dbc.Row([
        dbc.Col([
                dbc.Label("Model Year", html_for="model-year"),
                dcc.Dropdown(id="model-year", options=list(range(2000,current_year + 1)),placeholder="Select Model Year"),
                dbc.Label("Ad Title", html_for="ad-title"),
                dcc.Input(id="ad-title", type="text", placeholder="Enter Ad Title"),
                dbc.Label("Ad Description", html_for="ad-description"),
                dcc.Input(id="ad-description", type="text", placeholder="Enter Ad Description"),
                dbc.Label("Country", html_for="country"),
                dcc.Dropdown(id="country",
                             options=[{"label": "Canada", "value": "Canada"},
                                      {"label": "United States", "value": "United States"}],
                             placeholder="Select Country"),
                dbc.Label("Post Date", html_for="post-date"),
                dcc.DatePickerSingle(id="post-date", placeholder="Select Post Date")
        ]),
    ]),

    dbc.Button("Submit for Prediction", id="submit-prediction", color="primary", className="mt-3"),

    dbc.Row([
        # dbc.Col([
        #     dcc.Graph(id="plotly-graph"),
        # ], width=6),
        dbc.Col([
            html.Div(id="kpi", className="text-center")
        ], width=6),
    ], className="mt-4"),
])


@app.callback(
    Output("kpi", "children"),
    [Input("submit-prediction", "n_clicks")],
    [State("model-year", "value"),
     State("ad-title", "value"),
     State("ad-description", "value"),
     State("country", "value"),
     State("post-date", "date")]
)
def update_output(n_clicks, model_year, ad_title, ad_description, country, post_date):
    if n_clicks is None:
        return html.H3("Please Enter Data For Price Prediction")
    
    # Replace with the actual API endpoint and request format
    api_url = "https://bikebuddy-api.azurewebsites.net/text-predict"
    data = [{
        "ad_title": f"{model_year} {ad_title}",
        "ad_description": ad_description,
        "location": f"Unknown, {country}",
        "original_post_date": post_date,
    }]
    
    response = requests.post(api_url, json=data)
    response_data = response.json()
    kpi_value = response_data["predictions"][0]

    # figure = go.Figure(
    #     data=[go.Scatter(x=x_data, y=y_data, mode="lines+markers")],
    #     layout=go.Layout(title="Plotly Graph")
    # )

    kpi_element = html.H3(f"Predicted Price: ${round(kpi_value, 2)} CAD", className="text-center")

    return kpi_element


if __name__ == "__main__":
    app.run_server(debug=True)

