import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import cuml
import umap

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

# -----------------------------Data ----------------------------------------------------------
df_valid_gluon_inspection = pd.read_csv(
    "data/df_valid_gluon_inspection.csv", index_col=None
)
valid_gluon_embeddings = np.load("data/valid_gluon_embeddings.npy")
metrics = [
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "cosine",
    "correlation",
    "mahalanobis",
]


# ----------------------------Helper functions------------------------------------------------
def wrap_to_width(text):
    """
    Helper func to wrap text for hover data
    inserts <br> where needed for HTML line breaks
    """
    max_width_in_words = 10
    if len(text.split()) > max_width_in_words:
        num_chunks = np.ceil(len(text.split()) // max_width_in_words)
        chunks = np.array_split(text.split(), num_chunks)
        return "<br>".join([" ".join(chunk[:]) for chunk in chunks])
    else:
        return text


# -----------------------------Layout-----------------------------------------------------
app.layout = dbc.Col(
    [
        dbc.Row(
            [
                html.Label("Select Column to Color By"),
                dcc.Dropdown(
                    id="color-by",
                    options=df_valid_gluon_inspection.columns.to_list(),
                    value="category",
                ),
                html.Label("Select min_dist for UMAP"),
                dcc.Slider(id="min-dist", min=0, max=1, value=0.1),
                html.Label("Select n_neighbors for UMAP"),
                dcc.Slider(id="n-neighbors", min=1, max=100, value=10),
                html.Label("Select n_epochs for UMAP"),
                dcc.Slider(id="n-epochs", min=0, step=10, max=500, value=200),
                html.Label("Select spread for UMAP"),
                dcc.Slider(id="spread", min=0, max=5, value=1),
                html.Label("Select learning rate for UMAP"),
                dcc.Slider(id="learning-rate", step=0.1, min=0, max=5, value=1),
                html.Label("Select distance metric for UMAP"),
                dcc.Dropdown(id="metric", options=metrics, value=metrics[0]),
                dcc.Markdown(
                    """
            [Clicked Ad Link](/)
        """,
                    id="ad-link",
                ),
            ]
        ),
        dbc.Row([dcc.Graph(id="umap-visual")]),
    ]
)


# -----------------------------Callbacks---------------------------------------------------
@app.callback(
    Output("umap-visual", "figure"),
    [
        Input("color-by", "value"),
        Input("min-dist", "value"),
        Input("n-neighbors", "value"),
        Input("metric", "value"),
        Input("n-epochs", "value"),
        Input("spread", "value"),
        Input("learning-rate", "value"),
    ],
)
def build_umap_visual(
    color_by, min_dist, n_neighbors, metric, n_epochs, spread, learning_rate
):
    reducer = cuml.UMAP(
        random_state=42,
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_epochs=n_epochs,
        spread=spread,
        learning_rate=learning_rate,
    )
    valid_gluon_umap = reducer.fit_transform(valid_gluon_embeddings)

    # if df_valid_gluon_inspection[color_by].unique().shape[0] > 10000:
    #     color_by = pd.cut(df_valid_gluon_inspection[color_by], bins=20).astype(str)
    # else:

    color_by_data = df_valid_gluon_inspection[color_by]

    fig = px.scatter_3d(
        x=valid_gluon_umap[:, 0],
        y=valid_gluon_umap[:, 1],
        z=valid_gluon_umap[:, 2],
        color=color_by_data,
        hover_data={
            "URL": (":.s", df_valid_gluon_inspection.url.values),
            "Ad Title": (":.s", df_valid_gluon_inspection.ad_title.values),
            "Description": (
                ":.s",
                df_valid_gluon_inspection.description.apply(wrap_to_width).values,
            ),
            "Original Post Date": (
                ":.s",
                df_valid_gluon_inspection.original_post_date.values,
            ),
            "Age in Days at Post": (
                ":.s",
                df_valid_gluon_inspection.add_age__age_at_post.values,
            ),
            "Actual Price ($CAD)": (
                ":.s",
                df_valid_gluon_inspection.price__price_cpi_adjusted_CAD.values,
            ),
            "Predicted Price ($CAD)": (":.s", df_valid_gluon_inspection.pred.values),
            "Model Residual (+= under predicted)": (
                ":.s",
                df_valid_gluon_inspection.resid.values,
            ),
            color_by: (":.s", color_by_data),
        },
        height=1000,
        width=2000,
    )
    fig.update_traces(
        hovertemplate="""
    <b>Ad Title</b>=%{customdata[1]:.s}<br>
    <b>Description</b>=%{customdata[2]:.s}<br>
    <b>Original Post Date</b>=%{customdata[3]:.s}<br>
    <b>Age in Days at Post</b>=%{customdata[4]:.s}<br>
    <b>Actual Price ($CAD)</b>=%{customdata[5]:.s}<br>
    <b>Predicted Price ($CAD)</b>=%{customdata[6]:.s}<br>
    <b>Model Residual (+= under predicted)</b>=%{customdata[7]:.s}<br>
    <b>Colored value=%{customdata[8]:.s} </b>
    <extra></extra>
    """
    )

    return fig


@app.callback(Output("ad-link", "children"), [Input("umap-visual", "clickData")])
def open_url(clickData):
    if clickData:
        url = clickData["points"][0]["customdata"][0]
        return f"""[Link to Clicked Ad: {url}]({url})"""
    else:
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True, port=8060)
