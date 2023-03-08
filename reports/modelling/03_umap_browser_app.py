import json
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import cuml
from cuml.metrics import pairwise_distances

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

# -----------------------------Data ----------------------------------------------------------
df_gluon_inspection = pd.read_csv("data/df_gluon_inspection.csv", index_col=None)
gluon_embeddings = np.load("data/gluon_embeddings.npy")
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


def get_nearest_examples(
    selected_point_embedding: np.array,
    num_neighbors: int,
    embeddings_lookup: np.array,
    df_metadata: pd.DataFrame,
) -> tuple:
    distance_matrix = pairwise_distances(
        X=embeddings_lookup, Y=selected_point_embedding, metric="cosine"
    )
    closest_indices = np.argsort(distance_matrix, axis=0)[:num_neighbors].reshape(-1)
    closest_distances = distance_matrix[closest_indices].reshape(-1)
    return (
        embeddings_lookup[closest_indices],
        df_metadata.iloc[closest_indices, :].assign(distance=closest_distances),
    )


def build_3d_scatter(
    gluon_umap: np.array, df_gluon_inspection: pd.DataFrame, color_by: str
) -> px.scatter_3d:
    color_by_data = df_gluon_inspection[color_by]
    fig = px.scatter_3d(
        x=gluon_umap[:, 0],
        y=gluon_umap[:, 1],
        z=gluon_umap[:, 2],
        color=color_by_data,
        # custom_data=df_gluon_inspection.url.values,
        hover_data={
            "URL": (":.s", df_gluon_inspection.url.values),
            "Ad Title": (":.s", df_gluon_inspection.ad_title.values),
            "Description": (
                ":.s",
                df_gluon_inspection.description.apply(wrap_to_width).values,
            ),
            "Original Post Date": (
                ":.s",
                df_gluon_inspection.original_post_date.values,
            ),
            "Age in Days at Post": (
                ":.s",
                df_gluon_inspection.add_age__age_at_post.values,
            ),
            "Actual Price ($CAD)": (
                ":.s",
                df_gluon_inspection.price__price_cpi_adjusted_CAD.values,
            ),
            "Predicted Price ($CAD)": (":.s", df_gluon_inspection.pred.values),
            "Model Residual (+= under predicted)": (
                ":.s",
                df_gluon_inspection.resid.values,
            ),
            color_by: (":.s", color_by_data),
        },
        height=800,
        # width=2000,
    )

    # Format hover data
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


# -----------------------------Layout-----------------------------------------------------
app.layout = dbc.Col(
    [
        dbc.Row(
            [
                html.Label("Select Column to Color By"),
                dcc.Dropdown(
                    id="color-by",
                    options=df_gluon_inspection.columns.to_list(),
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
                html.Pre(id="debug-output"),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Train Set"),
                        dcc.Loading(
                            type="cube", children=[dcc.Graph(id="umap-visual-train")]
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.H2("Valid Set"),
                        html.Label(
                            "Click data point to get nearest neighbors in train set"
                        ),
                        dcc.Loading(
                            type="cube", children=[dcc.Graph(id="umap-visual-valid")]
                        ),
                    ]
                ),
            ]
        ),
    ]
)


# -----------------------------Callbacks---------------------------------------------------


# @app.callback(
#     Output("debug-output", "children"), [Input("umap-visual-valid", "clickData")]
# )
# def get_selected_point(clicked_data):
#     return json.dumps(clicked_data["points"], indent=2)


@app.callback(
    [
        Output("umap-visual-train", "figure"),
        Output("umap-visual-valid", "figure"),
    ],
    [
        Input("color-by", "value"),
        Input("min-dist", "value"),
        Input("n-neighbors", "value"),
        Input("metric", "value"),
        Input("n-epochs", "value"),
        Input("spread", "value"),
        Input("learning-rate", "value"),
        Input("umap-visual-valid", "clickData"),
    ],
)
def build_umap_visual(
    color_by,
    min_dist,
    n_neighbors,
    metric,
    n_epochs,
    spread,
    learning_rate,
    clicked_valid_point,
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
    gluon_umap = reducer.fit_transform(gluon_embeddings)

    # If validation point clicked, return embeddings for nearest neighbours in embedded space
    # before UMAP, within train set.
    if clicked_valid_point:
        num_neighbors = 30
        selected_url = clicked_valid_point["points"][0]["customdata"][0]
        (
            nearest_neighbor_embeddings,
            nearest_neighbor_metadata,
        ) = get_nearest_examples(
            selected_point_embedding=gluon_embeddings[
                df_gluon_inspection.query("url == @selected_url").index
            ],
            num_neighbors=num_neighbors,
            embeddings_lookup=gluon_embeddings[
                df_gluon_inspection.query("split=='train'").index
            ],
            df_metadata=df_gluon_inspection,
        )

        # Hacky, but returned df has distance column added
        train_fig = build_3d_scatter(
            nearest_neighbor_embeddings, nearest_neighbor_metadata, color_by="distance"
        )
    else:
        train_fig = build_3d_scatter(
            gluon_umap[df_gluon_inspection.query("split=='train'").index],
            df_gluon_inspection.query("split=='train'"),
            color_by,
        )
    valid_fig = build_3d_scatter(
        gluon_umap[df_gluon_inspection.query("split=='valid'").index],
        df_gluon_inspection.query("split=='valid'"),
        color_by,
    )
    return train_fig, valid_fig


@app.callback(Output("ad-link", "children"), [Input("umap-visual-valid", "clickData")])
def open_url(clickData):
    if clickData:
        url = clickData["points"][0]["customdata"][0]
        return f"""[Link to Clicked Ad: {url}]({url})"""
    else:
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True, port=8060)
