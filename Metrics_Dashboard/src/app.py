import pandas as pd
import pathlib
import joblib
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input,Output
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# load data
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
test_data = pd.read_csv(DATA_PATH.joinpath("test_data.csv"))

# split data
target = "Exited"
X = test_data.drop(columns=["RowNumber", "Exited"])
y = test_data[target]

# load model
MODEL_PATH = PATH.joinpath("model").resolve()
model = joblib.load(MODEL_PATH.joinpath("model.pkl"))

# predict probabilities
prediction_probabilites = model.predict_proba(X)[:,1]

# start dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0"
        }
    ]
)

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# app layout
thresholds = dcc.Slider(0, 1, 0.1, value=0.5, id="threshold", className="mb-4") # probability thresholds

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.H1("Churn Model Performance Metrics", className="text-center mb-4"),
            width=12)]),

    dbc.Row([
        dbc.Col([
            html.B("Probability"),
            thresholds
            ], width=12)
    ]),

    dbc.Row([
        dbc.Col([dcc.Graph(id="matrix")], lg=5, md=5, sm=12),

        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div(id="metrics"),
                ], lg=6, md=6, sm=12),
                dbc.Col([
                    html.H4("Costs"),
                    html.Label("Retention:", className="me-1"),
                    dcc.Input(value=50, type="number", id="retention-cost", style={"width": "80px"}, className="mb-1"),
                    html.Br(),
                    html.Label("Acqusition:", className="me-1"),
                    dcc.Input(value=200, type="number", id="aquisition-cost", style={"width": "80px"}, className="mb-1")
                ], lg=6, md=6, sm=12)
            ], align="top"),
            
            dbc.Row(id="costs", align="top")
            
        ], lg=5, md=5, sm=12, align="center")

    ], align="center")

], fluid=True)

@app.callback(Output("matrix", "figure"), Input("threshold", "value"))
def plot_matrix(threshold):
    predictions = (prediction_probabilites >= threshold).astype(int)
    matrix = confusion_matrix(y, predictions)

    fig = px.imshow(
        matrix,
        labels=dict(x="Predicted Values", y="True Values"),
        x = ["False", "True"],
        y = ["False", "True"],
        text_auto=True,
        aspect="auto")
    fig.update_xaxes(side="top")
    fig.update_traces(dict(showscale=False, coloraxis=None))

    return fig

@app.callback(
    [
        Output("metrics", "children"),
        Output("costs", "children")
    ],
    [
        Input("threshold", "value"),
        Input("retention-cost", "value"),
        Input("aquisition-cost", "value")
    ])
def score(threshold, crc, cac):
    """
    crc: customer retention cost
    cac: customer aquisition cost
    """
    predictions = (prediction_probabilites >= threshold).astype(int)
    matrix = confusion_matrix(y, predictions)

    accuracy = accuracy_score(y, predictions).round(2)
    precision = precision_score(y, predictions).round(2)
    recall = recall_score(y, predictions).round(2)

    scores = [
        html.H4("Metrics"),
        html.P(f"Accuracy Score: {accuracy}", className="d-block"),
        html.P(f"Precision Score: {precision}", className="d-block"),
        html.P(f"Recall Score: {recall}", className="d-block"),
    ]

    retention_cost = matrix.sum(axis=0)[1] * crc # crc cost with model
    aquisition_cost = matrix[1][0] * cac # cac with model
    retention_waste = matrix[0][1] * crc # amounted wasted on customer retention due to model precision
    # amount wasted on customer aqusition without model
    aquisition_waste = (matrix.sum(axis=1)[1] * cac) - aquisition_cost

    costs = [
        dbc.Col([
            html.H4("Amount Spent"),
            html.P(f"Retention: €{retention_cost:,.2f}", className="d-block"),
            html.P(f"Acquisition: €{aquisition_cost:,.2f}", className="d-block"),
            html.P(f"Total: €{retention_cost + aquisition_cost}")
        ], sm=12, md=4, lg=4),
        dbc.Col([
            html.H4("Amount Wasted"),
            html.P(f"Retention With Model: €{retention_waste:,.2f}"),
            html.P(f"Acqusition Without Model: €{aquisition_waste:,.2f}")
        ], sm=12, md=4, lg=4),
        dbc.Col([
            html.H4("Amount Saved"),
            html.P(
                    f"""
                    The model saves €{aquisition_waste-retention_cost:,.2f}
                    out of €{matrix.sum(axis=1)[1] * cac:,.2f} likely to be 
                    spent on customer acquisition.
                    """
                 )
        ], sm=12, md=4, lg=4)
    ]

    return scores, costs

 
if __name__=="__main__":
    app.run(debug=False)