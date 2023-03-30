from flask import Flask, render_template, request

# for plotting the data
import json
import pandas as pd
import plotly
import plotly.express as px

app = Flask(__name__, template_folder='templates', static_folder='static')

# variables for the filtering data viz portion
hour = [i for i in range(0,24)]
dow = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
month = ['January', 'February', 'March', 'April', 
        'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December']
year = [i for i in range(2012,2024)]
col_severe = ['Injury (Complaint of Pain)', 'Injury (Other Visible)', 
            'Injury (Severe)', 'Fatal']

# MAIN PAGE FUNCTIONS
@app.route('/')
def index():
    return render_template('index.html')

# VISUALIZATION FUNCTIONS 
@app.route('/visualize')
def visualize():
    return render_template('visualize.html', 
                           hour_len = len(hour), hour_list=hour,
                           dow_len = len(dow), dow_list=dow,
                           month_len = len(month), month_list=month,
                           year_len = len(year), year_list=year,
                           severe_len = len(col_severe), severe_list=col_severe,)

# ACTUAL VIZ CODE
@app.route('/cb_viz/getAccidents', methods=['GET'])
def cb_viz(): 
    graph_filter = request.args.to_dict()
    return graph(graph_filter['hour'],
                graph_filter['dow'],
                graph_filter['month'],
                graph_filter['year'],
                graph_filter['severe'])

def graph(hour, dow, month, year, severe):
    hour_list = [int(x) for x in hour.split(",")]
    dow_list = [x for x in dow.split(",")]
    month_list = [x for x in month.split(",")]
    year_list = [int(x) for x in year.split(",")]
    severe_list = [x for x in severe.split(",")]

    df = pd.read_csv('static/datasf_crashes_cleaned.csv')
    df = df[['tb_latitude', 'tb_longitude', 'primary_rd', 'hour', 
             'collision_time','accident_year','month','day_of_week', 
             'collision_severity', 'distance']]
    df = df[df['hour'].isin(hour_list) 
            & df['day_of_week'].isin(dow_list) 
            & df['month'].isin(month_list) 
            & df['accident_year'].isin(year_list) 
            & df['collision_severity'].isin(severe_list)]

    fig = px.scatter_mapbox(df, lat="tb_latitude", lon="tb_longitude", color="collision_severity", 
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            hover_name='primary_rd', hover_data=['collision_time','accident_year','month','day_of_week'],
                            zoom=11.15)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Create a JSON representation of the graph
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # graphJSON = plotly.io.to_json(fig, engine='orjson')
    return graphJSON


# PREDICTION FUNCTIONS
@app.route('/predict')
def predict():
    return render_template('predict.html')

