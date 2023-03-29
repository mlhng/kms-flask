from flask import Flask, render_template, request

# for plotting the data
import json
import pandas as pd
import plotly
import plotly.express as px
import numpy as np
import yfinance as yf

app = Flask(__name__, template_folder='templates', static_folder='static')
app.debug = True
datasf = pd.read_csv('data\datasf_crashes_cleaned.csv')

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/visualize')
# def visualize():
#     return render_template('visualize.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

@app.route('/callback/<endpoint>')
def cb(endpoint):   
    if endpoint == "getStock":
        return gm(request.args.get('data'),
                  request.args.get('period'),
                  request.args.get('interval'))
    elif endpoint == "getInfo":
        stock = request.args.get('data')
        st = yf.Ticker(stock)
        return json.dumps(st.info)
    else:
        return "Bad endpoint", 400

# Return the JSON data for the Plotly graph
def gm(stock,period, interval):
    st = yf.Ticker(stock)
  
    # Create a line graph
    df = st.history(period=(period), interval=interval)
    df=df.reset_index()
    df.columns = ['Date-Time']+list(df.columns[1:])
    max = (df['Open'].max())
    min = (df['Open'].min())
    range = max - min
    margin = range * 0.05
    max = max + margin
    min = min - margin
    fig = px.area(df, x='Date-Time', y="Open",
        hover_data=("Open","Close","Volume"), 
        range_y=(min,max), template="seaborn" )


    # Create a JSON representation of the graph
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


# ACTUAL VIZ CODE
@app.route('/cb_viz/<endpoint>')
def cb_viz(endpoint):   
    if endpoint == "getAccidents":
        return graph(request.args.get('hour'),
                  request.args.get('dow'),
                  request.args.get('month'),
                  request.args.get('year'),
                  request.args.get('col_severe'))
    else:
        return "Bad endpoint", 400

def graph(hour, dow, month, year, col_severe):
    df = datasf[['tb_latitude', 'tb_longitude', 'primary_rd', 'hour', 
             'collision_time','accident_year','month','day_of_week', 
             'collision_severity', 'distance']]
    df = df[df['hour'].isin(hour)]
    df = df[df['day_of_week'].isin(dow)]
    df = df[df['month'].isin(month)]
    df = df[df['accident_year'].isin(year)]
    df = df[df['collision_severity'].isin(col_severe)]

    fig = px.scatter_mapbox(df, lat="tb_latitude", lon="tb_longitude", color="collision_severity", 
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            hover_name='primary_rd', hover_data=['collision_time','accident_year','month','day_of_week'],
                            zoom=11.15)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Create a JSON representation of the graph
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON