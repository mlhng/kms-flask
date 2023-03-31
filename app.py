from flask import Flask, render_template, request

# for plotting the data
import json
import pandas as pd
import plotly
import plotly.express as px
import geopy as gpy

app = Flask(__name__, template_folder='templates', static_folder='static')

# variables for the filtering data viz portion
hour = [i for i in range(0,24)]
dow = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
month = ['January', 'February', 'March', 'April', 
        'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December']
year = [i for i in range(2012,2024)]
col_severe = ['Fatal', 'Injury (Severe)', 'Injury (Other Visible)', 
            'Injury (Complaint of Pain)']

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

#### PREDICTION PORTION
@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/ready_inputs/getAddress', methods=['GET'])
def verify_address():
    return ready_input(request.args.get('address'))

# test  2800 Turk Blvd, San Francisco, CA 94118
# test 444 De Haro St, San Francisco, CA 94107 (discord)
# test 185 Berry St, San Francisco, CA 94107 (lyft)
# test 1515 3rd St, San Francisco, CA 94158 (uber)
def ready_input(street):
    response = {'msg': None, 'address': None, 'coord': None}

    # attempt to try to convert the inputted address into a coordinate
    try: 
        geolocator = gpy.Nominatim(user_agent="keep_me_safe")
        location = geolocator.geocode(street)

        if "San Francisco" not in location.address :
            response['msg'] = f'{location.address} is not in San Francisco. Try Again.'
            response['address'] = create_graph(geolocator.geocode("1 Dr Carlton B Goodlett Pl"))
            response['coord'] = None
        else:
            location_msg = str(location.address)
            response['msg'] = location_msg
            response['address'] = create_graph(location)
            response['coord'] = str((location.latitude, location.longitude))
    # if it fails, output a message for the user saying it's not correct
    except:
        location = geolocator.geocode("1 Dr Carlton B Goodlett Pl")

        response['msg'] = "Invalid address... Is there a typo? Make sure this is a San Francisco address with no unit number!"
        response['address'] = create_graph(geolocator.geocode("1 Dr Carlton B Goodlett Pl"))
        response['coord'] = None

    response_json = json.dumps(response)
    return response_json
    
def create_graph(location):
    inputted_point = pd.DataFrame({
        'latitude' : [float(location.latitude)],
        'longitude': [float(location.longitude)]
    })

    fig = px.scatter_mapbox(inputted_point, 
                            lat="latitude", 
                            lon="longitude", 
                            mapbox_style='carto-positron',
                            zoom=11.75, height=600, width=800)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_traces(marker={'size': 15, 
                                'color': 'red',
                            })

    # Create a JSON representation of the graph
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
