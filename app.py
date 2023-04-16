from flask import Flask, render_template, request

# for plotting the data
import json
import pandas as pd
import plotly
import plotly.express as px

# for prediction
import requests
import datetime
import numpy as np
import itertools
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

configs = {'googlekey' :"AIzaSyB0MG9PRhLJcoVAeyxSbB3rtknrBvazWw4"}
model = joblib.load('static/xgboost.pkl')
model_columns = joblib.load('static/xgboost_columns.pkl')
accident_data = pd.read_csv('static/data/accidents_only.csv')

app = Flask(__name__, template_folder='templates', static_folder='static')

# variables for the filtering data viz portion
hour = [i for i in range(0,24)]
minute = [i for i in range(0,60)]
dow = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
day = [i for i in range(1,32)]
month = ['January', 'February', 'March', 'April', 
        'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December']
year = [i for i in range(2012,2023)]
col_severe = ['Fatal', 'Injury (Severe)', 'Injury (Other Visible)', 
            'Injury (Complaint of Pain)']

# get coordindates
def get_coord(address, country):
    parameters = {'address': address, 'country': country, 'key': configs['googlekey']}

    default_URL = "https://maps.googleapis.com/maps/api/geocode/json"
    default_point = requests.get(url=default_URL, params=parameters)
    default_json = default_point.json()

    if default_json['status'] == "OK":
        return (default_json['results'][0]['geometry']['location'])
default_location = get_coord("121 Spear Street", "USA")

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

    df = pd.read_csv('static/data/datasf_crashes_cleaned.csv')
    df = df[['tb_latitude', 'tb_longitude', 'primary_rd', 'hour', 
             'collision_time','accident_year','month','day_of_week', 
             'collision_severity', 'distance']]
    df = df[df['hour'].isin(hour_list) 
            & df['day_of_week'].isin(dow_list) 
            & df['month'].isin(month_list) 
            & df['accident_year'].isin(year_list) 
            & df['collision_severity'].isin(severe_list)]

    fig = px.scatter_mapbox(df, lat="tb_latitude", lon="tb_longitude", 
                            color="collision_severity", 
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            hover_name='primary_rd', 
                            hover_data=['collision_time','accident_year','month','day_of_week'],
                            zoom=11.5)
    fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    # Create a JSON representation of the graph
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # graphJSON = plotly.io.to_json(fig, engine='orjson')
    return graphJSON

#### PREDICTION PORTION
# default
# city hall 1 Dr Carlton B Goodlett Pl
# SF MOMA 151 3rd St, San Francisco

# test invalid address (some random string)

# show if outside of SF
# ischool 111 South Dr

### testing different accident routes in SF
# UCSF 2800 Turk Blvd, San Francisco, CA 94118
# Discord 444 De Haro St, San Francisco, CA 94107 
# Lyft 185 Berry St, San Francisco, CA 94107

### no accidents
# start 2-298 Dalewood Way, San Francisco
# end 290 Melrose Ave, San Francisco

# demonstrate hover information

@app.route('/predict')
def predict():
    return render_template('predict.html', 
                           min_len = len(minute), min_list = minute,
                           hour_len = len(hour), hour_list=hour,
                           dow_len = len(dow), dow_list=dow,
                           day_len = len(day), day_list = day,
                           month_len = len(month), month_list=month,
                           year_len = len(year), year_list=year)

@app.route('/route_predict/getRoute', methods=['GET'])
def route_predict(): 
    # get the two routes from the user here
    route_dict = request.args.to_dict()
    return verify_route(route_dict['addr1'],
                        route_dict['addr2'],
                        route_dict['user_datetime'])

def verify_route(origin, destination, user_dt):
    response = {'msg': None, 'address': None, 'real_route': None}
    parameters = {'origin': origin, 'destination': destination, 'key': configs['googlekey'], 'mode': 'walking'}
    
    # origin can be address i.e. 24 Sussex Drive Ottawa ON or lat lon, 41.43206, -81.38992 
    # this'll depend on how we're taking people's addresses 
    # we can add departure time as well and or arrival time
    api_URL = "https://maps.googleapis.com/maps/api/directions/json"
    route = requests.get(url=api_URL, params=parameters)
    route_json = route.json()

    if route_json['status'] == 'OK':
        if 'San Francisco' in route_json['routes'][0]['legs'][0]['start_address'] and 'San Francisco' in route_json['routes'][0]['legs'][0]['end_address']:
            did_accident_happen, graphJson = model_time(route_json, user_dt)
            if did_accident_happen:
                accident_count = json.loads(graphJson)

                response['msg'] = f"we have predicted {str(len(accident_count['data'][1]['lat']))} accidents!"
                response['address'] = graphJson
                response['real_route'] = True
            else:
                response['msg'] = 'we predict no accidents!\nYou\'re SaFe!'
                response['address'] = graphJson
                response['real_route'] = True
        else:
            response['msg'] = 'The start and/or the end address is not in San Francisco. Please make sure they are.'
            response['address'] = create_graph(default_location)
            response['real_route'] = False
    else:
        response['msg'] = "Invalid address... Is there a typo? Make sure this is a San Francisco address with no unit number!"
        response['address'] = create_graph(default_location)
        response['real_route'] = False

    response_json = json.dumps(response)
    return response_json

def create_graph(location):
    inputted_point = pd.DataFrame({
        'latitude' : [float(location['lat'])],
        'longitude': [float(location['lng'])]
    })

    fig = px.scatter_mapbox(inputted_point, 
                            lat="latitude", 
                            lon="longitude", 
                            mapbox_style='stamen-terrain',
                            zoom=12)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_traces(marker={'size': 15, 
                            'color': 'red',
                        })

    # Create a JSON representation of the graph
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

#Google Directions API Call 
def collect_coords(route_json):    
    waypoints = route_json['routes'][0]['legs']

    waypoint_lats = []
    waypoint_lons = []
    num_waypoints = 0

    # find cluster of interest from google api route
    for leg in waypoints:
        for step in leg['steps']:
            start_loc = step['start_location']
            waypoint_lats.append(start_loc['lat'])
            waypoint_lons.append(start_loc['lng'])
            num_waypoints += 1

    waypoint_lats = tuple(waypoint_lats)
    waypoint_lons = tuple(waypoint_lons)

    return waypoint_lats, waypoint_lons, num_waypoints

def calc_distance(accident_dataset, lats, longs, google_count_lat_long):
	# to do: we should probably use geopy instead here cuz it makes more sense to me 
    # load all cluster accident waypoints to check against proximity
    accident_point_counts = len(accident_dataset.index)

   # approximate radius of earth in km
    R = 6373.0
    new = accident_dataset._append([accident_dataset] * (google_count_lat_long - 1), ignore_index=True)  # repeat data frame (9746*waypoints_count) times
    lats_r = list(
        itertools.chain.from_iterable(itertools.repeat(x, accident_point_counts) for x in lats))  # repeat 9746 times
    longs_r = list(itertools.chain.from_iterable(itertools.repeat(x, accident_point_counts) for x in longs))

    # append
    new['lat2'] = np.radians(lats_r)
    new['long2'] = np.radians(longs_r)

    # cal radiun50m
    new['lat1'] = np.radians(new['tb_latitude'])
    new['long1'] = np.radians(new['tb_longitude'])
    new['dlon'] = new['long2'] - new['long1']
    new['dlat'] = new['lat2'] - new['lat1']

    new['a'] = np.sin(new['dlat'] / 2) ** 2 + np.cos(new['lat1']) * np.cos(new['lat2']) * np.sin(new['dlon'] / 2) ** 2
    new['distance'] = R * (2 * np.arctan2(np.sqrt(new['a']), np.sqrt(1 - new['a'])))

    return new

def model_pred(lats, longs, new_df):
    new_df = new_df.astype({'school_within_300':'category',
                'bus_stop_within_300':'category',
                'speedlimit':'category',
                'tb_latitude':'float',
                'tb_longitude':'float',
                'day_of_year':'category',
                'accident_year':'category',
                'distance':'category',
                'weather_1':'category',
                'collision_severity':'category',
                'type_of_collision':'category',
                'road_surface':'category',
                'road_cond_1':'category',
                'lighting':'category',
                'number_killed':'int',
                'number_injured':'int',
                'party1_type':'category',
                'party1_dir_of_travel':'category',
                'party1_move_pre_acc':'category',
                'party2_type':'category',
                'party2_dir_of_travel':'category',
                'party2_move_pre_acc':'category',
                'Neighborhoods':'category',
                'Current Police Districts':'category',
                'Current Supervisor Districts':'category',
                'Analysis Neighborhoods':'category',
                'hour':'category',
                'tavg':'category',
                'tmin':'category',
                'tmax':'category',
                'prcp':'category',
                'snow':'category',
                'wdir':'category',
                'wspd':'category',
                'pres':'category',
                'CLUSTER_ID':'category'})
    # do prediction for current datetime for all
    prob = pd.DataFrame(model.predict_proba(new_df), columns=['No', 'probability'])
    prob.set_index(new_df[['tb_latitude','tb_longitude']].index, inplace=True)
    prob.drop(columns='No',axis=1, inplace=True)
    # #merge with long lat
    output = prob.merge(new_df[['tb_latitude', 'tb_longitude']], how='outer',left_index=True,right_index=True)

    # #drop duplicates of same lat long (multiple accidents)
    output["tb_latitude"] = round(output["tb_latitude"], 5)
    output["tb_longitude"] = round(output["tb_longitude"], 5)
    output = output.drop_duplicates(subset=['tb_latitude', 'tb_longitude'], keep="last")

    # to json
    processed_results = []
    for index, row in output.iterrows():
        lat = float(row['tb_latitude'])
        long = float(row['tb_longitude'])
        prob = float(row['probability'])

        result = {'lat': lat, 'lng': long, 'probability': prob}
        processed_results.append(result)

    return model_graph(lats, longs, processed_results)


def model_graph(lats, longs, accident_output):
    test_df = pd.DataFrame(accident_output)

    fig = go.Figure(go.Scattermapbox(
        mode = "markers+lines",
        lon = longs,
        lat = lats,
        marker = {'size': 10},
        name="Route Waypoint"
    ))

    fig.add_trace(go.Scattermapbox(
        mode = "markers",
        lon = test_df.lng,
        lat = test_df.lat,
        text=test_df.probability,
        marker = {'size': 15},
        name="Potential Accident"
    ))

    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': longs[0], 'lat': lats[0]},
            'style': "stamen-terrain",
            'zoom': 14})

    fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
    
    # Create a JSON representation of the graph
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

#### PREDICTION PORTION
# test 2800 Turk Blvd, San Francisco, CA 94118
# test 444 De Haro St, San Francisco, CA 94107 (discord)
# test 185 Berry St, San Francisco, CA 94107 (lyft)
# test 1515 3rd St, San Francisco, CA 94158 (uber)

### no accidents
# start 2-298 Dalewood Way, San Francisco
# end 290 Melrose Ave, San Francisco

# MAIN MODEL PREDICTION
def model_time(route_json, tm):
    #parse time
    datetime_object = datetime.strptime(str(tm), '%Y/%b/%d %H:%M:%S')

    # get route planning
    lats, longs, google_count_lat_long = collect_coords(route_json)

    #calculate distance between past accident points and route
    dist = calc_distance(accident_data, lats, longs, google_count_lat_long)

    # filter for past accident points with distance <50m - route cluster
    dat = dist[dist['distance'] < 0.050]
    #if no cluster, exit
    if len(dat) == 0:
        return False, model_graph(lats=lats, longs=longs, accident_output={"lat": [], "lng": [], 'probability': []})
    else:
        # changing hour
        dat.loc[:,'hour'] = datetime_object.hour

        # changing day of year
        dat.loc[:,'day_of_year'] = datetime_object.timetuple().tm_yday

        # changing year
        year_col_list = ['2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']
        dat.loc[:,'accident_year'] = datetime_object.year

        final_df = dat
        #run model predicitions
        final_df.drop(columns = ['a', 'lat2', 'Unnamed: 0', 'long2', 'long1', 'dlat', 'lat1', 'dlon'], axis=1, inplace=True)
        final_df.drop(columns=['accident'], axis=1, inplace=True)

        return True, model_pred(lats, longs, final_df)
