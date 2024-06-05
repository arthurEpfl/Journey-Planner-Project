import streamlit as st
import pandas as pd
import datetime
import pydeck as pdk
import utils
import algo
import ml


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def colorStart(start):
    stops_data.loc[stops_data['stop_name'] == start, 'color'] = "#4FFF44"
    update_colors()
    
    
def colorStop(stop):
    stops_data.loc[stops_data['stop_name'] == stop, 'color'] = "#FF4B24"
    update_colors()
    
    
def init_data():
    stops_data = pd.read_csv(utils.stops_regions_path)

    stops_data = stops_data.astype({"stop_id": 'string'})
    stops_data = stops_data.astype({"stop_name": 'string'})
    stops_data = stops_data.astype({"name": 'string'})

    stops_data["color"] = "#656EF9"
    
    stops_data["color_rgb"] = stops_data["color"].apply(hex_to_rgb)
    
    return stops_data


def update_colors():
    stops_data["color_rgb"] = stops_data["color"].apply(hex_to_rgb)
    
    
def getLatLon(stop):
    return list(stops_data.loc[stops_data['stop_name'] == stop, ['stop_lon', 'stop_lat']].values[0])    
    
def getMeanLatLon(datafram):
    mean_lat = datafram['stop_lat'].mean()
    mean_lon = datafram['stop_lon'].mean()
    return mean_lat, mean_lon 
    
    
def drawMap(lat = 47.010507, lon = 7.954816  , zoom = 7):
    
    view_state = pdk.ViewState(
        latitude=lat, 
        longitude=lon,
        zoom=zoom
    )
    
    # ScatterplotLayer for dots at each stop
    dot_layer = pdk.Layer(
        type='ScatterplotLayer',
        data=stops_data,
        pickable=True,
        get_position=['stop_lon', 'stop_lat'],
        get_radius=1,  # Radius in meters
        get_color='color_rgb',  
        radius_scale=1,
        radius_min_pixels=5,
        radius_max_pixels=10
    )

    path_layer = None
    if (start_choice != None and arrival_choice != None):
        
        start_stop_id = stops_data.loc[stops_data['stop_name'] == start_choice, 'stop_id'].values[0]
        arrival_stop_id = stops_data.loc[stops_data['stop_name'] == arrival_choice, 'stop_id'].values[0]
        
        routes = algo.find_fastest_routes(
            G = st.session_state['graph'], 
            stop_id_a = str(start_stop_id), 
            stop_id_b = str(arrival_stop_id), 
            latest_arrival_time = st.session_state['time_arrival'],
            k = 2
        )
    
        st.session_state['path_data'] = algo.getPath(routes, stops_data)

        st.session_state['path_data']['color'] = st.session_state['path_data']['color'].apply(hex_to_rgb)
                        
        path_layer = pdk.Layer(
            type='PathLayer',
            data=st.session_state['path_data'],
            pickable=True,
            get_color='color',
            width_scale=1,
            width_min_pixels=3,
            get_path='path',
            get_width=5
        )
        
    r = pdk.Deck(layers=[dot_layer, path_layer], initial_view_state=view_state, tooltip={'text': '{stop_name}'}, map_style='mapbox://styles/mapbox/streets-v11')

    st.pydeck_chart(r)
   
weather = utils.weather    

stops_data = init_data()

st.title('Robust Journey Planner')

if "model" not in st.session_state:
    st.session_state['model'] = ml.loadModel()
    
if "graph" not in st.session_state:
    st.session_state['graph'] = algo.loadDataAndGraph()
    
    
st.sidebar.write("## Choose your Trip information")

city_choice = st.sidebar.selectbox('Choose your city', stops_data['name'].unique(), placeholder="Select city...", index=None)
if city_choice != None:
    city_choice_id = stops_data.loc[stops_data['name'] == city_choice, 'objectid'].values[0]
    stops_data = stops_data.loc[stops_data['objectid'] == city_choice_id]

start_choice = st.sidebar.selectbox('Choose your departure stop', stops_data['stop_name'].unique(), placeholder="Select departure stop...", index=None,)
colorStart(start_choice)

arrival_choice = st.sidebar.selectbox('Choose your arrival stop', stops_data['stop_name'].unique(), placeholder="Select arrival stop...", index=None,)
colorStop(arrival_choice)

today = datetime.datetime.now()

time_arrival = st.sidebar.time_input('Latest Arrival Time', value=today + datetime.timedelta(hours=3))
st.session_state['time_arrival'] = int(datetime.datetime.combine(datetime.datetime.now().date(), time_arrival).timestamp())

weather = st.sidebar.radio('Whats the weather ?',weather)

temperature = st.sidebar.slider('Temperature in °C', -10, 30, 15)

if city_choice != None:
    lat, lon = getMeanLatLon(stops_data)
    zoom = 12
    drawMap(lat, lon, zoom)
else:
    drawMap()    

st.divider()

# if st.sidebar.button('Plan my trip', disabled=(start_choice == None or arrival_choice == None)):
if start_choice != None and arrival_choice != None:
    with st.spinner('Wait for it...'):
        st.write("Your trip from", start_choice, "to", arrival_choice, "is planned for", today, "and you will arrive latest at", time_arrival, "with a", weather, "weather.")
                
        probabilities, probabilityLsit = algo.getDelayProbability(
            st.session_state['path_data'],
            clf = st.session_state['model'], 
            temperature = temperature, 
            precip_hrly = utils.weather[weather],
            arrival_time=st.session_state['time_arrival'],
            stops_data=stops_data
            )   

        index = 0
        for name, segment, time_for_change, mean_of_transport, time in zip(st.session_state['path_data']['stop_name'], st.session_state['path_data']['segments'], st.session_state['path_data']['timeChanges'], st.session_state['path_data']['meanOfTransport'], st.session_state['path_data']['total_time']):
            st.divider()
            #PATH 1 
            st.write(f"The {name} path proposed is:")
            sum = 0
            for i in range(len(segment)):
                sum += time_for_change[i] % 7
                st.write(f"Travel from {segment[i][0]} to {segment[i][1]} with the following mean of transport :  {mean_of_transport[i]}. You have {time_for_change[i] % 7}min to change the probability of a successful change is {probabilityLsit[index][i]}")
            
            st.write(f"Total time of the path is {sum} min and the overall probability of success is {probabilities[index]}")
            index += 1

        

        
