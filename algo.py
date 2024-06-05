import pandas as pd
import utils
import ml
import numpy as np
import networkx as nx
from tqdm import tqdm
import heapq
import datetime



def getPath(pathData, stops_data):
    
    pathTime = []
    pathCoord = []
    all_segments = []
    meanOfTransport = []
    pathChangeTimes = []
    stopIds = []
    arrivalTimes = []
    
    now = datetime.datetime.now().date()
    current_date_midnight = datetime.datetime.combine(now, datetime.datetime.min.time())
    timestamp = current_date_midnight.timestamp()    
    
    for options in pathData:
        pathTime.append(options[1] // 60)
        
        path = options[0]
        
        indivPathCoord = []
        for stops in path:
            stop_id = stops[0]
            coord = list(stops_data.loc[stops_data['stop_id'] == str(stop_id), ['stop_lon', 'stop_lat']].values[0])
            indivPathCoord.append(coord)
        
        pathCoord.append(indivPathCoord) 
                
        zipped_stops = [[path[i][1], path[i+1][1]] for i in range(len(path)-1)]
        all_segments.append(zipped_stops)
        
        transports = [path[i+1][2] for i in range(len(path)-1)]
        meanOfTransport.append(transports)
        
        stops = [path[i+1][0] for i in range(len(path)-1)]
        stopIds.append(stops)
        
        arrivals = [int(timestamp + path[i+1][4]) for i in range(len(path)-1)]
        arrivalTimes.append(arrivals)
        
        changeTime = []
        for i in range(len(path)-1):
            arrival = path[i+1][4]
            departure = path[i][3]
            if type(arrival) != float:
                arrival = 0
            if type(departure) != float:
                departure = 0
            
            changeTime.append(((arrival - departure) % 3600) // 60)
        pathChangeTimes.append(changeTime)
           
    max_length = max(len(sublist) for sublist in pathCoord)
    padded_list = []
    for sublist in pathCoord:
        padded_sublist = sublist + [sublist[-1]] * (max_length - len(sublist))
        padded_list.append(padded_sublist)
        
                  
    df = pd.DataFrame({
        'path': padded_list,
        'color': ["#000CFF", "#00A2FF"],
        'stop_name': ["Optimal Path", "Alternative"],
        'timeChanges' : pathChangeTimes,
        'segments' : all_segments,
        'meanOfTransport' : meanOfTransport,
        "stop_ids" : stopIds,
        "arrival_times" : arrivalTimes,
        "total_time" : pathTime
    })   
    
        
    return df
    
    
def loadDataAndGraph():
    stop_pairs_within_500m = pd.read_pickle(utils.stop_pairs_within_500m_path)
    final_df = pd.read_pickle(utils.final_data_path)
    
    G = nx.DiGraph()

    # Add nodes
    for index, row in tqdm(final_df.iterrows(), total=final_df.shape[0], desc='Adding nodes'):
        G.add_node(row['arrival_stop_id'], 
                stop_name=row['stop_name'],
                stop_lat=row['stop_lat'], 
                stop_lon=row['stop_lon'], 
                parent_station=row['parent_station'])
        
    # Pre-sort and group the dataframe by 'trip_id' with sorted 'stop_sequence'
    sorted_final_df = final_df.sort_values(by=['trip_id', 'stop_sequence'])
    grouped_by_trip = sorted_final_df.groupby('trip_id')

    # Add edges with route description, processing grouped trips
    for trip_id, trip_data in tqdm(grouped_by_trip, desc='Processing trips'):
        previous_row = None
        for idx, row in trip_data.iterrows():
            if previous_row is not None:
                weight = (row['arrival_time_seconds'] - previous_row['departure_time_seconds'])

                edge_data = G.get_edge_data(previous_row['stop_id'], row['stop_id'], default=None)
                if edge_data is None or (
                    edge_data['departure_time_seconds'] != previous_row['departure_time_seconds'] or 
                    edge_data['arrival_time_seconds'] != row['arrival_time_seconds']):
                    
                    G.add_edge(previous_row['stop_id'], row['stop_id'], 
                            departure_time_seconds=previous_row['departure_time_seconds'],
                            arrival_time_seconds=row['arrival_time_seconds'], 
                            weight=weight,
                            route_id = row['route_id'],
                            route_desc=row['route_desc'])
            previous_row = row


    # Add walking edges
    # We create one walking edge for each existing transit connection
    # This ensures that for every direct transit connection between two stops, 
    # there's an alternative walking connection available at the same departure time
    for _, row in tqdm(stop_pairs_within_500m.iterrows(), total=stop_pairs_within_500m.shape[0], desc='Adding walking edges'):
        stop_id1 = row['stop_id1']
        stop_id2 = row['stop_id2']
        
        # Ensure both stops are in the graph
        if G.has_node(stop_id1) and G.has_node(stop_id2):
            # Get all existing transit connections between the two stops
            existing_edges = list(G.edges(stop_id1, data=True))
            
            for _, _, edge_data in existing_edges:
                departure_time_seconds = edge_data['departure_time_seconds']
                transfer_time_seconds = row['transfer_time_seconds']
                arrival_time_seconds = departure_time_seconds + transfer_time_seconds

                # Check if a walking edge already exists with the same departure time
                # This prevents duplicate walking edges for the same departure time
                walking_edge_data = G.get_edge_data(stop_id1, stop_id2, default={})
                if walking_edge_data.get('departure_time_seconds') != departure_time_seconds:
                    # Add the walking edge with necessary attributes
                    G.add_edge(stop_id1, stop_id2,
                            departure_time_seconds=departure_time_seconds,
                            arrival_time_seconds=arrival_time_seconds,
                            weight=transfer_time_seconds,
                            route_id="WALKING",
                            route_desc="walk")
                    
    return G
                    
                    
def find_fastest_routes(G, stop_id_a, stop_id_b, latest_arrival_time, k):
    # Priority queue to store (negative of departure time from stop_id_a, path as detailed list)
    pq = []
    # Dictionary to store the best known arrival times at each stop
    best_arrival_times = {node: float('inf') for node in G.nodes}

    # Initialize the queue with the departure connections from stop_id_a
    for neighbor in G.neighbors(stop_id_a):
        edge_data = G.get_edge_data(stop_id_a, neighbor)
        departure_time = edge_data['arrival_time_seconds'] - edge_data['weight']
        if edge_data['arrival_time_seconds'] <= latest_arrival_time:
            initial_path_segment = [
                (stop_id_a, G.nodes[stop_id_a]['stop_name'], None, None, None),  # No route info for initial stop
                (neighbor, G.nodes[neighbor]['stop_name'], edge_data['route_desc'], departure_time, edge_data['arrival_time_seconds'])
            ]
            heapq.heappush(pq, (-departure_time, initial_path_segment))
            best_arrival_times[neighbor] = edge_data['arrival_time_seconds']

    routes = []

    # Process the priority queue
    while pq and len(routes) < k:
        neg_dep_time, path = heapq.heappop(pq)
        current_stop_info = path[-1]
        current_stop = current_stop_info[0]
        current_arrival_time = current_stop_info[4]  # Arrival time at current stop

        # If we've reached the destination and the time condition is satisfied
        if current_stop == stop_id_b and current_arrival_time <= latest_arrival_time:
            routes.append((path, -neg_dep_time))
            continue

        # Explore neighbors
        last_edge_was_walking = (len(path) > 1 and G.get_edge_data(path[-2][0], current_stop)['route_id'] == "WALKING")
        for neighbor in G.neighbors(current_stop):
            edge_data = G.get_edge_data(current_stop, neighbor)
            new_arrival_time = edge_data['arrival_time_seconds']
            departure_time = new_arrival_time - edge_data['weight']
            
            # Check if consecutive walking edges are being added
            if last_edge_was_walking and edge_data['route_id'] == "WALKING":
                continue  # Skip adding this edge

            # Continue only if the new arrival time is within limits and improves the best known time
            if new_arrival_time <= latest_arrival_time and new_arrival_time < best_arrival_times[neighbor]:
                best_arrival_times[neighbor] = new_arrival_time
                new_path_segment = (neighbor, G.nodes[neighbor]['stop_name'], edge_data['route_desc'], departure_time, new_arrival_time)
                new_path = path + [new_path_segment]
                heapq.heappush(pq, (-departure_time, new_path))

    # Sort the found routes by the latest departure time
    routes.sort(key=lambda x: x[1], reverse=True)
    return [(route, dep_time) for route, dep_time in routes[:k]]
    
    
    
def getDelayProbability(path_data, stops_data, clf, temperature, precip_hrly, arrival_time):
    probabilities = []
    probabilitesList = []
    
    
    for times, stops in zip(path_data['timeChanges'] , path_data['stop_ids']):
        
        predictions = []
        for i in range(len(times)):
        
            predictedDelay = ml.getDelay(
                clf, 
                bpuic = stops[i].split(":")[0],
                stop_lon = stops_data.loc[stops_data['stop_id'] == stops[i], 'stop_lon'].values[0],
                stop_lat = stops_data.loc[stops_data['stop_id'] == stops[i], 'stop_lat'].values[0],
                avg_delay = utils.avg_delay,
                stddev_delay = utils.std_dev,
                temp = temperature,
                max_precip_hrly = precip_hrly,
                ankunftszeit = arrival_time
                )
            
            #adjust times
            if times[i] == 0:
                times[i] = 10
            timeForChange = times[i] - predictedDelay

            if timeForChange < 0:
                timeForChange = 5
            predictions.append(utils.probability_of_success.get(timeForChange, 1))
            
        probabilitesList.append(predictions)
        
        probabilities.append(np.prod(predictions))
    
    return probabilities, probabilitesList