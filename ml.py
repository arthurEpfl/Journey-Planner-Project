import random
import pickle
import pandas as pd
import utils


def getFakeDelay():
    return random.randint(0, 20)


def loadModel():
    with open(utils.model_path, 'rb') as f:
        clf = pickle.load(f)
    return clf


def getInput(bpuic, stop_lon, stop_lat, avg_delay, stddev_delay, temp, max_precip_hrly, ankunftszeit):
    data = {
        'bpuic': [bpuic],
        'stop_lon': [stop_lon],
        'stop_lat': [stop_lat],
        'avg_delay': [avg_delay],
        'stddev_delay': [stddev_delay],
        'temp': [temp],
        'max_precip_hrly': [max_precip_hrly],
        'ankunftszeit': [ankunftszeit]
    }
    
    return pd.DataFrame(data)

def getDelay(clf, bpuic, stop_lon, stop_lat, avg_delay, stddev_delay, temp, max_precip_hrly, ankunftszeit):
    
    input_data = getInput(
        bpuic = bpuic,
        stop_lon = stop_lon,
        stop_lat = stop_lat,
        avg_delay = avg_delay,
        stddev_delay = stddev_delay,
        temp = temp,
        max_precip_hrly = max_precip_hrly,
        ankunftszeit = ankunftszeit
    )
    
    # print(input_data.head())
    
    y_pred = clf.predict(input_data)
    return y_pred[0]
