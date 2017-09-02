import numpy as np
import pandas as pd
import pandana as pdna
import osmnet as osm
import urbanaccess as ua
import pandana.network
from simpledbf import Dbf5
import shapefile
from json import dumps
import Queue
import time
import os

#input
#GTFS_PATH = r'./input/gtfs_bus'
GTFS_PATH = '/Volumes/WD-zqh/bus_trajectory/input/gtfs_bus/'
DBF_PATH = "./input/tl_2010_06037_tabblock10/tl_2010_06037_tabblock10.dbf"
JOB_PATH = "./input/ca_od_main_JT00_2010.csv"
SHAPEFILE_PATH = './input/tl_2010_06037_tabblock10/tl_2010_06037_tabblock10'
GROUP_SHAPE_PATH = "./input/tl_2010_06037_bg10/tl_2010_06037_bg10"

time_range = ["06:00:00","09:00:00"]
bounding_box = (-118.85, 33.67, -117.67, 34.32) # big
#bounding_box = (-118.30, 34.0, -118.25, 34.06) # small

#output
OUTPUT_PATH = "./output/"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# step 1, build network
NET_PATH = "./output/net.h5"
gtfs_df = ua.gtfs.load.gtfsfeed_to_df(GTFS_PATH,
                                      bbox= bounding_box,
                                      validation = True,
                                      remove_stops_outsidebbox=True)

ua_net = ua.gtfs.network.create_transit_net(gtfs_df, 
                                   day='monday', 
                                   timerange= time_range, 
                                   overwrite_existing_stop_times_int=False, 
                                   use_existing_stop_times_int=True, 
                                   save_processed_gtfs=False, 
                                   save_dir='data', 
                                   save_filename=None)
ua.gtfs.headways.headways(gtfs_df,time_range)

osm_data = ua.osm.load.ua_network_from_bbox(bbox= bounding_box,
                                           network_type='walk', 
                                           timeout=180, 
                                           memory=None, 
                                           max_query_area_size=2500000000L, 
                                           remove_lcn=True)

all_net = ua.osm.network.create_osm_net(osm_edges=osm_data[1], osm_nodes=osm_data[0], travel_speed_mph=3, network_type='walk')

all_net.transit_nodes = ua_net.transit_nodes
all_net.transit_edges = ua_net.transit_edges
ua_integrated_net = ua.network.integrate_network(all_net, 
                             headways=True, 
                             urbanaccess_gtfsfeeds_df=gtfs_df, 
                             headway_statistic='mean')

#ua.network.save_network(urbanaccess_network=ua_net, filename='out/net_ua.h5', overwrite_key = True);
ua = None
imp = pd.DataFrame(ua_integrated_net.net_edges['weight'])
net = pdna.network.Network(ua_integrated_net.net_nodes.x, ua_integrated_net.net_nodes.y,ua_integrated_net.net_edges.from_int, ua_integrated_net.net_edges.to_int, imp, False)
net.save_hdf5(NET_PATH)
ua_integrated_net = None

# step 2, compute closest nodes
print "compute closest nodes ..."
start = time.time()
CLOSEST_NODES_FILE = "./output/block_closest_node.csv"
dbf = Dbf5(DBF_PATH);
data_db = dbf.to_dataframe();
census_block_db = data_db[['GEOID10', 'INTPTLAT10', 'INTPTLON10']];
census_block_db['INTPTLAT10'] = census_block_db.apply(lambda x : float(x['INTPTLAT10']), axis = 1)
census_block_db['INTPTLON10'] = census_block_db.apply(lambda x : float(x['INTPTLON10']), axis = 1)
census_block_db['node_id'] = net.get_node_ids(census_block_db['INTPTLON10'],census_block_db['INTPTLAT10'])
census_block_db['net_lat'] = census_block_db.apply(lambda x : net.nodes_df.loc[x['node_id'], 'y'], axis = 1)
census_block_db['net_lon'] = census_block_db.apply(lambda x : net.nodes_df.loc[x['node_id'], 'x'], axis = 1)
census_block_db = census_block_db.set_index('GEOID10')

def outsideBox(geoid):
    lat = census_block_db.loc[geoid,'INTPTLAT10']
    lon = census_block_db.loc[geoid,'INTPTLON10']
    if lat < bounding_box[1] or lat > bounding_box[3] or lon < bounding_box[0] or lon > bounding_box[2]:
        return 0
    return 1

census_block_db['in_box'] = census_block_db.apply(lambda x : outsideBox(x.name), axis = 1)
census_block_sub = census_block_db[census_block_db.in_box == 1]
census_block_sub.to_csv(CLOSEST_NODES_FILE);
end = time.time()
print "compute closest nodes ", end - start, 's'

# step 3, economic compute
print "compute jobs ..."
start = time.time()
census_block_db = pd.read_csv(CLOSEST_NODES_FILE);
economic_data = pd.read_csv(JOB_PATH)
economic_data = economic_data[['w_geocode', 'S000', 'SA01', 'SA02', 'SA03', 'SE01', 'SE02', 'SE03']]
economic_data = economic_data.groupby('w_geocode').sum()
economic_data.reset_index(inplace = True)

economic_pd = economic_data[['w_geocode','S000','SE01','SE02','SE03']].merge(census_block_db[['GEOID10','node_id','net_lat','net_lon']], left_on='w_geocode', right_on='GEOID10')
economic_pd = economic_pd[['w_geocode','S000','SE01','SE02', 'SE03', 'node_id']]

def aggregrate():
    name = 'block_jobs_all'
    net.set(economic_pd['node_id'], variable=economic_pd.S000.astype(float), name=name)
    Distance = [20,40,60]
    start = time.time()
    for distance in Distance:
        print distance, " minutes"
        jobs_num = net.aggregate(distance, type='sum', decay='flat', name=name)
        jobs_pd = jobs_num.to_frame()
        jobs_pd.reset_index(inplace = True)
        jobs_pd = jobs_pd.merge(census_block_db[['node_id','net_lat','net_lon']], left_on='id_int', right_on='node_id',how='right')
        jobs_pd.to_csv('./output/jobs_all_'+ str(distance) + '.csv')

aggregrate()
end = time.time()
print "compute jobs ", end - start, 's'

# step 4, reachable blocks
print "compute reachable blocks ..."
start = time.time()
edges = net.edges_df
net_dict = {}
for i in range(0, len(edges)):
    edge = edges.iloc[i]
    if net_dict.has_key(int(edge['from'])):
        net_dict.get(int(edge['from'])).append((int(edge['to']), edge['weight']))
    else:
        net_dict[int(edge['from'])] = [(int(edge['to']),edge['weight'])]

def bfs(start, distance):
    queue = Queue.PriorityQueue()
    visited = set()
    queue.put((0, start))
    #visited.add(start)
    while not queue.empty():
        cur = queue.get() # (dist, id)
        if cur[1] in visited:
            continue;
        visited.add(cur[1])
        next_nodes = net_dict.get(cur[1]) 
        if next_nodes is None:
            continue
        for node in next_nodes:  # (id, weight)
            dist = cur[0] + node[1]
            if dist > distance:
                continue
            queue.put((dist, node[0]))
    return visited

census_block_db = pd.read_csv(CLOSEST_NODES_FILE,dtype={'GEOID10':str})
id_dict = {}
for i in range(0, len(census_block_db)):
    row = census_block_db.iloc[i]
    node_id = row['node_id']
    geo_id = row['GEOID10']
    if id_dict.has_key(node_id):
        id_dict.get(node_id).append(geo_id)
    else:
        id_dict[node_id] = [geo_id]

distance = 20
NEAREST_PATH = './output/nearest' + str(distance) + '/'
if not os.path.exists(NEAREST_PATH):
    os.makedirs(NEAREST_PATH)

for i in range(0, len(census_block_db)):
    row = census_block_db.iloc[i]
    file_name = NEAREST_PATH + str(row['GEOID10']) + ".json"
    node_id = row['node_id']
    nearest = bfs(node_id, distance)
    n_list = []
    for n_id in nearest:
        if id_dict.has_key(n_id):
            n_list = n_list + id_dict[n_id]
    with open(file_name, "w") as geojson:
        geojson.write(dumps({"nearest": n_list}, indent=2) + "\n")
end = time.time()
print "compute reachable blocks ", end - start, 's'

# step 5, interactive data
print "compute interactive data ..."
def inBox(lat, lon):
    if lat < bounding_box[1] or lat > bounding_box[3] or lon < bounding_box[0] or lon > bounding_box[2]:
        return False
    return True

start = time.time()
def generateJobs():
    jobs_20_pd = pd.read_csv('./output/jobs_all_20.csv',dtype={'0':int})
    jobs_20_pd.rename(columns={'0':'job_20'}, inplace=True)
    jobs_20_pd = jobs_20_pd.drop_duplicates(subset='node_id',keep='last')
    jobs_40_pd = pd.read_csv('./output/jobs_all_40.csv', dtype={'0':int})
    jobs_40_pd.rename(columns={'0':'job_40'}, inplace=True)
    jobs_40_pd = jobs_40_pd.drop_duplicates(subset='node_id',keep='last')
    jobs_60_pd = pd.read_csv('./output/jobs_all_60.csv',dtype={'0':int})
    jobs_60_pd.rename(columns={'0':'job_60'}, inplace=True)
    jobs_60_pd = jobs_60_pd.drop_duplicates(subset='node_id',keep='last')
    jobs_pd = jobs_20_pd[['node_id', 'job_20']].merge(jobs_40_pd[['node_id','job_40']], on='node_id',how='inner')
    jobs_pd = jobs_pd.merge(jobs_60_pd[['node_id','job_60']], on='node_id')
    census_block_db = pd.read_csv(CLOSEST_NODES_FILE,dtype={'GEOID10':str})
    jobs_pd2 = jobs_pd.merge(census_block_db[['node_id','GEOID10','INTPTLAT10','INTPTLON10']], on='node_id',how='right')
    jobs_pd2.set_index('GEOID10', inplace=True)
    return jobs_pd2

jobs_pd2 = generateJobs()

def shape2json(fname, outfile="./output/blocks.json"):
    reader = shapefile.Reader(fname)
    fields = reader.fields[1:]
    field_names = [field[0] for field in fields]

    data = []
    for sr in reader.shapeRecords():
        atr = dict(zip(field_names, sr.record))
        geom = sr.shape.__geo_interface__
        lat = float(atr['INTPTLAT10'])
        lon = float(atr['INTPTLON10'])
        if inBox(lat, lon):
            jobs = jobs_pd2.loc[atr['GEOID10']]
            jobs_dict = {"id":atr['GEOID10'],"min20":int(jobs['job_20']),"min40":int(jobs['job_40']), "min60":int(jobs['job_60'])}
            data.append(dict(type="Feature", geometry=geom, properties=jobs_dict))
            #data.append(dict(type="Feature", geometry=geom))

    with open(outfile, "w") as geojson:
        geojson.write(dumps({"type": "FeatureCollection",
                             "features": data}, indent=2) + "\n")

shape2json(SHAPEFILE_PATH)

def groupShape2Json(fname, outfile="./output/groups.json"):
    reader = shapefile.Reader(fname)
    data = []
    for sr in reader.shapeRecords():
        geom = sr.shape.__geo_interface__
        lat = float(sr.record[10])
        lon = float(sr.record[11])
        #if inBox(lat, lon):
        id_dict = {"id":sr.record[4]}
        data.append(dict(type="Feature", geometry=geom, properties=id_dict))

    with open(outfile, "w") as geojson:
        geojson.write(dumps({"type": "FeatureCollection", "features": data}, indent=2) + "\n")

groupShape2Json(GROUP_SHAPE_PATH)
end = time.time()
print "compute interactive data ", end - start, 's'
print "All Done"
