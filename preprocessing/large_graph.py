import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--output_path', type=str, default='/m100_work/ICT22_ESP_0/vblasone/PREPROCESSED_TEST_CODE/')
parser.add_argument('--log_file', type=str, default='log.txt')
parser.add_argument('--target_path_file', type=str, default='/m100_work/ICT22_ESP_0/vblasone/GRIPHO/gripho-v1_1h_TSmin30pct_2001-2016_cut.nc')
parser.add_argument('--topo_path_file', type=str, default='/m100_work/ICT22_ESP_0/vblasone/TOPO/GMTED_DEM_30s_remapdis_GRIPHO.nc')
parser.add_argument('--input_path_file', type=str, default='/m100_work/ICT22_ESP_0/vblasone/SLICED/q_sliced.nc')

# lat lon grid values
parser.add_argument('--lon_min', type=float, default=6.50)
parser.add_argument('--lon_max', type=float, default=14.00)
parser.add_argument('--lat_min', type=float, default=43.75)
parser.add_argument('--lat_max', type=float, default=47.25)

def select_from_gripho(lon_min, lon_max, lat_min, lat_max, lon, lat, pr, geo):
    bool_lon = np.logical_and(lon >= lon_min, lon <= lon_max)
    bool_lat = np.logical_and(lat >= lat_min, lat <= lat_max)
    bool_both = np.logical_and(bool_lon, bool_lat)
    selected_pr = np.array([pr[i][bool_both] for i in range(TIME_DIM)])

    return lon[bool_both], lat[bool_both], selected_pr, geo[bool_both]

if __name__ == '__main__':

    args = parser.parse_args()

    TIME_DIM = 140256
    SPATIAL_POINTS_DIM = 2107
    LON_MIN = 6.5
    LON_MAX = 18.75
    LAT_MIN = 36.5
    LAT_MAX =  47.25
    INTERVAL = 0.25

    LON_DIFF_MAX = 0.25 / 8 * 2
    LAT_DIFF_MAX = 0.25 / 10 * 2

    gripho = xr.open_dataset(args.target_path_file)
    era5_q = xr.open_dataset(args.input_path_file)
    topo = xr.open_dataset(args.topo_path_file)

    lon = gripho.lon.to_numpy()
    lat = gripho.lat.to_numpy()
    pr = gripho.pr.to_numpy()
    z = topo.z.to_numpy()

    #lon_coarse_grid_gripho_array = np.arange(LON_MIN, LON_MAX, INTERVAL)
    #lat_coarse_grid_gripho_array = np.arange(LAT_MIN, LAT_MAX, INTERVAL)

    with open(args.output_path + args.log_file, 'w') as f:
        f.write(f"\nStarting the preprocessing.")

    gnn_target = {}
    gnn_data = {}
    start = time.time()
    
    lon_min = 6.50
    lon_max = 14.00
    lat_min = 43.75
    lat_max = 47.25

            
    # select the region of interest to define the output graph
    lon_sel, lat_sel, pr_sel, z_sel = select_from_gripho(lon_min, lon_max,
        lat_min, lat_max, lon, lat, pr, z)

    pr_target = []
    lon_target = []
    lat_target = []
    z_target = []

    for s in range(pr_sel.shape[1]):
        # for each (lon, lat) point check that there is at least one value not nan
        if not np.isnan(pr_sel[:,s]).all():
            pr_target.append(pr_sel[:,s])
            lon_target.append(lon_sel[s])
            lat_target.append(lat_sel[s])
            z_target.append(z_sel[s])

    pr_target = np.array(pr_target)
    pr_target = pr_target.swapaxes(0,1)

    x = np.stack((lon_target, lat_target, z_target), axis=-1)
    edge_index = np.empty((2,0), dtype=int)

    for ii, xi in enumerate(x):
        for jj, xj in enumerate(x):
            if not np.array_equal(xi, xj) and np.abs(xi[0] - xj[0]) < LON_DIFF_MAX and np.abs(xi[1] - xj[1]) < LAT_DIFF_MAX:
                edge_index = np.concatenate((edge_index, np.array([[ii], [jj]])), axis=-1, dtype=int)
    gnn_data = {'x': x, 'edge_index': edge_index}

    with open(log_file, 'a') as f:
        f.write(f"\nPreprocessing took {time.time() - start} seconds")    

    with open(log_file, 'a') as f:
        f.write(f"\nStarting to write the file.")    

    with open(output_path+"north_italy_graph.pkl", 'wb') as f:
        pickle.dump(gnn_data, f)

    with open(log_file, 'a') as f:
        f.write(f"\nDone! :)")  



