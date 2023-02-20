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
parser.add_argument('--interval', type=float, default=0.25)
parser.add_argument('--time_dim', type=float, default=140256)


def cut_window(lon_min, lon_max, lat_min, lat_max, lon, lat, z, pr, time_dim):
    bool_lon = np.logical_and(lon >= lon_min, lon <= lon_max)
    bool_lat = np.logical_and(lat >= lat_min, lat <= lat_max)
    bool_both = np.logical_and(bool_lon, bool_lat)
    lon_sel = lon[bool_both]
    lat_sel = lat[bool_both]
    z_sel = z[bool_both]
    pr_sel = np.array(pr[:,bool_both])
    return lon_sel, lat_sel, z_sel, pr_sel

def select_nodes(lon_centre, lat_centre, lon, lat, cell_idx, cell_idx_array, offset, time_dim):
    bool_lon = np.logical_and(lon >= lon_centre, lon <= lon_centre+offset)
    bool_lat = np.logical_and(lat >= lat_centre, lat <= lat_centre+offset)
    bool_both = np.logical_and(bool_lon, bool_lat)
    cell_idx_array[bool_both] = cell_idx
    return cell_idx_array

def write_log(s, args, mode='a'):
    with open(args.output_path + args.log_file, mode) as f:
        f.write(s)

if __name__ == '__main__':

    args = parser.parse_args()

    write_log("Start!\n", args, 'w')

    LON_DIFF_MAX = 0.25 / 8 * 2
    LAT_DIFF_MAX = 0.25 / 10 * 2

    gripho = xr.open_dataset(args.target_path_file)
    topo = xr.open_dataset(args.topo_path_file)

    lon = gripho.lon.to_numpy()
    lat = gripho.lat.to_numpy()
    pr = gripho.pr.to_numpy()
    z = topo.z.to_numpy()

    write_log("Cutting the window...\n", args)

    # cut gripho and topo to the desired window
    lon_sel, lat_sel, z_sel, pr_sel = cut_window(args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon, lat, z, pr, args.time_dim)

    write_log("Done!\n", args)

    gnn_target = {}
    gnn_data = {}
    cell_idx_array = np.zeros(pr_sel.shape[1])
    
    lon_low_res_array = np.arange(args.lon_min, args.lon_max, args.interval)
    lat_low_res_array = np.arange(args.lat_min, args.lat_max, args.interval)
    lon_low_res_dim = lon_low_res_array.shape[0]

    # start the preprocessing
    write_log(f"Starting the preprocessing.\n", args)
    start = time.time()
    
    for i, lat_low_res in enumerate(lat_low_res_array):
        for j, lon_low_res in enumerate(lon_low_res_array):
            
            cell_idx = i * lon_low_res_dim + j
            cell_idx_array = select_nodes(lon_low_res, lat_low_res, lon_sel, lat_sel, cell_idx, cell_idx_array, args.interval, args.time_dim)
            
    end = time.time()
    write_log(f'loop took {end - start} s\n', args)

    with open('cell_idx_array.pkl', 'wb') as f:
        pickle.dump(cell_idx_array, f)

    sys.exit()

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



