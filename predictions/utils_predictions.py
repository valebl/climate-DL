import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib

def create_zones(zones_file='/leonardo_work/ICT23_ESP_0/SHARED/climate-DL/preprocessing/Italia.txt'):
    zones = []
    with open(zones_file) as f:
        lines = f.read()
        for zone in lines.split(';'):
            zones.append(zone)
    for i in range(len(zones)):
        zones[i] = zones[i].split('\n')
        for j in range(len(zones[i])):
            zones[i][j] = zones[i][j].split(',')
        if [''] in zones[i]:
            zones[i].remove([''])
    for i in range(len(zones)):
        for j in range(len(zones[i])):
            if '' in zones[i][j]:
                zones[i][j].remove('')
            if zones[i][j] == []:
                del zones[i][j]
                continue
            for k in range(len(zones[i][j])):
                zones[i][j][k] = float(zones[i][j][k])
    return zones

def plot_italy(zones, ax, color='k', alpha_fill=0.1, linewidth=1):
    j = 0
    for zone in zones:
        x_zone = [zone[i][0] for i in range(len(zone)) if i > 0]
        y_zone = [zone[i][1] for i in range(len(zone)) if i > 0]
        ax.fill(x_zone, y_zone, color, alpha=alpha_fill)
        ax.plot(x_zone, y_zone, color, alpha=1, linewidth=1)
        j += 1
        
def draw_rectangle(x_min, x_max, y_min, y_max, color, ax, fill=False, fill_color=None, alpha=0.5):
    y_grid = [y_min, y_min, y_max, y_max, y_min]
    x_grid = [x_min, x_max, x_max, x_min, x_min]
    ax.plot(x_grid, y_grid, color=color)
    if fill:
        if fill_color==None:
            fill_color = color
        ax.fill(x_grid, y_grid, color=fill_color, alpha=alpha)

def plot_maps(pos, pr_pred, pr, zones, save_path, save_file_name, 
        x_size, y_size, font_size_title, font_size=None, pr_min=None, pr_max=None, aggr=None, title="", 
        cmap='turbo', idx_start=0, idx_end=-1, legend_title="pr", xlim=None, ylim=None, cbar_y=1,
        cbar_title_size=None, cbar_pad=50, subtitle_y=1):
    
    if font_size is None:
        plt.rcParams.update({'font.size': int(16 // 7 * y_size)})
    else:
        plt.rcParams.update({'font.size': int(font_size)})
    
    if cbar_title_size is None:
        cbar_title_size = int(24 // 7 * y_size)
    else:
        cbar_title_size = cbar_title_size

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(x_size*2,y_size))

    lon = pos[:,0]
    lat = pos[:,1]
    
    pr_min = pr_min if pr_min is not None else min(pr_pred.min(), pr.min())
    pr_max = pr_max if pr_max is not None else max(pr_pred.max(), pr.max())

    #v = pr_pred[:,idx_start:idx_end]
    if aggr is not None:
        v_s = aggr(pr_pred, axis=1)
    else:
        v_s = pr_pred
    
    im = ax[0].scatter(lon,lat,c=v_s, marker="s", s=150, vmin=pr_min, vmax=pr_max, cmap=cmap)

    plot_italy(zones, color='black', ax=ax[0], alpha_fill=0)
    ax[0].set_xlim([lon.min()-0.25,lon.max()+0.25])
    ax[0].set_ylim([lat.min()-0.25,lat.max()+0.25])
    # ax[0].set_xlim([12.5, 14.1])
    # ax[0].set_ylim([45.4, 46.8])
    ax[0].set_title("PREDICTIONS")
    
    #v = pr[:,idx_start:idx_end]
    if aggr is not None:
        v_s = aggr(pr, axis=1)
    else:
        v_s = pr
    
    im = ax[1].scatter(lon,lat,c=v_s, marker="s", s=150, vmin=pr_min, vmax=pr_max, cmap=cmap)

    if xlim is not None:
        ax[0].set_xlim(xlim)
    if ylim is not None:
        ax[0].set_ylim(ylim)

    plot_italy(zones, color='black', ax=ax[1], alpha_fill=0)
    ax[1].set_xlim([lon.min()-0.25,lon.max()+0.25])
    ax[1].set_ylim([lat.min()-0.25,lat.max()+0.25])
    # ax[1].set_xlim([12.5, 14.1])
    # ax[1].set_ylim([45.4, 46.8])
    ax[1].set_title("GRIPHO")

    cbar = fig.colorbar(im, ax=ax, aspect=25, pad=0.025)
    cbar.ax.set_title(legend_title, rotation=0, fontsize=cbar_title_size, pad=cbar_pad)
    _ = fig.suptitle(title, fontsize=font_size_title, x=0.45, y=subtitle_y)

    if xlim is not None:
        ax[1].set_xlim(xlim)
    if ylim is not None:
        ax[1].set_ylim(ylim)

    plt.savefig(f'{save_path}{save_file_name}', dpi=400, bbox_inches='tight', pad_inches=0.0)

def date_to_day(year_start, month_start, day_start, year_end, month_end, day_end, first_year):
    day_of_year_start = datetime(year_start, month_start, day_start).timetuple().tm_yday
    day_of_year_end = datetime(year_end, month_end, day_end).timetuple().tm_yday
    start_idx = (date(year_start, month_start, day_start) - date(first_year, 1, 1)).days * 24
    end_idx = (date(year_end, month_end, day_end) - date(first_year, 1, 1)).days * 24 + 24
    return start_idx, end_idx


def extremes_cmap():
    c_lists = [[247, 255, 255],
               [238, 255, 255],
               [230, 255, 255],
               [209, 246, 255],
               [157, 217, 255],
               [105, 187, 255],
               [52, 157, 255],
               [25, 142, 216],
               [17, 137, 147],
               [9, 135, 79],
               [1, 129, 10],
               [12, 146, 12],
               [25, 167, 25],
               [38, 187, 38],
               [58, 203, 48],
               [113, 193, 35],
               [168, 182, 21],
               [233, 171, 8],
               [255, 146, 0],
               [255, 102, 0], 
               [255, 57, 0],
               [255, 13, 0],
               [236, 0, 0],
               [203, 0, 0],
               [164, 0, 0],
               [137, 0, 0]]

    for j, c_list_top in enumerate(c_lists[1:]):
        c_list_bot = c_lists[j]
        c = np.ones((8,4))
        for i in range(3):
            c[:,i] = np.linspace(c_list_bot[i]/255, c_list_top[i]/255, c.shape[0])
        if j == 0:
            cmap = c
        else:
            cmap = np.vstack((cmap, c))
    cmap = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
    return cmap

