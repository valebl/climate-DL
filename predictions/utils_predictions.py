import matplotlib.pyplot as plt


def create_zones(zones_file='/m100_work/ICT23_ESP_C/vblasone/precipitation-maps/Italia.txt'):
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

def plot_maps(pos, pr_pred, pr, pr_min, pr_max, zones, save_path, save_file_name, aggr=None, title="", cmap='turbo', idx_start=1+31*24, idx_end=-1, legend_title="pr"):
    
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,6))

    lon = pos[:,0]
    lat = pos[:,1]
    
    v = pr_pred[:,idx_start:idx_end]
    v_s = aggr(v, axis=1)
    im = ax[0].scatter(lon,lat,c=v_s, marker="s", s=110, vmin=pr_min, vmax=pr_max, cmap=cmap)

    plot_italy(zones, color='black', ax=ax[0], alpha_fill=0)
    ax[0].set_xlim([lon.min()-0.25,lon.max()+0.25])
    ax[0].set_ylim([lat.min()-0.25,lat.max()+0.25])
    # ax[0].set_xlim([12.5, 14.1])
    # ax[0].set_ylim([45.4, 46.8])
    ax[0].set_title("PREDICTIONS")
    
    v = pr[:,idx_start:idx_end]
    v_s = aggr(v, axis=1)
    im = ax[1].scatter(lon,lat,c=v_s, marker="s", s=110, vmin=pr_min, vmax=pr_max, cmap=cmap)

    plot_italy(zones, color='black', ax=ax[1], alpha_fill=0)
    ax[1].set_xlim([lon.min()-0.25,lon.max()+0.25])
    ax[1].set_ylim([lat.min()-0.25,lat.max()+0.25])
    # ax[1].set_xlim([12.5, 14.1])
    # ax[1].set_ylim([45.4, 46.8])
    ax[1].set_title("GRIPHO")

    cbar = fig.colorbar(im, ax=ax, aspect=25, pad=0.025)
    cbar.ax.set_title(legend_title, rotation=0, fontsize=18, pad=20)
    _ = fig.suptitle(title, fontsize=22, x=0.45, y=1)

    plt.savefig(f'{save_path}{save_file_name}', dpi=800, bbox_inches='tight', pad_inches=0.0)