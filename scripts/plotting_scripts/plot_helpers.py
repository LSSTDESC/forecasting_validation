import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cmasher as cmr
# Set Seaborn style to avoid grid
sns.set_style("white")
# Set Matplotlib's default style
plt.style.use('default')


def get_colors(num_colors, cmap="cmr.pride", cmap_range=(0.15, 0.85)):
    return cmr.take_cmap_colors(cmap, num_colors, cmap_range=cmap_range, return_fmt='hex')


def plot_resolution_sweep_subplots(data_resolutions, labels, y_label, title, forecast_year,
                                   precision=None, stability_steps=10, marker_size=5,
                                   extra_info="", subtitle_padding=1.02):
    resolutions = sorted(data_resolutions.keys())
    num_plots = len(labels)
    colors = get_colors(num_plots)

    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 2. * num_plots), sharex=True)
    fig.suptitle(f"{title} - Forecast Year {forecast_year}", fontsize=18, y=subtitle_padding)
    axes = [axes] if num_plots == 1 else axes  # Make `axes` iterable

    for i, label in enumerate(labels):
        data_values = [data_resolutions[res][i] for res in resolutions]
        avg_value = np.mean(data_values)
        axes[i].plot(resolutions, data_values, '-o', markersize=marker_size, color=colors[i])

        if precision:
            margin = avg_value * (precision / 100)
            upper_band, lower_band = avg_value + margin, avg_value - margin
            axes[i].fill_between(resolutions, lower_band, upper_band, color='lightgray',
                                 alpha=0.3, label=f"{precision}% margin")

        # Stabilization check helper function
        def check_stabilization(resolutions, data_values, lower_band, upper_band, stability_steps):
            stable_count = 0
            for res, value in zip(resolutions, data_values):
                if lower_band <= value <= upper_band:
                    stable_count += 1
                    if stable_count >= stability_steps:
                        return res
                else:
                    stable_count = 0
            return None

        stable_res = check_stabilization(resolutions, data_values, lower_band, upper_band, stability_steps)
        if stable_res:
            axes[i].axvline(stable_res, color='red', linestyle='--')
            axes[i].text(
                stable_res, avg_value, f'{stable_res}',
                color='red', va='top', ha='right', fontsize=12,
                rotation=90,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  # Customize color and transparency
            )
        axes[i].set_ylabel(f"{y_label} {label}", fontsize=14)
        axes[i].legend(loc='upper right', fontsize=12, frameon=False)

    axes[-1].set_xlabel("redshift resolution", fontsize=16)
    fig_name_pdf = f"{title.replace(' ', '_').lower()}_{forecast_year}_{extra_info}.pdf".replace("__", "_")
    fig_name_png = f"{title.replace(' ', '_').lower()}_{forecast_year}_{extra_info}.png".replace("__", "_")
    plt.savefig(f"plots_output/{fig_name_pdf}")
    plt.savefig(f"plots_output/{fig_name_png}")
    plt.tight_layout()
    plt.show()


def plot_stabilization_heatmap(heatmap_data, x_labels, y_labels, title, forecast_year, annotate_max=False):
    cmap = cmr.get_sub_cmap('cmr.pride', 0.15, 0.85)
    plt.figure(figsize=(10, len(y_labels) * 0.5))
    ax = sns.heatmap(heatmap_data, annot=not annotate_max, fmt=".0f", cmap=cmap,
                     cbar_kws={'label': 'Stabilization Resolution'}, xticklabels=x_labels, yticklabels=y_labels,
                     )

    if annotate_max:
        for x_idx in range(len(x_labels)):
            max_row_idx = np.nanargmax(heatmap_data[:, x_idx])
            max_value = heatmap_data[max_row_idx, x_idx]
            ax.text(x_idx + 0.5, max_row_idx + 0.5, f"{int(max_value)}", ha='center', va='center', color='white',
                    fontsize=10)

    ax.set_title(f"{title} - Forecast Year {forecast_year}", fontsize=18)
    ax.set_xlabel("$z_\\mathrm{{max}}$", fontsize=16)
    ax.set_ylabel("Index", fontsize=16)

    fig_name_pdf = f"{title.replace(' ', '_').lower()}_{forecast_year}.pdf"
    fig_name_png = f"{title.replace(' ', '_').lower()}_{forecast_year}.png"
    plt.savefig(f"plots_output/{fig_name_pdf}")
    plt.savefig(f"plots_output/{fig_name_png}")
    plt.show()