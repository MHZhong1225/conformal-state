# base_plots.py
import os, sys, inspect
from itertools import groupby
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle as pkl
import seaborn as sns
import pdb
from plotting_utils import *


def to_1d_y(data_y, col=None):
    """
    Convert data['y'] to a 1D numpy array.
    - If data_y is a DataFrame: pick `col` if provided, else first column.
    - If data_y is a Series: use it.
    - Else: np.asarray and flatten.
    """
    if isinstance(data_y, pd.DataFrame):
        if col is not None:
            y_series = data_y[col]
        else:
            y_series = data_y.iloc[:, 0]
        y = y_series.to_numpy(dtype=float)
        idx = y_series.index
    elif isinstance(data_y, pd.Series):
        y = data_y.to_numpy(dtype=float)
        idx = data_y.index
    else:
        y = np.asarray(data_y, dtype=float)
        idx = None

    y = np.asarray(y, dtype=float).reshape(-1)
    return y, idx


def stack_sets_2col(sets_list):
    """
    Ensure sets are stacked into shape (T,2) float array.
    """
    arr = np.stack(sets_list)
    arr = np.asarray(arr, dtype=float)

    # Handle accidental shapes like (T,1,2) or (T,2,1)
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 2:
        arr = arr[:, 0, :]
    if arr.ndim == 3 and arr.shape[1] == 2 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected sets stacked to (T,2), got shape {arr.shape}")
    return arr


if __name__ == "__main__":
    # Open file
    results_filename = sys.argv[1]
    with open(results_filename, 'rb') as handle:
        all_results = pkl.load(handle)

    dataset_name = results_filename.split('.')[-2].split('/')[-1]
    model_names = list(all_results.keys())

    # Remap names
    method_title_map = {
        'Trail': 'Trail',
        'ACI': 'ACI',
        'ACI (clipped)': 'ACI (clipped)',
        # 'Quantile': 'Conformal P',
        # 'Quantile+Integrator (log)': 'Conformal PI',
        'Quantile+Integrator (log)+Scorecaster': 'Conformal PID',
        'CPTC': 'CPTC',
        'DSS-CC': 'Ours'
    }

    for model_name in model_names:
        results = all_results[model_name]

        # If results contains ACI (clipped), delete it
        if 'ACI (clipped)' in results.keys():
            del results['ACI (clipped)']
        # Filter out Quantile and Quantile+Integrator (log) as requested
        if 'Quantile' in results.keys():
            del results['Quantile']
        if 'Quantile+Integrator (log)' in results.keys():
            del results['Quantile+Integrator (log)']

        plots_folder = "./plots/" + dataset_name + "/" + model_name + "/"
        os.makedirs(plots_folder, exist_ok=True)

        # Set style
        method_keys = list(results.keys())
        cmap_lines = sns.color_palette("husl", len(method_keys))
        sns.set_theme(context="notebook", palette=cmap_lines, style="white", font_scale=4)
        sns.set_style({'axes.spines.right': False, 'axes.spines.top': False})

        # Process results
        alpha = results["alpha"]
        scores = results["scores"]
        T_burnin = results["T_burnin"]
        real_data = results["real_data"]
        multiple_series = results["multiple_series"]
        quantiles_given = results["quantiles_given"]
        score_function_name = results["score_function_name"]
        asymmetric = results["asymmetric"]

        # -------------------------
        # REAL DATA HANDLING (FIXED)
        # -------------------------
        if real_data:
            forecasts = results["forecasts"]
            data = results["data"]

            # --- FIX 1: forecasts[0] KeyError when Series ---
            if isinstance(forecasts, (pd.Series, pd.DataFrame)):
                first_forecast = forecasts.iloc[0]
            else:
                first_forecast = forecasts[0]
            listlike_forecast = is_listlike(first_forecast)

            # --- FIX 2: force y to 1D vector (avoid pandas alignment -> (T,T)) ---
            # If you want a specific column instead of first column, set y_col="colname"
            y_col = None
            y_vec, y_index = to_1d_y(data["y"], col=y_col)

            # Remove from results dict to avoid pickle bloat later
            del results["forecasts"]
            del results["data"]
            try:
                log = results["log"]
                del results["log"]
            except:
                pass

        # clean meta keys
        del results["alpha"]
        del results["scores"]
        del results["T_burnin"]
        del results["real_data"]
        del results["multiple_series"]
        del results["quantiles_given"]
        del results["score_function_name"]
        del results["asymmetric"]

        """
            START PLOTTING
        """
        nrows = max([len(list(results[key].keys())) for key in results.keys()])
        ncols = len(list(results.keys()))

        # -------------------------
        # COVERAGE COMPUTATION (FIXED)
        # -------------------------
        coverages = {}
        for key in results.keys():
            for lr in list(results[key].keys()):
                if real_data:
                    sets_arr = stack_sets_2col(results[key][lr]["sets"][T_burnin+1:])
                    low = sets_arr[:, 0]
                    high = sets_arr[:, 1]

                    # align lengths safely
                    start = T_burnin + 1
                    y_slice = y_vec[start:start + len(low)]
                    n = min(len(y_slice), len(low), len(high))
                    y_slice = y_slice[:n]
                    low = low[:n]
                    high = high[:n]

                    results[key][lr]["covered"] = (low <= y_slice) & (y_slice <= high)
                else:
                    # keep original logic for synthetic
                    results[key][lr]["covered"] = results[key][lr]["q"][T_burnin+1:] >= scores[T_burnin+1:]

            coverages[key] = {lr: results[key][lr]["covered"].astype(int).mean()
                              for lr in list(results[key].keys())}

        print(coverages)

        # Plot coverage
        linewidth = 2
        transparency = 0.7

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            sharey=True,
            figsize=(ncols * 10.1, nrows * 6.4),
            squeeze=False
        )

        i = 0
        # NOTE: scores shape might be Series/DataFrame/list; keep your old behavior:
        xlabels_nonscores = range(T_burnin+1, scores.shape[0]) if hasattr(scores, "shape") else range(T_burnin+1, len(scores))

        for key in results.keys():
            color = cmap_lines[i]
            j = 0
            for lr in results[key].keys():
                label = f"lr={lr}, cvg={100*coverages[key][lr]:.1f}%" if lr is not None else f"cvg={100*coverages[key][lr]:.1f}%"
                cov = results[key][lr]["covered"]
                cov_ma = moving_average(cov)

                axs[j, i].plot(
                    xlabels_nonscores[T_burnin:],
                    cov_ma[T_burnin:],
                    label=label,
                    linewidth=linewidth,
                    color=color,
                    alpha=transparency
                )
                axs[j, i].axhline(1 - alpha, color='#888888', linestyle='--', linewidth=linewidth)
                axs[j, i].legend(handlelength=0.0, handletextpad=-0.1)
                j += 1

            axs[0, i].set_title(method_title_map[key])
            i += 1

        fig.supxlabel('Time')
        fig.supylabel('Coverage')
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, wspace=0.2)
        plt.savefig(plots_folder + "coverage.pdf")

        # Size plots (zoomed in)! Only visualize the 'upper' score, i.e., the last one in the array.
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols + 1, sharex=True, sharey=True,
                                figsize=((ncols + 1) * 10.1, nrows * 6.4), squeeze=False)

        last = 50
        i = 1
        upper_scores = np.stack(scores)[-last:, -1] if len(np.stack(scores).shape) > 1 else np.asarray(scores)[-last:]
        low_clip = np.nanmin(upper_scores) * 0.9
        high_clip = np.nanmax(upper_scores) * 1.1

        for key in results.keys():
            color = cmap_lines[i - 1]
            j = 0
            for lr in results[key].keys():
                upper_quantiles = np.stack(results[key][lr]["q"])[-last:, -1] if len(np.stack(scores).shape) > 1 else np.asarray(results[key][lr]["q"])[-last:]
                upper_quantiles = np.clip(upper_quantiles, low_clip, high_clip)[-last:]

                axs[j, i].plot(xlabels_nonscores[-last:], upper_scores,
                               linewidth=linewidth, alpha=transparency / 4, color=cmap_lines[-1])
                label = f"lr={lr}" if lr is not None else None
                axs[j, i].plot(xlabels_nonscores[-last:], upper_quantiles,
                               linewidth=linewidth, color=color, alpha=transparency, label=label)
                if label is not None:
                    axs[j, i].legend(handlelength=0.0, handletextpad=-0.1)
                j += 1

            axs[0, i].set_title(method_title_map[key])
            i += 1

        axs[0, 0].plot(xlabels_nonscores[-last:], upper_scores,
                       linewidth=linewidth, alpha=transparency, color=cmap_lines[-1])
        axs[0, 0].set_title('Scores')
        plt.ylim([low_clip, high_clip])
        fig.supxlabel('Time')
        fig.supylabel(r'$q_t$')
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, wspace=0.2, hspace=0.1)
        plt.savefig(plots_folder + "size_zoomed.pdf")

        # Plot sets (zoomed)
        if real_data:
            sns.set_theme(context="notebook", palette=cmap_lines, style="white", font_scale=4)
            sns.set_style({'axes.spines.right': False, 'axes.spines.top': False})

            if listlike_forecast:
                forecasts_zoomed = [forecast[-last:] for forecast in forecasts]
            else:
                forecasts_zoomed = forecasts[-last:]

            # FIX: use y_vec (1D)
            y_zoomed = y_vec[-last:]

            fig, axs = plt.subplots(nrows=nrows, ncols=ncols + 1, sharex=True, sharey=True,
                                    figsize=((ncols + 1) * 10.1, nrows * 6.4), squeeze=False)

            i = 1
            y_clip_low = np.nanmin(y_zoomed) * 0.8
            y_clip_high = np.nanmax(y_zoomed) * 1.2

            for key in results.keys():
                color = lighten_color(desaturate_color(cmap_lines[i - 1], saturation=0.3), 0.5)
                j = 0
                for lr in results[key].keys():
                    sets_zoomed = stack_sets_2col(results[key][lr]["sets"])[-last:]
                    sets_zoomed = np.clip(sets_zoomed, y_clip_low, y_clip_high)
                    results[key][lr]["sets_zoomed"] = sets_zoomed

                    axs[j, i].plot(np.arange(y_zoomed.shape[0]), y_zoomed, color='black', alpha=0.2)
                    label = f"lr={lr}" if lr is not None else None
                    axs[j, i].fill_between(np.arange(y_zoomed.shape[0]),
                                           sets_zoomed[:, 0], sets_zoomed[:, 1],
                                           color=color, alpha=transparency, label=label)
                    if label is not None:
                        axs[j, i].legend(handlelength=0.0, handletextpad=-0.1)
                    j += 1

                axs[0, i].set_title(method_title_map[key])
                i += 1

            axs[0, 0].plot(np.arange(y_zoomed.shape[0]), y_zoomed,
                           linewidth=linewidth, alpha=transparency, color='black', label="ground truth")
            axs[0, 0].legend()

            if listlike_forecast:
                axs[1, 0].plot(np.array(forecasts_zoomed[1]),
                               linewidth=linewidth, alpha=transparency, color='green', label="forecast")
            else:
                axs[1, 0].plot(np.asarray(forecasts_zoomed),
                               linewidth=linewidth, alpha=transparency, color='green', label="forecast")
            axs[1, 0].legend()

            axs[0, 0].set_title("y")
            plt.ylim([y_clip_low, y_clip_high])
            fig.supxlabel('Time')
            fig.supylabel(r'$\mathcal{C}_t$')
            plt.tight_layout(pad=0.05)
            plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, wspace=0.2)
            plt.savefig(plots_folder + "sets_zoomed.pdf")

        # Plot sets (full)
        if real_data:
            sns.set_theme(context="notebook", palette=cmap_lines, style="white", font_scale=4)
            sns.set_style({'axes.spines.right': False, 'axes.spines.top': False})

            if listlike_forecast:
                forecasts_zoomed = [forecast[T_burnin+1:] for forecast in forecasts]
            else:
                forecasts_zoomed = forecasts[T_burnin+1:]

            # FIX: use y_vec (1D)
            y_zoomed = y_vec[T_burnin+1:]

            fig, axs = plt.subplots(nrows=nrows, ncols=ncols + 1, sharex=True, sharey=True,
                                    figsize=((ncols + 1) * 10.1, nrows * 6.4), squeeze=False)

            i = 1
            y_clip_low = np.nanmin(y_zoomed) * 0.8
            y_clip_high = np.nanmax(y_zoomed) * 1.2

            for key in results.keys():
                color = lighten_color(desaturate_color(cmap_lines[i - 1], saturation=0.3), 0.5)
                j = 0
                for lr in results[key].keys():
                    sets_zoomed = stack_sets_2col(results[key][lr]["sets"])[T_burnin+1:]
                    sets_zoomed = np.clip(sets_zoomed, y_clip_low, y_clip_high)
                    results[key][lr]["sets_zoomed"] = sets_zoomed

                    axs[j, i].plot(np.arange(y_zoomed.shape[0]), y_zoomed, color='black', alpha=0.2)
                    label = f"lr={lr}" if lr is not None else None
                    axs[j, i].fill_between(np.arange(y_zoomed.shape[0]),
                                           sets_zoomed[:, 0], sets_zoomed[:, 1],
                                           color=color, alpha=transparency, label=label)
                    if label is not None:
                        axs[j, i].legend(handlelength=0.0, handletextpad=-0.1)
                    j += 1

                axs[0, i].set_title(method_title_map[key])
                i += 1

            axs[0, 0].plot(y_zoomed, linewidth=linewidth, alpha=transparency, color='black', label="ground truth")
            axs[0, 0].legend()

            if listlike_forecast:
                axs[1, 0].plot(np.array([forecast[1] for forecast in forecasts]),
                               linewidth=linewidth, alpha=transparency, color='green', label=r'$1-\alpha/2$ forecast')
                axs[2, 0].plot(np.array([forecast[0] for forecast in forecasts]),
                               linewidth=linewidth, alpha=transparency, color='green', label=r'$\alpha/2$ forecast')
                axs[2, 0].legend()
            else:
                axs[1, 0].plot(np.asarray(forecasts_zoomed),
                               linewidth=linewidth, alpha=transparency, color='green', label="forecast")
            axs[1, 0].legend()

            axs[0, 0].set_title("y")
            plt.ylim([y_clip_low, y_clip_high])
            fig.supxlabel('Time')
            fig.supylabel(r'$\mathcal{C}_t$')
            plt.tight_layout(pad=0.05)
            plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, wspace=0.2)
            plt.savefig(plots_folder + "sets.pdf")