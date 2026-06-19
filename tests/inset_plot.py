# inset_plot.py
import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from plotting_utils import moving_average, plot_time_series, longest_true_sequence
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import pickle as pkl
from matplotlib.dates import date2num
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle
import pdb
import matplotlib.patheffects as pe
import matplotlib.ticker as mtick


def dataframe_to_latex(df):
    # Clean names
    df = df.copy()
    df.replace({'ar': 'AR', 'theta': 'Theta', 'transformer': 'Transformer', 'prophet': 'Prophet'}, inplace=True)

    metric_cols = [c for c in df.columns if c not in ['Model type', 'Method']]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce') 

    df_melted = df.melt(id_vars=['Model type', 'Method'], var_name='Metric', value_name='Value')

    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

    # Pivot (mean is OK because numeric)
    df_pivot = df_melted.pivot_table(
        index='Metric',
        columns=['Model type', 'Method'],
        values='Value',
        aggfunc='mean'
    )

    metric_order = [
        'Marginal coverage',
        'Longest err sequence',
        'Average set size',
        'Median set size',
        '75% quantile set size',
        '90% quantile set size',
        '95% quantile set size'
    ]
    df_pivot = df_pivot.reindex(metric_order)

    # Build LaTeX table
    latex_table = "\\begin{tabular}{l" + "l"*len(df_pivot.columns) + "}\n\\toprule\n"

    model_types = df_pivot.columns.get_level_values(0).unique()
    for model in model_types:
        n_methods = sum(df_pivot.columns.get_level_values(0) == model)
        latex_table += f"& \\multicolumn{{{n_methods}}}{{c}}{{{model}}}"
    latex_table += " \\\\\n"

    latex_table += "& " + " & ".join(df_pivot.columns.get_level_values(1)) + " \\\\\n"
    latex_table += "\\midrule\n"

    # Format values
    for metric, row in df_pivot.iterrows():
        formatted = []
        for val in row.values:
            if pd.isna(val):
                formatted.append("")
            elif np.isinf(val):
                formatted.append("Inf")
            else:
                formatted.append("{:.3g}".format(val))
        latex_table += metric + " & " + " & ".join(formatted) + " \\\\\n"

    latex_table = latex_table.replace("%", "\\%")
    latex_table += "\\bottomrule\n\\end{tabular}\n"
    return latex_table

def stack_sets_2col(sets_list):
    """
    Convert sets (list of per-time arrays) to shape (T,2) float array.
    Handles shapes like (T,2,1) by squeeze.
    """
    arr = np.stack(sets_list)
    arr = np.asarray(arr, dtype=float)
    arr = np.squeeze(arr)  # (T,2,1)->(T,2)

    # Sometimes squeeze can turn (T,1,2)->(T,2) as well
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected sets shape (T,2), got {arr.shape}")
    return arr


def to_1d_series(y_obj):
    """
    Convert y_obj to a pandas Series (1D) with a DatetimeIndex if possible.
    If y_obj is DataFrame -> take first column.
    """
    if isinstance(y_obj, pd.DataFrame):
        y_series = y_obj.iloc[:, 0]
    elif isinstance(y_obj, pd.Series):
        y_series = y_obj
    else:
        y_series = pd.Series(y_obj)
    return y_series


def align_all(datetimes, y, low1, high1, low2, high2, covered1, covered2):
    """
    Force everything to 1D and align all lengths by truncating to the same min length n.
    """
    # Ensure 1D numpy
    y = np.asarray(y, dtype=float).reshape(-1)
    low1 = np.asarray(low1, dtype=float).reshape(-1)
    high1 = np.asarray(high1, dtype=float).reshape(-1)
    low2 = np.asarray(low2, dtype=float).reshape(-1)
    high2 = np.asarray(high2, dtype=float).reshape(-1)
    covered1 = np.asarray(covered1, dtype=float).reshape(-1)
    covered2 = np.asarray(covered2, dtype=float).reshape(-1)

    # datetimes can be DatetimeIndex
    n = min(len(datetimes), len(y), len(low1), len(high1), len(low2), len(high2), len(covered1), len(covered2))

    datetimes = datetimes[:n]
    y = y[:n]
    low1, high1 = low1[:n], high1[:n]
    low2, high2 = low2[:n], high2[:n]
    covered1, covered2 = covered1[:n], covered2[:n]

    return datetimes, y, low1, high1, low2, high2, covered1, covered2


def plot_everything(
    coverages_list, sets_list, titles_list, y, alpha,
    window_start, window_end, window_loc,
    coverage_inset, set_inset, miscoverage_scatterplot,
    savename, model_name, datetimes=None
):
    fig, axs = plt.subplots(
        nrows=2, ncols=len(coverages_list),
        figsize=(10 * len(coverages_list), 6),
        sharex=True, sharey=False
    )

    plot_time_series(
        fig, axs[0, :], coverages_list,
        window_start, window_end, window_loc,
        False, y, "#138085",
        coverage_inset, False, datetimes,
        hline=1 - alpha
    )
    plot_time_series(
        fig, axs[1, :], sets_list,
        window_start, window_end, window_loc,
        True, y, "#EEB362",
        set_inset, miscoverage_scatterplot, datetimes
    )

    axs[0, 0].set_ylabel('Coverage', fontsize=20)
    axs[1, 0].set_ylabel('Sets', fontsize=20)

    axs[0, 0].set_title(titles_list[0], fontsize=20)
    if len(titles_list) > 1:
        axs[0, 1].set_title(titles_list[1], fontsize=20)

    ymin = min([ax.get_ylim()[0] for ax in axs[0, :]])
    ymax = max([ax.get_ylim()[1] for ax in axs[0, :]])

    for ax in axs[0, :]:
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([0.5, 0.75, 1.0])

    axs[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    axs[0, 0].yaxis.set_tick_params(labelsize=13)
    if axs.shape[1] > 1:
        axs[0, 1].set_yticklabels([])

    ymin = min([ax.get_ylim()[0] for ax in axs[1, :]])
    ymax = max([ax.get_ylim()[1] for ax in axs[1, :]])
    for ax in axs[1, :]:
        ax.set_ylim([ymin, ymax + 0.1 * np.abs(ymax)])

    if axs.shape[1] > 1:
        axs[1, 1].set_yticklabels([])

    axs[1, 0].yaxis.set_tick_params(labelsize=13)
    axs[0, 0].yaxis.set_tick_params(labelsize=13)
    axs[1, 0].xaxis.set_tick_params(labelsize=13)
    if axs.shape[1] > 1:
        axs[1, 1].xaxis.set_tick_params(labelsize=13)

    fig.autofmt_xdate()
    plt.subplots_adjust(left=0.1, bottom=0.15)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time", fontsize=20, labelpad=40)

    os.makedirs('./plots/1v1/' + model_name, exist_ok=True)
    plt.savefig('./plots/1v1/' + model_name + "/" + savename + '.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot time series data.')
    parser.add_argument('--filename', help='Path to pickle file containing time series data.')
    parser.add_argument('--key1', help='First key for time series data extraction.')
    parser.add_argument('--lr1', help='Learning rate associated with first key.', type=float)
    parser.add_argument('--key2', help='Second key for time series data extraction.')
    parser.add_argument('--lr2', help='Learning rate associated with second key.', type=float)
    parser.add_argument('--window_length', help='Length of inset window.', default=60, type=int)
    parser.add_argument('--window_start', help='Start of inset window.', default=None, type=int)
    parser.add_argument('--window_loc', help='Location of inset window.', default='upper right', type=str)
    parser.add_argument('--coverage_average_length', help='Length of moving average window for coverage.', default=50, type=int)
    parser.add_argument('--coverage_average_burnin', help='How long to wait before displaying moving average of coverages', default=0, type=int)
    parser.add_argument('--coverage_inset', dest='coverage_inset', default=False, action='store_true')
    parser.add_argument('--set_inset', dest='set_inset', default=False, action='store_true')
    parser.add_argument('--miscoverage_scatterplot', dest='miscoverage_scatterplot', default=False, action='store_true')

    args = parser.parse_args()

    method_title_map = {
        'ACI': 'ACI',
        'ACI (clipped)': 'ACI (clipped)',
        'Quantile': 'Conformal P Control' if args.lr1 != 0 else 'Base Forecaster',
        'Quantile+Integrator (log)': 'Conformal PI Control',
        'Quantile+Integrator (log)+Scorecaster': 'Conformal PID Control',
        'DSS-CI': 'Ours'
    }

    args.window_start = args.window_start if args.window_start is not None else -args.window_length
    datasetname = args.filename.split('/')[-1].split('.')[0]

    with open(args.filename, 'rb') as f:
        all_data = pkl.load(f)

    model_names = list(all_data.keys())
    df_list_for_table = []

    for model_name in model_names:
        data = all_data[model_name]

        T_burnin = data['T_burnin']
        alpha = data['alpha']

        # ----------------------------
        # y -> 1D series + numpy + datetimes
        # ----------------------------
        y_series = to_1d_series(data['data']['y'])
        # print("y type/shape:", type(y_series), y_series.shape)

        y_all = y_series.to_numpy(dtype=float).reshape(-1)
        dt_all = y_series.index

        y = y_all[T_burnin + 1:]
        datetimes = dt_all[T_burnin + 1:]

        # ----------------------------
        # sets -> (T,2)
        # ----------------------------
        sets1_arr = stack_sets_2col(data[args.key1][args.lr1]['sets'])[T_burnin + 1:]
        sets2_arr = stack_sets_2col(data[args.key2][args.lr2]['sets'])[T_burnin + 1:]

        low1, high1 = sets1_arr[:, 0], sets1_arr[:, 1]
        low2, high2 = sets2_arr[:, 0], sets2_arr[:, 1]

        # ----------------------------
        # coverage moving average
        # ----------------------------
        covered1_raw = ((y >= low1) & (y <= high1)).astype(float)
        covered2_raw = ((y >= low2) & (y <= high2)).astype(float)

        covered1 = moving_average(covered1_raw, args.coverage_average_length)
        covered2 = moving_average(covered2_raw, args.coverage_average_length)

        b = args.coverage_average_burnin
        covered1 = covered1[b:]
        covered2 = covered2[b:]

        y = y[b:]
        datetimes = datetimes[b:]
        low1, high1 = low1[b:], high1[b:]
        low2, high2 = low2[b:], high2[b:]

        datetimes, y, low1, high1, low2, high2, covered1, covered2 = align_all(
            datetimes, y, low1, high1, low2, high2, covered1, covered2
        )

        sets1 = [low1, high1]
        sets2 = [low2, high2]

        time_series1 = pd.Series(covered1, index=datetimes)
        time_series2 = pd.Series(covered2, index=datetimes)


        sizes1 = high1 - low1
        sizes2 = high2 - low2

        # miscoverage mask
        mask1 = (y < low1) | (y > high1)
        mask2 = (y < low2) | (y > high2)

        df_list_for_table.append(pd.DataFrame({
            'Model type': model_name,
            'Method': method_title_map[args.key1],
            'Marginal coverage': float(np.mean((y >= low1) & (y <= high1))),
            'Longest err sequence': int(longest_true_sequence(mask1)),
            # 'Average set size': float(np.mean(sizes1)) if not np.any(np.isinf(sizes1)) else 'Inf',
            'Average set size': float(np.mean(sizes1)) if not np.any(np.isinf(sizes1)) else np.inf,
            'Median set size': float(np.median(np.nan_to_num(sizes1, nan=np.inf))),
            '75% quantile set size': float(np.quantile(np.nan_to_num(sizes1, nan=np.inf), 0.75)),
            '90% quantile set size': float(np.quantile(np.nan_to_num(sizes1, nan=np.inf), 0.90)),
            '95% quantile set size': float(np.quantile(np.nan_to_num(sizes1, nan=np.inf), 0.95)),
        }, index=[0]))

        df_list_for_table.append(pd.DataFrame({
            'Model type': model_name,
            'Method': method_title_map[args.key2],
            'Marginal coverage': float(np.mean((y >= low2) & (y <= high2))),
            'Longest err sequence': int(longest_true_sequence(mask2)),
            # 'Average set size': float(np.mean(sizes2)) if not np.any(np.isinf(sizes2)) else 'Inf',
            'Average set size': float(np.mean(sizes2)) if not np.any(np.isinf(sizes2)) else np.inf,
            'Median set size': float(np.median(np.nan_to_num(sizes2, nan=np.inf))),
            '75% quantile set size': float(np.quantile(np.nan_to_num(sizes2, nan=np.inf), 0.75)),
            '90% quantile set size': float(np.quantile(np.nan_to_num(sizes2, nan=np.inf), 0.90)),
            '95% quantile set size': float(np.quantile(np.nan_to_num(sizes2, nan=np.inf), 0.95)),
        }, index=[0]))

        # plot
        window_start = args.window_start
        window_end = args.window_start + args.window_length

        savename = (
            datasetname + '_' + model_name + '_' +
            args.key1 + '_lr' + str(args.lr1) + '_' +
            args.key2 + '_lr' + str(args.lr2) + '_window' +
            str(args.window_length) + '_start' + str(args.window_start) +
            str(args.coverage_inset) + str(args.set_inset)
        )

        plot_everything(
            [time_series1, time_series2],
            [sets1, sets2],
            [method_title_map[args.key1], method_title_map[args.key2]],
            y,
            alpha,
            window_start,
            window_end,
            args.window_loc,
            args.coverage_inset,
            args.set_inset,
            args.miscoverage_scatterplot,
            savename,
            model_name,
            datetimes=datetimes
        )


    # if len(df_list_for_table) == 0:
    #     print("WARNING: df_list_for_table is empty; skip latex table.")
    #     sys.exit(0)

    df = pd.concat(df_list_for_table, ignore_index=True)
    latex_table = dataframe_to_latex(df)

    os.makedirs('./plots/1v1/', exist_ok=True)
    out_tex = './plots/1v1/' + datasetname + "_" + f"{args.key1}_lr{args.lr1}_{args.key2}_lr{args.lr2}" + '.tex'
    with open(out_tex, 'w') as f:
        f.write(latex_table)

    # print("Wrote latex table:", out_tex)