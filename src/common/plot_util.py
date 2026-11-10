import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import confusion_matrix
from src.common.data_util import *
import seaborn as sns


def _build_bins(data, bins_interval, x_bound, centered):
    if x_bound is not None and len(x_bound) == 3:
        b_min, b_step, b_max = x_bound
        start = b_min - (b_step / 2) if centered else b_min
        return np.arange(start, b_max + b_step, b_step), b_step
    if x_bound is not None and len(x_bound) == 2:
        b_min, b_max = x_bound
        start = b_min - (bins_interval / 2) if centered else b_min
        return np.arange(start, b_max + bins_interval, bins_interval), bins_interval
    if centered:
        return np.arange(min(data) - (bins_interval / 2),
                         max(data) + bins_interval + (bins_interval / 2),
                         bins_interval), bins_interval
    return np.arange(min(data) - bins_interval,
                     max(data) + bins_interval,
                     bins_interval), bins_interval
def _render_histogram(ax, spec, fontsize):
    data = spec["data"]
    bins_interval = spec.get("bins_interval", 10)
    x_bound = spec.get("x_bound", None)
    centered = spec.get("centered", False)
    explicit_x_ticks = spec.get("explicit_x_ticks", False)
    x_tick_interval = spec.get("x_tick_interval", None)
    log_x_labels = spec.get("log_x_labels", False)   # <-- new
    log_y = spec.get("log_y", False)

    bins, step = _build_bins(data, bins_interval, x_bound, centered)
    ax.hist(data, bins=bins, edgecolor='black', color='steelblue', log=log_y)

    if x_bound is not None and len(x_bound) == 2:
        ax.set_xlim(x_bound[0], x_bound[1])
    else:
        ax.set_xlim(bins[0], bins[-1])

    if x_tick_interval is not None:
        b_min = x_bound[0] if x_bound else bins[0]
        b_max = x_bound[1] if (x_bound and len(x_bound) == 2) else bins[-1]
        tick_positions = np.arange(b_min, b_max + x_tick_interval, x_tick_interval)
        ax.set_xticks(tick_positions)
        if log_x_labels:
            ax.set_xticklabels([f"$10^{{{int(t)}}}$" for t in tick_positions], fontsize=fontsize * 0.85)
        else:
            ax.set_xticklabels([str(int(t)) for t in tick_positions], fontsize=fontsize * 0.85)
    elif explicit_x_ticks:
        if centered:
            bin_centers = bins[:-1] + step / 2
            labels = [int(c) if c == int(c) else c for c in bin_centers]
            ax.set_xticks(bin_centers)
            ax.set_xticklabels(labels, fontsize=fontsize * 0.85)
        else:
            ax.set_xticks(bins)

    # log axis can't start at 0; 0.8 keeps single-count bins visible
    ax.set_ylim(bottom=0.8 if log_y else 0)



def _render_bar(ax, spec, fontsize):
    data = spec["data"]
    x_categories = [str(k) for k in data.keys()]
    y_values = list(data.values())
    color = spec.get("color", "steelblue")
    labels_to_hide = spec.get("x_labels_to_hide", [])
    skip_every_n = spec.get("skip_every_n", None)
    log_y = spec.get("log_y", False)
    ylim = spec.get("ylim", None)

    ax.bar(x_categories, y_values, color=color, edgecolor='black', log=log_y)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # log axis can't start at 0; 0.8 keeps single-count bars visible
        ax.set_ylim(bottom=0.8 if log_y else 0)

    if labels_to_hide:
        display = [label if label not in labels_to_hide else ""
                   for label in x_categories]
        ax.set_xticklabels(display, fontsize=fontsize * 0.85)
    elif skip_every_n is not None:
        display = [label if i % skip_every_n == 0 else ""
                   for i, label in enumerate(x_categories)]
        ax.set_xticklabels(display, fontsize=fontsize * 0.85)

def _render_grouped_bar(ax, spec, fontsize):

    groups = spec["groups"]
    metrics = spec["metrics"]
    width = spec.get("bar_width", 0.25)

    x = np.arange(len(groups))
    group_labels = [g["label"] for g in groups]

    for j, metric in enumerate(metrics):
        ys = [g[metric["key"]] for g in groups]
        offset = (j - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, ys, width,
               label=metric["label"],
               color=metric["color"],
               edgecolor="black",
               hatch=metric.get("hatch", ""),
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=fontsize * 0.85)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    if spec.get("show_legend", False):
        ax.legend(loc=spec.get("legend_loc", "upper left"),
                  fontsize=fontsize * 0.8, framealpha=0.9)


def _render_line(ax, spec, fontsize):

    for s in spec["series"]:
        ax.plot(s["x"], s["y"],
                label=s.get("label", ""),
                marker=s.get("marker", "o"),
                linestyle=s.get("ls", "-"),
                color=s.get("color", None),
                markersize=s.get("markersize", 7),
                linewidth=s.get("linewidth", 2))

    ax.set_xlim(spec.get("xlim", (-0.05, 1.05)))
    ax.set_ylim(spec.get("ylim", (0, 105)))

    if "xticks" in spec:
        ax.set_xticks(spec["xticks"])
        ax.set_xticklabels([str(t) for t in spec["xticks"]], fontsize=fontsize * 0.85)

    ax.grid(True, linestyle='--', alpha=0.6)

    if spec.get("show_legend", False):
        ax.legend(loc=spec.get("legend_loc", "lower right"),
                  fontsize=fontsize * 0.8, framealpha=0.9)


def _render_heatmap(ax, spec, fontsize, cbar_ax=None):
    cm = spec["cm"]
    display_labels = spec.get("display_labels", [])
    vmin = spec.get("vmin", 0)
    vmax = spec.get("vmax", cm.max())
    show_x = spec.get("show_xticklabels", True)
    show_y = spec.get("show_yticklabels", True)
    show_cbar = spec.get("show_cbar", False)

    sns.heatmap(cm, ax=ax, cmap="Blues", vmin=vmin, vmax=vmax,
                annot=True, fmt="d",
                cbar=show_cbar,
                cbar_ax=cbar_ax if show_cbar else None,
                annot_kws={"size": spec.get("annot_size", 10)},
                xticklabels=display_labels if show_x else False,
                yticklabels=display_labels if show_y else False)

    if show_x:
        ax.set_xticklabels(display_labels, fontsize=fontsize * 0.85,
                           rotation=45, ha='right', rotation_mode='anchor')
    else:
        ax.set_xticklabels([])

    if show_y:
        ax.set_yticklabels(display_labels, fontsize=fontsize * 0.85,
                           rotation=0, va='center')
    else:
        ax.set_yticklabels([])
def _render_waterfall(ax, spec, fontsize):
    # traces are stacked top-to-bottom in the order given, each shifted down by `offset`
    traces = spec["traces"]
    offset = spec["offset"]
    color = spec.get("color", "steelblue")

    for i, trace in enumerate(traces):
        shift = -i * offset
        x = trace["x"]
        ax.fill_between(x, trace["lower"] + shift, trace["upper"] + shift,
                        color=color, alpha=0.25, linewidth=0)
        ax.plot(x, trace["mean"] + shift, color=color, linewidth=2)
        ax.text(x[-1] + (x[-1] - x[0]) * 0.02, trace["mean"][-1] + shift,
                trace['label'],
                fontsize=fontsize * 0.85, va='center', ha='left')

    if "xlim" in spec:
        ax.set_xlim(spec["xlim"])
    if "ylim" in spec:
        ax.set_ylim(spec["ylim"])

    # amplitude scale bar replaces the y axis, since stacked offsets make tick values meaningless
    scale = spec.get("scale_bar")
    if scale is not None:
        x0, x1 = ax.get_xlim()
        y0, _ = ax.get_ylim()
        bar_x = x0 + (x1 - x0) * 0.03
        bar_y = y0 + offset * 0.15
        ax.plot([bar_x, bar_x], [bar_y, bar_y + scale], color='black', linewidth=2.5)
        ax.text(bar_x + (x1 - x0) * 0.015, bar_y + scale / 2, f"{scale:g} µV",
                fontsize=fontsize * 0.8, va='center', ha='left')

    ax.set_yticks([])
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    if "panel_label" in spec:
        ax.text(0.03, 0.98, spec["panel_label"], transform=ax.transAxes,
                fontsize=fontsize * 1.15, fontweight='bold', ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="black", alpha=0.85))

    if spec.get("band_label"):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        ax.legend(handles=[Line2D([0], [0], color=color, linewidth=2, label="Grand average"),
                           Patch(facecolor=color, alpha=0.25, label=spec["band_label"])],
                  loc="upper right", fontsize=fontsize * 0.8, framealpha=0.9)

def _render_sequence_waterfall(ax, spec, fontsize):
    # one row per tested level (highest at the top); repeats at the same level are overlaid
    traces = spec["traces"]
    offset = spec["offset"]
    color = spec.get("color", "steelblue")
    threshold_color = spec.get("threshold_color", "#d62728")

    levels = sorted({t["db"] for t in traces}, reverse=True)
    level_shift = {db: -i * offset for i, db in enumerate(levels)}

    for trace in traces:
        x, y = trace["x"], trace["y"] + level_shift[trace["db"]]
        trace_color = threshold_color if trace.get("is_threshold") else color
        ax.plot(x, y, color=trace_color, linewidth=1.6, alpha=0.9)

        latency = trace.get("latency_5")
        if latency is not None and not np.isnan(latency) and x[0] <= latency <= x[-1]:
            ax.plot(latency, np.interp(latency, x, y) + offset * 0.18, marker='v',
                    color='black', markersize=fontsize * 0.45, linestyle='none')

    x_end = traces[0]["x"][-1]
    x_span = traces[0]["x"][-1] - traces[0]["x"][0]
    for db in levels:
        level_traces = [t for t in traces if t["db"] == db]
        label = f"{db:g} dB" + (f" (×{len(level_traces)})" if len(level_traces) > 1 else "")
        has_threshold = any(t.get("is_threshold") for t in level_traces)
        ax.text(x_end + x_span * 0.02, level_traces[-1]["y"][-1] + level_shift[db], label,
                fontsize=fontsize * 0.8, va='center', ha='left',
                color=threshold_color if has_threshold else 'black',
                fontweight='bold' if has_threshold else 'normal')

    if "xlim" in spec:
        ax.set_xlim(spec["xlim"])
    if "ylim" in spec:
        ax.set_ylim(spec["ylim"])

    scale = spec.get("scale_bar")
    if scale is not None:
        x0, x1 = ax.get_xlim()
        y0, _ = ax.get_ylim()
        bar_x = x0 + (x1 - x0) * 0.03
        bar_y = y0 + offset * 0.1
        ax.plot([bar_x, bar_x], [bar_y, bar_y + scale], color='black', linewidth=2.5)
        ax.text(bar_x + (x1 - x0) * 0.015, bar_y + scale / 2, f"{scale:g} µV",
                fontsize=fontsize * 0.75, va='center', ha='left')

    ax.set_yticks([])
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    if spec.get("show_legend", False):
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], color=color, linewidth=2, label="Waveform"),
                           Line2D([0], [0], color=threshold_color, linewidth=2, label="Threshold waveform"),
                           Line2D([0], [0], marker='v', color='black', linestyle='none',
                                  label="Clinician-marked wave V")],
                  loc="lower right", fontsize=fontsize * 0.7, framealpha=0.9)

def create_panel(panel_spec, save_path=None, figsize=None, fontsize=14,
                 shared_cbar=False, cbar_label="",
                 dpi=300, save_padding=0.2,
                 shared_x_axis=False,
                 hspace=None, wspace=None,
                 row_labels=None,
                 row_label_loc="upper left"):     # ← new params

    rows = len(panel_spec)
    cols = max(len(row) for row in panel_spec)

    if figsize is None:
        figsize = (6 * cols, 4.5 * rows)

    if hspace is None:
        hspace = 0.35 if not shared_x_axis else 0.20
    if wspace is None:
        wspace = 0.30

    if shared_cbar:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(rows, cols + 1,
                      width_ratios=[1] * cols + [0.04],
                      wspace=wspace, hspace=hspace,
                      left=0.08, right=0.97, top=0.95, bottom=0.12)
        cbar_ax = fig.add_subplot(gs[:, -1])
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(rows)])
    else:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        cbar_ax = None
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

    # maps a location string to (x, y, ha, va) in axes-fraction coordinates
    loc_map = {
        "upper left":  (0.03, 0.95, "left",  "top"),
        "upper right": (0.97, 0.95, "right", "top"),
        "lower left":  (0.03, 0.05, "left",  "bottom"),
        "lower right": (0.97, 0.05, "right", "bottom"),
    }

    for r, row in enumerate(panel_spec):
        for c, spec in enumerate(row):
            ax = axes[r][c]
            plot_type = spec.get("type")

            is_last = (r == len(panel_spec) - 1 and c == len(row) - 1)
            is_bottom_row = (r == rows - 1)
            this_cbar_ax = cbar_ax if (shared_cbar and is_last) else None

            if plot_type == "histogram":
                _render_histogram(ax, spec, fontsize)
            elif plot_type == "bar":
                _render_bar(ax, spec, fontsize)
            elif plot_type == "grouped_bar":
                _render_grouped_bar(ax, spec, fontsize)
            elif plot_type == "line":
                _render_line(ax, spec, fontsize)
            elif plot_type == "heatmap":
                _render_heatmap(ax, spec, fontsize, cbar_ax=this_cbar_ax)
            elif plot_type == "waterfall":
                _render_waterfall(ax, spec, fontsize)
            elif plot_type == "sequence_waterfall":
                _render_sequence_waterfall(ax, spec, fontsize)

            if shared_x_axis and not is_bottom_row:
                ax.set_xlabel("", fontsize=fontsize, labelpad=10)
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(spec.get("x_label", ""), fontsize=fontsize, labelpad=10)
                ax.tick_params(axis='x', labelsize=fontsize * 0.85)

            ax.set_ylabel(spec.get("y_label", ""), fontsize=fontsize, labelpad=10)
            ax.tick_params(axis='y', labelsize=fontsize * 0.85)

        for c in range(len(row), cols):
            axes[r][c].set_visible(False)

        # ── new: row label as an in-panel text box on the leftmost axis ──
        if row_labels is not None and r < len(row_labels):
            x, y, ha, va = loc_map.get(row_label_loc, loc_map["upper left"])
            axes[r][0].text(
                x, y, row_labels[r],
                transform=axes[r][0].transAxes,
                fontsize=fontsize * 1.15, fontweight='bold',
                ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="black", alpha=0.85)
            )

    if shared_cbar and cbar_ax is not None:
        cbar_ax.tick_params(labelsize=fontsize * 0.8)
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        cbar_ax.yaxis.set_major_formatter(formatter)
        if cbar_label:
            cbar_ax.set_ylabel(cbar_label, fontsize=fontsize)

    if not shared_cbar:
        fig.tight_layout()
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=save_padding)
    plt.show()

def custom_sort_key(label):
    s = str(label).lower().strip()
    if s == "<-30":  return -1000
    elif s == ">20": return 1000
    elif s == "stop": return 2000
    else:
        try:    return float(s)
        except: return 3000


def create_freq_graph(values, save_path=None):
    value_counts = values.value_counts()
    sorted_labels = sorted(value_counts.index.tolist(), key=custom_sort_key)
    value_counts = value_counts.reindex(sorted_labels)
    create_panel([[{"type": "bar", "data": value_counts.to_dict(),
                    "x_label": "Stimulus decisions [dB or termination]",
                    "y_label": "Occurrences"}]],
                 save_path=save_path, figsize=(10, 7), fontsize=18)


def create_bar_chart(dictionary, x_label, y_label, title, save_path=None):
    create_panel([[{"type": "bar", "data": dictionary,
                    "x_label": x_label, "y_label": y_label}]],
                 save_path=save_path, figsize=(15, 10), fontsize=36)


def create_histogram(list_data, bins_interval, x_label, y_label, title,
                     explicit_x_ticks=False, save_path=None, x_bound=None,
                     centered=False, x_tick_interval=None, log_x_labels=False):
    if len(list_data) == 0:
        print(f"Graph for {title} had empty list, skipping...")
        return
    create_panel([[{"type": "histogram", "data": list_data,
                    "bins_interval": bins_interval, "x_bound": x_bound,
                    "centered": centered, "explicit_x_ticks": explicit_x_ticks,
                    "x_tick_interval": x_tick_interval,
                    "log_x_labels": log_x_labels,
                    "x_label": x_label, "y_label": y_label}]],
                 save_path=save_path, figsize=(15, 10), fontsize=36)

def create_joint_confusion_matrix(model_dict, reverse_mapping_path, save_path, indices_to_blank=None):
    if indices_to_blank is None:
        indices_to_blank = [2, 4, 6, 8, 10]

    with open(reverse_mapping_path, "r") as f:
        reverse_map = {int(k): v for k, v in json.load(f).items()}

    dfs = {}
    all_true_labels = set()
    for title, csv_path in model_dict.items():
        df = pd.read_csv(csv_path)
        dfs[title] = df
        all_true_labels.update(df['true'].tolist())

    int_labels = sorted(list(all_true_labels))
    display_labels = [reverse_map.get(i, str(i)) for i in int_labels]
    for idx in indices_to_blank:
        if idx < len(display_labels):
            display_labels[idx] = ""

    cms, titles = [], []
    for title, df in dfs.items():
        cms.append(confusion_matrix(df['true'], df['predictions'], labels=int_labels))
        titles.append(title)

    num_models = len(cms)
    max_cols = 3
    cols = min(max_cols, num_models)
    rows = (num_models + cols - 1) // cols
    n_labels = len(display_labels)
    fontsize = 24

    # size each matrix so it's square: each cell ~0.28in, plus margins for labels
    cell_size = 0.48
    matrix_inch = n_labels * cell_size          # square matrix side length
    y_label_margin = 1.8      # left margin for y tick labels
    x_label_margin = 2.0      # bottom margin for rotated x tick labels (now needed on every row)
    shared_label_margin = 0.6  # extra space for the shared "Predicted Label" text
    cbar_width = 0.5
    col_gap = 0.15            # tighter column gap
    row_gap_extra = 1.2       # extra vertical room since every row now carries x tick labels

    fig_width  = cols * matrix_inch + y_label_margin + cbar_width + col_gap * (cols - 1)
    fig_height = rows * matrix_inch + x_label_margin + shared_label_margin + row_gap_extra * (rows - 1)

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(
        rows, cols + 1,
        width_ratios=[matrix_inch] * cols + [cbar_width * 0.4],
        wspace=col_gap / matrix_inch,
        hspace=(row_gap_extra + x_label_margin * 0.3) / matrix_inch,   # widened to fit per-row x labels
        left=y_label_margin / fig_width,
        right=1.0 - (cbar_width / fig_width),
        top=0.99,
        bottom=(x_label_margin + shared_label_margin) / fig_height
    )

    cbar_ax = fig.add_subplot(gs[:, -1])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(rows)])

    # force square axes
    for ax_row in axes:
        for ax in ax_row:
            ax.set_aspect('equal')

    vmin = 0
    vmax = max(cm.max() for cm in cms) / 5

    flat_idx = 0
    for r in range(rows):
        for c in range(cols):
            if flat_idx >= num_models:
                axes[r][c].set_visible(False)
                flat_idx += 1
                continue

            ax = axes[r][c]
            is_first_col = (c == 0)
            show_cbar = (flat_idx == num_models - 1)

            sns.heatmap(
                cms[flat_idx], ax=ax, cmap="Blues",
                vmin=vmin, vmax=vmax,
                annot=True, fmt="d",
                cbar=show_cbar,
                cbar_ax=cbar_ax if show_cbar else None,
                annot_kws={"size": 13},
                xticklabels=display_labels,
                yticklabels=display_labels if is_first_col else False
            )

            # every row now shows its own x tick labels
            ax.set_xticklabels(
                display_labels, fontsize=fontsize,
                rotation=45, ha='right', rotation_mode='anchor'
            )
            ax.set_xlabel("")   # per-axis label removed — shared label added below instead

            if is_first_col:
                ax.set_yticklabels(display_labels, fontsize=fontsize, rotation=0, va='center')
                ax.set_ylabel("True Label", fontsize=fontsize, labelpad=12)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

            flat_idx += 1
    middle_col_idx = cols // 2
    middle_col_left = y_label_margin + middle_col_idx * (matrix_inch + col_gap)
    middle_col_center = middle_col_left + matrix_inch / 2
    plot_area_center = middle_col_center / fig_width

    fig.text(plot_area_center, 0.02, "Predicted Label",
             ha='center', va='bottom', fontsize=fontsize)
    cbar_ax.tick_params(labelsize=fontsize)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    cbar_ax.yaxis.set_major_formatter(formatter)

    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.show()


def create_joint_sequence_accuracy_plot(model_dict, save_path):
    line_metrics = [
        {"key": "accuracy",       "label": "Top-1 Accuracy",  "marker": "o", "ls": "-",  "color": "#1f77b4"},
        {"key": "top3_acc",       "label": "Top-3 Accuracy",  "marker": "s", "ls": "--", "color": "#ff7f0e"},
        {"key": "pm10db_correct", "label": "±10 dB Accuracy", "marker": "D", "ls": ":",  "color": "#2ca02c"},
    ]

    panel_spec = []
    items = list(model_dict.items())
    max_cols = 2
    cols = min(max_cols, len(items))

    for row_start in range(0, len(items), cols):
        row = []
        for model_name, json_path in items[row_start:row_start + cols]:
            with open(json_path, 'r') as f:
                data = json.load(f)

            points = sorted(
                [{"x": float(k.strip('%')) / 100.0, **{m["key"]: v.get(m["key"], 0) for m in line_metrics}}
                 for k, v in data.items() if k.endswith('%')],
                key=lambda d: d["x"]
            )
            xs = [p["x"] for p in points]

            is_first = (row_start == 0 and len(row) == 0)
            row.append({
                "type": "line",
                "series": [{"x": xs, "y": [p[m["key"]] for p in points],
                             "label": m["label"], "marker": m["marker"],
                             "ls": m["ls"], "color": m["color"]} for m in line_metrics],
                "xlim": (-0.05, 1.05), "ylim": (0, 105),
                "xticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "show_legend": is_first,
                "legend_loc": "lower right",
                "x_label": "Waveform position in sequence",
                "y_label": "Accuracy" if len(row) == 0 else "",
            })
        panel_spec.append(row)

    create_panel(panel_spec, save_path=save_path,
                 figsize=(6 * cols, 5 * len(panel_spec)),
                 fontsize=14, dpi=300)


def create_joint_freq_accuracy_plot(model_dict, save_path):
    bar_metrics = [
        {"key": "accuracy",       "label": "Top-1 Accuracy",  "color": "#1f77b4", "hatch": ""},
        {"key": "top3_acc",       "label": "Top-3 Accuracy",  "color": "#ff7f0e", "hatch": "///"},
        {"key": "pm10db_correct", "label": "±10 dB Accuracy", "color": "#2ca02c", "hatch": "..."},
    ]

    panel_spec = []
    items = list(model_dict.items())
    max_cols = 3
    cols = min(max_cols, len(items))

    for row_start in range(0, len(items), cols):
        row = []
        for i, (model_name, json_path) in enumerate(items[row_start:row_start + cols]):
            with open(json_path, 'r') as f:
                data = json.load(f)

            groups = sorted(
                [{"label": f"{int(k.replace('Hz','').strip()) / 1000:g}",
                  **{m["key"]: v.get(m["key"], 0) for m in bar_metrics}}
                 for k, v in data.items() if "Hz" in k],
                key=lambda d: float(d["label"])
            )

            is_first = (row_start == 0 and i == 0)
            row.append({
                "type": "grouped_bar",
                "groups": groups,
                "metrics": bar_metrics,
                "bar_width": 0.25,
                "show_legend": is_first,
                "legend_loc": "upper left",
                "x_label": "Frequency [kHz]",
                "y_label": "Accuracy [%]" if i == 0 else "",
            })
        panel_spec.append(row)

    create_panel(panel_spec, save_path=save_path,
                 figsize=(6 * cols, 5 * len(panel_spec)),
                 fontsize=24, dpi=300, row_labels=['SCH', 'IHDD'], row_label_loc="upper right")

def hist_plot(cross_vals):
    bins = np.arange(-1, 1.1, 0.1)
    plt.xticks(np.arange(-1, 1.1, 0.1))
    plt.xlim(-1, 1)
    plt.hist(cross_vals, bins=bins)
    plt.show()


def plot_waveform(epoch, waveforms, record_id=None, decibel=None, freq=None, ear=None,
                  features=None, y_bounds=None, x_bounds=None, amplitude_5=None, latency_5=None):
    if x_bounds is not None:
        bound_1 = int((x_bounds[0] - WAVEFORM_START_MS) / (epoch / len(waveforms[0])))
        bound_2 = int((x_bounds[1] - WAVEFORM_START_MS) / (epoch / len(waveforms[0])))
        epoch_values = np.linspace(x_bounds[0], x_bounds[1], len(waveforms[0][bound_1:bound_2]))
    else:
        epoch_values = waveform_times(epoch, len(waveforms[0]))
    raw_data = np.concatenate(waveforms).ravel().astype(float)
    y_min, y_max = float(np.min(raw_data)), float(np.max(raw_data))

    if features is not None:
        for i, feature_value in enumerate(features):
            if feature_value != 0:
                y_val = feature_value.item()
                x_val = (epoch / 8) * (i + 1)
                plt.plot(x_val, y_val, 'ro')
                y_min = min(y_min, y_val)
                y_max = max(y_max, y_val)

    for waveform in waveforms:
        if x_bounds is not None:
            bound_1 = int((x_bounds[0] - WAVEFORM_START_MS) / (epoch / len(waveforms[0])))
            bound_2 = int((x_bounds[1] - WAVEFORM_START_MS) / (epoch / len(waveforms[0])))
            waveform = waveform[bound_1:bound_2]
        plt.plot(epoch_values, np.array(waveform, dtype=float))

    if y_bounds is not None:
        plt.ylim(y_bounds[0], y_bounds[1])
    else:
        plt.ylim(y_min - 0.2, y_max + 0.2)

    if latency_5 is not None and amplitude_5 is not None:
        try:
            lat_5, amp_5 = float(latency_5), float(amplitude_5)
            if not np.isnan(lat_5) and not np.isnan(amp_5):
                plt.vlines(lat_5, y_min, y_max, colors='black', linestyles='--', linewidth=2, zorder=5)
        except (ValueError, TypeError):
            pass

    plt.title(f"Record {record_id}, {decibel} dB, {freq} Hz, {ear} Ear")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


def plot_record(record_id, record_df, waveform_df, features=None):
    waveform = [get_waveform(record_id, waveform_df)]
    record = record_df[record_df['id'] == record_id]
    plot_waveform(record['epoch'].iloc[0], waveform, record_id,
                  record['db'].iloc[0], record['freq'].iloc[0], record['ear'].iloc[0],
                  features, amplitude_5=record['amplitude_5'].iloc[0],
                  latency_5=record['latency_5'].iloc[0])


def plot_txt(path):
    with open(path, "r") as f:
        waveform = list(map(lambda x: float(x.strip()), f.readlines()))
        plot_waveform(26.67, np.array(waveform), None, None, None, None, None)


def plot_pca_waveforms(data_set, reconstructed_waveforms, index):
    original = data_set[index]
    reconstructed = reconstructed_waveforms[index]
    rmsd = np.sqrt(np.mean((original - reconstructed) ** 2))
    mad = np.mean(np.abs(original - reconstructed))
    plt.figure(figsize=(10, 6))
    plt.plot(original, label='Original Waveform', linestyle='--')
    plt.plot(reconstructed, label='Reconstructed Waveform', linestyle='-')
    plt.legend()
    plt.title("Original vs Reconstructed Waveform")
    plt.text(0.8, 0.06, f'RMSD: {rmsd:.3f}', transform=plt.gca().transAxes, fontsize=12, color='blue')
    plt.text(0.8, 0.02, f'MAD: {mad:.3f}', transform=plt.gca().transAxes, fontsize=12, color='green')
    plt.show()


def plot_cm(cm, title, x_title, y_title, tick_label, save_path, should_save, holdout_validation_cm=False):
    if cm is None:
        print("No CM was provided - returning!")
        return
    display_labels = list(tick_label)
    for idx in [2, 4, 6, 8, 10]:
        if idx < len(display_labels):
            display_labels[idx] = ""
    create_panel([[{
        "type": "heatmap",
        "cm": cm,
        "display_labels": display_labels,
        "vmin": 0, "vmax": 1000,
        "show_cbar": True,
        "show_xticklabels": True,
        "show_yticklabels": True,
        "x_label": x_title,
        "y_label": y_title,
    }]], figsize=(12, 10), fontsize=18,
        save_path=save_path if should_save else None,
        shared_cbar=True)


def waterfall_plot_sequence(seq):
    plt.figure(figsize=(10, 10))
    clean_waves = [np.array(item[0], dtype=float) for item in seq]
    max_wave_height = max(np.ptp(w) for w in clean_waves) if clean_waves else 1
    base_gap = max_wave_height * 0.4
    db_gap_factor = max_wave_height * 0.05
    colors = plt.cm.viridis(np.linspace(0, 1, len(seq)))
    current_shift, previous_db = 0.0, None

    for i, waveform_pts in enumerate(clean_waves):
        epoch = float(seq[i][1])
        try:    amp_5, lat_5 = float(seq[i][2]), float(seq[i][3])
        except: amp_5, lat_5 = np.nan, np.nan
        try:    db_val = float(seq[i][4])
        except: db_val = 0.0

        if i > 0:
            current_shift -= base_gap + abs(db_val - previous_db) * db_gap_factor
        previous_db = db_val

        time_axis = waveform_times(epoch, len(waveform_pts))
        shifted = waveform_pts + current_shift
        plt.plot(time_axis, shifted, color=colors[i], linewidth=1.5)

        if not np.isnan(amp_5) and not np.isnan(lat_5):
            mh = base_gap * 0.5
            plt.vlines(lat_5, (amp_5 + current_shift) - mh, (amp_5 + current_shift) + mh,
                       colors='black', linestyles=':', linewidth=2)

        plt.text(epoch + epoch * 0.02, shifted[-1], f"{int(db_val)} dB",
                 color='black', fontsize=10, verticalalignment='center')

    plt.title("Clinical ABR Sequence")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.yticks([])
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def waterfall_plot_sequence_non_sequential(seq, start_time=0.0):
    print("Printing Record IDs for Waveforms...")
    plt.figure(figsize=(10, 10))
    clean_waves = [np.array(item[0], dtype=float) for item in seq]
    max_wave_height = max(np.ptp(w) for w in clean_waves) if clean_waves else 1
    db_scale_factor = (max_wave_height * 1.5) / 10.0
    intra_group_offset = max_wave_height * 0.2
    colors = plt.cm.viridis(np.linspace(0, 1, len(seq)))
    db_counts = {}

    for i, waveform_pts in enumerate(clean_waves):
        print(f"Waveform Record ID: {seq[i][5]}, dB: {seq[i][4]}")
        duration = float(seq[i][1])
        time_axis = np.linspace(start_time, start_time + duration, len(waveform_pts))
        try:    amp_5, lat_5 = float(seq[i][2]), float(seq[i][3])
        except: amp_5, lat_5 = np.nan, np.nan
        try:    db_val = float(seq[i][4])
        except: db_val = 0.0

        base_y = db_val * db_scale_factor
        count = db_counts.get(db_val, 0)
        db_counts[db_val] = count + 1
        shifted = waveform_pts + base_y - count * intra_group_offset

        plt.plot(time_axis, shifted, color=colors[i], linewidth=1.5, zorder=db_val)

        if not np.isnan(amp_5) and not np.isnan(lat_5):
            mh = max_wave_height * 0.3
            plt.vlines(lat_5, (amp_5 + base_y - count * intra_group_offset) - mh,
                       (amp_5 + base_y - count * intra_group_offset) + mh,
                       colors='black', linestyles=':', linewidth=2, zorder=db_val + 1)

        plt.text(time_axis[-1] + duration * 0.02, shifted[-1],
                 f"{int(db_val)} dB", color='black', fontsize=12, verticalalignment='center')

    plt.xlabel("Latency (ms)", fontsize=20)
    plt.ylabel("Amplitude", fontsize=20)
    plt.tick_params(axis='both', labelsize=16)
    plt.yticks([])
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_binary_confusion_matrix(csv_path, save_path=None, title="Binary Confusion Matrix"):
    df = pd.read_csv(csv_path)
    cm = confusion_matrix(df['true'].astype(int), df['predictions'].astype(int), labels=[0, 1])
    create_panel([[{
        "type": "heatmap",
        "cm": cm,
        "display_labels": ["False", "True"],
        "vmin": 0, "vmax": cm.max(),
        "show_cbar": True,
        "show_xticklabels": True,
        "show_yticklabels": True,
        "annot_size": 22,
        "x_label": "Predicted Label",
        "y_label": "True Label",
    }]], figsize=(10, 9), fontsize=22,
        save_path=save_path, shared_cbar=True)


def label_counts(label_ids):
    vc = label_ids.value_counts()
    sorted_keys = sorted(vc.index.tolist(), key=custom_sort_key)
    return vc.reindex(sorted_keys).to_dict()

def create_dataset_comparison_panel(sch_seq_lengths, uw_seq_lengths,
                                     sch_post_click_dbs, uw_post_click_dbs,
                                     sch_label_ids, uw_label_ids,
                                     sch_averages, uw_averages,
                                     save_path=None, figsize=(12, 7), fontsize=10):
    panel_spec = [
        [
            {"type": "histogram", "data": sch_seq_lengths,
             "bins_interval": 2, "x_bound": [0, 28], "x_tick_interval": 4,
             "x_label": "Number of Waveforms", "y_label": "Number of Sequences"},

            {"type": "histogram", "data": sch_post_click_dbs,
             "bins_interval": 10, "centered": True, "x_bound": [-40, 120], "x_tick_interval": 20,
             "x_label": "Starting Level [dB HL]", "y_label": "Occurrences"},

            {"type": "bar", "data": label_counts(sch_label_ids),
             "x_label": "Stimulus decisions [dB or termination]", "y_label": "Occurrences",
             "x_labels_to_hide": ["-25", "-15", "-5", "5", "15", "25", "<-30", ">20"]},

            {"type": "histogram", "data": sch_averages,
             "bins_interval": 100, "x_bound": [0, 4000],
             "x_label": "Number of Averages", "y_label": "Number of Occurrences"},
        ],
        [
            {"type": "histogram", "data": uw_seq_lengths,
             "bins_interval": 2, "x_bound": [0, 28], "x_tick_interval": 4,
             "x_label": "Number of Waveforms", "y_label": "Number of Sequences"},

            {"type": "histogram", "data": uw_post_click_dbs,
             "bins_interval": 10, "centered": True, "x_bound": [-40, 120], "x_tick_interval": 20,
             "x_label": "Starting Level [dB HL]", "y_label": "Occurrences"},

            {"type": "bar", "data": label_counts(uw_label_ids),
             "x_label": "Stimulus decisions [dB or termination]", "y_label": "Occurrences",
             "x_labels_to_hide": ["-25", "-15", "-5", "5", "15", "25", "<-30", ">20"]},

            {"type": "histogram", "data": uw_averages,
             "bins_interval": 100, "x_bound": [0, 4000],
             "x_label": "Number of Averages", "y_label": "Number of Occurrences"},
        ],
    ]

    create_panel(panel_spec, save_path=save_path, figsize=figsize, fontsize=fontsize,
                 save_padding=1.4, shared_x_axis=False,
                 row_labels=["SCH", "IHDD"], row_label_loc="upper right")

def create_filter_comparison_panel(sch_low_filters, uw_low_filters,
                                   sch_high_filters, uw_high_filters,
                                   save_path=None, figsize=(16, 12), fontsize=20):
    # low_filter = high-pass cutoff, high_filter = low-pass cutoff (Bio-logic naming)
    # each column shares one category set (zeros included) so SCH/IHDD bars line up
    low_categories = sorted(set(sch_low_filters) | set(uw_low_filters))
    high_categories = sorted(set(sch_high_filters) | set(uw_high_filters))

    def counts(values, categories):
        vc = pd.Series(values).value_counts()
        return {int(c): int(vc.get(c, 0)) for c in categories}

    panels = [
        [counts(sch_low_filters, low_categories), counts(sch_high_filters, high_categories)],
        [counts(uw_low_filters, low_categories), counts(uw_high_filters, high_categories)],
    ]

    # one log-scale y range for all four panels so bar heights are directly comparable
    max_count = max(v for row in panels for d in row for v in d.values())
    ylim = (0.8, 10 ** math.ceil(math.log10(max_count)))

    x_labels = ["Low Filter (High-Pass Cutoff) [Hz]", "High Filter (Low-Pass Cutoff) [Hz]"]
    panel_spec = [
        [{"type": "bar", "data": data, "log_y": True, "ylim": ylim,
          "x_label": x_labels[c], "y_label": "Number of Records"}
         for c, data in enumerate(row)]
        for row in panels
    ]

    create_panel(panel_spec, save_path=save_path, figsize=figsize, fontsize=fontsize,
                 row_labels=["SCH", "IHDD"], row_label_loc="upper right")

def create_waveform_waterfall_panel(dataset_traces, save_path=None, band_label="± 1 SD",
                                    xlim=(0, 15), figsize=(18, 10), fontsize=20, show_legend=False):
    # dataset_traces: {panel name: [{"label", "n", "x", "mean", "lower", "upper"}, ...]}
    all_traces = [t for traces in dataset_traces.values() for t in traces]

    # shared spacing and y range so both datasets are drawn on the same amplitude scale
    offset = max(np.max(t["upper"]) - np.min(t["lower"]) for t in all_traces) * 0.75
    n_levels = max(len(traces) for traces in dataset_traces.values())
    top = max(np.max(t["upper"]) for t in all_traces)
    bottom = min(np.min(t["lower"]) for t in all_traces) - (n_levels - 1) * offset
    pad = offset * 0.25
    ylim = (bottom - pad, top + pad * 2)

    # round the scale bar to a readable value around a third of the spacing
    scale_bar = float(f"{offset / 3:.1g}")

    panel_spec = [[
        {"type": "waterfall", "traces": traces, "offset": offset,
         "xlim": xlim, "ylim": ylim, "scale_bar": scale_bar,
         "panel_label": name,
         "band_label": band_label if show_legend and i == 0 else None,
         "x_label": "Time [ms]", "y_label": ""}
        for i, (name, traces) in enumerate(dataset_traces.items())
    ]]

    create_panel(panel_spec, save_path=save_path, figsize=figsize, fontsize=fontsize, wspace=0.35)

def create_sequence_waterfall_panel(rows, row_labels, save_path=None,
                                    xlim=(0, 15), figsize=(24, 16), fontsize=20):
    # rows: [[{"traces": [{"db", "x", "y", "latency_5", "is_threshold"}], "threshold"}, ...], ...]
    all_traces = [t for row in rows for seq in row for t in seq["traces"]]

    # shared level spacing and y range across every panel so amplitudes are comparable
    offset = np.percentile([np.ptp(t["y"]) for t in all_traces], 90) * 0.9
    n_levels = max(len({t["db"] for t in seq["traces"]}) for row in rows for seq in row)
    top = max(np.max(t["y"]) for t in all_traces)
    ylim = (-(n_levels - 1) * offset - offset * 0.9, top + offset * 0.5)
    scale_bar = float(f"{offset / 2:.1g}")

    panel_spec = [
        [{"type": "sequence_waterfall", "traces": seq["traces"], "threshold": seq["threshold"],
          "offset": offset, "xlim": xlim, "ylim": ylim, "scale_bar": scale_bar,
          "show_legend": (r == len(rows) - 1 and c == len(row) - 1),
          "x_label": "Time [ms]" if r == len(rows) - 1 else "", "y_label": ""}
         for c, seq in enumerate(row)]
        for r, row in enumerate(rows)
    ]

    create_panel(panel_spec, save_path=save_path, figsize=figsize, fontsize=fontsize,
                 wspace=0.3, hspace=0.25, row_labels=row_labels, row_label_loc="upper left")