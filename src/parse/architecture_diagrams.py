# Architecture diagrams for the stimulus-change models (DNN / LSTM / Transformer),
# drawn in the style of the "Attention Is All You Need" figure. Layer sizes are read
# from the saved model configs so the diagrams stay in sync with what is trained.
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
CONFIG_DIR = f"{ROOT_DIR}/exported_features/stimulus_change/config"
OUTPUT_DIR = f"{ROOT_DIR}/data/feature_tables/sch/plots/architecture"

COLORS = {
    "input": "#f9dfe3",      # inputs / embeddings
    "linear": "#e3e1f7",     # linear layers / MLPs
    "core": "#fde0c0",       # the sequence-mixing block (attention or LSTM)
    "ff": "#cfe8f7",         # feed forward / activation
    "norm": "#f1f5c4",       # add & norm / batch norm
    "dropout": "#ececec",
    "softmax": "#d9f0d3",
    "container": "#f3f3f3",
    "key": "#fbfbfb",
    "edge": "#333333",
    "mask_on": "#e8a25c",
    "mask_off": "#ffffff",
}
FONT = 10
BOX_W = 4.6
ACTIVATION_NAMES = {"relu": "ReLU", "tanh": "Tanh", "gelu": "GELU", "leaky_relu": "LeakyReLU"}
CX = 5.0

# shared vertical layout so the three models line up when drawn side by side;
# gaps between boxes are kept >= ~0.45 so every arrow stays visible
Y = {
    "key_top": -0.35, "input_text": 0.35, "concat": 1.3, "input_mlp": 2.5, "projection": 3.7,
    "block_bottom": 4.4, "block_top": 9.4,
    "output_mlp": 10.3, "linear": 11.4, "softmax": 12.4, "output_text": 13.15, "title": 14.3,
    "bottom": -3.6,
}


def load_config(name):
    with open(f"{CONFIG_DIR}/{name}_config.json") as f:
        return json.load(f)


def sizes(layers):
    return ", ".join(str(h) for h in layers)


def mlp_definition(name, in_dim, layers, ops):
    # one key entry: what a named MLP box is made of, with every width
    repeat = f"{len(layers)} × " if len(layers) > 1 else ""
    widths = " → ".join(str(d) for d in [in_dim] + list(layers))
    return f"{name}  ({widths})", f"{repeat}[{' → '.join(ops)}]"


def box(ax, cy, h, text, color, cx=CX, w=BOX_W, fontsize=FONT):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=COLORS["edge"], lw=1.2, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, linespacing=1.2, zorder=4)
    return {"cx": cx, "cy": cy, "w": w, "h": h,
            "top": cy + h / 2, "bottom": cy - h / 2, "left": cx - w / 2, "right": cx + w / 2}


def arrow(ax, start, end, color=COLORS["edge"], lw=1.3, ls='-', connection=None):
    kwargs = {"connectionstyle": connection} if connection else {}
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=11, lw=lw,
                                 color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=2, **kwargs))


def line(ax, points, color=COLORS["edge"], lw=1.3, ls='-'):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2, solid_capstyle='round')


def routed_arrow(ax, points):
    # polyline whose final segment carries the arrow head
    line(ax, points[:-1])
    arrow(ax, points[-2], points[-1])


def container(ax, label, x0=1.95, x1=8.3):
    ax.add_patch(FancyBboxPatch((x0, Y["block_bottom"]), x1 - x0, Y["block_top"] - Y["block_bottom"],
                                boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc=COLORS["container"], ec=COLORS["edge"], lw=1.4, zorder=1))
    ax.text(x0 - 0.25, (Y["block_bottom"] + Y["block_top"]) / 2, label,
            ha='right', va='center', fontsize=FONT + 1)


def residual(ax, from_y, into):
    # skip connection: leaves the main path below a sub-block and re-enters at the Add & Norm box
    x_side = into["left"] - 0.45
    routed_arrow(ax, [(CX, from_y), (x_side, from_y), (x_side, into["cy"]), (into["left"], into["cy"])])


def concat_node(ax, cy, embed_dim):
    # the positional embedding is concatenated with the features, not added
    ax.add_patch(Circle((CX, cy), 0.22, fc='white', ec=COLORS["edge"], lw=1.3, zorder=3))
    ax.text(CX, cy - 0.01, r"$\Vert$", ha='center', va='center', fontsize=FONT + 1, zorder=4)
    pos = box(ax, cy, 0.75, f"Learned Positional\nEmbedding ({embed_dim})", COLORS["input"], cx=8.45, w=2.9,
              fontsize=FONT - 1.5)
    arrow(ax, (pos["left"], cy), (CX + 0.22, cy))
    ax.text(CX - 0.35, cy, "concat", ha='right', va='center', fontsize=FONT - 2, style='italic')


def causal_mask_icon(ax, x0, cy, n=4, cell=0.13):
    y0 = cy - n * cell / 2
    for row in range(n):
        for col in range(n):
            visible = col <= row
            ax.add_patch(Rectangle((x0 + col * cell, y0 + (n - 1 - row) * cell), cell, cell,
                                   fc=COLORS["mask_on"] if visible else COLORS["mask_off"],
                                   ec=COLORS["edge"], lw=0.5, zorder=3))
    ax.text(x0 + n * cell / 2, y0 - 0.12, "causal\nmask", ha='center', va='top', fontsize=FONT - 3)


def input_label(ax, text):
    ax.text(CX, Y["input_text"], text, ha='center', va='center', fontsize=FONT)


def mlp_key(ax, entries):
    # separate box under the diagram spelling out what each MLP box contains
    line_h, pad = 0.36, 0.25
    n_lines = 2 * len(entries)
    height = n_lines * line_h + 2 * pad
    x0, x1 = 0.45, 9.55
    top = Y["key_top"]
    ax.add_patch(FancyBboxPatch((x0, top - height), x1 - x0, height,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=COLORS["key"], ec="#999999", lw=1.0, ls='--', zorder=1))
    y = top - pad + line_h / 2
    for name, recipe in entries:
        y -= line_h
        ax.text(x0 + 0.25, y, name, ha='left', va='center', fontsize=FONT - 1.5, fontweight='bold')
        y -= line_h
        ax.text(x0 + 0.55, y, recipe, ha='left', va='center', fontsize=FONT - 1.5)


def output_head(ax, post_fc_layers, output_size, below_top, condition):
    mlp = box(ax, Y["output_mlp"], 0.7, f"Output MLP ({sizes(post_fc_layers)})", COLORS["linear"])
    arrow(ax, (CX, below_top), (CX, mlp["bottom"]))
    linear = box(ax, Y["linear"], 0.5, f"Linear (→ {output_size})", COLORS["linear"])
    arrow(ax, (CX, mlp["top"]), (CX, linear["bottom"]))
    softmax = box(ax, Y["softmax"], 0.5, "Softmax", COLORS["softmax"])
    arrow(ax, (CX, linear["top"]), (CX, softmax["bottom"]))
    arrow(ax, (CX, softmax["top"]), (CX, Y["output_text"] - 0.1))
    ax.text(CX, Y["output_text"], f"Next stimulus decision\nP(ŷ$_t$ | {condition})",
            ha='center', va='bottom', fontsize=FONT)


def setup_axes(ax, title=None):
    ax.set_xlim(0, 10)
    ax.set_ylim(Y["bottom"], Y["title"] + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.text(CX, Y["title"], title, ha='center', va='bottom', fontsize=FONT + 4, fontweight='bold')


def draw_dnn(ax, config, title=None):
    setup_axes(ax, title)
    input_label(ax, f"x$_t$: {config['input_size']} features of the current waveform")
    pre = box(ax, Y["input_mlp"], 0.7, f"Input MLP ({sizes(config['pre_fc_layers'])})", COLORS["linear"])
    arrow(ax, (CX, Y["input_text"] + 0.25), (CX, pre["bottom"]))

    act = ACTIVATION_NAMES[config['activation']]
    hidden = config["hidden_layers"]
    container(ax, f"{len(hidden)}×\n({sizes(hidden)})")
    steps = [("Linear", COLORS["linear"]), ("BatchNorm", COLORS["norm"]),
             (act, COLORS["ff"]), (f"Dropout ({config['dropout']})", COLORS["dropout"])]
    previous_top = pre["top"]
    for i, (text, color) in enumerate(steps):
        b = box(ax, 5.0 + i * 1.2, 0.55, text, color)
        arrow(ax, (CX, previous_top), (CX, b["bottom"]))
        previous_top = b["top"]

    output_head(ax, config["post_fc_layers"], config["output_size"], previous_top, "x$_t$")

    # every DNN layer (input, hidden and output) is the same Linear → BN → act → Dropout block
    ops = ["Linear", "BatchNorm", act, f"Dropout {config['dropout']}"]
    mlp_key(ax, [
        mlp_definition("Input MLP", config["input_size"], config["pre_fc_layers"], ops),
        mlp_definition("Output MLP", hidden[-1], config["post_fc_layers"], ops),
    ])


def draw_lstm(ax, config, title=None):
    setup_axes(ax, title)
    input_label(ax, f"x$_t$: {config['input_size']} features per waveform, t = 1 … T")
    arrow(ax, (CX, Y["input_text"] + 0.25), (CX, Y["concat"] - 0.22))
    concat_node(ax, Y["concat"], config["pos_embedding_dim"])
    concat_dim = config["input_size"] + config["pos_embedding_dim"]

    container(ax, f"{config['lstm_layers']}×")
    cell = box(ax, 6.2, 1.4, f"LSTM Layer\n(hidden {config['lstm_hidden_size']})", COLORS["core"], w=3.0)
    arrow(ax, (CX, Y["concat"] + 0.22), (CX, cell["bottom"]))
    ax.text(CX + 0.15, 3.2, f"{concat_dim}", ha='left', va='center', fontsize=FONT - 2, style='italic')

    # recurrent state carried from the previous waveform
    arrow(ax, (cell["right"], cell["cy"] + 0.35), (cell["right"], cell["cy"] - 0.35),
          connection="arc3,rad=-1.6")
    ax.text(cell["right"] + 0.7, cell["cy"], "h$_{t-1}$\nc$_{t-1}$", ha='left', va='center',
            fontsize=FONT - 1)

    drop = box(ax, 8.5, 0.5, f"Dropout ({config['dropout']})", COLORS["dropout"])
    arrow(ax, (CX, cell["top"]), (CX, drop["bottom"]))

    output_head(ax, config["post_fc_layers"], config["output_size"], drop["top"], "x$_1$ … x$_t$")

    ops = ["Linear", ACTIVATION_NAMES[config['activation']], f"Dropout {config['dropout']}"]
    mlp_key(ax, [mlp_definition("Output MLP", config["lstm_hidden_size"], config["post_fc_layers"], ops)])


def draw_transformer(ax, config, title=None):
    setup_axes(ax, title)
    input_label(ax, f"x$_t$: {config['input_size']} features per waveform, t = 1 … T")
    arrow(ax, (CX, Y["input_text"] + 0.25), (CX, Y["concat"] - 0.22))
    concat_node(ax, Y["concat"], config["pos_embedding_dim"])

    pre = box(ax, Y["input_mlp"], 0.7, f"Input MLP ({sizes(config['pre_fc_layers'])})", COLORS["linear"])
    arrow(ax, (CX, Y["concat"] + 0.22), (CX, pre["bottom"]))
    proj = box(ax, Y["projection"], 0.5, f"Linear (→ {config['transformer_dim']})", COLORS["linear"])
    arrow(ax, (CX, pre["top"]), (CX, proj["bottom"]))

    container(ax, f"{config['transformer_layers']}×")
    attention = box(ax, 5.6, 0.9, f"Masked Multi-Head\nSelf-Attention ({config['transformer_heads']} heads)",
                    COLORS["core"])
    # query / key / value branches, as in the original figure
    fork_y = 4.85
    line(ax, [(CX, proj["top"]), (CX, fork_y)])
    line(ax, [(CX - 0.8, fork_y), (CX + 0.8, fork_y)])
    for dx in (-0.8, 0.0, 0.8):
        arrow(ax, (CX + dx, fork_y), (CX + dx, attention["bottom"]))
    causal_mask_icon(ax, attention["right"] + 0.12, attention["cy"])

    norm1 = box(ax, 6.7, 0.42, "Add & Norm", COLORS["norm"])
    arrow(ax, (CX, attention["top"]), (CX, norm1["bottom"]))
    residual(ax, 4.62, norm1)

    act = ACTIVATION_NAMES[config['activation']]
    ff = box(ax, 7.75, 0.75, f"Feed Forward\n({config['transformer_ff']}, {act})", COLORS["ff"])
    arrow(ax, (CX, norm1["top"]), (CX, ff["bottom"]))
    norm2 = box(ax, 8.8, 0.42, "Add & Norm", COLORS["norm"])
    arrow(ax, (CX, ff["top"]), (CX, norm2["bottom"]))
    residual(ax, (norm1["top"] + ff["bottom"]) / 2, norm2)

    output_head(ax, config["post_fc_layers"], config["output_size"], norm2["top"], "x$_1$ … x$_t$")

    # the feed-forward is nn.TransformerEncoderLayer's own: Linear → act → Dropout → Linear
    ops = ["Linear", act, f"Dropout {config['dropout']}"]
    d, ff_dim = config["transformer_dim"], config["transformer_ff"]
    mlp_key(ax, [
        mlp_definition("Input MLP", config["input_size"] + config["pos_embedding_dim"],
                       config["pre_fc_layers"], ops),
        (f"Feed Forward  ({d} → {ff_dim} → {d})", f"Linear → {act} → Dropout {config['dropout']} → Linear"),
        mlp_definition("Output MLP", d, config["post_fc_layers"], ops),
    ])


def draw_history_panel(ax, kind, n_steps=5, focus=4):
    # unrolled view over waveforms 1..n_steps showing what each model can use for its decision
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.0, 5.6)
    ax.set_aspect('equal')
    ax.axis('off')
    titles = {"dnn": "DNN", "lstm": "LSTM", "transformer": "Transformer"}
    captions = {
        "dnn": "Each waveform is classified on its own",
        "lstm": "Earlier waveforms reach step t only\nthrough the hidden state",
        "transformer": "Step t attends directly to every earlier\nwaveform; later ones are masked",
    }
    core_text = {"dnn": "DNN", "lstm": "LSTM", "transformer": "Attn"}
    ax.text(5, 5.35, titles[kind], ha='center', va='bottom', fontsize=FONT + 3, fontweight='bold')
    ax.text(5, -0.55, captions[kind], ha='center', va='top', fontsize=FONT - 0.5)

    xs = [1 + 2 * i for i in range(n_steps)]
    inputs, cores, outputs = [], [], []
    for t, x in enumerate(xs, start=1):
        inputs.append(box(ax, 0.6, 0.6, f"x$_{t}$", COLORS["input"], cx=x, w=1.2))
        cores.append(box(ax, 2.55, 0.8, core_text[kind], COLORS["core"], cx=x, w=1.3))
        outputs.append(box(ax, 4.5, 0.6, f"ŷ$_{t}$", COLORS["softmax"], cx=x, w=1.2))
        arrow(ax, (x, cores[-1]["top"]), (x, outputs[-1]["bottom"]))

    if kind == "dnn":
        for inp, core in zip(inputs, cores):
            arrow(ax, (inp["cx"], inp["top"]), (core["cx"], core["bottom"]))
    elif kind == "lstm":
        for inp, core in zip(inputs, cores):
            arrow(ax, (inp["cx"], inp["top"]), (core["cx"], core["bottom"]))
        for left, right in zip(cores[:-1], cores[1:]):
            arrow(ax, (left["right"], left["cy"]), (right["left"], right["cy"]))
            ax.text((left["right"] + right["left"]) / 2, left["cy"] + 0.12, "h", ha='center', va='bottom',
                    fontsize=FONT - 1, style='italic')
    else:
        target = cores[focus - 1]
        for t, (inp, core) in enumerate(zip(inputs, cores), start=1):
            if t != focus:
                arrow(ax, (inp["cx"], inp["top"]), (core["cx"], core["bottom"]), color="#bbbbbb", lw=1.0)
        for t, inp in enumerate(inputs, start=1):
            visible = t <= focus
            arrow(ax, (inp["cx"], inp["top"]), (target["cx"] + (t - focus) * 0.12, target["bottom"]),
                  color=COLORS["mask_on"] if visible else "#bbbbbb", lw=1.6 if visible else 1.0,
                  ls='-' if visible else (0, (3, 3)))


def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUTPUT_DIR}/{name}.{ext}", dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def export_architecture_diagrams():
    configs = {name: load_config(name) for name in ("dnn", "lstm", "transformer")}
    drawers = {"dnn": draw_dnn, "lstm": draw_lstm, "transformer": draw_transformer}
    titles = {"dnn": "DNN", "lstm": "LSTM", "transformer": "Transformer"}

    # one figure per model
    for name, draw in drawers.items():
        fig, ax = plt.subplots(figsize=(5.2, 9.2))
        draw(ax, configs[name])
        save(fig, f"{name}_architecture")

    # all three side by side; shared parts sit at the same height so the swapped core stands out
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 10))
    for ax, (name, draw) in zip(axes, drawers.items()):
        draw(ax, configs[name], title=titles[name])
    fig.subplots_adjust(wspace=0.05)
    save(fig, "architecture_comparison")

    # unrolled over time: what history each model can use
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, kind in zip(axes, ("dnn", "lstm", "transformer")):
        draw_history_panel(ax, kind)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "sequence_history_comparison")


if __name__ == "__main__":
    export_architecture_diagrams()
