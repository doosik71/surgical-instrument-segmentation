"""Generate an HTML model summary with an embedded SVG architecture diagram."""

from __future__ import annotations

import argparse
import html
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import FlexibleUNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import AppSettings


DEFAULT_INPUT_SHAPE = (1, 3, 480, 736)


@dataclass(slots=True)
class ModuleExecution:
    """Runtime summary for one executed module."""

    name: str
    module_type: str
    depth: int
    input_shape: str
    output_shape: str
    direct_params: int
    total_params: int


@dataclass(slots=True)
class TreeNode:
    """Module hierarchy node used for HTML rendering."""

    name: str
    path: str
    module_type: str = ""
    depth: int = 0
    input_shape: str = "-"
    output_shape: str = "-"
    direct_params: int = 0
    total_params: int = 0
    children: list["TreeNode"] = field(default_factory=list)


def build_model() -> FlexibleUNet:
    """Create the exact architecture used by the application."""
    model = FlexibleUNet(
        in_channels=3,
        out_channels=2,
        backbone="efficientnet-b2",
        spatial_dims=2,
        pretrained=False,
        is_pad=False,
        pre_conv=None,
    )
    model.eval()
    return model


def try_load_weights(model: torch.nn.Module, model_path: Path) -> str:
    """Load local weights if available, otherwise keep the random initialization."""
    if not model_path.exists():
        return f"Skipped: weights not found at {model_path}"

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return f"Loaded weights from {model_path}"


def tensor_shape_repr(value: Any) -> str:
    """Convert tensor-like runtime values into a compact shape string."""
    if isinstance(value, torch.Tensor):
        return "x".join(str(dim) for dim in value.shape)
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(tensor_shape_repr(item) for item in value)
        bracket_left, bracket_right = ("[", "]") if isinstance(value, list) else ("(", ")")
        return f"{bracket_left}{inner}{bracket_right}"
    if isinstance(value, dict):
        inner = ", ".join(f"{key}: {tensor_shape_repr(item)}" for key, item in value.items())
        return "{" + inner + "}"
    return type(value).__name__


def format_count(value: int) -> str:
    """Return a human-friendly integer with separators."""
    return f"{value:,}"


def collect_module_executions(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
) -> OrderedDict[str, ModuleExecution]:
    """Run one dummy forward pass and record input/output shapes per module."""
    executions: OrderedDict[str, ModuleExecution] = OrderedDict()
    named_modules = dict(model.named_modules())
    handles: list[Any] = []

    for name, module in named_modules.items():
        if not name:
            continue

        def hook(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            *,
            module_name: str = name,
        ) -> None:
            if module_name in executions:
                return

            target_module = named_modules[module_name]
            executions[module_name] = ModuleExecution(
                name=module_name,
                module_type=type(target_module).__name__,
                depth=module_name.count(".") + 1,
                input_shape=tensor_shape_repr(inputs),
                output_shape=tensor_shape_repr(output),
                direct_params=sum(p.numel() for p in target_module.parameters(recurse=False)),
                total_params=sum(p.numel() for p in target_module.parameters()),
            )

        handles.append(module.register_forward_hook(hook))

    sample = torch.randn(*input_shape)
    with torch.no_grad():
        model(sample)

    for handle in handles:
        handle.remove()

    return executions


def build_module_tree(
    model: torch.nn.Module,
    executions: OrderedDict[str, ModuleExecution],
) -> TreeNode:
    """Merge static module hierarchy with runtime shape metadata."""
    root = TreeNode(name="model", path="", module_type=type(model).__name__, total_params=sum(p.numel() for p in model.parameters()))
    lookup: dict[str, TreeNode] = {"": root}

    for name, module in model.named_modules():
        if not name:
            continue

        parts = name.split(".")
        parent_path = ".".join(parts[:-1])
        node = TreeNode(
            name=parts[-1],
            path=name,
            module_type=type(module).__name__,
            depth=len(parts),
            direct_params=sum(p.numel() for p in module.parameters(recurse=False)),
            total_params=sum(p.numel() for p in module.parameters()),
        )
        execution = executions.get(name)
        if execution is not None:
            node.input_shape = execution.input_shape
            node.output_shape = execution.output_shape

        lookup[name] = node
        lookup[parent_path].children.append(node)

    return root


def render_tree_html(node: TreeNode) -> str:
    """Render the module hierarchy as nested HTML details blocks."""
    if node.path:
        title = (
            f"<span class='tree-name'>{html.escape(node.path)}</span>"
            f"<span class='tree-meta'>{html.escape(node.module_type)} | "
            f"in {html.escape(node.input_shape)} | out {html.escape(node.output_shape)} | "
            f"params {format_count(node.total_params)}</span>"
        )
    else:
        title = (
            f"<span class='tree-name'>{html.escape(node.module_type)}</span>"
            f"<span class='tree-meta'>params {format_count(node.total_params)}</span>"
        )

    if not node.children:
        return f"<div class='tree-leaf'>{title}</div>"

    children_html = "".join(render_tree_html(child) for child in node.children)
    return f"<details class='tree-node' open><summary>{title}</summary>{children_html}</details>"


def render_table_rows(records: list[ModuleExecution]) -> str:
    """Render module execution rows for an HTML table."""
    rows: list[str] = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(record.name)}</td>"
            f"<td>{html.escape(record.module_type)}</td>"
            f"<td>{record.depth}</td>"
            f"<td>{html.escape(record.input_shape)}</td>"
            f"<td>{html.escape(record.output_shape)}</td>"
            f"<td>{format_count(record.direct_params)}</td>"
            f"<td>{format_count(record.total_params)}</td>"
            "</tr>"
        )
    return "".join(rows)


def get_stage_record(executions: OrderedDict[str, ModuleExecution], name: str) -> ModuleExecution:
    """Return a required stage record."""
    try:
        return executions[name]
    except KeyError as exc:
        raise KeyError(f"Missing runtime record for stage '{name}'") from exc


def stage_label(title: str, record: ModuleExecution) -> str:
    """Return a concise multi-line label for the SVG node."""
    return f"{title}\n{record.module_type}\nout {record.output_shape}"


def svg_text(x: float, y: float, lines: list[str], css_class: str = "node-title") -> str:
    """Render multi-line SVG text using tspans."""
    rendered = [f"<text x='{x}' y='{y}' class='{css_class}'>"]
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else "1.2em"
        rendered.append(f"<tspan x='{x}' dy='{dy}'>{html.escape(line)}</tspan>")
    rendered.append("</text>")
    return "".join(rendered)


def render_network_svg(executions: OrderedDict[str, ModuleExecution]) -> str:
    """Render a U-Net style architecture diagram."""
    stage_positions = {
        "input": (80, 270, 170, 86, "input"),
        "stem": (300, 270, 170, 86, "stem"),
        "encoder._blocks.0": (520, 70, 170, 86, "encoder"),
        "encoder._blocks.1": (520, 170, 170, 86, "encoder"),
        "encoder._blocks.2": (520, 270, 170, 86, "encoder"),
        "encoder._blocks.3": (520, 370, 170, 86, "encoder"),
        "encoder._blocks.4": (520, 470, 170, 86, "encoder"),
        "encoder._blocks.5": (520, 570, 170, 86, "encoder"),
        "encoder._blocks.6": (740, 570, 170, 86, "bottleneck"),
        "decoder.blocks.0": (960, 470, 190, 86, "decoder"),
        "decoder.blocks.1": (960, 370, 190, 86, "decoder"),
        "decoder.blocks.2": (960, 270, 190, 86, "decoder"),
        "decoder.blocks.3": (960, 170, 190, 86, "decoder"),
        "decoder.blocks.4": (960, 70, 190, 86, "decoder"),
        "segmentation_head.0": (1200, 70, 180, 86, "head"),
        "output": (1430, 70, 180, 86, "output"),
    }

    stage_labels = {
        "input": ["Input", "Tensor", "out 1x3x480x736"],
        "stem": [
            "Stem",
            "Conv2d + BN + Swish",
            f"out {get_stage_record(executions, 'encoder._bn0').output_shape}",
        ],
        "encoder._blocks.0": ["Encoder Block 0", "Sequential", f"out {get_stage_record(executions, 'encoder._blocks.0').output_shape}"],
        "encoder._blocks.1": ["Encoder Block 1", "Sequential", f"out {get_stage_record(executions, 'encoder._blocks.1').output_shape}"],
        "encoder._blocks.2": ["Encoder Block 2", "Sequential", f"out {get_stage_record(executions, 'encoder._blocks.2').output_shape}"],
        "encoder._blocks.3": ["Encoder Block 3", "Sequential", f"out {get_stage_record(executions, 'encoder._blocks.3').output_shape}"],
        "encoder._blocks.4": ["Encoder Block 4", "Sequential", f"out {get_stage_record(executions, 'encoder._blocks.4').output_shape}"],
        "encoder._blocks.5": ["Encoder Block 5", "Sequential", f"out {get_stage_record(executions, 'encoder._blocks.5').output_shape}"],
        "encoder._blocks.6": ["Encoder Block 6", "Sequential", f"out {get_stage_record(executions, 'encoder._blocks.6').output_shape}"],
        "decoder.blocks.0": [
            "Decoder Block 0",
            "UpSample + Concat + TwoConv",
            f"out {get_stage_record(executions, 'decoder.blocks.0').output_shape}",
        ],
        "decoder.blocks.1": [
            "Decoder Block 1",
            "UpSample + Concat + TwoConv",
            f"out {get_stage_record(executions, 'decoder.blocks.1').output_shape}",
        ],
        "decoder.blocks.2": [
            "Decoder Block 2",
            "UpSample + Concat + TwoConv",
            f"out {get_stage_record(executions, 'decoder.blocks.2').output_shape}",
        ],
        "decoder.blocks.3": [
            "Decoder Block 3",
            "UpSample + Concat + TwoConv",
            f"out {get_stage_record(executions, 'decoder.blocks.3').output_shape}",
        ],
        "decoder.blocks.4": [
            "Decoder Block 4",
            "UpSample + TwoConv",
            f"out {get_stage_record(executions, 'decoder.blocks.4').output_shape}",
        ],
        "segmentation_head.0": [
            "Segmentation Head",
            "1x1 Conv2d",
            f"out {get_stage_record(executions, 'segmentation_head.0').output_shape}",
        ],
        "output": ["Output", "Foreground logits", "out 1x2x480x736"],
    }

    main_flow = [
        ("input", "stem"),
        ("stem", "encoder._blocks.0"),
        ("encoder._blocks.0", "encoder._blocks.1"),
        ("encoder._blocks.1", "encoder._blocks.2"),
        ("encoder._blocks.2", "encoder._blocks.3"),
        ("encoder._blocks.3", "encoder._blocks.4"),
        ("encoder._blocks.4", "encoder._blocks.5"),
        ("encoder._blocks.5", "encoder._blocks.6"),
        ("encoder._blocks.6", "decoder.blocks.0"),
        ("decoder.blocks.0", "decoder.blocks.1"),
        ("decoder.blocks.1", "decoder.blocks.2"),
        ("decoder.blocks.2", "decoder.blocks.3"),
        ("decoder.blocks.3", "decoder.blocks.4"),
        ("decoder.blocks.4", "segmentation_head.0"),
        ("segmentation_head.0", "output"),
    ]

    skip_edges = [
        ("encoder._blocks.4", "decoder.blocks.0", "skip 120ch"),
        ("encoder._blocks.2", "decoder.blocks.1", "skip 48ch"),
        ("encoder._blocks.1", "decoder.blocks.2", "skip 24ch"),
        ("encoder._blocks.0", "decoder.blocks.3", "skip 16ch"),
    ]

    svg_parts = [
        "<svg viewBox='0 0 1660 760' role='img' aria-label='FlexibleUNet architecture summary'>",
        "<defs>",
        "<marker id='arrow' markerWidth='10' markerHeight='10' refX='9' refY='3' orient='auto'>",
        "<path d='M0,0 L0,6 L9,3 z' fill='#2f4858' />",
        "</marker>",
        "<marker id='arrow-soft' markerWidth='10' markerHeight='10' refX='9' refY='3' orient='auto'>",
        "<path d='M0,0 L0,6 L9,3 z' fill='#bc6c25' />",
        "</marker>",
        "</defs>",
        "<rect x='0' y='0' width='1660' height='760' rx='28' class='svg-bg' />",
        "<text x='56' y='52' class='svg-title'>MONAI FlexibleUNet with EfficientNet-B2 encoder</text>",
        "<text x='56' y='80' class='svg-subtitle'>Solid arrows show the main forward path. Dashed arrows show encoder-to-decoder skip connections.</text>",
    ]

    for start, end in main_flow:
        x1, y1, w1, h1, _ = stage_positions[start]
        x2, y2, _, h2, _ = stage_positions[end]
        svg_parts.append(
            "<path class='edge-main' "
            f"d='M{x1 + w1} {y1 + h1 / 2} C {x1 + w1 + 50} {y1 + h1 / 2}, {x2 - 50} {y2 + h2 / 2}, {x2} {y2 + h2 / 2}' />"
        )

    for start, end, label in skip_edges:
        x1, y1, w1, h1, _ = stage_positions[start]
        x2, y2, _, h2, _ = stage_positions[end]
        mid_x = (x1 + w1 + x2) / 2
        mid_y = min(y1, y2) - 22
        svg_parts.append(
            "<path class='edge-skip' "
            f"d='M{x1 + w1} {y1 + h1 / 2} C {mid_x} {y1 + h1 / 2}, {mid_x} {y2 + h2 / 2}, {x2} {y2 + h2 / 2}' />"
        )
        svg_parts.append(f"<text x='{mid_x - 28}' y='{mid_y}' class='edge-label'>{html.escape(label)}</text>")

    for key, (x, y, w, h, tone) in stage_positions.items():
        svg_parts.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='20' class='node node-{tone}' />")
        svg_parts.append(svg_text(x + 18, y + 28, stage_labels[key]))

    decoder_notes = [
        ("decoder.blocks.0", "352 up + 120 skip -> 472 in -> 256 out"),
        ("decoder.blocks.1", "256 up + 48 skip -> 304 in -> 128 out"),
        ("decoder.blocks.2", "128 up + 24 skip -> 152 in -> 64 out"),
        ("decoder.blocks.3", "64 up + 16 skip -> 80 in -> 32 out"),
        ("decoder.blocks.4", "32 up -> 32 in -> 16 out"),
    ]
    for stage_name, label in decoder_notes:
        x, y, _, _, _ = stage_positions[stage_name]
        svg_parts.append(f"<text x='{x + 10}' y='{y + 104}' class='decoder-note'>{html.escape(label)}</text>")

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def render_stage_rows(executions: OrderedDict[str, ModuleExecution]) -> str:
    """Render the main stage summary table."""
    rows: list[str] = []
    descriptions = {
        "encoder._conv_stem": "Input stem convolution after padding.",
        "encoder._blocks.0": "First EfficientNet stage; first retained skip feature.",
        "encoder._blocks.1": "Downsamples to 120x184 and widens to 24 channels.",
        "encoder._blocks.2": "Downsamples to 60x92; third skip source.",
        "encoder._blocks.3": "Intermediate encoder refinement at 30x46.",
        "encoder._blocks.4": "Selected skip source for decoder block 0.",
        "encoder._blocks.5": "Downsamples to 15x23 before bottleneck refinement.",
        "encoder._blocks.6": "Deepest encoder stage and decoder input.",
        "decoder.blocks.0": "Upsample bottleneck, concatenate 120-channel skip, then TwoConv.",
        "decoder.blocks.1": "Upsample and fuse 48-channel skip feature.",
        "decoder.blocks.2": "Upsample and fuse 24-channel skip feature.",
        "decoder.blocks.3": "Upsample and fuse 16-channel skip feature.",
        "decoder.blocks.4": "Final upsample without skip connection.",
        "segmentation_head.0": "1x1 convolution that produces 2-class logits.",
    }

    order = list(descriptions)
    for name in order:
        record = get_stage_record(executions, name)
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(record.module_type)}</td>"
            f"<td>{html.escape(record.input_shape)}</td>"
            f"<td>{html.escape(record.output_shape)}</td>"
            f"<td>{format_count(record.total_params)}</td>"
            f"<td>{html.escape(descriptions[name])}</td>"
            "</tr>"
        )
    return "".join(rows)


def generate_html(
    model: torch.nn.Module,
    executions: OrderedDict[str, ModuleExecution],
    output_path: Path,
    weights_status: str,
    input_shape: tuple[int, int, int, int],
) -> str:
    """Build the final self-contained HTML report."""
    module_tree = build_module_tree(model, executions)
    all_records = list(executions.values())
    svg_markup = render_network_svg(executions)
    total_params = sum(p.numel() for p in model.parameters())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Model Summary</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe6;
      --panel: #fffaf2;
      --ink: #1f2a30;
      --muted: #59656f;
      --line: #dbcdb4;
      --accent: #bc6c25;
      --encoder: #5fa8d3;
      --decoder: #7fb069;
      --head: #e76f51;
      --input: #6d597a;
      --bottleneck: #264653;
      --output: #ef476f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      background:
        radial-gradient(circle at top left, #fff6df 0, transparent 24rem),
        linear-gradient(180deg, #f7f3ea 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 32px 24px 60px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    p {{
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.55;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      margin-bottom: 24px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .stat, .panel {{
      background: rgba(255, 250, 242, 0.88);
      border: 1px solid rgba(219, 205, 180, 0.9);
      border-radius: 20px;
      padding: 18px 20px;
      box-shadow: 0 14px 30px rgba(46, 62, 72, 0.06);
      backdrop-filter: blur(6px);
    }}
    .stat-label {{
      font-size: 0.9rem;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .stat-value {{
      font-size: 1.25rem;
      font-weight: 700;
      word-break: break-word;
    }}
    .diagram-panel {{
      overflow-x: auto;
      padding: 12px;
    }}
    svg {{
      width: 100%;
      min-width: 1480px;
      height: auto;
      display: block;
    }}
    .svg-bg {{
      fill: #fffef8;
      stroke: #d7c4a1;
      stroke-width: 1.5;
    }}
    .svg-title {{
      font-size: 28px;
      font-weight: 700;
      fill: var(--ink);
    }}
    .svg-subtitle {{
      font-size: 15px;
      fill: var(--muted);
    }}
    .node {{
      stroke-width: 2;
      filter: drop-shadow(0 8px 14px rgba(0, 0, 0, 0.08));
    }}
    .node-input {{ fill: #efe8f5; stroke: var(--input); }}
    .node-stem {{ fill: #f4eef8; stroke: var(--input); }}
    .node-encoder {{ fill: #e8f4fb; stroke: var(--encoder); }}
    .node-bottleneck {{ fill: #e5ecef; stroke: var(--bottleneck); }}
    .node-decoder {{ fill: #edf7e8; stroke: var(--decoder); }}
    .node-head {{ fill: #fdebe6; stroke: var(--head); }}
    .node-output {{ fill: #fde6ee; stroke: var(--output); }}
    .node-title {{
      fill: var(--ink);
      font-size: 14px;
      font-weight: 600;
    }}
    .edge-main {{
      fill: none;
      stroke: #2f4858;
      stroke-width: 3;
      marker-end: url(#arrow);
    }}
    .edge-skip {{
      fill: none;
      stroke: var(--accent);
      stroke-width: 2.5;
      stroke-dasharray: 8 8;
      marker-end: url(#arrow-soft);
    }}
    .edge-label, .decoder-note {{
      fill: var(--muted);
      font-size: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 10px 12px;
      vertical-align: top;
    }}
    th {{
      color: var(--ink);
      background: rgba(255, 247, 231, 0.72);
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    details.panel {{
      padding: 0;
      overflow: hidden;
    }}
    details.panel > summary {{
      list-style: none;
      cursor: pointer;
      padding: 18px 20px;
      font-weight: 700;
    }}
    details.panel > summary::-webkit-details-marker {{
      display: none;
    }}
    details.panel > div {{
      padding: 0 20px 20px;
    }}
    .tree-node {{
      margin-left: 12px;
      border-left: 1px solid var(--line);
      padding-left: 12px;
    }}
    .tree-node > summary {{
      cursor: pointer;
      padding: 8px 0;
      list-style: none;
    }}
    .tree-node > summary::-webkit-details-marker {{
      display: none;
    }}
    .tree-leaf {{
      margin-left: 24px;
      padding: 6px 0;
    }}
    .tree-name {{
      display: inline-block;
      min-width: 320px;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.92rem;
      color: #23313a;
    }}
    .tree-meta {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .grid {{
      display: grid;
      gap: 18px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div>
        <h1>FlexibleUNet architecture summary</h1>
        <p>This report was generated from the same MONAI architecture used by the application. Shapes come from an actual forward pass with a dummy input tensor, so the stage outputs and decoder concatenation widths match the runtime graph.</p>
      </div>
      <div class="stats">
        <div class="stat">
          <div class="stat-label">Backbone</div>
          <div class="stat-value">EfficientNet-B2</div>
        </div>
        <div class="stat">
          <div class="stat-label">Input tensor</div>
          <div class="stat-value">{'x'.join(str(dim) for dim in input_shape)}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Total parameters</div>
          <div class="stat-value">{format_count(total_params)}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Executed modules recorded</div>
          <div class="stat-value">{format_count(len(all_records))}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Weights</div>
          <div class="stat-value">{html.escape(weights_status)}</div>
        </div>
        <div class="stat">
          <div class="stat-label">Output file</div>
          <div class="stat-value">{html.escape(str(output_path))}</div>
        </div>
      </div>
    </section>

    <section class="panel diagram-panel">
      <h2>SVG Overview</h2>
      <p>The architecture is a U-Net style decoder on top of an EfficientNet-B2 encoder. The dashed connections are the retained skip tensors used during decoder fusion.</p>
      {svg_markup}
    </section>

    <section class="grid">
      <div class="panel table-wrap">
        <h2>Stage Summary</h2>
        <table>
          <thead>
            <tr>
              <th>Stage</th>
              <th>Type</th>
              <th>Input</th>
              <th>Output</th>
              <th>Total params</th>
              <th>Connection role</th>
            </tr>
          </thead>
          <tbody>
            {render_stage_rows(executions)}
          </tbody>
        </table>
      </div>

      <details class="panel" open>
        <summary>Full Module Hierarchy</summary>
        <div>
          <p>Each node shows module path, module type, runtime input and output shape, and total parameter count for that subtree.</p>
          {render_tree_html(module_tree)}
        </div>
      </details>

      <details class="panel">
        <summary>Executed Module Table</summary>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Depth</th>
                <th>Input</th>
                <th>Output</th>
                <th>Direct params</th>
                <th>Total params</th>
              </tr>
            </thead>
            <tbody>
              {render_table_rows(all_records)}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/model_summary.html"),
        help="Output HTML path.",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Do not load data/model/models/model.pt before generating the report.",
    )
    parser.add_argument(
        "--input-shape",
        nargs=4,
        type=int,
        metavar=("N", "C", "H", "W"),
        default=DEFAULT_INPUT_SHAPE,
        help="Dummy input shape used for the runtime forward pass.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the self-contained HTML architecture report."""
    args = parse_args()
    settings = AppSettings.from_env()
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    if args.skip_weights:
        weights_status = "Skipped: --skip-weights was provided"
    else:
        weights_status = try_load_weights(model, settings.local_model_path)

    input_shape = tuple(args.input_shape)
    executions = collect_module_executions(model, input_shape)
    html_report = generate_html(model, executions, output_path.resolve(), weights_status, input_shape)
    output_path.write_text(html_report, encoding="utf-8")

    print(f"[INFO] Wrote model summary to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
