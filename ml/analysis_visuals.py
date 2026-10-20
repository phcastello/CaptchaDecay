"""Deterministic, local-only diagnostic galleries and plots from actual results.

Original images are read at their native resolution and never written. Pillow
is the plotting fallback when the optional Matplotlib dependency is unavailable.
No inference, training, network access, or image generation happens here.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
import importlib.util
import math
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont

from .metrics import levenshtein


def _count(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a nonnegative integer.") from error
    if not math.isfinite(number) or number < 0 or number != int(number):
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(number)


def _distance(row):
    if "edit_distance" in row:
        return _count(row["edit_distance"], "edit_distance")
    return levenshtein(row["expected"], row["predicted"])


def _stratum(row, correct):
    length = _count(row.get("expected_length", len(row["expected"])), "expected_length")
    if correct:
        return (length,)
    operations = tuple(
        _count(row.get(name, row.get(alias, 0)), name) > 0
        for name, alias in (("substitutions", "sub"), ("deletions", "del"), ("insertions", "ins"))
    )
    return (length, *operations)


def _stratified(rows, amount, rng, correct):
    """Take a seeded round-robin quota across strata before taking more rows."""
    groups = defaultdict(list)
    for row in rows:
        groups[_stratum(row, correct)].append(row)
    keys = sorted(groups)
    for key in keys:
        groups[key].sort(key=lambda row: row["filename"])
        rng.shuffle(groups[key])
    rng.shuffle(keys)
    selected = []
    while len(selected) < amount and keys:
        remaining_keys = []
        for key in keys:
            if len(selected) == amount:
                break
            selected.append(groups[key].pop())
            if groups[key]:
                remaining_keys.append(key)
        keys = remaining_keys
    return selected


def select_gallery(rows: Sequence[Mapping], seed=42, max_correct=20, max_incorrect=20) -> list[dict]:
    """Copy at most 20 correct and 20 incorrect unique rows for visual inspection.

    Correct examples are stratified by reference length. Incorrect examples are
    stratified by reference length and the presence of each edit operation. Up
    to four incorrect slots are reserved for the largest edit distances and
    explicitly marked ``extremo``; all other selected rows are ``amostra``.
    The gallery is illustrative, not a prevalence estimate for the dataset.
    """
    max_correct = min(20, _count(max_correct, "max_correct"))
    max_incorrect = min(20, _count(max_incorrect, "max_incorrect"))
    unique = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Gallery rows must be mappings.")
        if not all(isinstance(row.get(key), str) for key in ("filename", "expected", "predicted")):
            raise ValueError("Gallery rows require string filename, expected, and predicted fields.")
        if not row["filename"]:
            raise ValueError("Gallery filename cannot be empty.")
        name_key = row["filename"].casefold()
        if name_key in unique:
            if dict(unique[name_key]) != dict(row):
                raise ValueError(f"Conflicting duplicate gallery filename: {row['filename']!r}.")
            continue
        unique[name_key] = dict(row)
    ordered = sorted(unique.values(), key=lambda row: row["filename"])
    correct = [row for row in ordered if row["expected"] == row["predicted"]]
    incorrect = [row for row in ordered if row["expected"] != row["predicted"]]
    rng = random.Random(seed)
    result = [dict(row, selection="amostra") for row in _stratified(correct, max_correct, rng, True)]
    # Avoid using the entire incorrect quota for extremes when a small quota is
    # requested: retain a representative half where possible.
    extreme_count = min(4, max_incorrect // 2, len(incorrect))
    extremes = sorted(incorrect, key=lambda row: (-_distance(row), row["filename"]))[:extreme_count]
    extreme_names = {row["filename"].casefold() for row in extremes}
    candidates = [row for row in incorrect if row["filename"].casefold() not in extreme_names]
    representative = _stratified(candidates, max_incorrect - extreme_count, rng, False)
    result.extend(dict(row, selection="amostra") for row in representative)
    result.extend(dict(row, selection="extremo") for row in extremes)
    return result


def _font(size=14):
    for candidate in (
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(text, draw, font, width):
    """Wrap even long unspaced predictions without changing their content."""
    lines, current = [], ""
    for char in text:
        if current and draw.textlength(current + char, font=font) > width:
            lines.append(current)
            current = char
        else:
            current += char
    lines.append(current)
    return lines


def generate_gallery(rows, image_paths: Mapping, output: Path, seed=42, *, paired=False) -> list[dict]:
    """Write a contact sheet of selected native 215 x 80 local images."""
    selected = ([dict(row) for row in sorted(rows, key=lambda row: row["filename"])]
                if paired else select_gallery(rows, seed=seed))
    output = Path(output)
    sources = {Path(path).resolve() for path in image_paths.values()}
    if output.resolve() in sources:
        raise ValueError("Gallery output must not overwrite an original image.")
    normal_font, heading_font = _font(14), _font(20)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    card_width, gap, columns, top = 310, 12, 4, 90
    captions = []
    max_lines = 0
    for row in selected:
        if paired:
            text = [f"arquivo: {row['filename']}", f"ground truth: {row['expected']}",
                    f"A: {row['predicted_a'] or '<vazio>'} | {'acerto' if row['correct_a'] else 'erro'} | d={row['edit_distance_a']}",
                    f"B: {row['predicted_b'] or '<vazio>'} | {'acerto' if row['correct_b'] else 'erro'} | d={row['edit_distance_b']}"]
        else:
            correct = row["expected"] == row["predicted"]
            text = [
            f"arquivo: {row['filename']}",
            f"esperado: {row['expected'] or '<vazio>'}",
            f"previsto: {row['predicted'] or '<vazio>'}",
            f"distância: {_distance(row)} | {'correto' if correct else 'incorreto'}",
            f"seleção: {row['selection']}",
            ]
        lines = [line for value in text for line in _wrap(value, probe, normal_font, card_width - 20)]
        captions.append(lines)
        max_lines = max(max_lines, len(lines))
    card_height = 104 + max_lines * 19
    count_rows = max(1, math.ceil(len(selected) / columns))
    canvas = Image.new("RGB", (columns * (card_width + gap) + gap,
                               top + count_rows * (card_height + gap) + gap), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), "Transferência sintético → real — validação" if paired else "Galeria de diagnóstico — validação sintética", font=heading_font, fill="#172033")
    draw.text((gap, 40), "Verde: ambos acertam | Vermelho: ambos erram | Azul: só B acerta | Laranja: só A acerta"
              if paired else "Seleção estratificada por comprimento/tipo de erro; extremos identificados.",
              font=normal_font, fill="#334155")
    draw.text((gap, 61), f"{len(selected)} {'imagens elegíveis; labels fixados antes das previsões' if paired else 'exemplos ilustrativos'}; originais em 215 × 80 pixels.",
              font=normal_font, fill="#334155")
    if not selected:
        draw.text((gap + 10, top + 20), "Sem exemplos disponíveis.", font=heading_font, fill="#475569")
    for index, (row, lines) in enumerate(zip(selected, captions)):
        filename = row["filename"]
        if filename not in image_paths:
            raise ValueError(f"Missing original image path for gallery filename {filename!r}.")
        x = gap + (index % columns) * (card_width + gap)
        y = top + (index // columns) * (card_height + gap)
        if paired:
            border = {(True, True): "#15803d", (False, False): "#b91c1c",
                      (False, True): "#2563eb", (True, False): "#b45309"}[row["correct_a"], row["correct_b"]]
        else:
            correct = row["expected"] == row["predicted"]
            border = "#15803d" if correct else "#b91c1c"
            if row["selection"] == "extremo":
                border = "#b45309"
        draw.rectangle((x, y, x + card_width, y + card_height), fill="white", outline=border, width=2)
        with Image.open(image_paths[filename]) as original:
            if original.size != (215, 80) or original.format != "PNG":
                raise ValueError(f"Expected original 215 x 80 PNG for gallery image {filename!r}.")
            canvas.paste(original.convert("RGB"), (x + (card_width - 215) // 2, y + 8))
        for line_index, line in enumerate(lines):
            draw.text((x + 10, y + 98 + line_index * 19), line, font=normal_font, fill="#172033")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    return selected


def _summary_value(summary, key, default):
    return summary.get(key, default) if isinstance(summary, Mapping) else getattr(summary, key, default)


def _plot_data(summary):
    confusion = _summary_value(summary, "confusion_matrix", {})
    characters = list(confusion.get("characters", []))
    if not all(isinstance(char, str) for char in characters) or len(set(characters)) != len(characters):
        raise ValueError("Confusion matrix characters must be unique strings.")
    raw_counts = confusion.get("counts", [])
    if len(raw_counts) != len(characters) or any(len(row) != len(characters) for row in raw_counts):
        raise ValueError("Confusion matrix dimensions do not match its characters.")
    counts = [[_count(value, "confusion count") for value in row] for row in raw_counts]
    raw_by_length = _summary_value(summary, "by_length", {})
    by_length = []
    for length, stats in sorted(raw_by_length.items(), key=lambda item: int(item[0])):
        item = {"length": _count(length, "expected length")}
        for key in ("num_images", "num_correct", "substitutions", "deletions", "insertions"):
            item[key] = _count(stats.get(key, 0), key)
        for key in ("exact_match", "cer"):
            item[key] = float(stats.get(key, 0))
            if not math.isfinite(item[key]) or item[key] < 0 or (key == "exact_match" and item[key] > 1):
                raise ValueError(f"Invalid {key} in by-length statistics.")
        by_length.append(item)
    return characters, counts, by_length


def _shade(value, maximum):
    ratio = value / maximum if maximum else 0
    return tuple(round(low + (high - low) * ratio) for low, high in zip((247, 250, 252), (30, 64, 175)))


def _pillow_confusion(characters, counts, output):
    size = len(characters)
    cell = 27 if size <= 36 else 20
    side = max(360, size * cell)
    margin, panel_gap, top = 66, 76, 120
    width = margin * 2 + side * 2 + panel_gap
    canvas = Image.new("RGB", (width, top + side + 95), "white")
    draw = ImageDraw.Draw(canvas)
    font, small, heading = _font(14), _font(11), _font(19)
    draw.text((20, 12), "Confusão de caracteres — contagens absolutas", font=heading, fill="#172033")
    draw.text((20, 40), "Linhas = esperado; colunas = previsto. Acertos ficam na diagonal do painel esquerdo.",
              font=font, fill="#334155")
    draw.text((20, 62), "Inserções e deleções são contabilizadas separadamente no gráfico por comprimento.",
              font=font, fill="#334155")
    if not size:
        draw.text((20, 115), "Sem dados de alinhamento.", font=heading, fill="#475569")
    for panel in range(2):
        x_start = margin + panel * (side + panel_gap)
        draw.text((x_start, 90), "Todos os alinhamentos" if panel == 0 else "Somente substituições (diagonal = 0)",
                  font=font, fill="#172033")
        maximum = max((value for row_index, row in enumerate(counts)
                       for col_index, value in enumerate(row) if panel == 0 or row_index != col_index), default=0)
        for row_index, row in enumerate(counts):
            label = characters[row_index]
            y = top + row_index * cell
            draw.text((x_start - 22, y + (cell - 14) // 2), label, font=font, fill="#172033")
            draw.text((x_start + row_index * cell + 7, top - 20), label, font=font, fill="#172033")
            for col_index, original_value in enumerate(row):
                value = 0 if panel == 1 and row_index == col_index else original_value
                x = x_start + col_index * cell
                draw.rectangle((x, y, x + cell, y + cell), fill=_shade(value, maximum), outline="#dbe1e8")
                if value:
                    text = str(value)
                    text_width = draw.textlength(text, font=small)
                    if text_width <= cell - 2:
                        draw.text((x + (cell - text_width) / 2, y + (cell - 12) // 2), text,
                                  font=small, fill="white" if maximum and value / maximum > 0.55 else "#172033")
        legend_y = top + side + 23
        for index in range(120):
            draw.line((x_start + index, legend_y, x_start + index, legend_y + 12),
                      fill=_shade(index, 119))
        draw.text((x_start, legend_y + 16), f"0 a {maximum} ocorrências", font=font, fill="#334155")
    canvas.save(output, format="PNG")


def _pillow_by_length(by_length, output):
    canvas = Image.new("RGB", (1080, 850), "white")
    draw = ImageDraw.Draw(canvas)
    font, small, heading = _font(16), _font(13), _font(21)
    draw.text((24, 15), "Resultados por comprimento real do label", font=heading, fill="#172033")
    if not by_length:
        draw.text((24, 70), "Sem dados por comprimento.", font=font, fill="#475569")
        canvas.save(output, format="PNG")
        return
    x0, x1, top, bottom = 85, 1035, 110, 380
    peak = max(max(item["exact_match"], item["cer"]) * 100 for item in by_length)
    percent_max = max(100, math.ceil(peak / 25) * 25)
    draw.text((x0, 65), "Exact match e CER (%)", font=font, fill="#172033")
    for value in range(0, percent_max + 1, max(25, percent_max // 5)):
        y = bottom - (bottom - top) * value / percent_max
        draw.line((x0, y, x1, y), fill="#e2e8f0")
        draw.text((x0 - 58, y - 8), f"{value}%", font=small, fill="#475569")
    group_width = (x1 - x0) / len(by_length)
    bar_width = min(90, group_width / 3)
    for index, item in enumerate(by_length):
        center = x0 + group_width * (index + 0.5)
        for series, color, offset in (("exact_match", "#2563eb", -bar_width), ("cer", "#ea580c", 4)):
            value = item[series] * 100
            height = (bottom - top) * value / percent_max
            draw.rectangle((center + offset, bottom - height, center + offset + bar_width - 4, bottom), fill=color)
            draw.text((center + offset, bottom - height - 20), f"{value:.2f}%", font=small, fill="#172033")
        draw.text((center - 45, bottom + 12), f"{item['length']} (n={item['num_images']})", font=small, fill="#172033")
    draw.text((x0, 425), "Azul: exact match   |   Laranja: CER agregado; pode ultrapassar 100%.",
              font=small, fill="#334155")
    top, bottom = 540, 775
    edit_max = max(sum(item[key] for key in ("substitutions", "deletions", "insertions")) for item in by_length)
    scale = max(1, edit_max)
    draw.text((x0, 475), "Operações de edição (contagens absolutas)", font=font, fill="#172033")
    draw.text((x0, 506), "Vermelho: substituições   |   Roxo: deleções   |   Verde: inserções", font=small, fill="#334155")
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = bottom - (bottom - top) * fraction
        draw.line((x0, y, x1, y), fill="#e2e8f0")
        draw.text((x0 - 62, y - 8), f"{scale * fraction:g}", font=small, fill="#475569")
    for index, item in enumerate(by_length):
        center = x0 + group_width * (index + 0.5)
        cumulative = 0
        width = min(120, group_width * 0.5)
        for key, color in (("substitutions", "#dc2626"), ("deletions", "#9333ea"), ("insertions", "#059669")):
            amount = item[key]
            y_bottom = bottom - (bottom - top) * cumulative / scale
            cumulative += amount
            y_top = bottom - (bottom - top) * cumulative / scale
            if amount:
                draw.rectangle((center - width / 2, y_top, center + width / 2, y_bottom), fill=color)
                if y_bottom - y_top >= 17:
                    draw.text((center - 12, (y_top + y_bottom) / 2 - 8), str(amount), font=small, fill="white")
        draw.text((center - 14, bottom - (bottom - top) * cumulative / scale - 20), str(cumulative),
                  font=small, fill="#172033")
        draw.text((center - 35, bottom + 12), str(item["length"]), font=small, fill="#172033")
    canvas.save(output, format="PNG")


def _matplotlib_plots(characters, counts, by_length, confusion_output, length_output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    fig.suptitle("Confusão de caracteres — contagens absolutas; linhas = esperado, colunas = previsto")
    for panel, axis in enumerate(axes):
        axis.set_title("Todos os alinhamentos (acertos na diagonal)" if panel == 0 else "Somente substituições")
        if not characters:
            axis.text(0.5, 0.5, "Sem dados de alinhamento.", ha="center", va="center")
            axis.set_axis_off()
            continue
        matrix = np.array(counts, dtype=np.int64)
        if panel:
            np.fill_diagonal(matrix, 0)
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        axis.set_xticks(range(len(characters)), characters)
        axis.set_yticks(range(len(characters)), characters)
        axis.set_xlabel("Previsto")
        axis.set_ylabel("Esperado")
        for row_index, col_index in zip(*np.nonzero(matrix)):
            value = int(matrix[row_index, col_index])
            axis.text(col_index, row_index, str(value), ha="center", va="center", fontsize=6,
                      color="white" if value > matrix.max() * 0.55 else "black")
        fig.colorbar(image, ax=axis, shrink=0.65, label="Ocorrências")
    fig.savefig(confusion_output, dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    fig.suptitle("Resultados por comprimento real do label")
    if not by_length:
        for axis in axes:
            axis.text(0.5, 0.5, "Sem dados por comprimento.", ha="center", va="center")
            axis.set_axis_off()
    else:
        positions = np.arange(len(by_length))
        axes[0].bar(positions - 0.18, [item["exact_match"] * 100 for item in by_length], width=0.36,
                    label="Exact match", color="#2563eb")
        axes[0].bar(positions + 0.18, [item["cer"] * 100 for item in by_length], width=0.36,
                    label="CER agregado", color="#ea580c")
        axes[0].set_ylabel("Percentual (%)")
        axes[0].set_xticks(positions, [f"{item['length']} (n={item['num_images']})" for item in by_length])
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.2)
        bottom = np.zeros(len(by_length))
        for key, label, color in (("substitutions", "Substituições", "#dc2626"),
                                  ("deletions", "Deleções", "#9333ea"), ("insertions", "Inserções", "#059669")):
            values = np.array([item[key] for item in by_length])
            axes[1].bar(positions, values, bottom=bottom, label=label, color=color)
            bottom += values
        axes[1].set_xticks(positions, [str(item["length"]) for item in by_length])
        axes[1].set_xlabel("Comprimento real")
        axes[1].set_ylabel("Operações de edição (contagens)")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(length_output, dpi=140)
    plt.close(fig)


def generate_plots(summary, output_dir: Path) -> list[Path]:
    """Plot the full actual confusion counts and reference-length statistics."""
    characters, counts, by_length = _plot_data(summary)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_output = output_dir / "confusion-matrix.png"
    length_output = output_dir / "by-length.png"
    if importlib.util.find_spec("matplotlib") is not None:
        _matplotlib_plots(characters, counts, by_length, confusion_output, length_output)
    else:
        _pillow_confusion(characters, counts, confusion_output)
        _pillow_by_length(by_length, length_output)
    return [confusion_output, length_output]
