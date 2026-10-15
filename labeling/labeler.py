"""Rotulagem manual local; nenhuma previsão ou transformação dos PNGs."""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["filename", "label", "split", "sha256", "source", "split_seed"]


class ConcurrentEditError(ValueError):
    """A versão no disco mudou ou há outra gravação em andamento."""


def read_optional(path):
    return path.read_bytes() if path.exists() else None


def atomic_write(path, data, expected, validate):
    """Compare novamente depois da escrita temporária, antes de os.replace."""
    if read_optional(path) != expected:
        raise ConcurrentEditError(f"{path.name} foi alterado externamente. Recarregue os dados.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".labeling-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        written = temporary.read_bytes()
        if written != data:
            raise OSError("O conteúdo temporário diverge da escrita solicitada; original preservado.")
        validate(written)
        if read_optional(path) != expected:
            raise ConcurrentEditError(f"{path.name} foi alterado externamente. Recarregue os dados.")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_csv(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
    if reader.fieldnames != COLUMNS:
        raise ValueError("Cabeçalho inválido: esperado filename,label,split,sha256,source,split_seed.")
    rows, names = [], set()
    for row in reader:
        name = row.get("filename", "")
        if (None in row or any(v is None for v in row.values()) or not name
                or name in {".", ".."} or "/" in name or "\\" in name or ":" in name
                or Path(name).suffix.lower() != ".png" or name in names
                or row["split"] not in {"train", "validation", "test"}
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ValueError("Linha inválida, incompleta ou duplicada em labels.csv.")
        names.add(name)
        rows.append(row)
    return rows


def parse_state(data):
    if data is None:
        return {"version": 1, "review": {}}
    state = json.loads(data)
    if (not isinstance(state, dict) or set(state) != {"version", "review"}
            or state["version"] != 1 or not isinstance(state["review"], dict)
            or any(not isinstance(key, str) or value is not True
                   for key, value in state["review"].items())):
        raise ValueError("labeling_state.json inválido; estado preservado.")
    return state


class LabelStore:
    """Persistência independente de Tk; abre PNGs somente dos splits autorizados."""

    def __init__(self, root=ROOT / "dataset/real", splits=("train", "validation"), *, allow_test=False):
        self.root = Path(root).resolve()
        self.splits = tuple(dict.fromkeys(splits))
        if not self.splits or set(self.splits) - {"train", "validation", "test"}:
            raise ValueError("Selecione train, validation ou test.")
        if "test" in self.splits and not allow_test:
            raise ValueError("O teste é isolado para avaliação final. Use --split test --allow-test-labeling.")
        self.csv_path = self.root / "labels.csv"
        self.state_path = self.root / "labeling_state.json"
        self.reload()

    @contextmanager
    def writing(self):
        path = self.root / ".labeling.lock"
        try:
            stream = path.open("x", encoding="utf-8")
        except FileExistsError:
            raise ConcurrentEditError("Outra gravação está em andamento (.labeling.lock). Recarregue após concluída.") from None
        try:
            with stream:
                stream.write(str(os.getpid()))
                stream.flush()
                yield
        finally:
            path.unlink()

    def reload(self):
        csv_bytes = self.csv_path.read_bytes()
        rows = parse_csv(csv_bytes)
        selected = [r for r in rows if r["split"] in self.splits]
        for row in selected:
            path = self.root / row["split"] / row["filename"]
            if (path.resolve().parent != (self.root / row["split"]).resolve()
                    or path.resolve().parent.name != row["split"]):
                raise ValueError(f"Imagem fora do split selecionado: {path}")
            # Refuse a train/validation directory redirected into test via symlink.
            if row["split"] != "test" and (self.root / "test").resolve() in path.resolve().parents:
                raise ValueError(f"Split selecionado aponta para test: {path}")
            self.validate_image(path, row["sha256"])
        state_bytes = read_optional(self.state_path)
        state = parse_state(state_bytes)
        # Only publish the new snapshot once every validation has succeeded.
        self.rows, self.selected = rows, selected
        self.csv_bytes, self.state_bytes, self.state = csv_bytes, state_bytes, state

    @staticmethod
    def identity(row):
        return f"{row['split']}/{row['filename']}:{row['sha256']}"

    @staticmethod
    def validate_image(path, digest):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"SHA-256 divergente: {path}")
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Imagem não é PNG: {path}")
            image.verify()

    def checked_row(self, filename):
        row = next((r for r in self.selected if r["filename"] == filename), None)
        if row is None:
            raise ValueError("Imagem não pertence aos splits autorizados.")
        self.validate_image(self.root / row["split"] / filename, row["sha256"])
        return row

    def status(self, row):
        if self.identity(row) in self.state["review"]:
            return "revisão"
        return "rotulada" if row["label"] else "pendente"

    def filtered(self, mode="Pendentes"):
        return [r for r in self.selected if mode == "Todas"
                or (mode == "Revisão" and self.status(r) == "revisão")
                or (mode == "Pendentes" and self.status(r) == "pendente")]

    def save_label(self, filename, label):
        row = self.checked_row(filename)
        label = label.strip()  # Preserve case, internal spaces and arbitrary lengths/characters.
        changed = [dict(r, label=label) if r is row else dict(r) for r in self.rows]
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(changed)
        data = text.getvalue().encode("utf-8")
        with self.writing():
            atomic_write(self.csv_path, data, self.csv_bytes,
                         lambda written: self.validate_written_csv(written, changed))
        row["label"] = label
        self.csv_bytes = data

    @staticmethod
    def validate_written_csv(written, expected_rows):
        if parse_csv(written) != expected_rows:
            raise ValueError("Falha ao validar o CSV temporário; original preservado.")

    def set_review(self, filename, marked=True):
        row = self.checked_row(filename)
        review = dict(self.state["review"])
        if marked:
            review[self.identity(row)] = True
        else:
            review.pop(self.identity(row), None)
        state = {"version": 1, "review": review}
        data = (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        with self.writing():
            if self.csv_path.read_bytes() != self.csv_bytes:
                raise ConcurrentEditError("labels.csv foi alterado externamente. Recarregue os dados.")
            atomic_write(self.state_path, data, self.state_bytes, parse_state)
        self.state, self.state_bytes = state, data


class LabelerApp:
    def __init__(self, window, store):
        import tkinter as tk
        from tkinter import ttk
        self.window, self.store = window, store
        self.index = 0
        window.title("CaptchaDecay — Rotulagem manual")
        window.geometry(f"{min(900, window.winfo_screenwidth() - 60)}x{min(720, window.winfo_screenheight() - 100)}")
        window.minsize(640, 540)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(2, weight=1)
        top = ttk.Frame(window, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Exibir:").pack(side="left")
        self.mode = tk.StringVar(value="Pendentes")
        filters = ttk.Combobox(top, textvariable=self.mode, values=["Pendentes", "Revisão", "Todas"],
                               state="readonly", width=12)
        filters.pack(side="left", padx=8)
        filters.bind("<<ComboboxSelected>>", lambda event: self.refresh(reset=True))
        self.smooth = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Interpolação suave", variable=self.smooth,
                        command=self.draw_images).pack(side="left", padx=8)
        ttk.Button(top, text="Recarregar CSV", command=self.reload).pack(side="right")
        self.info = tk.StringVar()
        self.info_label = ttk.Label(window, textvariable=self.info, padding=(12, 0), wraplength=800)
        self.info_label.grid(row=1, column=0, sticky="ew")
        self.canvas = tk.Canvas(window, background="#e8eaed", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=12, pady=10)
        self.canvas.bind("<Configure>", lambda event: self.draw_images())
        form = ttk.Frame(window, padding=(12, 0, 12, 12))
        form.grid(row=3, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Label:").grid(row=0, column=0, padx=(0, 8))
        self.value = tk.StringVar()
        self.entry = ttk.Entry(form, textvariable=self.value, font=("TkDefaultFont", 18))
        self.entry.grid(row=0, column=1, sticky="ew")
        buttons = ttk.Frame(window, padding=(12, 0, 12, 6))
        buttons.grid(row=4, column=0, sticky="ew")
        for column, (title, command) in enumerate([
            ("Anterior", lambda: self.navigate(-1)), ("Próxima", lambda: self.navigate(1)),
            ("Pular", self.skip), ("Marcar revisão", self.review),
            ("Salvar", lambda: self.save(False)), ("Salvar e próxima", self.save),
        ]):
            ttk.Button(buttons, text=title, command=command).grid(row=column // 3, column=column % 3,
                                                                padx=3, pady=3, sticky="ew")
        for column in range(3):
            buttons.columnconfigure(column, weight=1)
        self.notice = tk.StringVar(value="Enter salva e avança. Navegar/pular descarta o texto não salvo.")
        self.notice_label = ttk.Label(window, textvariable=self.notice, padding=12, wraplength=800)
        self.notice_label.grid(row=5, column=0, sticky="ew")
        window.bind("<Configure>", self.resize_text)
        for sequence, action in [
            ("<Return>", self.save), ("<Control-s>", lambda: self.save(False)),
            ("<Control-Left>", lambda: self.navigate(-1)),
            ("<Control-Right>", lambda: self.navigate(1)), ("<Escape>", self.skip),
        ]:
            # Entry bindings run before native cursor movement and return 'break'.
            self.entry.bind(sequence, lambda event, callback=action: self.shortcut(callback))
        self.image = None
        self.refresh(reset=True)

    def resize_text(self, event):
        if event.widget is self.window:
            for label in (self.info_label, self.notice_label):
                label.configure(wraplength=max(200, event.width - 30))

    @staticmethod
    def shortcut(action):
        action()
        return "break"

    @property
    def current(self):
        return self.visible[self.index] if self.visible else None

    def refresh(self, reset=False):
        self.visible = self.store.filtered(self.mode.get())
        self.index = 0 if reset else min(self.index, max(0, len(self.visible) - 1))
        row = self.current
        counts = {s: sum(self.store.status(r) == s for r in self.store.selected)
                  for s in ("rotulada", "pendente", "revisão")}
        if row:
            path = self.store.root / row["split"] / row["filename"]
            try:
                self.store.checked_row(row["filename"])
                with Image.open(path) as image:
                    self.image = image.convert("RGB")
            except (OSError, ValueError) as error:
                self.image = None
                self.error(error)
            self.info.set(f"Split: {row['split']}  |  Imagem: {self.index + 1}/{len(self.visible)}  |  "
                          f"Estado: {self.store.status(row)}\n{row['filename']}\n"
                          f"Rotuladas: {counts['rotulada']}/{len(self.store.selected)}  |  "
                          f"Pendentes: {counts['pendente']}  |  Revisão: {counts['revisão']}")
            self.value.set(row["label"])
        else:
            self.image = None
            self.value.set("")
            self.info.set(f"Nenhuma imagem neste filtro. Rotuladas: {counts['rotulada']}; "
                          f"pendentes: {counts['pendente']}; revisão: {counts['revisão']}. Use Todas/Revisão.")
        self.draw_images()
        self.entry.focus_set()
        self.entry.icursor("end")

    def draw_images(self):
        from PIL import ImageTk
        self.canvas.delete("all")
        if self.image is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 50 or height < 50:
            return
        resample = Image.Resampling.LANCZOS if self.smooth.get() else Image.Resampling.NEAREST
        original = self.image.copy()
        original.thumbnail((width - 24, max(1, height // 3 - 25)), resample)
        scale = min(3, (width - 24) / self.image.width,
                    max(1, height - original.height - 64) / self.image.height)
        enlarged = self.image.resize((max(1, round(self.image.width * scale)),
                                      max(1, round(self.image.height * scale))), resample)
        self.photos = [ImageTk.PhotoImage(original), ImageTk.PhotoImage(enlarged)]
        self.canvas.create_text(width // 2, 12, text=f"Original: {self.image.width} × {self.image.height}")
        self.canvas.create_image(width // 2, 26, image=self.photos[0], anchor="n")
        y = 38 + original.height
        self.canvas.create_text(width // 2, y, text=f"Ampliação: {scale:.2f}× (ajustada à janela)")
        self.canvas.create_image(width // 2, y + 14, image=self.photos[1], anchor="n")

    def error(self, error):
        from tkinter import messagebox
        self.notice.set(str(error))
        messagebox.showerror("Não foi possível concluir", str(error), parent=self.window)

    def reload(self):
        try:
            self.store.reload()
            self.refresh(reset=True)
            self.notice.set("Dados recarregados; texto não salvo foi descartado.")
        except (OSError, ValueError, csv.Error) as error:
            self.error(error)

    def navigate(self, delta):
        if self.visible:
            self.index = (self.index + delta) % len(self.visible)
        self.refresh()

    def skip(self):
        self.navigate(1)

    def review(self):
        if self.current:
            try:
                self.store.set_review(self.current["filename"])
            except (OSError, ValueError) as error:
                self.error(error)
                return
            self.notice.set("Marcada para revisão. Texto digitado não foi salvo; eventual label anterior permanece.")
            self.refresh()

    def save(self, advance=True):
        if not self.current:
            return
        filename = self.current["filename"]
        was_review = self.store.status(self.current) == "revisão"
        try:
            self.store.save_label(filename, self.value.get())
        except (OSError, ValueError, csv.Error) as error:
            self.error(error)
            return
        if was_review:
            try:
                self.store.set_review(filename, False)
            except (OSError, ValueError) as error:
                self.error(ValueError(f"Label SALVO, mas a marca de revisão permanece: {error}"))
                self.refresh()
                return
        self.notice.set("Label salvo." if self.value.get().strip() else "Label vazio salvo: imagem não rotulada.")
        if advance:
            old = [r["filename"] for r in self.visible]
            remaining = [r["filename"] for r in self.store.filtered(self.mode.get())]
            if filename in remaining:
                self.index = (remaining.index(filename) + 1) % len(remaining)
            elif remaining:
                self.index = next((remaining.index(name) for name in old[self.index + 1:] + old[:self.index]
                                   if name in remaining), 0)
        # Saving without advancing keeps the selected image available for correction.
        elif filename not in [r["filename"] for r in self.store.filtered(self.mode.get())]:
            self.mode.set("Todas")
            self.index = [r["filename"] for r in self.store.selected].index(filename)
        self.refresh()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/real")
    parser.add_argument("--split", nargs="+", choices=["train", "validation", "test"], default=["train", "validation"])
    parser.add_argument("--allow-test-labeling", action="store_true")
    args = parser.parse_args(argv)
    try:
        store = LabelStore(args.dataset, args.split, allow_test=args.allow_test_labeling)
        import tkinter as tk
        try:
            window = tk.Tk()
        except tk.TclError as error:
            raise ValueError(f"Não foi possível abrir o ambiente gráfico Tk: {error}") from error
        LabelerApp(window, store)
        window.mainloop()
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Erro: {error}\n")
    except ImportError:
        parser.exit(1, "Tkinter indisponível. Instale o componente Tk do Python (python3-tk no Linux).\n")


if __name__ == "__main__":
    main()
