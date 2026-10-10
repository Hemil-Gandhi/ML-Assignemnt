"""Simplified Tkinter GUI: cardiovascular form (loop-generated) + PTB-XL upload."""
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import load_ptbxl_model, load_ptbxl_thresholds, load_tabular_models
from src.prediction import (
    cardiovascular_majority_vote,
    load_ecg_record,
    predict_cardiovascular,
    predict_ptbxl,
)

NUMERIC_FIELDS = [
    ("Age", "age_years", "int"),
    ("Height (cm)", "height", "int"),
    ("Weight (kg)", "weight", "float"),
    ("Systolic BP", "ap_hi", "int"),
    ("Diastolic BP", "ap_lo", "int"),
]

CHOICE_FIELDS = [
    ("Gender", "gender", ["Female", "Male"]),
    ("Cholesterol", "cholesterol", ["Normal", "Above Normal", "Well Above Normal"]),
    ("Glucose", "gluc", ["Normal", "Above Normal", "Well Above Normal"]),
    ("Smoking", "smoke", ["No", "Yes"]),
    ("Alcohol", "alco", ["No", "Yes"]),
    ("Active", "active", ["No", "Yes"]),
]

FIELD_LABELS = {key: label for label, key, _ in NUMERIC_FIELDS}

CHOICE_MAP = {
    "gender": {"Female": 2, "Male": 1},
    "cholesterol": {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3},
    "gluc": {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3},
    "smoke": {"No": 0, "Yes": 1},
    "alco": {"No": 0, "Yes": 1},
    "active": {"No": 0, "Yes": 1},
}


class HeartDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease ML Application")
        self.root.geometry("800x650")
        try:
            self.tabular_models = load_tabular_models()
            self.ptbxl_model = load_ptbxl_model()
            self.thresholds = load_ptbxl_thresholds()
        except Exception as error:
            messagebox.showerror("Model Loading Error", str(error))
            self.root.destroy()
            return
        self.prediction_type = tk.StringVar(value="Cardiovascular")
        self.ecg_path = None
        self.entries = {}
        self.vars = {}
        self._build_shell()
        self.show_cardio()

    # ----- shell -----
    def _build_shell(self):
        ttk.Label(self.root, text="Heart Disease ML Application",
                  font=("Arial", 18, "bold")).pack(pady=(15, 0))
        ttk.Label(self.root, text="Simplified prediction demo",
                  font=("Arial", 10)).pack(pady=(0, 10))
        type_frame = ttk.LabelFrame(self.root, text="Prediction Type", padding=10)
        type_frame.pack(fill="x", padx=20, pady=5)
        for text, val in [("Cardiovascular", "Cardiovascular"), ("PTB-XL ECG", "PTB-XL")]:
            ttk.Radiobutton(type_frame, text=text, variable=self.prediction_type,
                            value=val, command=self._switch).pack(side="left", padx=15)
        self.content = ttk.Frame(self.root)
        self.content.pack(fill="both", expand=True, padx=20)

    def _switch(self):
        self.show_cardio() if self.prediction_type.get() == "Cardiovascular" else self.show_ecg()

    def _clear(self):
        for w in self.content.winfo_children():
            w.destroy()

    # ----- cardio form -----
    def show_cardio(self):
        self._clear()
        form = ttk.LabelFrame(self.content, text="Patient Information", padding=10)
        form.pack(fill="x", pady=5)
        for row, (label, key, kind) in enumerate(NUMERIC_FIELDS):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            e = ttk.Entry(form, width=20)
            e.grid(row=row, column=1, padx=8, pady=4)
            self.entries[key] = (e, kind)
        for row, (label, key, values) in enumerate(CHOICE_FIELDS):
            r = row if row < 5 else 5
            c = 2 if row < 5 else 0  # last row wraps
            ttk.Label(form, text=label).grid(row=r if c == 2 else 6, column=c,
                                             sticky="w", padx=8, pady=4)
            var = tk.StringVar(value=values[0] if key != "active" else "Yes")
            ttk.Combobox(form, textvariable=var, values=values,
                         state="readonly", width=18).grid(
                row=r if c == 2 else 6, column=c + 1, padx=8, pady=4)
            self.vars[key] = var
        btns = ttk.Frame(self.content)
        btns.pack(pady=10)
        ttk.Button(btns, text="Predict", command=self.do_cardio).pack(side="left", padx=8)
        ttk.Button(btns, text="Reset", command=self.show_cardio).pack(side="left", padx=8)
        self.results = ttk.LabelFrame(self.content, text="Results", padding=10)
        self.results.pack(fill="both", expand=True, pady=5)

    def _read_patient(self):
        patient = {}
        for key, (entry, kind) in self.entries.items():
            val = entry.get().strip()
            label = FIELD_LABELS.get(key, key)
            if not val:
                raise ValueError(f"{label} is required.")
            try:
                patient[key] = int(val) if kind == "int" else float(val)
            except ValueError:
                raise ValueError(f"{label} must be a number.") from None
        for key, var in self.vars.items():
            patient[key] = CHOICE_MAP[key][var.get()]
        return patient

    def do_cardio(self):
        try:
            results = predict_cardiovascular(self.tabular_models, self._read_patient())
            overall = cardiovascular_majority_vote(results)
            for w in self.results.winfo_children():
                w.destroy()
            for name, r in results.items():
                txt = "Positive" if r["prediction"] == 1 else "Negative"
                ttk.Label(self.results, text=f"{name}: {txt} ({r['probability']:.1%})").pack(anchor="w")
            ttk.Separator(self.results, orient="horizontal").pack(fill="x", pady=8)
            overall_txt = "Positive" if overall["prediction"] == 1 else "Negative"
            ttk.Label(self.results, text=f"Overall: {overall_txt} "
                      f"(+{overall['positive_votes']}/-{overall['negative_votes']})",
                      font=("Arial", 12, "bold")).pack()
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    # ----- ecg form -----
    def show_ecg(self):
        self._clear()
        self.ecg_path = None
        up = ttk.LabelFrame(self.content, text="PTB-XL ECG", padding=15)
        up.pack(fill="x", pady=10)
        ttk.Label(up, text="Select a .hea record (matching .dat required).").pack(pady=5)
        ttk.Button(up, text="Select ECG Record", command=self.pick_ecg).pack()
        self.ecg_label = ttk.Label(up, text="No ECG record selected.")
        self.ecg_label.pack(pady=10)
        btns = ttk.Frame(self.content)
        btns.pack(pady=5)
        ttk.Button(btns, text="Predict", command=self.do_ecg).pack(side="left", padx=8)
        ttk.Button(btns, text="Reset", command=self.show_ecg).pack(side="left", padx=8)
        self.results = ttk.LabelFrame(self.content, text="ECG Results", padding=10)
        self.results.pack(fill="both", expand=True, pady=5)

    def pick_ecg(self):
        p = filedialog.askopenfilename(title="Select ECG",
                                       filetypes=[("WFDB Header", "*.hea"), ("All", "*.*")])
        if not p:
            return
        path = Path(p)
        if path.suffix.lower() != ".hea" or not path.with_suffix(".dat").exists():
            messagebox.showerror("Invalid File", "Pick a .hea with a matching .dat file.")
            return
        self.ecg_path = path.with_suffix("")
        self.ecg_label.config(text=f"Selected: {path.name}")

    def do_ecg(self):
        if self.ecg_path is None:
            messagebox.showerror("No ECG", "Select an ECG record first.")
            return
        try:
            results = predict_ptbxl(self.ptbxl_model, load_ecg_record(self.ecg_path),
                                    self.thresholds)
            for w in self.results.winfo_children():
                w.destroy()
            for cls, r in results.items():
                ttk.Label(self.results,
                          text=f"{cls}: {'Positive' if r['positive'] else 'Negative'} "
                               f"({r['probability']:.1%})").pack(anchor="w")
        except Exception as e:
            messagebox.showerror("ECG Error", str(e))


def main():
    root = tk.Tk()
    HeartDiseaseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
