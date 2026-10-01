import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from src.models import (
    load_ptbxl_model,
    load_ptbxl_thresholds,
    load_tabular_models,
)

from src.prediction import (
    load_ecg_record,
    predict_cardiovascular,
    predict_ptbxl,
    cardiovascular_majority_vote,
)


class HeartDiseaseApp:
    """Tkinter application for cardiovascular and PTB-XL prediction."""

    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease ML Application")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        # -----------------------------------------------------
        # Load models
        # -----------------------------------------------------

        try:
            self.tabular_models = load_tabular_models()
            self.ptbxl_model = load_ptbxl_model()
            self.thresholds = load_ptbxl_thresholds()

        except Exception as error:
            messagebox.showerror(
                "Model Loading Error",
                f"Could not load the models.\n\n{error}",
            )
            self.root.destroy()
            return

        # -----------------------------------------------------
        # Application state
        # -----------------------------------------------------

        self.prediction_type = tk.StringVar(value="Cardiovascular")

        self.ecg_path = None

        # -----------------------------------------------------
        # Build interface
        # -----------------------------------------------------

        self.create_widgets()

    # =========================================================
    # Main interface
    # =========================================================

    def create_widgets(self):
        title = ttk.Label(
            self.root,
            text="Heart Disease ML Application",
            font=("Arial", 20, "bold"),
        )

        title.pack(pady=(20, 5))

        subtitle = ttk.Label(
            self.root,
            text="Machine Learning Based Prediction System",
            font=("Arial", 11),
        )

        subtitle.pack(pady=(0, 15))

        # -----------------------------------------------------
        # Prediction type
        # -----------------------------------------------------

        type_frame = ttk.LabelFrame(
            self.root,
            text="Prediction Type",
            padding=15,
        )

        type_frame.pack(
            fill="x",
            padx=25,
            pady=10,
        )

        ttk.Radiobutton(
            type_frame,
            text="Cardiovascular Disease",
            variable=self.prediction_type,
            value="Cardiovascular",
            command=self.switch_prediction_type,
        ).pack(
            side="left",
            padx=20,
        )

        ttk.Radiobutton(
            type_frame,
            text="PTB-XL ECG",
            variable=self.prediction_type,
            value="PTB-XL",
            command=self.switch_prediction_type,
        ).pack(
            side="left",
            padx=20,
        )

        # -----------------------------------------------------
        # Content container
        # -----------------------------------------------------

        self.content_frame = ttk.Frame(self.root)

        self.content_frame.pack(
            fill="both",
            expand=True,
            padx=25,
        )

        self.create_cardiovascular_form()

    # =========================================================
    # Cardiovascular form
    # =========================================================

    def create_cardiovascular_form(self):
        self.clear_content()

        form_frame = ttk.LabelFrame(
            self.content_frame,
            text="Patient Information",
            padding=15,
        )

        form_frame.pack(
            fill="x",
            pady=10,
        )

        self.entries = {}

        fields = [
            ("Age", "age_years"),
            ("Height (cm)", "height"),
            ("Weight (kg)", "weight"),
            ("Systolic BP", "ap_hi"),
            ("Diastolic BP", "ap_lo"),
        ]

        for row, (label, key) in enumerate(fields):
            ttk.Label(
                form_frame,
                text=label,
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=10,
                pady=6,
            )

            entry = ttk.Entry(
                form_frame,
                width=25,
            )

            entry.grid(
                row=row,
                column=1,
                padx=10,
                pady=6,
            )

            self.entries[key] = entry

        # -----------------------------------------------------
        # Gender
        # -----------------------------------------------------

        ttk.Label(
            form_frame,
            text="Gender",
        ).grid(
            row=0,
            column=2,
            sticky="w",
            padx=10,
            pady=6,
        )

        self.gender_var = tk.StringVar(value="Female")

        gender_box = ttk.Combobox(
            form_frame,
            textvariable=self.gender_var,
            values=["Female", "Male"],
            state="readonly",
            width=22,
        )

        gender_box.grid(
            row=0,
            column=3,
            padx=10,
            pady=6,
        )

        # -----------------------------------------------------
        # Cholesterol
        # -----------------------------------------------------

        ttk.Label(
            form_frame,
            text="Cholesterol",
        ).grid(
            row=1,
            column=2,
            sticky="w",
            padx=10,
            pady=6,
        )

        self.cholesterol_var = tk.StringVar(value="Normal")

        cholesterol_box = ttk.Combobox(
            form_frame,
            textvariable=self.cholesterol_var,
            values=[
                "Normal",
                "Above Normal",
                "Well Above Normal",
            ],
            state="readonly",
            width=22,
        )

        cholesterol_box.grid(
            row=1,
            column=3,
            padx=10,
            pady=6,
        )

        # -----------------------------------------------------
        # Glucose
        # -----------------------------------------------------

        ttk.Label(
            form_frame,
            text="Glucose",
        ).grid(
            row=2,
            column=2,
            sticky="w",
            padx=10,
            pady=6,
        )

        self.gluc_var = tk.StringVar(value="Normal")

        gluc_box = ttk.Combobox(
            form_frame,
            textvariable=self.gluc_var,
            values=[
                "Normal",
                "Above Normal",
                "Well Above Normal",
            ],
            state="readonly",
            width=22,
        )

        gluc_box.grid(
            row=2,
            column=3,
            padx=10,
            pady=6,
        )

        # -----------------------------------------------------
        # Smoking
        # -----------------------------------------------------

        ttk.Label(
            form_frame,
            text="Smoking",
        ).grid(
            row=3,
            column=2,
            sticky="w",
            padx=10,
            pady=6,
        )

        self.smoke_var = tk.StringVar(value="No")

        smoke_box = ttk.Combobox(
            form_frame,
            textvariable=self.smoke_var,
            values=["No", "Yes"],
            state="readonly",
            width=22,
        )

        smoke_box.grid(
            row=3,
            column=3,
            padx=10,
            pady=6,
        )

        # -----------------------------------------------------
        # Alcohol
        # -----------------------------------------------------

        ttk.Label(
            form_frame,
            text="Alcohol Consumption",
        ).grid(
            row=4,
            column=2,
            sticky="w",
            padx=10,
            pady=6,
        )

        self.alco_var = tk.StringVar(value="No")

        alco_box = ttk.Combobox(
            form_frame,
            textvariable=self.alco_var,
            values=["No", "Yes"],
            state="readonly",
            width=22,
        )

        alco_box.grid(
            row=4,
            column=3,
            padx=10,
            pady=6,
        )

        # -----------------------------------------------------
        # Physical activity
        # -----------------------------------------------------

        ttk.Label(
            form_frame,
            text="Physically Active",
        ).grid(
            row=5,
            column=2,
            sticky="w",
            padx=10,
            pady=6,
        )

        self.active_var = tk.StringVar(value="Yes")

        active_box = ttk.Combobox(
            form_frame,
            textvariable=self.active_var,
            values=["No", "Yes"],
            state="readonly",
            width=22,
        )

        active_box.grid(
            row=5,
            column=3,
            padx=10,
            pady=6,
        )

        # -----------------------------------------------------
        # Buttons
        # -----------------------------------------------------

        button_frame = ttk.Frame(self.content_frame)

        button_frame.pack(pady=15)

        ttk.Button(
            button_frame,
            text="Predict",
            command=self.predict_cardiovascular,
        ).pack(
            side="left",
            padx=10,
        )

        ttk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
        ).pack(
            side="left",
            padx=10,
        )

        # -----------------------------------------------------
        # Results
        # -----------------------------------------------------

        self.results_frame = ttk.LabelFrame(
            self.content_frame,
            text="Prediction Results",
            padding=15,
        )

        self.results_frame.pack(
            fill="both",
            expand=True,
            pady=5,
        )

    # =========================================================
    # PTB-XL interface
    # =========================================================

    def create_ptbxl_form(self):
        self.clear_content()

        upload_frame = ttk.LabelFrame(
            self.content_frame,
            text="PTB-XL ECG",
            padding=20,
        )

        upload_frame.pack(
            fill="x",
            pady=20,
        )

        ttk.Label(
            upload_frame,
            text=(
                "Select a PTB-XL ECG record (.hea).\n"
                "The corresponding .dat file must be present."
            ),
        ).pack(pady=(0, 15))

        ttk.Button(
            upload_frame,
            text="Select ECG Record",
            command=self.select_ecg,
        ).pack()

        self.ecg_label = ttk.Label(
            upload_frame,
            text="No ECG record selected.",
        )

        self.ecg_label.pack(pady=15)

        button_frame = ttk.Frame(self.content_frame)

        button_frame.pack(pady=10)

        ttk.Button(
            button_frame,
            text="Predict",
            command=self.predict_ptbxl,
        ).pack(
            side="left",
            padx=10,
        )

        ttk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
        ).pack(
            side="left",
            padx=10,
        )

        self.results_frame = ttk.LabelFrame(
            self.content_frame,
            text="ECG Prediction Results",
            padding=15,
        )

        self.results_frame.pack(
            fill="both",
            expand=True,
            pady=10,
        )

    # =========================================================
    # Switch prediction type
    # =========================================================

    def switch_prediction_type(self):
        if self.prediction_type.get() == "Cardiovascular":
            self.create_cardiovascular_form()
        else:
            self.create_ptbxl_form()

    # =========================================================
    # Cardiovascular prediction
    # =========================================================

    def predict_cardiovascular(self):
        try:
            patient = {
                "age_years": self.get_integer(
                    "age_years",
                    "Age",
                ),
                "gender": (1 if self.gender_var.get() == "Male" else 2),
                "height": self.get_integer(
                    "height",
                    "Height",
                ),
                "weight": self.get_float(
                    "weight",
                    "Weight",
                ),
                "ap_hi": self.get_integer(
                    "ap_hi",
                    "Systolic BP",
                ),
                "ap_lo": self.get_integer(
                    "ap_lo",
                    "Diastolic BP",
                ),
                "cholesterol": {
                    "Normal": 1,
                    "Above Normal": 2,
                    "Well Above Normal": 3,
                }[self.cholesterol_var.get()],
                "gluc": {
                    "Normal": 1,
                    "Above Normal": 2,
                    "Well Above Normal": 3,
                }[self.gluc_var.get()],
                "smoke": (1 if self.smoke_var.get() == "Yes" else 0),
                "alco": (1 if self.alco_var.get() == "Yes" else 0),
                "active": (1 if self.active_var.get() == "Yes" else 0),
            }

            results = predict_cardiovascular(
                self.tabular_models,
                patient,
            )

            overall = cardiovascular_majority_vote(results)

            self.display_cardiovascular_results(
                results,
                overall,
            )

        except ValueError as error:
            messagebox.showerror(
                "Invalid Input",
                str(error),
            )

        except Exception as error:
            messagebox.showerror(
                "Prediction Error",
                str(error),
            )

    # =========================================================
    # ECG file selection
    # =========================================================

    def select_ecg(self):
        file_path = filedialog.askopenfilename(
            title="Select PTB-XL ECG Record",
            filetypes=[
                ("WFDB Header", "*.hea"),
                ("All Files", "*.*"),
            ],
        )

        if not file_path:
            return

        path = Path(file_path)

        if path.suffix.lower() != ".hea":
            messagebox.showerror(
                "Invalid File",
                "Please select a PTB-XL .hea file.",
            )
            return

        dat_path = path.with_suffix(".dat")

        if not dat_path.exists():
            messagebox.showerror(
                "Missing ECG Data",
                (
                    "The corresponding .dat file was not found.\n\n"
                    f"Expected:\n{dat_path}"
                ),
            )
            return

        self.ecg_path = path.with_suffix("")

        self.ecg_label.config(text=f"Selected: {path.name}")

    # =========================================================
    # PTB-XL prediction
    # =========================================================

    def predict_ptbxl(self):
        if self.ecg_path is None:
            messagebox.showerror(
                "No ECG Selected",
                "Please select a PTB-XL ECG record first.",
            )
            return

        try:
            signal = load_ecg_record(self.ecg_path)

            results = predict_ptbxl(
                self.ptbxl_model,
                signal,
                self.thresholds,
            )

            self.display_ptbxl_results(results)

        except Exception as error:
            messagebox.showerror(
                "ECG Prediction Error",
                str(error),
            )

    # =========================================================
    # Display cardiovascular results
    # =========================================================

    def display_cardiovascular_results(
        self,
        results,
        overall,
    ):
        self.clear_results()

        for model_name, result in results.items():

            prediction_text = "Positive" if result["prediction"] == 1 else "Negative"

            probability = result["probability"]

            ttk.Label(
                self.results_frame,
                text=(f"{model_name}: " f"{prediction_text} " f"({probability:.2%})"),
                font=("Arial", 11),
            ).pack(
                anchor="w",
                pady=4,
            )

        ttk.Separator(
            self.results_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=10,
        )

        overall_text = "Positive" if overall["prediction"] == 1 else "Negative"

        ttk.Label(
            self.results_frame,
            text=f"Overall Result: {overall_text}",
            font=("Arial", 14, "bold"),
        ).pack(pady=5)

        ttk.Label(
            self.results_frame,
            text=(
                f"Positive votes: "
                f"{overall['positive_votes']}    "
                f"Negative votes: "
                f"{overall['negative_votes']}"
            ),
        ).pack()

    # =========================================================
    # Display PTB-XL results
    # =========================================================

    def display_ptbxl_results(self, results):
        self.clear_results()

        for class_name, result in results.items():

            status = "Positive" if result["positive"] else "Negative"

            ttk.Label(
                self.results_frame,
                text=(f"{class_name}: " f"{status} " f"({result['probability']:.2%})"),
                font=("Arial", 11),
            ).pack(
                anchor="w",
                pady=4,
            )

    # =========================================================
    # Input helpers
    # =========================================================

    def get_integer(self, key, field_name):
        value = self.entries[key].get().strip()

        if not value:
            raise ValueError(f"{field_name} is required.")

        try:
            return int(value)

        except ValueError:
            raise ValueError(f"{field_name} must be an integer.")

    def get_float(self, key, field_name):
        value = self.entries[key].get().strip()

        if not value:
            raise ValueError(f"{field_name} is required.")

        try:
            return float(value)

        except ValueError:
            raise ValueError(f"{field_name} must be a number.")

    # =========================================================
    # Reset
    # =========================================================

    def reset(self):
        self.ecg_path = None

        if self.prediction_type.get() == "Cardiovascular":
            self.create_cardiovascular_form()
        else:
            self.create_ptbxl_form()

    # =========================================================
    # Utility methods
    # =========================================================

    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def clear_results(self):
        if hasattr(self, "results_frame"):
            for widget in self.results_frame.winfo_children():
                widget.destroy()


def main():
    root = tk.Tk()

    HeartDiseaseApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
