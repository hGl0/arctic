import os
import logging
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QComboBox, QPushButton,
    QListWidget, QListWidgetItem, QMessageBox, QCheckBox,
    QHBoxLayout, QLineEdit
)
from PyQt6.QtCore import Qt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime

from filter_window import FilterWindow

from vortexclust.io.cleaner import to_date

def select_scaler(name):
    if name == 'StandardScaler': return StandardScaler()
    if name == 'RobustScaler': return RobustScaler()
    if name == 'MinMaxScaler': return MinMaxScaler()
    return None


month_map = {
    'January': 1, 'February': 2, 'March': 3,
    'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9,
    'October': 10, 'November': 11, 'December': 12
}

class ScaleWindow(QWidget):
    def __init__(self, df, output_path):
        super().__init__()
        self.df = df
        self.output_path = output_path
        self.setWindowTitle("Preprocessing: Scaling")
        self.setGeometry(300, 300, 600, 600)

        layout = QVBoxLayout()

        # Time column selection
        layout.addWidget(QLabel("Select Time Column:"))
        self.time_selector = QComboBox()
        self.time_selector.addItems(df.columns)
        layout.addWidget(self.time_selector)

        # Time format selection
        COMMON_FORMATS = [
            "%d.%m.%Y-%H:%M:%S",
            "%d.%m.%Y-%H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%d/%m/%Y %H:%M",
        ]
        self.format_label = QLabel("Select Time Format:")
        self.format_selector = QComboBox()
        self.format_selector.addItems(COMMON_FORMATS)
        self.custom_format_input = QLineEdit()
        self.custom_format_input.setPlaceholderText("Or enter custom format")

        format_layout = QHBoxLayout()
        format_layout.addWidget(self.format_selector)
        format_layout.addWidget(QLabel("or"))
        format_layout.addWidget(self.custom_format_input)
        layout.addWidget(QLabel("Time Format:"))
        layout.addLayout(format_layout)

        # Scaler selection
        layout.addWidget(QLabel("Select Scaler:"))
        self.scaler_selector = QComboBox()
        self.scaler_selector.addItems(["None", "StandardScaler", "RobustScaler", "MinMaxScaler"])
        layout.addWidget(self.scaler_selector)

        # Time constraints
        self.use_constraints = QCheckBox("Apply Time Constraints")
        layout.addWidget(self.use_constraints)

        self.start_date_input = QLineEdit()
        self.start_date_input.setPlaceholderText("dd/mm/yyyy")
        self.end_date_input = QLineEdit()
        self.end_date_input.setPlaceholderText("dd/mm/yyyy")

        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Start Date:"))
        date_layout.addWidget(self.start_date_input)
        date_layout.addWidget(QLabel("End Date:"))
        date_layout.addWidget(self.end_date_input)
        layout.addLayout(date_layout)

        self.month_selector = QListWidget()
        self.month_selector.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        for m in month_map:
            item = QListWidgetItem(str(m))
            self.month_selector.addItem(item)

        layout.addWidget(QLabel("Select Months of Interest (optional):"))
        layout.addWidget(self.month_selector)

        # Features for plotting
        layout.addWidget(QLabel("Select Features to Plot:"))
        self.feature_selector = QListWidget()
        self.feature_selector.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for col in df.columns:
            item = QListWidgetItem(col)
            self.feature_selector.addItem(item)
        layout.addWidget(self.feature_selector)

        # Days to plot
        self.days_input = QLineEdit()
        self.days_input.setPlaceholderText("Number of days to plot (optional)")
        layout.addWidget(QLabel("Limit Number of Days to Plot (from start date):"))
        layout.addWidget(self.days_input)

        # Process button
        process_btn = QPushButton("Apply Scaling and Plot")
        process_btn.clicked.connect(self.scale_data)
        filter_btn = QPushButton("Continue with filtering")
        filter_btn.clicked.connect(self.start_filter)
        layout.addWidget(process_btn)
        layout.addWidget(filter_btn)

        self.setLayout(layout)

    def scale_data(self):
        df = self.df.copy()
        time_col = self.time_selector.currentText()
        format_from_dropdown = self.format_selector.currentText()
        format_from_custom = self.custom_format_input.text().strip()
        time_format = format_from_custom if format_from_custom else format_from_dropdown

        # convert to datetime with fallback
        try:
            to_date(df, time_col, format=time_format)
        except Exception as e:
            try:
                to_date(df, time_col, format='mixed')
                QMessageBox.warning(
                    self,
                    "Format Warning",
                    f"Failed to convert using provided format:\n{e}\n\nFalling back to mixed format detection."
                )
            except Exception as fallback_e:
                QMessageBox.critical(self, "Error", f"Failed to convert time column with fallback:\n{fallback_e}")
                return

        # Apply constraints if selected
        if self.use_constraints.isChecked():
            try:
                start_str = self.start_date_input.text().strip()
                end_str = self.end_date_input.text().strip()
                start_dt = datetime.strptime(start_str, "%d/%m/%Y") if start_str else df[time_col].min()
                end_dt = datetime.strptime(end_str, "%d/%m/%Y") if end_str else df[time_col].max()

                actual_min = df[time_col].min()
                actual_max = df[time_col].max()
                clipped = False
                if pd.Timestamp(start_dt).date() < pd.Timestamp(actual_min).date():
                    start_dt = actual_min
                    clipped = True
                if end_dt > actual_max:
                    end_dt = actual_max
                    clipped = True

                if clipped:
                    QMessageBox.warning(
                        self,
                        "Date Constraint Warning",
                        f"Selected dates exceed available data range.\n"
                        f"Using available range from {actual_min} to {actual_max}.\n"
                        f"{df[time_col].nunique()} values detected. "
                    )

                df = df[(df[time_col] >= start_dt) & (df[time_col] <= end_dt)]

                selected = [item.text() for item in self.month_selector.selectedItems()]
                months = [month_map[name] for name in selected]

                if months:
                    df = df[df[time_col].dt.month.isin(months)]
                    QMessageBox.information(self, "Selected Months", f"Selected months: {df[time_col].dt.month.nunique()}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Invalid date format. Please use dd/mm/yyyy.\n{e}")
                return

        # Scaling
        scaler_name = self.scaler_selector.currentText()
        scaler = select_scaler(scaler_name)

        if scaler:
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            df_scaled = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)
        else:
            df_scaled = df.copy()

        # reset df to df_scaled
        self.df = df_scaled

        # Plotting
        selected_features = [item.text() for item in self.feature_selector.selectedItems()]
        if len(selected_features) % 2 != 0:
            QMessageBox.warning(self, "Feature Selection", "Please select an even number of features for paired plotting.")
            return

        days_str = self.days_input.text().strip()
        if days_str.isdigit():
            num_days = int(days_str)
        else:
            num_days = -1
        if num_days > df_scaled.shape[0]:
            QMessageBox.warning(self, "Feature Selection", f"Please select a number of days to plot.\n The maximum number are {df_scaled.shape[0]}.")

        n_pairs = len(selected_features) // 2
        fig, axes = plt.subplots(n_pairs, figsize=(10, 4 * n_pairs))
        if n_pairs == 1:
            axes = [axes]

        for i in range(len(axes)):
            f1, f2 = selected_features[2*i], selected_features[2*i + 1]
            axes[i].plot(df_scaled[f1][:num_days], label=f1)
            axes[i].plot(df_scaled[f2][:num_days], label=f2)
            axes[i].legend(loc='upper left')

        plt.tight_layout()
        os.makedirs(self.output_path, exist_ok=True)
        out_path = os.path.join(self.output_path, "scaled_features.png")
        plt.savefig(out_path)
        plt.close()

        QMessageBox.information(self, "Success", f"Scaled features plotted and saved to:\n{out_path}")

    def start_filter(self):
        # Proceed to scaling
        df_scaled = self.df.copy()
        self.filter_window = FilterWindow(df_scaled, self.output_path)
        self.filter_window.show()
        self.hide()
