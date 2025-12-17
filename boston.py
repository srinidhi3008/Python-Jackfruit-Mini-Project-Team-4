import wx
import wx.grid
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

class AppState:
    def __init__(self):
        self.df = None
        self.x = None
        self.y = None
        self.x_scaled = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None

state = AppState()

class DataPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.load_btn = wx.Button(self, label="Load 'Boston Dataset.csv'")
        self.load_btn.Bind(wx.EVT_BUTTON, self.on_load)
        
        self.log = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        font = wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.log.SetFont(font)
        
        main_sizer.Add(self.load_btn, 0, wx.ALL, 10)
        main_sizer.Add(wx.StaticText(self, label="Data Inspection Output:"), 0, wx.LEFT, 10)
        main_sizer.Add(self.log, 1, wx.EXPAND | wx.ALL, 10)
        
        self.SetSizer(main_sizer)

    def on_load(self, event):
        try:
            filename = 'Boston Dataset.csv'
            
            if not os.path.exists(filename):
                self.log.WriteText(f"Error: '{filename}' not found in the current directory.\n")
                return

            state.df = pd.read_csv(filename)
            
            self.log.Clear()
            self.log.WriteText("=== DATA LOADED SUCCESSFULLY ===\n\n")
            
            self.log.WriteText("--- df.head() ---\n")
            self.log.WriteText(state.df.head().to_string() + "\n\n")
            
            self.log.WriteText("--- df.describe() ---\n")
            self.log.WriteText(state.df.describe().to_string() + "\n\n")
            
            self.log.WriteText("--- df.isnull().sum() ---\n")
            self.log.WriteText(str(state.df.isnull().sum()) + "\n\n")

            state.x = state.df.drop("medv", axis=1)
            state.y = state.df["medv"]
            state.feature_names = state.x.columns

            scaler = StandardScaler()
            state.x_scaled = scaler.fit_transform(state.x)
            
            state.x_train, state.x_test, state.y_train, state.y_test = train_test_split(
                state.x_scaled, state.y, test_size=0.2, random_state=42
            )
            
            self.log.WriteText("--- PREPROCESSING COMPLETE ---\n")
            self.log.WriteText("Data scaled using StandardScaler.\n")
            self.log.WriteText("Data split into Train/Test sets (80/20).\n")
            
            wx.MessageBox("Data Loaded and Processed!", "Success", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            self.log.WriteText(f"Error: {e}\n")
            wx.MessageBox(f"Error: {e}", "Error", wx.OK | wx.ICON_ERROR)


class VizPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self, -1, self.figure)
        
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_corr = wx.Button(self, label="Show Correlation Heatmap")
        self.btn_hist = wx.Button(self, label="Show Price Histogram")
        
        self.btn_corr.Bind(wx.EVT_BUTTON, self.plot_heatmap)
        self.btn_hist.Bind(wx.EVT_BUTTON, self.plot_hist)
        
        btn_sizer.Add(self.btn_corr, 0, wx.ALL, 5)
        btn_sizer.Add(self.btn_hist, 0, wx.ALL, 5)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(btn_sizer, 0, wx.CENTER | wx.TOP, 10)
        main_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 10)
        
        self.SetSizer(main_sizer)

    def check_data_loaded(self):
        if state.df is None:
            wx.MessageBox("Please load data in the 'Data' tab first.", "Warning", wx.OK | wx.ICON_WARNING)
            return False
        return True

    def plot_heatmap(self, event):
        if not self.check_data_loaded(): return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        corr = state.df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": 8}, fmt=".2f")
        ax.set_title("Correlation Matrix")
        
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_hist(self, event):
        if not self.check_data_loaded(): return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        
        ax.hist(state.df["medv"], bins=30, edgecolor='black')
        ax.set_title("Distribution of House Prices")
        ax.set_xlabel("Price ($1000s)")
        ax.set_ylabel("Frequency")
        
        self.figure.tight_layout()
        self.canvas.draw()

class ModelPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.train_btn = wx.Button(self, label="Train Linear Regression Model")
        self.train_btn.Bind(wx.EVT_BUTTON, self.on_train)
        
        self.predict_btn = wx.Button(self, label="Predict Test Sample")
        self.predict_btn.Bind(wx.EVT_BUTTON, self.on_predict)
        self.predict_btn.Disable()
        
        self.text_out = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 100))
        
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self, -1, self.figure)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        btn_sizer.Add(self.train_btn, 0, wx.ALL, 5)
        btn_sizer.Add(self.predict_btn, 0, wx.ALL, 5)
        
        sizer.Add(btn_sizer, 0, wx.CENTER | wx.TOP, 10)
        sizer.Add(wx.StaticText(self, label="Model Evaluation:"), 0, wx.LEFT | wx.TOP, 10)
        sizer.Add(self.text_out, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 10)
        
        self.SetSizer(sizer)

    def on_train(self, event):
        if state.x_train is None:
            wx.MessageBox("Please load data in the 'Data' tab first.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        try:
            state.model = LinearRegression()
            state.model.fit(state.x_train, state.y_train)

            y_pred = state.model.predict(state.x_test)
            mse = mean_squared_error(state.y_test, y_pred)
            r2 = r2_score(state.y_test, y_pred)
            
            self.text_out.SetValue(f"Model Training Complete.\n\nMean Squared Error: {mse:.4f}\nR² Score: {r2:.4f}")
            self.predict_btn.Enable()
            
            self.plot_importance()
            
        except Exception as e:
            wx.MessageBox(f"Training Error: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def plot_importance(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        coef = pd.Series(state.model.coef_, index=state.feature_names).sort_values()
        
        x_positions = range(len(coef))
        ax.bar(x_positions, coef.values)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(coef.index, rotation=90)
        
        ax.set_title("Feature Importance")
        ax.axhline(0, color='black', linewidth=0.8)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def on_predict(self, event):
        sample_house = state.x_test[0].reshape(1, -1)
        predicted_price = state.model.predict(sample_house)
        actual_price = state.y_test.iloc[0]
        
        msg = (f"Sample: First row of Test Set (x_test[0])\n\n"
               f"Predicted Price: ${predicted_price[0]:.2f} (1000s)\n"
               f"Actual Price:    ${actual_price:.2f} (1000s)")
        
        wx.MessageBox(msg, "Prediction Result", wx.OK | wx.ICON_INFORMATION)


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Boston Housing Predictor", size=(900, 750))
        
        panel = wx.Panel(self)
        notebook = wx.Notebook(panel)
        
        self.tab1 = DataPanel(notebook)
        self.tab2 = VizPanel(notebook)
        self.tab3 = ModelPanel(notebook)
        
        notebook.AddPage(self.tab1, "1. Data Loading")
        notebook.AddPage(self.tab2, "2. Visualizations")
        notebook.AddPage(self.tab3, "3. Model & Predict")
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(notebook, 1, wx.EXPAND)
        panel.SetSizer(sizer)
        
        self.CreateStatusBar()
        self.SetStatusText("Ready. Load data to begin.")
        self.Center()

if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()