#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lib_format as fm
import lib_analyza as anl

import time as Time
# import threading as thr
# import multiprocessing as mtp

get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
import numpy as np
import pandas as pd

get_ipython().system('pip install plotly')
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

get_ipython().system('pip install dash==1.19.0  ')
get_ipython().system('pip install dash_bootstrap_components ')
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output, State

get_ipython().system('pip install webbrowser')
import webbrowser 


# In[2]:


directory = "data/"
merania = fm.intro(directory)


def generate_options(merania):
    meranie = []
    for i in merania:
        meranie.append({"label": "Dataset "+i, "value": i})
    return meranie


meranie = generate_options(merania)
merania, meranie


# In[3]:


datasets = meranie

settings = [{'label': 'My.QLS', 'value': 'My'},
            {'label': 'Custom.QLS', 'value': 'Custom'}]


# In[4]:


file = merania[1]
time, mag, n = fm.load(file, directory)

dataset = pd.DataFrame(
    {
        "Time": time,
        "Magnetická intenzita": mag,
        "Magnetická intenzita (normalizovaná)": None,
        "Magnetická intenzita (FFT)": None,
        "Magnetická intenzita (SavGol)": None,
    }
)

dataset["Magnetická intenzita (normalizovaná)"] = anl.norm(dataset["Magnetická intenzita"])


# In[5]:


fig0 = go.Figure()

fig0.layout = {
    "title": "Názov Grafu",
    "title_font_color": "#009670",
    "template": "simple_white",  # simple_white
    "plot_bgcolor": "rgba(255,255,255,0)",  # rgba(255,255,255,1)
    "paper_bgcolor": "rgba(255,255,255,0)",
    "legend": {
        "x": 0,  # 0
        "y": 1,  # 1
        "bgcolor": "#2f4b7c",  # "rgba(255,255,255,1)"
        "bordercolor": "#665191",  # Black
        "borderwidth": 1,  # 1
    },
    "xaxis": {
        "color": "#a05195",
        "linecolor": "#a05195",
        "title": "Názov x-ovej osi",
        "ticklen": 5,  # 5
        "zeroline": False,  # False
        "rangeslider": {"visible": True},
    },
    "yaxis": {
        "color": "#a05195",
        "linecolor": "#a05195",
        "title": "Názov y-ovej osi",
        "ticklen": 10,  # 5
        "zeroline": False,  # False
    },
}


fig0.add_trace(
    go.Scatter(
        x=dataset["Time"],  # x
        y=dataset["Magnetická intenzita (normalizovaná)"],  # y
        line={
            "color": "#009670",  # rgba(0, 158, 115, 1)
            "width": 1,  # 1
            "dash": "solid",  # solid
        },
        mode="lines",
        name="názov čiary",  # Mag_small
        marker={"color": "#009670"},  # "color": 'rgba(0, 114, 178, 1)"
    )
)

fig0.show("notebook")


# In[6]:


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])  # BOOTSTRAP, https://codebeautify.org/python-formatter-beautifier


# In[7]:


@app.callback(Output("output1", "value"), [Input("data_options", "value")])
def change_dataset(tem):
    file = tem
    time, mag, n = fm.load(file, directory)
    global dataset 
    dataset = pd.DataFrame(
        {
            "Time": time,
            "Magnetická intenzita": mag,
            "Magnetická intenzita (normalizovaná)": 0,
            "Magnetická intenzita (FFT)": 0,
            "Magnetická intenzita (SavGol)": 0,
        }
    )
    dataset["Magnetická intenzita (normalizovaná)"] = anl.norm(dataset["Magnetická intenzita"])
    return tem


# In[8]:


dataset


# In[9]:


# a_slider <=> a_input
@app.callback(
    Output("a_input", "value"),
    Output("a_slider", "value"),
    Input("a_input", "value"),
    Input("a_slider", "value"),
)
def callback(input_value, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = input_value if trigger_id == "a_input" else slider_value
    return value, value


# a_slider <=> a_input
@app.callback(
    Output("σ_input", "value"),
    Output("σ_slider", "value"),
    Input("σ_input", "value"),
    Input("σ_slider", "value"),
)
def callback(input_value, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = input_value if trigger_id == "σ_input" else slider_value
    return value, value


# win_slider <=> win_input
@app.callback(
    Output("win_input", "value"),
    Output("win_slider", "value"),
    Input("win_input", "value"),
    Input("win_slider", "value"),
)
def callback(input_value, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = input_value if trigger_id == "win_input" else slider_value
    return value, value


# pol_slider <=> pol_input
@app.callback(
    Output("pol_input", "value"),
    Output("pol_slider", "value"),
    Input("pol_input", "value"),
    Input("pol_slider", "value"),
)
def callback(input_value, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = input_value if trigger_id == "pol_input" else slider_value
    return value, value


# tr+_slider <=> tr+_input
@app.callback(
    Output("tr+_input", "value"),
    Output("tr+_slider", "value"),
    Input("tr+_input", "value"),
    Input("tr+_slider", "value"),
)
def callback(input_value, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = input_value if trigger_id == "tr+_input" else slider_value
    return value, value


# tr-_slider <=> tr-_input
@app.callback(
    Output("tr-_input", "value"),
    Output("tr-_slider", "value"),
    Input("tr-_input", "value"),
    Input("tr-_slider", "value"),
)
def callback(input_value, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = input_value if trigger_id == "tr-_input" else slider_value
    return value, value


# pro_slider <=> pro_input
@app.callback(
    Output("pro_input", "value"),
    Output("pro_slider", "value"),
    Input("pro_input", "value"),
    Input("pro_slider", "value"),
)
def callback(input_value, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = input_value if trigger_id == "pro_input" else slider_value
    return value, value


# wid_slider <=> wid_input
@app.callback(
    Output("wid_input", "value"),
    Output("wid_slider", "value"),
    Input("wid_input", "value"),
    Input("wid_slider", "value"),
)
def callback(input_value, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = input_value if trigger_id == "wid_input" else slider_value
    return value, value


# dis_slider <=> dis_input
@app.callback(
    Output("dis_input", "value"),
    Output("dis_slider", "value"),
    Input("dis_input", "value"),
    Input("dis_slider", "value"),
)
def callback(input_value, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = input_value if trigger_id == "dis_input" else slider_value
    return value, value


# hei_slider <=> hei_input
@app.callback(
    Output("hei_input", "value"),
    Output("hei_slider", "value"),
    Input("hei_input", "value"),
    Input("hei_slider", "value"),
)
def callback(input_value, slider_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = input_value if trigger_id == "hei_input" else slider_value
    return value, value


# In[10]:


@app.callback(Output("fig1", "figure"), [Input("start_button", "n_click"), Input("res", "value")])
def make_fig1(n_click, value):
    figure = go.Figure()

    figure.layout = {
        "title": "Závislosť mag. poľa od času, normalizované",
        "title_font_color": "#009670",
        "template": "simple_white",  # simple_white
        "plot_bgcolor": "rgba(255,255,255,0)",  # rgba(255,255,255,1)
        "paper_bgcolor": "rgba(255,255,255,0)",
        "legend": {
            "x": 0,  # 0
            "y": 1,  # 1
            "bgcolor": "#2f4b7c",  # "rgba(255,255,255,1)"
            "bordercolor": "#665191",  # Black
            "borderwidth": 1,  # 1
        },
        "xaxis": {
            "color": "#a05195",
            "linecolor": "#a05195",
            "title": "Čas [s]",
            "ticklen": 5,  # 5
            "zeroline": False,  # False
            "rangeslider": {"visible": True},
        },
        "yaxis": {
            "color": "#a05195",
            "linecolor": "#a05195",
            "title": "Intenzita [AU]",
            "ticklen": 10,  # 5
            "zeroline": False,  # False
        },
    }

    figure.add_trace(
        go.Scatter(
            x=dataset["Time"][::value],  # x
            y=dataset["Magnetická intenzita (normalizovaná)"][::value],  # y
            line={
                "color": "#009670",  # rgba(0, 158, 115, 1)
                "width": 1,  # 1
                "dash": "solid",  # solid
            },
            mode="lines",
            name="vývoj mag. intenzity",  # Mag_small
            marker={"color": "#009670"},  # "color": 'rgba(0, 114, 178, 1)"
        )
    )

    return figure


# In[11]:


@app.callback(Output("fig2", "figure"), [Input("start_button", "n_click"), Input("res", "value"), Input("a_input", "value"), Input("σ_input", "value")])
def make_fig2(n_click, value, a, sigma):
    
    f = np.fft.rfftfreq(dataset["Magnetická intenzita (normalizovaná)"].size)
    I = np.fft.rfft(dataset["Magnetická intenzita (normalizovaná)"], n=dataset.shape[0]) 
#     f = np.fft.rfftfreq(dataset["Magnetická intenzita"].size)
#     I = np.fft.rfft(dataset["Magnetická intenzita"], n=dataset.shape[0]) 
    
    
    temp = anl.norm(I.real)
    temp = temp * np.exp(-sigma*sigma*(f-a/10000)*(f-a/10000))
    temp = np.fft.irfft(temp, n=dataset.shape[0])
    dataset["Magnetická intenzita (FFT)"] = anl.norm(temp)
    
#     temp = I.real
#     temp = temp * np.exp(-sigma*sigma*(f-a/10000)*(f-a/10000))
#     temp = np.fft.irfft(temp, n=dataset.shape[0])
#     dataset["Magnetická intenzita (FFT)"] = temp
    
    figure = go.Figure()

    figure.layout = {
        "title": "FFT intenzita v závislosti frekvencie",
        "title_font_color": "#009670",
        "template": "simple_white",  # simple_white
        "plot_bgcolor": "rgba(255,255,255,0)",  # rgba(255,255,255,1)
        "paper_bgcolor": "rgba(255,255,255,0)",
        "showlegend": False,
        "legend": {
            "x": 0,  # 0
            "y": 1,  # 1
            "bgcolor": "#2f4b7c",  # "rgba(255,255,255,1)"
            "bordercolor": "#665191",  # Black
            "borderwidth": 1,  # 1
        },
        "xaxis": {
            "color": "#a05195",
            "linecolor": "#a05195",
            "title": "Frekvencia [AU]",
            "ticklen": 5,  # 5
            "zeroline": False,  # False
            "rangeslider": {"visible": True},
            "range": [-0,0.01],
        },
        "yaxis": {
            "color": "#a05195",
            "linecolor": "#a05195",
            "title": "Intenzita [AU]",
            "ticklen": 10,  # 5
            "zeroline": False,  # False
            "range": [-1,+1],
        },
    }

    figure.add_trace(
        go.Scatter(
            x=f[::value],  # x
            y=anl.norm(I.real)[::value],  # y
            line={
                "color": "#009670",  # rgba(0, 158, 115, 1)
                "width": 1,  # 1
                "dash": "solid",  # solid
            },
            mode="lines",
            name="FFT",  # Mag_small
            marker={"color": "#009670"},  # "color": 'rgba(0, 114, 178, 1)"
        )
    )
    
    figure.add_trace(
        go.Scatter(
            x=f[::value],  # x
            y=np.exp(-sigma*sigma*(f-a/10000)*(f-a/10000))[::value],  # y
            line={
                "color": "#ff7c43",  # rgba(0, 158, 115, 1)
                "width": 1,  # 1
                "dash": "solid",  # solid
            },
            mode="lines",
            name="Filter",  
            marker={"color": "#ff7c43"},  # "color": 'rgba(0, 114, 178, 1)"
        ))

    return figure


# In[12]:


@app.callback(Output("fig3", "figure"), [Input("start_button", "n_click"), Input("res", "value"), Input("a_input", "value"), Input("σ_input", "value"), Input("win_input", "value"), Input("pol_input", "value")])
def make_fig3(n_click, value, a, sigma, win, pol):
    
    f = np.fft.rfftfreq(dataset["Magnetická intenzita (normalizovaná)"].size)
    I = np.fft.rfft(dataset["Magnetická intenzita (normalizovaná)"], n=dataset.shape[0]) 
    
    temp = anl.norm(I)
    temp = temp * np.exp(-sigma*sigma*(f-a/10000)*(f-a/10000))
    temp = np.fft.irfft(temp, n=dataset.shape[0])
    temp = anl.norm(temp)

#     f = np.fft.rfftfreq(dataset["Magnetická intenzita"].size)
#     I = np.fft.rfft(dataset["Magnetická intenzita"], n=dataset.shape[0]) 
    
#     temp = I
#     temp = temp * np.exp(-sigma*sigma*(f-a/10000)*(f-a/10000))
#     temp = np.fft.irfft(temp, n=dataset.shape[0])
#     temp = temp
    
    dataset["Magnetická intenzita (SavGol)"] = signal.savgol_filter(temp, win, pol, mode="constant")
    
    figure = go.Figure()

    figure.layout = {
        "title": "Závislosť mag. poľa od času, po FFT a Sav-Gol filtre",
        "title_font_color": "#009670",
        "template": "simple_white",  # simple_white
        "plot_bgcolor": "rgba(255,255,255,0)",  # rgba(255,255,255,1)
        "paper_bgcolor": "rgba(255,255,255,0)",
        "showlegend": False,
        "legend": {
            "x": 0,  # 0
            "y": 1,  # 1
            "bgcolor": "#2f4b7c",  # "rgba(255,255,255,1)"
            "bordercolor": "#665191",  # Black
            "borderwidth": 1,  # 1
        },
        "xaxis": {
            "color": "#a05195",
            "linecolor": "#a05195",
            "title": "Čas [s]",
            "ticklen": 5,  # 5
            "zeroline": False,  # False
            "rangeslider": {"visible": True}
        },
        "yaxis": {
            "color": "#a05195",
            "linecolor": "#a05195",
            "title": "Intenzita [AU]",
            "ticklen": 10,  # 5
            "zeroline": False,  # False
            "range": [-1,+1],
        },
    }

    figure.add_trace(
        go.Scatter(
            x=dataset["Time"][::value],  # x
            y=anl.norm(dataset["Magnetická intenzita (SavGol)"])[::value],  # y
            line={
                "color": "#009670",  # rgba(0, 158, 115, 1)
                "width": 1,  # 1
                "dash": "solid",  # solid
            },
            mode="lines",
            name="vývoj mag. intenzity",  # Mag_small
            marker={"color": "#009670"},  # "color": 'rgba(0, 114, 178, 1)"
        )
    )
    
    return figure


# In[13]:


@app.callback(Output("fig4", "figure"), 
              [Input("start_button", "n_click"), 
               Input("res", "value"), 
               Input("a_input", "value"), 
               Input("σ_input", "value"), 
               Input("win_input", "value"), 
               Input("pol_input", "value"),
               Input("tr+_input", "value"), 
               Input("tr-_input", "value"),
               Input("dis_input", "value"), 
               Input("pro_input", "value"), 
               Input("wid_input", "value"), 
               Input("hei_input", "value"), 
              ])
def make_fig4(n_click, value, a, sigma, win, pol, trp, trm, dis, pro, wid, hei):
    
    f = np.fft.rfftfreq(dataset["Magnetická intenzita (normalizovaná)"].size)
    I = np.fft.rfft(dataset["Magnetická intenzita (normalizovaná)"], n=dataset.shape[0]) 
    
    temp = anl.norm(I)
    temp = temp * np.exp(-sigma*sigma*(f-a/10000)*(f-a/10000))
    temp = np.fft.irfft(temp, n=dataset.shape[0])
    temp = anl.norm(temp)
    
    temp = signal.savgol_filter(temp, win, pol, mode="constant")
    
    tempp, properties = signal.find_peaks(temp, height=hei, threshold=trp, distance=dis, prominence=pro, width=wid)
    tempm, properties = signal.find_peaks(-temp, height=hei, threshold=trm, distance=dis, prominence=pro, width=wid)

#     f = np.fft.rfftfreq(dataset["Magnetická intenzita"].size)
#     I = np.fft.rfft(dataset["Magnetická intenzita"], n=dataset.shape[0]) 
    
#     temp = I
#     temp = temp * np.exp(-sigma*sigma*(f-a/10000)*(f-a/10000))
#     temp = np.fft.irfft(temp, n=dataset.shape[0])
#     temp = temp
    
#     temp = signal.savgol_filter(temp, win, pol, mode="constant")
    
#     tempp, properties = signal.find_peaks(temp, height=hei, threshold=trp, distance=dis, prominence=pro, width=wid)
#     tempm, properties = signal.find_peaks(-temp, height=hei, threshold=trm, distance=dis, prominence=pro, width=wid)

    
#     temp = np.concatenate((tempp, tempm))
#     peaks_time = dataset["Time"][temp]
#     peaks_mag = dataset["Magnetická intenzita"][temp]
#     np.savetxt("output_E.txt", np.transpose(np.array([np.unique(peaks_time), np.unique(peaks_mag)])), delimiter='\t', newline="\n")
# #     fm.save_data(peaks_time, peaks_mag, directory="UFA_peak_finder_6.0", file="EEE")

    figure = go.Figure()

    figure.layout = {
        "title": "Závislosť mag. poľa od času, s vrcholmie",
        "title_font_color": "#009670",
        "template": "simple_white",  # simple_white
        "plot_bgcolor": "rgba(255,255,255,0)",  # rgba(255,255,255,1)
        "paper_bgcolor": "rgba(255,255,255,0)",
        "showlegend": False,
        "legend": {
            "x": 0,  # 0
            "y": 1,  # 1
            "bgcolor": "#2f4b7c",  # "rgba(255,255,255,1)"
            "bordercolor": "#665191",  # Black
            "borderwidth": 1,  # 1
        },
        "xaxis": {
            "color": "#a05195",
            "linecolor": "#a05195",
            "title": "Čas [s]",
            "ticklen": 5,  # 5
            "zeroline": False,  # False
            "rangeslider": {"visible": True}
        },
        "yaxis": {
            "color": "#a05195",
            "linecolor": "#a05195",
            "title": "Intenzita [AU]",
            "ticklen": 10,  # 5
            "zeroline": False,  # False
            "range": [-1,+1],
        },
    }

    figure.add_trace(
        go.Scatter(
            x=dataset["Time"][::value],  # x
            y=dataset["Magnetická intenzita (SavGol)"][::value],  # y
            line={
                "color": "#009670",  # rgba(0, 158, 115, 1)
                "width": 1,  # 1
                "dash": "solid",  # solid
            },
            mode="lines",
            name="vývoj mag. intenzity",  # Mag_small
            marker={"color": "#009670"},  # "color": 'rgba(0, 114, 178, 1)"
        )
    )
    
    figure.add_trace(
        go.Scatter(
            x=dataset["Time"][tempp],  # x
            y=dataset["Magnetická intenzita (SavGol)"][tempp],  # y
            line={
                "color": "#009670",  # rgba(0, 158, 115, 1)
                "width": 1,  # 1
                "dash": "solid",  # solid
            },
            mode="markers",
            name="vývoj mag. intenzity",  # Mag_small
            marker={"color": "#d45087"},  # "color": 'rgba(0, 114, 178, 1)"
        )
    )
    
    figure.add_trace(
        go.Scatter(
            x=dataset["Time"][tempm],  # x
            y=dataset["Magnetická intenzita (SavGol)"][tempm],  # y
            line={
                "color": "#009670",  # rgba(0, 158, 115, 1)
                "width": 1,  # 1
                "dash": "solid",  # solid
            },
            mode="markers",
            name="vývoj mag. intenzity",  # Mag_small
            marker={"color": "#d45087"},  # "color": 'rgba(0, 114, 178, 1)"
        )
    )
    
    temp = np.concatenate((tempp, tempm))
    peaks_time = dataset["Time"][temp]
    peaks_mag = dataset["Magnetická intenzita"][temp]
    np.savetxt("output/output_E.txt", np.transpose(np.array([np.unique(peaks_time), np.unique(peaks_mag)])), delimiter='\t', newline="\n")

    return figure


# In[14]:


# # "save_button"
# @app.callback(Output("output1", "value"),
#                 Input("save_button", "n_clicks"),
#               [State("res", "value"), 
#                State("a_input", "value"), 
#                State("σ_input", "value"), 
#                State("win_input", "value"), 
#                State("pol_input", "value"),
#                State("tr+_input", "value"), 
#                State("tr-_input", "value"),
#                State("dis_input", "value"), 
#                State("pro_input", "value"), 
#                State("wid_input", "value"), 
#                State("hei_input", "value"), 
#               ])
# def save(value, a, sigma, win, pol, trp, trm, dis, pro, wid, hei, n_click):
    
#     f = np.fft.rfftfreq(dataset["Magnetická intenzita (normalizovaná)"].size)
#     I = np.fft.rfft(dataset["Magnetická intenzita (normalizovaná)"], n=dataset.shape[0]) 
    
#     temp = anl.norm(I)
#     temp = temp * np.exp(-sigma*sigma*(f-a/10000)*(f-a/10000))
#     temp = np.fft.irfft(temp, n=dataset.shape[0])
#     temp = anl.norm(temp)
    
#     temp = signal.savgol_filter(temp, win, pol, mode="constant")
    
#     tempp, properties = signal.find_peaks(temp, height=hei, threshold=trp, distance=dis, prominence=pro, width=wid)
#     tempm, properties = signal.find_peaks(-temp, height=hei, threshold=trm, distance=dis, prominence=pro, width=wid)
    
#     temp = np.concatenate((tempp, tempm))
    
#     peaks_time = dataset["Time"][temp],  # x
#     peaks_mag = dataset["Magnetická intenzita"][temp]
        
#     fm.save_txt(peaks_time, peaks_mag)
#     return ""


# In[15]:


# #############################################
# @app.callback(Input("save_button", "n_click"), 
#               [State("res", "value"), 
#                State("a_input", "value"), 
#                State("σ_input", "value"), 
#                State("win_input", "value"), 
#                State("pol_input", "value"),
#                State("tr+_input", "value"), 
#                State("tr-_input", "value"),
#                State("dis_input", "value"), 
#                State("pro_input", "value"), 
#                State("wid_input", "value"), 
#                State("hei_input", "value"), 
#               ])
# def sace(n_click, value, a, sigma, win, pol, trp, trm, dis, pro, wid, hei):
#     Time.sleep(10)
#     if n_clicks == 0:pass
#     else:
#         f = np.fft.rfftfreq(dataset["Magnetická intenzita (normalizovaná)"].size)
#         I = np.fft.rfft(dataset["Magnetická intenzita (normalizovaná)"], n=dataset.shape[0]) 

#         temp = anl.norm(I)
#         temp = temp * np.exp(-sigma*sigma*(f-a/10000)*(f-a/10000))
#         temp = np.fft.irfft(temp, n=dataset.shape[0])
#         temp = anl.norm(temp)

#         temp = signal.savgol_filter(temp, win, pol, mode="constant")

#         tempp, properties = signal.find_peaks(temp, height=hei, threshold=trp, distance=dis, prominence=pro, width=wid)
#         tempm, properties = signal.find_peaks(-temp, height=hei, threshold=trm, distance=dis, prominence=pro, width=wid)

#         temp = np.concatenate((tempp, tempm))
#         peaks_time = dataset["Time"][temp]
#         peaks_mag = dataset["Magnetická intenzita"][temp]
#         np.savetxt("output_E.txt", np.transpose(np.array([np.unique(peaks_time), np.unique(peaks_mag)])), delimiter='\t', newline="\n")


# In[16]:


# Prvá lajna s Menom, logom a contribution

uvodna_linka = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.H1("SpearPy"), width="auto"),
                dbc.Col(dbc.CardImg(src="assets/img.png"), width="1"),
                dbc.Col(html.Div("by: Samuel Amrich"), width="auto"),
            ],
            align="end",
            justify="between",
        )
    ]
)


# In[17]:


# Druhá lajna s

nacitacia_linka = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        options=datasets,
                        id="data_options",
#                         value="E5",
                        multi=False,
                        style={"width": "200%"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dcc.Input(
                        id="path_input",
                        type="text",
                        value="UFA_peak_finder_6.0/data",
                        style={"width": "200%"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        "Start",
                        id="start_button",
                        color="success",
                        className="mr-2",
                        size="lg",
                        n_clicks = 0
                    ),
                    width="auto",
                ),
            ],
            align="end",
            justify="between",
        )
    ]
)


# In[18]:


rozlisovacia_linka = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div("Rozlišovacia presnosť:"), width=4),
                dbc.Col(
                    dbc.RadioItems(
                        id= "res",
                        options=[
                            {"label": "Accurate/ Slow (1:1)", "value": 1},
                            {"label": "Mediocre/ Fast (1:10)", "value": 10},
                            {"label": "erroneous/ superfast (1:100)", "value": 100},
                        ],
                        value=1,
                        inline=True,
                    ),
                    width=20,
                ),
            ],
            align="end",
            justify="center",
        )
    ]
)


# In[19]:


stavova_linka = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        options=settings,
                        value=None,
                        multi=False,
                        style={"width": "200%"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dcc.Input(
                        id="path_set",
                        type="text",
                        value="UFA_peak_finder_6.0/settings",
                        style={"width": "200%"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        "Load",
                        id="load_button",
                        color="success",
                        className="mr-1",
                        size="lg",
                    ),
                    width="auto",
                ),
            ],
            align="end",
            justify="between",
        )
    ]
)


# In[20]:


# progress_linka = html.Div(
#     [
#         dbc.Progress(id="progress", value=0, striped=True, animated=True),
#         dcc.Interval(id="interval", interval=250, n_intervals=0),
#     ]
# )


# @app.callback(Output("progress", "value"), [Input("interval", "n_intervals")])
# def advance_progress(n):
#     return min(n % 110, 100)


# In[21]:


# Graf 1 - Povodne data normalizovane inak nic

# graf_1 = html.Div(dcc.Graph(id='example-graph-1', figure=fig0))
graf_1 = dcc.Graph(id='fig1')


# In[22]:


vyhladzovacia_linka = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("a slider: "),
                        dcc.Slider(id="a_slider", min=0, max=10, value=0, step=0.1),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.Div("σ slider: "),
                        dcc.Slider(id="σ_slider", min=0, max=5000, value=500, step=10),
                    ],
                    width=6,
                ),
            ],
            align="end",
            justify="center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("a input: "),
                        dcc.Input(
                            id="a_input",
                            type="number",
                            min=0, 
                            max=10,
                            value=0,
                            style={"width": "200%"},
                        ),
                    ],
                    width="6",
                ),
                dbc.Col(
                    [
                        html.Div("σ input: "),
                        dcc.Input(
                            id="σ_input",
                            type="number",
                            min=0, 
                            max=5000,
                            value=500,
                            style={"width": "200%"},
                        ),
                    ],
                    width="6",
                )
            ],
            align="start",
            justify="start",
        )
    ]
)


# In[23]:


# Graf 2 - Data odsuemne cez fourierovku, ale ukazujem samotnu furierovku plus osekavaciu funkciu

# graf_2 = html.Div(dcc.Graph(id='example-graph-2', figure=fig0))
graf_2 = dcc.Graph(id='fig2')


# In[24]:


aproximacna_linka = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("win slider: "),
                        dcc.Slider(id="win_slider", min=1, max=501, value=101, step=2),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.Div("pol slider: "),
                        dcc.Slider(id="pol_slider", min=0, max=10, value=2, step=1),
                    ],
                    width=6,
                ),
            ],
            align="end",
            justify="between",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("win input: "),
                        dcc.Input(
                            id="win_input",
                            type="number",
                            min=0,
                            max=501,
                            value=101,
                            style={"width": "150%"},
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        html.Div("pol input: "),
                        dcc.Input(
                            id="pol_input",
                            type="number",
                            min=0,
                            max=10,
                            value=2,
                            style={"width": "150%"},
                        ),
                    ],
                    width="auto",
                )
            ],
            align="end",
            justify="between",
        ),
    ]
)


# In[25]:


#Graf 3 - Data vyhladene cez Savinsky golansky filter

# graf_3 = html.Div(dcc.Graph(id='example-graph-3', figure=fig0))
graf_3 = dcc.Graph(id='fig3')


# In[26]:


peak_linka = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("tr+ slider: "),
                        dcc.Slider(id="tr+_slider", min=0, max=0.0001, value=0, step=0.000001),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.Div("tr- slider: "),
                        dcc.Slider(id="tr-_slider", min=0, max=0.0001, value=0, step=0.000001),
                    ],
                    width=6,
                ),
            ],
            align="end",
            justify="between",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("dis slider: "),
                        dcc.Slider(id="dis_slider", min=0, max=10000, value=2000, step=10),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.Div("pro slider: "),
                        dcc.Slider(id="pro_slider", min=0, max=2, value=0.5, step=0.05),
                    ],
                    width=6,
                ),
            ],
            align="start",
            justify="between",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("wid slider: "),
                        dcc.Slider(id="wid_slider", min=0, max=1000, value=90, step=10),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.Div("hei slider: "),
                        dcc.Slider(id="hei_slider", min=0, max=1, value=0.1, step=0.05),
                    ],
                    width=6,
                ),
            ],
            align="start",
            justify="between",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("tr+ input: "),
                        dcc.Input(
                            id="tr+_input",
                            type="number",
                            min=0, 
                            max=0.0001,
                            value=0,
                            style={"width": "100%"},
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        html.Div("tr- input: "),
                        dcc.Input(
                            id="tr-_input",
                            type="number",
                            min=0, 
                            max=0.0001,
                            value=0,
                            style={"width": "100%"},
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        html.Div("dis input: "),
                        dcc.Input(
                            id="dis_input",
                            type="number",
                            min=0, 
                            max=10000,
                            value=2000,
                            style={"width": "100%"},
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        html.Div("pro input: "),
                        dcc.Input(
                            id="pro_input",
                            type="number",
                            min=0, 
                            max=2,
                            value=0.5,
                            style={"width": "100%"},
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        html.Div("wid input: "),
                        dcc.Input(
                            id="wid_input",
                            type="number",
                            min=0, 
                            max=1000,
                            value=90,
                            style={"width": "100%"},
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    [
                        html.Div("hei input: "),
                        dcc.Input(
                            id="hei_input",
                            type="number",
                            min=0, 
                            max=1,
                            value=0.1,
                            style={"width": "100%"},
                        ),
                    ],
                    width="auto",
                )
            ],
            align="end",
            justify="between",
        )
    ]
)


# In[27]:


# Graf 4 - Vysledny grafd s najdenymi srandami

# graf_4 = html.Div(dcc.Graph(id='example-graph-4', figure=fig0))
graf_4 = dcc.Graph(id='fig4')


# In[28]:


ukladacia_linka = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("Nastavenie uloženia: "),
                        dbc.Checklist(
                            options=[
                                {"label": ".TXT", "value": "txt"},
                                {"label": ".CSV", "value": "csv"},
                                {"label": ".PNG", "value": "png"},
                                {"label": ".PDF", "value": "pdf"},
                                {"label": ".QLS", "value": "qls"},
                            ],
                            value=["txt", "csv", "png", "pdf", "qls"],
                            id="save_switch",
                            inline=True,
                            switch=True,
                        ),
                    ],
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        "Save",
                        id="save_button",
                        color="success",
                        className="mr-1",
                        size="lg",
                    ),
                    width="auto",
#                     html.Button('Save', id='save_button', n_clicks=0),
                ),
            ],
            align="end",
            justify="between",
        )
    ]
)


# In[29]:


konecna_linka = html.Div([html.Div(id='output1'), html.Div(id='output2')])


# In[30]:


url = "http://127.0.0.1:8050/"
chrome_path = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"
webbrowser.get(chrome_path).open(url)


# In[31]:


app.layout = html.Div(
    [
        uvodna_linka,
        nacitacia_linka,
        rozlisovacia_linka,
        stavova_linka,
#         progress_linka,
        graf_1,
        vyhladzovacia_linka,
        graf_2,
        aproximacna_linka,
        graf_3,
        peak_linka,
        graf_4,
        ukladacia_linka,
        konecna_linka,
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)


# In[ ]:




