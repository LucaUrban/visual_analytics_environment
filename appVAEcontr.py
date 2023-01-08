import streamlit as st
import time
import base64
import os
from urllib.parse import quote as urlquote
from urllib.request import urlopen
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import json
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import medcouple
import math
import scipy.stats as stats
from scipy.special import softmax
import pymannkendall as mk
import csv

st.title("Visual Information Quality Environment")
st.write("In this part you can upload your csv file either dropping your file or browsing it. Then the application will start showing all of the charts for the Dataset. " +
         "To change the file to be analyzed you have to refresh the page.")
uploaded_file = st.file_uploader("Choose a file")
demo_data_radio = st.radio("What is the dataset you want to import:", ('Demo datset', 'ETER Dataset', 'Another dataset'))

if demo_data_radio == 'Demo datset' or uploaded_file is not None:
    if uploaded_file is not None:
        if demo_data_radio == 'ETER Dataset':
            table = pd.read_csv(uploaded_file, delimiter = ';', decimal = ',', low_memory = False)
            
            for col in table.columns:
                if table[col].dtypes == 'O' and not (col.startswith('Flag') or col.startswith('Notes')):
                    table[col] = table[col].apply(lambda x: x.replace(',', '.') if not pd.isna(x) else x)
            
            for token in ['a', 'c', 'm',  'nc', 's', 'x', 'xc', 'xr']:
                table.replace({token: np.nan}, inplace = True)
            
            for col in table.columns:
                table[col] = pd.to_numeric(table[col], errors = 'ignore')
        else:
            table = pd.read_csv(uploaded_file)
    else:
        table = pd.read_csv('https://raw.githubusercontent.com/LucaUrban/visual_analytics_environment/main/eter%20dataset%20demo.csv')

    # importing the NUTS codes for the europe in the three levels
    with urlopen('https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson') as response:
        eu_nut0 = json.load(response)
    with urlopen('https://raw.githubusercontent.com/eurostat/Nuts2json/master/2021/4326/60M/nutsrg_2.json') as response:
        eu_nut2 = json.load(response)
    with urlopen('https://raw.githubusercontent.com/eurostat/Nuts2json/master/2021/4326/60M/nutsrg_3.json') as response:
        eu_nut3 = json.load(response)
        
    lis_id_eu_nut0 = [el['properties']['ISO2'] for el in eu_nut0['features']]
    lis_id_eu_nut2 = [el['properties']['id'] for el in eu_nut2['features']]
    lis_id_eu_nut3 = [el['properties']['id'] for el in eu_nut3['features']]

    # functions
    def cr_metrics_table(flag_notes, set_entity, ones, twos = set()):
        if flag_notes_on:
            summ_table = pd.DataFrame([[str(len(twos.intersection(set_entity))) + ' over ' + str(len(twos)), str(round((100 * len(twos.intersection(set_entity))) / len(twos), 2)) + '%'], 
                                       [str(len(set_entity)) + ' / ' + str(len(ones.union(twos))), str(round(100 * (len(set_entity) / len(ones.union(twos))), 2)) + '%'], 
                                       [str(len(set_entity.difference(ones.union(twos)))), str(round((100 * len(set_entity.difference(ones.union(twos)))) / len(set_entity), 2)) + '%']], 
                                       columns = ['Absolute Values', 'In percentage'], 
                                       index = ['Accuracy', 'new/prev cases', 'Extra cases'])
        else:
            summ_table = pd.DataFrame([[str(len(ones.intersection(set_entity))) + ' over ' + str(len(ones)), str(round((100 * len(ones.intersection(set_entity))) / len(ones), 2)) + '%'], 
                                       [str(len(set_entity)) + ' / ' + str(len(ones)), str(round(100 * (len(set_entity) / len(ones)), 2)) + '%'], 
                                       [str(len(set_entity.difference(ones))), str(round((100 * len(set_entity.difference(ones))) / len(set_entity), 2)) + '%']], 
                                       columns = ['Absolute Values', 'In percentage'], 
                                       index = ['Accuracy', 'new/prev cases', 'Extra cases'])
        return summ_table
    
    def map_creation(res, nut_col, map_feature, color):
        if color == 'Portland': m_color = px.colors.diverging.Portland
        if color == 'Picnic': m_color = px.colors.diverging.Picnic
        if color == 'Geyser': m_color = px.colors.diverging.Geyser
        
        px.set_mapbox_access_token("pk.eyJ1IjoibHVjYXVyYmFuIiwiYSI6ImNrZm5seWZnZjA5MjUydXBjeGQ5ZDBtd2UifQ.T0o-wf5Yc0iTSeq-A9Q2ww")
        if len(res[nut_col][0]) == 2:
            map_box = px.choropleth_mapbox(res, geojson = eu_nut0, locations = res[nut_col], featureidkey = 'properties.ISO2', color = map_feature, 
                                           color_continuous_scale = m_color, range_color = (res[map_feature].min(), res[map_feature].max()),
                                           mapbox_style = "carto-positron", zoom = 3, center = {"lat": 47.4270826, "lon": 15.5322329}, opacity = 0.5,
                                           labels = {map_feature: map_feature})
        
        if len(res[nut_col][0]) == 4:
            map_box = px.choropleth_mapbox(res, geojson = eu_nut2, locations = res[nut_col], featureidkey = 'properties.id', color = map_feature, 
                                           color_continuous_scale = m_color, range_color = (res[map_feature].min(), res[map_feature].max()), 
                                           mapbox_style = "carto-positron", zoom = 3, center = {"lat": 47.4270826, "lon": 15.5322329}, opacity = 0.5,
                                           labels = {map_feature: map_feature})
        
        if len(res[nut_col][0]) == 5:
            map_box = px.choropleth_mapbox(res, geojson = eu_nut3, locations = res[nut_col], featureidkey = 'properties.id', color = map_feature, 
                                           color_continuous_scale = m_color, range_color = (res[map_feature].min(), res[map_feature].max()), 
                                           mapbox_style = "carto-positron", zoom = 3, center = {"lat": 47.4270826, "lon": 15.5322329}, opacity = 0.5, 
                                           labels = {map_feature: map_feature})
        return map_box
    
    def cr_dnwl_tab_cc(table, con_checks_id_col, time_col, con_checks_features, descr_col, flags_col):
        table_download = table.pivot(index = [con_checks_id_col], columns = [time_col], values = [con_checks_features])
        table_download.columns = table_download.columns.droplevel()
        table_download.rename(columns = str, inplace = True)
        t_col = [str(el) for el in sorted(table[time_col].unique())]; list_fin = []
        if flags_col != '':
            table_download = table_download.join(table[[con_checks_id_col] + descr_col + ['Class trend', flags_col, 'Prob inst ' + con_checks_features, 'Rupt. years']].groupby([con_checks_id_col]).agg(pd.Series.mode), 
                                                 on = con_checks_id_col)
            table_download.rename(columns = {'Class trend': 'Trend', flags_col: 'Existing flag', 'Prob inst ' + con_checks_features: 'Detected case'}, inplace = True)
            df_cols = descr_col + ['Variable'] + t_col + ['Trend', 'Existing flag', 'Detected case', 'Rupt. years']
        else:
            table_download = table_download.join(table[[con_checks_id_col] + descr_col + ['Class trend', 'Prob inst ' + con_checks_features, 'Rupt. years']].groupby([con_checks_id_col]).agg(pd.Series.mode), 
                                                 on = con_checks_id_col)
            table_download.rename(columns = {'Class trend': 'Trend', 'Prob inst ' + con_checks_features: 'Detected case'}, inplace = True)
            df_cols = descr_col + ['Variable'] + t_col + ['Trend', 'Detected case', 'Rupt. years']
        table_download['Variable'] = con_checks_features
        table_download = table_download[df_cols]
        table_download.replace({'Trend': {i+1 : list(dict_trend.keys())[i] for i in range(len(list(dict_trend.keys())))}}, inplace = True)
        return table_download.to_csv(sep = ';').encode('utf-8')

    # selection boxes columns
    col_an = [col for col in list(table) if len(table[col].unique()) < 10 or is_numeric_dtype(table[col])]
    col_obj = [col for col in list(table) if table[col].dtypes == 'O']
    col_mul = [col for col in list(table) if is_numeric_dtype(table[col])]
    lis_check = [{'label': col, 'value': col} for col in col_mul if col != col_mul[0]]

    widget = st.selectbox("what is the widget you want to display:",
                          ["Data View", "Geographical Analysis", "Mono dimensional Analysis", "Ratios Analysis", "Multi-dimensional Analysis", 
                           "Autocorrelation Analysis", "Feature Importance Analysis", "Anomalies check", "Consistency checks", "Time series forecasting"], 0)
    
    if widget == "Data View":
        # showing the table with the data
        st.header("Data View")
        st.write("Data contained into the dataset:", table)
    
    if widget == "Geographical Analysis":
        # map-box part
        st.sidebar.subheader("Map area")
        nut_col = st.sidebar.selectbox("Nut column", col_obj, 0)
        map_feature = st.sidebar.selectbox("Feature column", col_mul, 0)
        map_q = st.sidebar.number_input("Quantile value", 0, 100, 50)
        map_color = st.sidebar.selectbox("Map color-palette", ['Portland', 'Picnic', 'Geyser'], 0)

        st.header("Geographical Analysis")
        try:
            st.plotly_chart(map_creation(table[[nut_col, map_feature]].groupby(by = nut_col, as_index = False).quantile(map_q/100), 
                                         nut_col, map_feature, map_color), use_container_width=True)
        except:
            st.warning('You have to select a NUTS id column in the selection box after \"Nut column\" to produce the map')
    
    if widget == "Mono dimensional Analysis":
        # mono variable analysis part
        st.header("Monodimension Analysis")

        st.sidebar.subheader("Monovariable Area")
        monoVar_col = st.sidebar.selectbox("Monovariable feature", col_an, 6)
        monoVar_type = st.sidebar.selectbox("Chart type", ["gauge plot", "cdf plot", "pie chart"], 0)

        if monoVar_type == "gauge plot":
            q_grey_area = st.sidebar.number_input("Quantile value grey area", 0, 25, 5)
            monoVar_plot = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = table[monoVar_col].mean(),
                delta = {"reference": 2 * table[monoVar_col].mean() - table[monoVar_col].quantile(0.95)},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {'axis': {'range': [table[monoVar_col].min(), table[monoVar_col].max()]},
                         'steps' : [
                             {'range': [table[monoVar_col].min(), table[monoVar_col].quantile(q_grey_area/100)], 'color': "lightgray"},
                             {'range': [table[monoVar_col].quantile(1 - q_grey_area/100), table[monoVar_col].max()], 'color': "gray"}]},
                title = {'text': "Gauge plot for the variable: " + monoVar_col}))
        if monoVar_type == "cdf plot":
            monoVar_plot = px.ecdf(table, x = monoVar_col, marginal = "box")
        if monoVar_type == "pie chart":
            monoVar_plot = px.pie(table, names = monoVar_col, title = "Pie chart for the variable: " + monoVar_col)

        st.plotly_chart(monoVar_plot, use_container_width=True)
            
    if widget == "Time series forecasting":
        st.header("Time series forecasting")

        use_col = st.sidebar.selectbox("Chosen Variable", col_mul, 0)
        modality = st.sidebar.selectbox("Forecasting Method", ["Rolling Forecast", "Recurring Forecast"], 0)
        index = st.sidebar.selectbox("Index col", col_obj, 0)
        time = st.sidebar.selectbox("Time col", table.columns, 0)
 
        # pre-work
        data = table[[index, time, use_col]].sort_values(by=[time])
        res = np.array([]); ids = []
        for id in data[index].unique():
            el = data[data[index] == id][use_col]
            n = len(list(data[time].unique()))
            if el.shape[0] == n:
                res = np.concatenate([res, el.values])
                ids.append(id)
        res = res.reshape(res.shape[0]//n, n)
        col_mean = np.nanmean(res, axis = 1)

        #Find indices that you need to replace
        inds = np.where(np.isnan(res))

        #Place column means in the indices. Align the arrays using take
        res[inds] = np.take(col_mean, inds[1])
        
        # fit the init model and making the predictions
        pred_ar = np.array([]); pred_ma = np.array([]); pred_arma = np.array([]); pred_arima = np.array([])

        for i in range(res.shape[0]):
            pred_ar = np.append(pred_ar, AutoReg(res[i, 0:res.shape[1]-1], lags = 1).fit().predict(len(res), len(res)))
            pred_ma = np.append(pred_ma, ARIMA(res[i, 0:res.shape[1]-1], order=(0, 0, 1)).fit().predict(len(res), len(res)))
            pred_arma = np.append(pred_arma, ARIMA(res[i, 0:res.shape[1]-1], order=(2, 0, 1)).fit().predict(len(res), len(res)))
            pred_arima = np.append(pred_arima, ARIMA(res[i, 0:res.shape[1]-1], order=(1, 1, 1)).fit().predict(len(res), len(res), typ='levels'))
        
        # visual part
        mse_mins = np.array([mean_squared_error(pred_ar, res[:, res.shape[1]-1]), mean_squared_error(pred_ma, res[:, res.shape[1]-1]),
                             mean_squared_error(pred_arma, res[:, res.shape[1]-1]), mean_squared_error(pred_arima, res[:, res.shape[1]-1])])
        st.table(pd.DataFrame(mse_mins.reshape((1, 4)), columns = ['AR', 'MA', 'ARMA', 'ARIMA'], index = ['MSE error']))
         
        ch_model = st.selectbox("Model to apply", ['AR', 'MA', 'ARMA', 'ARIMA'])
        ch_id = st.selectbox("Id to forecast", ids)
        num_fut_pred = st.sidebar.number_input("Number of periods to forecast", 1, 10, 1)
        fig_forecasting = go.Figure()
        
        # forecasting
        par_for = []; rif = res[ids.index(ch_id)]
        for i in range(num_fut_pred + 1):
            # prediction based on the chosen model
            if ch_model == 'AR':
                pred = AutoReg(rif, lags = 1).fit().predict(len(rif), len(rif))[0]

            if ch_model == 'MA':
                pred = ARIMA(rif, order=(0, 0, 1)).fit().predict(len(rif), len(rif))[0]

            if ch_model == 'ARMA':
                pred = ARIMA(rif, order=(2, 0, 1)).fit().predict(len(rif), len(rif))[0]

            if ch_model == 'ARIMA':
                pred = ARIMA(rif, order=(1, 1, 1)).fit().predict(len(rif), len(rif))[0]
                
            par_for.append(pred); rif = np.append(rif, pred)
            # rolling forecasting
            if modality == "Rolling Forecast":
                rif = rif[1:]
        
        fig_forecasting.add_trace(go.Scatter(x = [max(list(data[time].unique())) + j for j in range(num_fut_pred + 1)], 
                                             y = [res[ids.index(ch_id), -1]] + par_for, mode = 'lines+markers', name = "Prediction", line = dict(color = 'firebrick')))
        fig_forecasting.add_trace(go.Scatter(x = list(data[time].unique()), y = data[data[index] == ch_id][use_col].values, mode = 'lines+markers', name = "Value", 
                                             line = dict(color = 'royalblue')))
        fig_forecasting.update_layout(xaxis_title = use_col, yaxis_title = time, title_text = "Values over time with future predictions")
        st.plotly_chart(fig_forecasting, use_container_width=True)
    
    if widget == "Ratios Analysis":
        # ratio analysis part
        st.header("Ratios Analysis")

        st.sidebar.subheader("Ratio Area")
        ratio_num = st.sidebar.multiselect("Variables ratio numerator", col_mul)
        ratio_den = st.sidebar.multiselect("Variables ratio denominator", col_mul)
        map_color = st.sidebar.selectbox("Map color-palette", ['Portland', 'Picnic', 'Geyser'], 0)
        
        new_ratio_name = st.text_input('Name of the new ratio', 'R_1')
            
        if len(ratio_num) == 1 and len(ratio_den) == 1:
            table = pd.concat([table, pd.DataFrame(np.divide(table[ratio_num].values, table[ratio_den].values), columns = [new_ratio_name])], axis = 1)
        else:
            table = pd.concat([table, pd.DataFrame(np.divide(np.nansum(table[ratio_num].values, axis = 1), np.nansum(table[ratio_den].values, axis = 1)), columns = [new_ratio_name])], axis = 1)
        table.loc[table[table[new_ratio_name] == np.inf].index, new_ratio_name] = np.nan
        
        ratio_plot = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = table[new_ratio_name].mean(),
            delta = {"reference": 2 * table[new_ratio_name].mean() - table[new_ratio_name].quantile(0.95)},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [table[new_ratio_name].min(), table[new_ratio_name].max()]},
                     'steps' : [
                         {'range': [table[new_ratio_name].min(), table[new_ratio_name].quantile(0.05)], 'color': "lightgray"},
                         {'range': [table[new_ratio_name].quantile(0.95), table[new_ratio_name].max()], 'color': "gray"}],},
            title = {'text': "Gauge plot for the variable: " + new_ratio_name}))
        
        st.plotly_chart(ratio_plot, use_container_width=True)

        # map pplot + violin plot on the aggregated results
        left, right = st.columns(2)
        with left: 
            ratio_vio_sel1 = st.selectbox("First category col (or nut id col)", col_obj, 0)
        with right:
            ratio_vio_sel2 = st.selectbox("Second category column", ['-'] + list(col_obj), 0)
            
        res = pd.DataFrame([[nut, table[table[ratio_vio_sel1] == nut][new_ratio_name].mean()] for nut in table[ratio_vio_sel1].unique()], 
                           columns = [ratio_vio_sel1, new_ratio_name])
        try: 
            st.plotly_chart(map_creation(res, ratio_vio_sel1, new_ratio_name, map_color), use_container_width=True)
        except:
            st.warning('You have to select a NUTS id column in the selection box after \"First category col (or nut id col)\" to produce the map')

        cou_sel = st.selectbox("Id to explore", ['All ids'] + list(table[ratio_vio_sel1].unique()), 0)
        if cou_sel == 'All ids':
            if ratio_vio_sel2 == '-':
                fig_vio = px.violin(table, y = new_ratio_name, box = True, points = 'suspectedoutliers', title = 'Violin plot for the created ratio', 
                                    hover_data = [ratio_vio_sel1, ratio_num[0], ratio_den[0]])
            else:
                table[ratio_vio_sel2].replace({np.nan: 'missing'}, inplace = True)
                fig_vio = px.violin(table, y = new_ratio_name, color = table[ratio_vio_sel2], box = True, points = 'suspectedoutliers', 
                                    title = 'Violin plot for the created ratio', hover_data = [ratio_vio_sel1, ratio_num[0], ratio_den[0]])
        else:
            if ratio_vio_sel2 == '-':
                fig_vio = px.violin(table[table[ratio_vio_sel1] == cou_sel], y = new_ratio_name, x = ratio_vio_sel1, box = True, points = 'suspectedoutliers', 
                                    title = 'Violin plot for the created ratio', hover_data = [ratio_vio_sel1, ratio_num[0], ratio_den[0]])
            else:
                table[ratio_vio_sel2].replace({np.nan: 'missing'}, inplace = True)
                fig_vio = px.violin(table[table[ratio_vio_sel1] == cou_sel], y = new_ratio_name, x = ratio_vio_sel1, color = table[table[ratio_vio_sel1] == cou_sel][ratio_vio_sel2], 
                                    box = True, points = 'suspectedoutliers', title = 'Violin plot for the created ratio', hover_data = [ratio_vio_sel1, ratio_num[0], ratio_den[0]])
        st.plotly_chart(fig_vio, use_container_width=True)

        st.write('If you want to download the result file with the new ratio clik on the following button:')
        st.download_button(label = "Download data with lables", data = table.to_csv(index = None, sep = ';').encode('utf-8'), file_name = 'result.csv', mime = 'text/csv')
    
    if widget == "Multi-dimensional Analysis":
        # multi variable analysis part
        st.header("Multi-dimension Analysis")

        st.sidebar.subheader("Multivariable Area")
        multi_index = st.sidebar.selectbox("Multivariable index col", col_obj, 1)
        multi_time = st.sidebar.selectbox("Multivariable time col", ['-'] + list(table.columns), 3)
        multiXax_col = st.sidebar.selectbox("Multivariable X axis col", col_mul, 1)
        multiYax_col = st.sidebar.selectbox("Multivariable Y axis col", col_mul, 2)
        cat_col = st.sidebar.selectbox("Category column", ['None'] + list(table.columns), 0)
        
        if multi_time != '-' and table[multi_time].dtype != 'O' and table[multi_time].max() != table[multi_time].min():
            multiSlider = st.sidebar.slider("Multivarible time value", int(table[multi_time].min()), int(table[multi_time].max()), int(table[multi_time].min()))
            dff = table[table[multi_time] == multiSlider]
        else: 
            dff = table
            
        if cat_col == 'None':
            multi_plot = px.scatter(x = dff[multiXax_col], y = dff[multiYax_col], hover_name = dff[multi_index])
        else:
            multi_plot = px.scatter(x = dff[multiXax_col], y = dff[multiYax_col], hover_name = dff[multi_index], color = dff[cat_col])
        multi_plot.update_traces(customdata = dff[multi_index])
        multi_plot.update_xaxes(title = multiXax_col)
        multi_plot.update_yaxes(title = multiYax_col)

        st.plotly_chart(multi_plot, use_container_width=True)

        if multi_time != '-' and table[multi_time].dtype != 'O' and table[multi_time].max() != table[multi_time].min():
            # time control charts
            el_id = st.selectbox("Id time control charts", table[multi_index].unique(), 0)

            dff_tcc = table[table[multi_index] == el_id][[multi_time, multiXax_col, multiYax_col]]
            for i in range(2):
                fig_tcc = go.Figure()
                if dff_tcc.shape[0] != 0:
                    x_barbar = table.groupby(by = multi_index).mean()[dff_tcc.columns[i+1]].mean()

                    x_LCL = x_barbar - (1.88 * (dff_tcc[dff_tcc.columns[i+1]].quantile(0.95) - dff_tcc[dff_tcc.columns[i+1]].quantile(0.05)))
                    x_UCL = x_barbar + (1.88 * (dff_tcc[dff_tcc.columns[i+1]].quantile(0.95) - dff_tcc[dff_tcc.columns[i+1]].quantile(0.05)))
                    
                    fig_tcc.add_trace(go.Scatter(x = dff_tcc[multi_time], y = dff_tcc[dff_tcc.columns[i+1]], mode = 'lines+markers', name = "Value"))
                    fig_tcc.add_trace(go.Scatter(x = dff_tcc[multi_time], y = [x_UCL for _ in range(dff_tcc[multi_time].shape[0])], mode = "lines", name = "Upper Bound"))
                    fig_tcc.add_trace(go.Scatter(x = dff_tcc[multi_time], y = [x_LCL for _ in range(dff_tcc[multi_time].shape[0])], mode = "lines", name = "Lower Bound"))

                    fig_tcc.update_xaxes(showgrid = False)
                    fig_tcc.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                                           xref = 'paper', yref = 'paper', showarrow=False, align = 'left',
                                           bgcolor = 'rgba(255, 255, 255, 0.5)', text = f'<b>{el_id}</b><br>{dff_tcc.columns[i+1]}')
                    fig_tcc.update_layout(xaxis_title = multi_time, yaxis_title = dff_tcc.columns[i+1])
                    fig_tcc.update_layout(height = 250, margin = {'l': 20, 'b': 30, 'r': 10, 't': 10})

                    st.plotly_chart(fig_tcc, use_container_width=True)
    
    if widget == "Autocorrelation Analysis":
        # crossfilter analysis part
        st.header("Autocorrelation Analysis")

        st.sidebar.subheader("Autocorrelation Area")
        cross_index = st.sidebar.selectbox("Autocorrelation index col", table.columns, 1)
        cross_time = st.sidebar.selectbox("Autocorrelation time col", table.columns, 3)
        cross_col = st.sidebar.selectbox("Autocorrelation X axis col", col_mul, 1)
        crossSlider = st.sidebar.slider("Autocorrelation time value", int(table[cross_time].min()), int(table[cross_time].max()-1), int(table[cross_time].min()))
        lag_val = st.sidebar.number_input("Time Lag Value", 1, 10, 1)
        cat_col = st.sidebar.selectbox("Category column", ['None'] + list(table.columns), 0)

        if cat_col == 'None':
            final_df_cross = pd.merge(table[table[cross_time] == crossSlider][[cross_index, cross_col]], 
                                      table[table[cross_time] == crossSlider + lag_val][[cross_index, cross_col]], 
                                      how = "inner", on = cross_index)
            cross_plot = px.scatter(x = final_df_cross[cross_col + "_x"], y = final_df_cross[cross_col + "_y"], hover_name = final_df_cross[cross_index])
        else:
            final_df_cross = pd.merge(table[table[cross_time] == crossSlider][[cross_index, cat_col, cross_col]], 
                                      table[table[cross_time] == crossSlider + lag_val][[cross_index, cat_col, cross_col]], 
                                      how = "inner", on = [cross_index, cat_col])
            cross_plot = px.scatter(x = final_df_cross[cross_col + "_x"], y = final_df_cross[cross_col + "_y"], 
                                    hover_name = final_df_cross[cross_index], color = final_df_cross[cat_col])

        cross_plot.update_xaxes(title = cross_col)
        cross_plot.update_yaxes(title = cross_col + f" After {lag_val} years")

        st.plotly_chart(cross_plot, use_container_width=True)

        st.subheader("Autocorrelation")
        st.write("Autocorrelation value: " + str(round(final_df_cross[cross_col + "_x"].corr(final_df_cross[cross_col + "_y"]), 5)))

        # difference timeseries plot
        el_id_diff = st.selectbox("Id deltas timeseries", table[cross_index].unique())

        dff_diff = table[table[cross_index] == el_id_diff][[cross_time, cross_col]]
        dff_diff.dropna(inplace = True)

        if dff_diff.shape[0] > 1:
            fig_diff = go.Figure()
            x = [[i, 0] for i in range(1, dff_diff.shape[0])]
            Y = [dff_diff[cross_col].iloc[dff_diff.shape[0] - i - 1] - dff_diff[cross_col].iloc[dff_diff.shape[0] - i] for i in range(1, dff_diff.shape[0])]
            reg = LinearRegression().fit(x, Y); coeff = reg.coef_; intercept = reg.intercept_

            fig_diff.add_trace(go.Scatter(x = [str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i]) + "-" + str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i - 1]) for i in range(1, dff_diff.shape[0])], 
                                     y = Y, mode = 'markers', name = "Value"))
            fig_diff.add_trace(go.Scatter(x = [str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i]) + "-" + str(dff_diff[cross_time].iloc[dff_diff.shape[0] - i - 1]) for i in range(1, dff_diff.shape[0])], 
                                     y = [intercept + (i * coeff[0]) for i in range(dff_diff.shape[0])], 
                                     mode = 'lines', name = "Regression"))
            fig_diff.update_xaxes(showgrid=False)
            fig_diff.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom', xref='paper', yref='paper', showarrow=False, align='left',
                                    bgcolor='rgba(255, 255, 255, 0.5)', text = f'<b>{el_id_diff}</b><br>{cross_col}')
            fig_diff.update_layout(xaxis_title = cross_time, yaxis_title = list(dff_diff)[1])
            fig_diff.update_layout(height = 400)

            st.plotly_chart(fig_diff, use_container_width = True)

            st.subheader("Regression Parameters")
            st.write(f"Intercept: {round(intercept, 4)} \nSlope: {round(coeff[0], 4)}")
        else:
            st.warning("It wasn\'t possible to create the plot because of the lack of data")
    
    if widget == "Feature Importance Analysis":
        # pareto chart with feature importance on ridge regressor
        st.sidebar.subheader("Feature Importance Area")
        feaImp_target = st.sidebar.selectbox("Feature Importance target", col_mul, 0)
        id_sel_col = st.sidebar.selectbox("ID/category column", table.columns, 0)
        ch_model = st.sidebar.selectbox("Model", ['Ridge Regression', 'Elastic Net Regression', 'Lasso Regression'], 0)

        st.header("Feature Importance Analysis")
        
        left, right = st.columns(2)
        with left: 
            fea_Imp_features = st.multiselect("Features:", col_mul)
        with right:
            id_sel = st.selectbox("Index/category selection:", ['All ids'] + list(table[id_sel_col].unique()), 0)
        
        if len(fea_Imp_features) >= 2:
            scaler = StandardScaler()
            if id_sel == 'All ids':
                target = table[feaImp_target]; train_nm = table[fea_Imp_features]
            else:
                target = table[table[id_sel_col] == id_sel][feaImp_target]; train_nm = table[table[id_sel_col] == id_sel][fea_Imp_features]

            # replacing the nan values and scaling target and features 
            target.replace({np.nan : 0}, inplace = True)
            for name_col in fea_Imp_features:
                train_nm[name_col].replace({np.nan : train_nm[name_col].mean()}, inplace = True)
            train_nm = scaler.fit_transform(train_nm)

            # Create figure with secondary y-axis
            reg_par = np.array([[.1, 1], [10, 100]]); dim_plots = reg_par.shape
            fig_tot = make_subplots(rows = dim_plots[0], cols = dim_plots[1], 
                                    specs = [[{"secondary_y": True} for i in range(dim_plots[0])] for j in range(dim_plots[1])],
                                    subplot_titles = tuple(f"Feature importance for alpha = {par}" for i in range(dim_plots[0]) for par in reg_par[i]))
                     
            for num_row in range(dim_plots[0]):
                for num_col in range(dim_plots[1]):
                    # Constructing the model
                    if ch_model == 'Ridge Regression': model = Ridge(alpha = reg_par[num_row][num_col], random_state=0)
                    if ch_model == 'Elastic Net Regression': model = ElasticNet(alpha = reg_par[num_row][num_col], random_state=0)
                    if ch_model == 'Lasso Regression': model = Lasso(alpha = reg_par[num_row][num_col], random_state=0)
                    
                    model.fit(train_nm, target)

                    importance = softmax(normalize([model.coef_]))[0]
                    dict_soft = {fea_Imp_features[i]: importance[i]*100 for i in range(importance.shape[0])}
                    dict_soft = {k: v for k, v in sorted(dict_soft.items(), key=lambda item: item[1], reverse = True)}
                    
                    fig_tot.add_trace(
                        go.Bar(x = list(dict_soft.keys()), y = list(dict_soft.values()), 
                               marker_color = 'rgb(158,202,225)', marker_line_color = 'rgb(8,48,107)', 
                               marker_line_width = 1.5, opacity = 0.6, name = 'Value'),
                        row = num_row + 1, col = num_col + 1, secondary_y = False
                    )

                    fig_tot.add_trace(
                        go.Scatter(x = list(dict_soft.keys()), y = np.cumsum(np.array(list(dict_soft.values()))), line_color = 'rgb(255, 150, 0)'),
                        row = num_row + 1, col = num_col + 1, secondary_y = True
                    )

                    # Add figure title
                    fig_tot.update_layout(
                        title_text = "Feature importances", showlegend = False
                    )

                    # Set x-axis title
                    fig_tot.update_xaxes(title_text = "Variables")

                    # Set y-axes titles
                    fig_tot.update_yaxes(title_text="<b>Value</b> of importance", secondary_y=False)
                    fig_tot.update_yaxes(title_text="<b>%</b> of importance", secondary_y=True)

            fig_tot.update_layout(height = 600)
            st.plotly_chart(fig_tot, use_container_width=True)
                  
            st.plotly_chart(px.imshow(table[fea_Imp_features].corr(), x = fea_Imp_features,  y = fea_Imp_features, labels = dict(color = "Corr Value"), 
                                      color_continuous_scale = px.colors.sequential.Hot), 
                            use_container_width=True)
        else:
            st.warning("Yuo have to choose at least two columns")
        
    if widget == "Anomalies check":
        use_col = st.sidebar.selectbox("Chosen Variable", col_mul, 0)
        map_color = st.selectbox("Map color-palette", ['Portland', 'Picnic', 'Geyser'], 0)
        var_clean = table[use_col].dropna().values
        
        # MLE normal
        mu_hat = var_clean.mean()
        sigma_hat = math.sqrt(((var_clean - var_clean.mean()) ** 2).sum() / var_clean.shape[0])

        # MLE exponential
        lambda_hat_exp = var_clean.shape[0] / var_clean.sum()

        # MLE log-normal
        mu_hat_log = (np.ma.log(var_clean)).sum() / var_clean.shape[0]
        sigma_hat_log = math.sqrt(((np.ma.log(var_clean) - mu_hat_log) ** 2).sum() / var_clean.shape[0])
        
        # MLE weibull
        a, alpha_hat, b, beta_hat = stats.exponweib.fit(var_clean, floc=0, fa=1)
        
        # computing the p-values for all the distributions
        result_norm = stats.kstest(var_clean, 'norm', (mu_hat, sigma_hat))
        result_exp = stats.kstest(var_clean, 'expon')
        result_lognorm = stats.kstest(var_clean, 'lognorm', (mu_hat_log, sigma_hat_log))
        result_weibull2 = stats.kstest(var_clean, 'dweibull', (beta_hat, alpha_hat))
        
        # visual part
        dis_fit = [[result_norm[1], result_exp[1], result_lognorm[1], result_weibull2[1]], 
                   [int(result_norm[1] > 0.05), int(result_exp[1] > 0.05), int(result_lognorm[1] > 0.05), int(result_weibull2[1] > 0.05)]]
        st.table(pd.DataFrame(dis_fit, columns = ['Normal', 'Exponential', 'Log-Norm', 'Weibul'], index = ['P-value', 'P > t']))

        ch_distr = st.selectbox("Distribution anomalies estimation", ['Normal', 'Exponential', 'Log-Norm', 'Weibull'])
        fig_distr = go.Figure(data = [go.Histogram(x = var_clean, 
                                                   xbins = dict(start = var_clean.min(), end = var_clean.max(), size = (var_clean.max() - var_clean.min()) / 25),
                                                   autobinx = False, 
                                                   histnorm = 'probability density')])
        
        x_pos = np.linspace(var_clean.min(), var_clean.max(), 25)
        if ch_distr == 'Normal':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.norm(mu_hat, sigma_hat).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Exponential':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.expon(lambda_hat_exp).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Log-Norm':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.lognorm(mu_hat_log, sigma_hat_log).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        if ch_distr == 'Weibull':
            fig_distr.add_trace(go.Scatter(x = x_pos, y = stats.dweibull(alpha_hat, beta_hat).pdf(x_pos), mode = 'lines+markers', name = "Est Distribution"))
        
        fig_distr.update_layout(title = 'Hist plot to comapre data with possible underlying distribution', xaxis_title = use_col + ' values', yaxis_title = use_col + ' PMF and ch. distr. PDF')
        st.plotly_chart(fig_distr, use_container_width=True)
         
        # outlier part
        st.markdown('In the next part there is the effective detection of the outliers contained into the data. The detection is made by the **Tukey\'s fences**. ' + 
                    'These fences are calculated by refering to the next formulas and the applied formula depend on the type of distribution chosen ' + 
                    'and the **skewness** of the data. \n If the data is **skewed** the fences are calculated in this way: ')
        st.latex(r'''[Q_{1} - (k \cdot ITQ), Q_{3} + (k \cdot ITQ)]''')
        st.markdown('If the data distribution is **not skewed** the formula changes with a correction term and becomes:')
        st.latex(r'''[Q_{1} - (k \cdot e^{-4mc} \cdot ITQ), Q_{3} + (k \cdot e^{3mc} \cdot ITQ)]''')
        st.markdown('If mc > 0 and: ')
        st.latex(r'''[Q_{1} - (k \cdot e^{-3mc} \cdot ITQ), Q_{3} + (k \cdot e^{4mc} \cdot ITQ)]''')
        st.markdown('If mc < 0. \n In these equations $Q_{1}$ represent the first quantile, while $Q_{3}$ represents the third quantile, $ITQ = Q_{3} - Q_{1}$, k is the ' + 
                    '**Tukey\'s constant** and mc is the value of the **MedCouple** function. A value is treated as an outlier if it doesen\'t fit into these intervals. ' + 
                    'In this application we make a distinction between strong and weak outlier. A strong outlier $o_{s}$ is a value that given a $t_{f}$ value for the ' + 
                    'fence\'s correction term: ')
        st.markdown('$o_{s} < Q_{1} - 2 \cdot t_{f}$ if it\'s a left outlier and $o_{s} > Q_{3} + 2 \cdot t_{f}$ if it\'s a right one')
        st.markdown('While a weak outlier $o_{w}$ is defined as: ')
        st.markdown('$o_{w} \in [Q_{1} - 2 \cdot t_{f}, Q_{1} - t_{f}]$ if it\'s a left outlier and $o_{w} \in [Q_{3} + t_{f}, Q_{3} + 2 \cdot t_{f}]$ if it\'s a right one')
        st.markdown('In the next numeric input you can insert the value of the **Tukey\'s constant** (is usually setted to 1,5), from the previous formulas we can derive ' + 
                    'that a smaller **k** will reduce the fence\'s size (you will find more outliers but less significant), while a bigger **k** will have an opposite effect,' + 
                    ' so this value must be chosen wisely.')
        
        tukey_const = st.number_input("Tukey’s constant value", 0.5, 7.5, 1.5)
        Q3 = table[use_col].quantile(0.75); Q1 = table[use_col].quantile(0.25); ITQ = Q3- Q1
        
        if stats.skewtest(var_clean)[1] >= 0.025:
            df_AllOut = table[(table[use_col] <= Q1 - (tukey_const * ITQ)) | (table[use_col] >= Q3 + (tukey_const * ITQ))]
            df_StLeftOut = table[table[use_col] < Q1 - (2 * tukey_const * ITQ)]
            df_WeLeftOut = table[(table[use_col] >= Q1 - (2 * tukey_const * ITQ)) & (table[use_col] <= Q1 - (tukey_const * ITQ))]
            df_WeRightOut = table[(table[use_col] >= Q3 + (tukey_const * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * ITQ))]
            df_StRightOut = table[table[use_col] > Q3 + (2 * tukey_const * ITQ)]
        else:
            # calculating the medcouple function for the tukey fence
            if var_clean.shape[0] > 5000:
                MC = np.array([medcouple(var_clean[np.random.choice(var_clean.shape[0], 5000)]) for _ in range(50)]).mean()
            else:
                MC = medcouple(var_clean)
            
            # calculating the tukey fence
            if MC > 0:
                df_AllOut = table[(table[use_col] <= Q1 - (tukey_const * math.exp(-4 * MC) * ITQ)) | (table[use_col] >= Q3 + (tukey_const * math.exp(3 * MC) * ITQ))]
                df_StLeftOut = table[table[use_col] < Q1 - (2 * tukey_const * math.exp(-4 * MC) * ITQ)]
                df_WeLeftOut = table[(table[use_col] >= Q1 - (2 * tukey_const * math.exp(-4 * MC) * ITQ)) & (table[use_col] <= Q1 - (tukey_const * math.exp(-4 * MC) * ITQ))]
                df_WeRightOut = table[(table[use_col] >= Q3 + (tukey_const * math.exp(3 * MC) * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * math.exp(3 * MC) * ITQ))]
                df_StRightOut = table[table[use_col] > Q3 + (2 * tukey_const * math.exp(3 * MC) * ITQ)]
            else:
                df_AllOut = table[(table[use_col] <= Q1 - (tukey_const * math.exp(-3 * MC) * ITQ)) | (table[use_col] >= Q3 + (tukey_const * math.exp(4 * MC) * ITQ))]
                df_StLeftOut = table[table[use_col] < Q1 - (2 * tukey_const * math.exp(-3 * MC) * ITQ)]
                df_WeLeftOut = table[(table[use_col] >= Q1 - (2 * tukey_const * math.exp(-3 * MC) * ITQ)) & (table[use_col] <= Q1 - (tukey_const * math.exp(-3 * MC) * ITQ))]
                df_WeRightOut = table[(table[use_col] >= Q3 + (tukey_const * math.exp(4 * MC) * ITQ)) & (table[use_col] <= Q3 + (2 * tukey_const * math.exp(4 * MC) * ITQ))]
                df_StRightOut = table[table[use_col] > Q3 + (2 * tukey_const * math.exp(4 * MC) * ITQ)]
                
        st.table(pd.DataFrame(np.array([df_StLeftOut.shape[0], df_WeLeftOut.shape[0], df_WeRightOut.shape[0], df_StRightOut.shape[0]]).reshape(1, 4),
                              index = ['Number'], columns = ['Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers']))
        
        # a more specific view of the ouliers by country or generic id and type
        left, right = st.columns(2)
        with left: 
            out_id_col = st.selectbox("Outlier index col", table.columns, 0)
        with right:
            out_type = st.selectbox("Outlier type", ['All', 'Strong left outliers', 'Weak left outliers', 'Weak right outliers', 'Strong right outliers'], 0)
        
        if out_type == 'All':
            res = pd.DataFrame([[nut, df_AllOut[df_AllOut[out_id_col] == nut].shape[0]] for nut in df_AllOut[out_id_col].unique()], 
                               columns = [out_id_col, 'Num. Out.'])
            try: 
                st.plotly_chart(map_creation(res, out_id_col, 'Num. Out.', map_color), use_container_width=True)
            except:
                st.warning('You have to select a NUTS id column in the selection box after \"Outlier index col\" to produce the map')
         
        if out_type == 'Strong left outliers':
            res = pd.DataFrame([[nut, df_StLeftOut[df_StLeftOut[out_id_col] == nut].shape[0]] for nut in df_StLeftOut[out_id_col].unique()], 
                               columns = [out_id_col, 'Num. Out.'])
            try: 
                st.plotly_chart(map_creation(res, out_id_col, 'Num. Out.', map_color), use_container_width=True)
            except:
                st.warning('You have to select a NUTS id column in the selection box after \"Outlier index col\" to produce the map')
            
        if out_type == 'Weak left outliers':
            res = pd.DataFrame([[nut, df_WeLeftOut[df_WeLeftOut[out_id_col] == nut].shape[0]] for nut in df_WeLeftOut[out_id_col].unique()], 
                               columns = [out_id_col, 'Num. Out.'])
            try: 
                st.plotly_chart(map_creation(res, out_id_col, 'Num. Out.', map_color), use_container_width=True)
            except:
                st.warning('You have to select a NUTS id column in the selection box after \"Outlier index col\" to produce the map')
            
        if out_type == 'Weak right outliers':
            res = pd.DataFrame([[nut, df_WeRightOut[df_WeRightOut[out_id_col] == nut].shape[0]] for nut in df_WeRightOut[out_id_col].unique()], 
                               columns = [out_id_col, 'Num. Out.'])
            try: 
                st.plotly_chart(map_creation(res, out_id_col, 'Num. Out.', map_color), use_container_width=True)
            except:
                st.warning('You have to select a NUTS id column in the selection box after \"Outlier index col\" to produce the map')
            
        if out_type == 'Strong right outliers':
            res = pd.DataFrame([[nut, df_StRightOut[df_StRightOut[out_id_col] == nut].shape[0]] for nut in df_StRightOut[out_id_col].unique()], 
                               columns = [out_id_col, 'Num. Out.'])
            try: 
                st.plotly_chart(map_creation(res, out_id_col, 'Num. Out.', map_color), use_container_width=True)
            except:
                st.warning('You have to select a NUTS id column in the selection box after \"Outlier index col\" to produce the map')
        
        out_cou = st.selectbox("Choose the specific value for the id", ['All ids'] + list(res[out_id_col]), 0)
         
        if out_cou == 'All ids': 
            st.write(df_AllOut)
        else:
            st.write(df_AllOut[df_AllOut[out_id_col] == out_cou]) 
        
    if widget == "Consistency checks":
        methodology = st.sidebar.selectbox("Analysis to apply", ['Multiannual analysis', 'Ratio analysis'], 0)
        if methodology == 'Ratio analysis':
            con_checks_id_col = st.sidebar.selectbox("Index col", table.columns, 0)
            country_sel_col = st.sidebar.selectbox("Country col", ['-'] + list(table.columns), 0)
            cat_sel_col = st.sidebar.selectbox("Category col", ['-'] + list(table.columns), 0)
            flag_issue_quantile = st.sidebar.number_input("Flags quantile (S2 and S3)", 0.0, 30.0, 5.0, 0.1)
            prob_cases_per = st.sidebar.number_input("Percentage problematic cases", 0.0, 100.0, 20.0)
            p_value_trend_per = st.sidebar.number_input("P-value percentage trend estimation", 5.0, 50.0, 10.0)

            new_ratio_radio = st.radio("Do you want to create a new ratio or you want to use an existing one:", ('Create a new ratio', 'Existing one'))
            if new_ratio_radio == 'Existing one':
                con_checks_feature = st.selectbox("Variable consistency checks:", col_mul)
            else:
                con_checks_feature = st.text_input('Write here the name of the new ratio', 'R_1')
                left1, right1 = st.columns(2)
                with left1:
                    ratio_num = st.multiselect("Numerator column", col_mul)
                with right1:
                    ratio_den = st.multiselect("Denominator column", col_mul)
                if len(ratio_num) == 1 and len(ratio_den) == 1:
                    table = pd.concat([table, pd.DataFrame(np.divide(table[ratio_num].values, table[ratio_den].values), columns = [con_checks_feature])], axis = 1)
                else:
                    table = pd.concat([table, pd.DataFrame(np.divide(np.nansum(table[ratio_num].values, axis = 1), np.nansum(table[ratio_den].values, axis = 1)), columns = [con_checks_feature])], axis = 1)
                table.loc[table[table[con_checks_feature] == np.inf].index, con_checks_feature] = np.nan
            flag_radio = st.radio("Do you want to use the flags:", ('Yes', 'No')); flags_col = ''
            if flag_radio == 'Yes':
                left1, right1 = st.columns(2)
                with left1:
                    flags_col = st.selectbox("Flag variable", table.columns)
                with right1:
                    notes_col = st.selectbox("Notes variable", ['-'] + list(table.columns))

            table['Class trend'] = 0
            for id_inst in table[con_checks_id_col].unique():
                # trend classification
                inst = table[table[con_checks_id_col] == id_inst][con_checks_feature].values[::-1]
                geo_mean_vec = np.delete(inst, np.where((inst == 0) | (np.isnan(inst))))
                if geo_mean_vec.shape[0] > 3:
                    mann_kend_res = mk.original_test(geo_mean_vec)
                    trend, p, tau = mann_kend_res.trend, mann_kend_res.p, mann_kend_res.Tau
                    if trend == 'increasing':
                        table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 5
                    if trend == 'decreasing':
                        table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 1
                    if trend == 'no trend':
                        if p <= p_value_trend_per/100 and tau >= 0:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 4
                        if p <= p_value_trend_per/100 and tau < 0:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 2
                        if p > p_value_trend_per/100:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 3

            dict_flags = dict(); countries = list(table[country_sel_col].unique())
            if cat_sel_col != '-':
                categories = list(table[cat_sel_col].unique())
                dict_flags[con_checks_feature] = dict()
                for cc in countries:
                    country_table = table[table[country_sel_col] == cc][[con_checks_id_col, con_checks_feature]]
                    inst_lower = set(country_table[country_table[con_checks_feature] <= country_table[con_checks_feature].quantile(flag_issue_quantile/100)][con_checks_id_col].values)
                    inst_upper = set(country_table[country_table[con_checks_feature] >= country_table[con_checks_feature].quantile(1 - (flag_issue_quantile/100))][con_checks_id_col].values)
                    dict_flags[con_checks_feature][cc] = inst_lower.union(inst_upper)
                for cat in categories:
                    cat_table = table[table[cat_sel_col] == cat][[con_checks_id_col, con_checks_feature]]
                    inst_lower = set(cat_table[cat_table[con_checks_feature] <= cat_table[con_checks_feature].quantile(flag_issue_quantile/100)][con_checks_id_col].values)
                    inst_upper = set(cat_table[cat_table[con_checks_feature] >= cat_table[con_checks_feature].quantile(1 - (flag_issue_quantile/100))][con_checks_id_col].values)
                    dict_flags[con_checks_feature][cat] = inst_lower.union(inst_upper)

                dict_check_flags = {}; set_app = set()
                for cc in countries:
                    set_app = set_app.union(dict_flags[con_checks_feature][cc])
                for cat in categories:
                    set_app = set_app.union(dict_flags[con_checks_feature][cat])
                dict_check_flags[con_checks_feature] = set_app
                
                table['Prob inst ' + con_checks_feature] = 0
                table.loc[table[table[con_checks_id_col].isin(dict_check_flags[con_checks_feature])].index, 'Prob inst ' + con_checks_feature] = 1

                # table reporting the cases by countries
                DV_fin_res = np.zeros((len(categories), len(countries)), dtype = int)
                for el in set_app:
                    DV_fin_res[categories.index(table[table[con_checks_id_col] == el][cat_sel_col].unique()[0]), 
                               countries.index(table[table[con_checks_id_col] == el][country_sel_col].unique()[0])] += 1
                    
                DV_fin_res = np.append(DV_fin_res, np.sum(DV_fin_res, axis = 1).reshape((len(categories), 1)), axis = 1)
                DV_fin_res = np.append(DV_fin_res, np.sum(DV_fin_res, axis = 0).reshape(1, len(countries) + 1), axis = 0)
                cou_0_cases = np.where(DV_fin_res[len(categories), :] == 0)
                DV_fin_res = np.delete(DV_fin_res, cou_0_cases, 1)
                countries = [i for j, i in enumerate(countries) if j not in cou_0_cases[0]]
                list_fin_res = DV_fin_res.tolist(); list_prob_cases = []
                for row in range(len(list_fin_res)):
                    for i in range(len(list_fin_res[row])):
                        if row != len(list_fin_res)-1 and i != len(list_fin_res[row])-1:
                            den = len(table[(table[country_sel_col] == countries[i]) & (table[cat_sel_col] == categories[row])][con_checks_id_col].unique())
                        if row == len(list_fin_res)-1 and i != len(list_fin_res[row])-1:
                            den = len(table[table[country_sel_col] == countries[i]][con_checks_id_col].unique())
                        if row != len(list_fin_res)-1 and i == len(list_fin_res[row])-1:
                            den = len(table[table[cat_sel_col] == categories[row]][con_checks_id_col].unique())
                        if row == len(list_fin_res)-1 and i == len(list_fin_res[row])-1:
                            den = len(table[con_checks_id_col].unique())
                        num = list_fin_res[row][i]
                        if den != 0:
                            num_app = round(100 * num/den, 2); list_fin_res[row][i] = str(list_fin_res[row][i]) + '\n(' + str(num_app) + '%)'
                        else:
                            num_app = 0; list_fin_res[row][i] = '0\n(0%)'
                        if i != len(list_fin_res[row])-1 and num_app >= prob_cases_per:
                            if row != len(list_fin_res)-1:
                                list_prob_cases.append([con_checks_feature, countries[i], categories[int(row % len(categories))], str(num_app) + '%', str(num) + ' / ' + str(den)])
                            else:
                                list_prob_cases.append(['Total', countries[i], 'All categories', str(num_app) + '%', str(num) + ' / ' + str(den)])

                flag_notes_on = False
                if flag_radio == 'Yes':
                    if table[flags_col].dtypes == 'O':
                        if notes_col == '-':
                            ones = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p')][con_checks_id_col].values)
                        else:
                            flag_notes_on = True
                            ones = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p') & (pd.isna(table[notes_col]))][con_checks_id_col].values).union(set(table[table[flags_col] == 'p'][con_checks_id_col].values))
                            twos = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p') & (-pd.isna(table[notes_col]))][con_checks_id_col].values)
                            ones = ones - (ones & twos)
                    else:
                        ones = set(table[table[flags_col] == 1][con_checks_id_col].values); twos = set(table[table[flags_col] == 2][con_checks_id_col].values)
                    if flag_notes_on:
                        summ_table = pd.DataFrame([[str(len(twos.intersection(dict_check_flags[con_checks_feature]))) + ' over ' + str(len(twos)), str(round((100 * len(twos.intersection(dict_check_flags[con_checks_feature]))) / len(twos), 2)) + '%'], 
                                                   [str(len(dict_check_flags[con_checks_feature])) + ' / ' + str(len(ones.union(twos))), str(round(100 * (len(dict_check_flags[con_checks_feature]) / len(ones.union(twos))), 2)) + '%'], 
                                                   [str(len(dict_check_flags[con_checks_feature].difference(ones.union(twos)))), str(round((100 * len(dict_check_flags[con_checks_feature].difference(ones.union(twos)))) / len(dict_check_flags[con_checks_feature]), 2)) + '%']], 
                                                   columns = ['Absolute Values', 'In percentage'], 
                                                   index = ['Accuracy', 'new/prev cases', 'Extra cases'])
                    else:
                        summ_table = pd.DataFrame([[str(len(ones.intersection(dict_check_flags[con_checks_feature]))) + ' over ' + str(len(ones)), str(round((100 * len(ones.intersection(dict_check_flags[con_checks_feature]))) / len(ones), 2)) + '%'], 
                                                   [str(len(dict_check_flags[con_checks_feature])) + ' / ' + str(len(ones)), str(round(100 * (len(dict_check_flags[con_checks_feature]) / len(ones)), 2)) + '%'], 
                                                   [str(len(dict_check_flags[con_checks_feature].difference(ones))), str(round((100 * len(dict_check_flags[con_checks_feature].difference(ones))) / len(dict_check_flags[con_checks_feature]), 2)) + '%']], 
                                                   columns = ['Absolute Values', 'In percentage'], 
                                                   index = ['Accuracy', 'new/prev cases', 'Extra cases'])
                    st.table(summ_table)

                table_fin_res = pd.DataFrame(list_fin_res, index = [con_checks_feature + ' (' + str(cat) + ')' for cat in categories] + ['Total'], columns = countries + ['Total'])
                st.table(table_fin_res)
                st.table(pd.DataFrame(list_prob_cases, columns = ['Variable', 'Country', 'Category', '% Value', 'Absolute values']))

                if len(list(table['Class trend'].unique())) > 1:
                    dict_trend = {'Strong decrease': [], 'Weak decrease': [], 'Undetermined trend': [], 'Weak increase': [], 'Strong increase': []}; set_trend = set()
                    for inst in dict_check_flags[con_checks_feature]:
                        class_tr = int(table[table[con_checks_id_col] == inst]['Class trend'].unique()[0])
                        if class_tr != 0:
                            dict_trend[list(dict_trend.keys())[class_tr-1]].append(inst)
                            if class_tr == 1 or class_tr == 3 or class_tr == 5:
                                set_trend.add(inst)
                    st.table(pd.DataFrame([len(v) for v in dict_trend.values()], index = dict_trend.keys(), columns = ['Number of institutions']))
                  
                    if flag_radio == 'Yes':
                        st.table(cr_metrics_table(flag_notes_on, set_trend, ones, twos))

                    trend_type = st.selectbox('Institution trend type', list(dict_trend.keys()), 0)
                    trend_inst = st.selectbox('Institution to vizualize', dict_trend[trend_type])
                    st.plotly_chart(px.line(table[table[con_checks_id_col] == trend_inst][[con_checks_feature, 'Reference year']], 
                                            x = 'Reference year', y = con_checks_feature), use_container_width=True)

                st.write('To download the results select a time variable and then click the Download data button')
                left1, right1 = st.columns(2)
                with left1:
                    time_col = st.selectbox("Time variable:", table.columns)
                with right1:
                    descr_col = st.multiselect("Select Descriptive columns to add to results (optional):", table.columns)

                st.download_button(label = "Download data with lables", file_name = 'result.csv', mime = 'text/csv',
                               data = cr_dnwl_tab_cc(table, con_checks_id_col, time_col, con_checks_features, descr_col, flags_col))
            else:
                st.warning('you have to choose a value for the field "Category selection column".')
        else:
            con_checks_id_col = st.sidebar.selectbox("Index col", table.columns, 0)
            time_col = st.sidebar.selectbox("Time column", table.columns)
            country_sel_col = st.sidebar.selectbox("Country col", ['-'] + list(table.columns), 0)
            cat_sel_col = st.sidebar.selectbox("Category col", ['-'] + list(table.columns), 0)
            retain_quantile = st.sidebar.number_input("Quantile to exclude from the calculation (S1)", 1.0, 10.0, 2.0, 0.1)
            flag_issue_quantile = st.sidebar.number_input("Flags quantile (S2 and S3)", 35.0, 100.0, 95.0, 0.1)
            prob_cases_per = st.sidebar.number_input("Percentage problematic cases", 0.0, 100.0, 20.0)
            p_value_trend_per = st.sidebar.number_input("P-value percentage trend estimation", 5.0, 50.0, 10.0)
            rupt_y_per = st.sidebar.number_input("Rupture years threshold", 5.0, 50.0, 25.0)

            con_checks_feature = st.selectbox("Variable consistency checks:", col_mul)
            flag_radio = st.radio("Do you want to use the flags:", ('Yes', 'No')); flags_col = ''
            if flag_radio == 'Yes':
                left1, right1 = st.columns(2)
                with left1:
                    flags_col = st.selectbox("Flag variable", table.columns)
                with right1:
                    notes_col = st.selectbox("Notes variable", ['-'] + list(table.columns))
            
            # creation of the datasets variable for the geomtric mean and DV computation
            table['Class trend'] = 0; table['Rupt. years'] = ''
            df_DV = table[[con_checks_id_col, time_col, con_checks_feature]].sort_values(by = [con_checks_id_col, time_col], ascending = [True, False])
            df_DV = df_DV[~pd.isna(df_DV[con_checks_feature])]
            df_gm = df_DV[df_DV[con_checks_feature] != 0][[con_checks_id_col, con_checks_feature]]
            
            # trend classification
            for id_inst in df_gm[con_checks_id_col].unique():
                inst = df_gm[df_gm[con_checks_id_col] == id_inst][con_checks_feature].values[::-1]
                if inst.shape[0] > 3:
                    mann_kend_res = mk.original_test(inst)
                    trend, p, tau = mann_kend_res.trend, mann_kend_res.p, mann_kend_res.Tau
                    if trend == 'increasing':
                        table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 5
                    if trend == 'decreasing':
                        table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 1
                    if trend == 'no trend':
                        if p <= p_value_trend_per/100 and tau >= 0:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 4
                        if p <= p_value_trend_per/100 and tau < 0:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 2
                        if p > p_value_trend_per/100:
                            table.loc[table[table[con_checks_id_col] == id_inst].index, 'Class trend'] = 3
            
            # computation of the geometric mean
            df_gm['Occ'] = df_gm.groupby(con_checks_id_col)[con_checks_id_col].transform('count')
            df_gm['Geo Mean'] = df_gm[con_checks_feature] ** (1 / df_gm['Occ'])
            df_gm = df_gm.groupby(by = con_checks_id_col).prod()['Geo Mean']
            df_gm = df_gm[df_gm >= df_gm.quantile(retain_quantile/100)]

            # computation of DV and problematic institutions
            df_DV['Deltas'] = df_DV[con_checks_feature].diff()
            df_DV.drop(df_DV.groupby(con_checks_id_col).head(1).index, axis=0, inplace = True)
            df_pos = df_DV[df_DV['Deltas'] > 0][[con_checks_id_col, 'Deltas']].groupby(con_checks_id_col).sum()
            df_neg = df_DV[df_DV['Deltas'] < 0][[con_checks_id_col, 'Deltas']].groupby(con_checks_id_col).sum()
            df_fin = df_pos.merge(df_neg, on = con_checks_id_col)
            df_fin['Delta prod'] = (df_fin['Deltas_x'] * df_fin['Deltas_y']).abs()
            df_fin = df_fin.merge(df_gm, on = con_checks_id_col)
            df_fin['DV'] = df_fin['Delta prod'] / df_fin['Geo Mean']
            df_fin = df_fin[df_fin['DV'] > df_fin['DV'].quantile(flag_issue_quantile/100)]
            ck_flags = set(df_fin.index); list_prob_cases = []

            # creation od the result tables
            list_countries = list(table[table[con_checks_id_col].isin(ck_flags)][country_sel_col].unique())
            if cat_sel_col == '-':
                DV_fin_res = np.array([len(set(table[table[country_sel_col] == country][con_checks_id_col]).intersection(ck_flags)) for country in list_countries]).reshape((1, len(list_countries)))
                DV_fin_res = np.append(DV_fin_res, np.array([np.sum(DV_fin_res, axis = 1)]), axis = 1)
                DV_fin_res = np.append(DV_fin_res, DV_fin_res, axis = 0)
                list_fin_res = DV_fin_res.tolist()
                for row in range(len(list_fin_res)):
                    for i in range(len(list_fin_res[row])):
                        if i != len(list_fin_res[row])-1:
                            den = len(table[table[country_sel_col] == list_countries[i]][con_checks_id_col].unique())
                        else:
                            den = len(table[con_checks_id_col].unique())
                        num = list_fin_res[row][i]
                        if den != 0:
                            num_app = round(100 * num/den, 2); list_fin_res[row][i] = str(list_fin_res[row][i]) + '\n(' + str(num_app) + '%)'
                        else:
                            num_app = 0; list_fin_res[row][i] = '0\n(0%)'
                        if i != len(list_fin_res[row])-1 and num_app >= prob_cases_per:
                            if row != len(list_fin_res)-1:
                                list_prob_cases.append([con_checks_feature, list_countries[i], str(num_app) + '%', str(num) + ' / ' + str(den)])
                            else:
                                list_prob_cases.append(['Total', list_countries[i], str(num_app) + '%', str(num) + ' / ' + str(den)])
                table_fin_res = pd.DataFrame(list_fin_res, index = [con_checks_feature, 'Total'], columns = list_countries + ['Total'])
            else:
                list_un_cat = list(table[table[con_checks_id_col].isin(ck_flags)][cat_sel_col].unique())
                DV_fin_res = np.array([[len(set(table[(table[country_sel_col] == country) & (table[cat_sel_col] == cat)][con_checks_id_col]).intersection(ck_flags)) 
                                        for country in list_countries] for cat in list_un_cat])
                tab_abs = np.array([[len(set(table[(table[country_sel_col] == country) & (table[cat_sel_col] == cat)][con_checks_id_col])) 
                                        for country in list_countries] for cat in list_un_cat])
                DV_fin_res = np.append(DV_fin_res, np.sum(DV_fin_res, axis = 1).reshape((len(list_un_cat), 1)), axis = 1)
                DV_fin_res = np.append(DV_fin_res, np.sum(DV_fin_res, axis = 0).reshape(1, len(list_countries)+1), axis = 0)
                tab_abs = np.append(tab_abs, np.sum(tab_abs, axis = 1).reshape((len(list_un_cat), 1)), axis = 1)
                tab_abs = np.append(tab_abs, np.sum(tab_abs, axis = 0).reshape(1, len(list_countries)+1), axis = 0)
                tab_per = 100*np.true_divide(DV_fin_res, tab_abs, out = np.zeros(DV_fin_res.shape, dtype=float), where = tab_abs!=0)
                list_fin_res = [[f'{DV_fin_res[i, j]}\n({tab_per[i, j]})' for i in range(DV_fin_res.shape[0])] for j in range(DV_fin_res.shape[1])]
                list_prob_cases = [[con_checks_feature, f'list_countries[{j}]', f'list_countries[{i}]', f'{round(tab_per[i, j], 2)}%', f'{DV_fin_res[i, j]}/{tab_abs[i, j]}'] if i < tab_per.shape[0]-2 or j < tab_per.shape[1]-2
                                   else ['Total', f'list_countries[{j}]', 'All categories', f'{round(tab_per[i, j], 2)}%', f'{DV_fin_res[i, j]}/{tab_abs[i, j]}']
                                   for i, j in np.argwhere(tab_per >= prob_cases_per/100)]

            table['Prob inst ' + con_checks_feature] = 0
            table.loc[table[table[con_checks_id_col].isin(df_fin.index)].index, 'Prob inst ' + con_checks_feature] = 1

            flag_notes_on = False
            if flag_radio == 'Yes':
                if table[flags_col].dtypes == 'O':
                    if notes_col == '-':
                        ones = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p')][con_checks_id_col].values)
                    else:
                        flag_notes_on = True
                        ones = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p') & (pd.isna(table[notes_col]))][con_checks_id_col].values).union(set(table[table[flags_col] == 'p'][con_checks_id_col].values))
                        twos = set(table[(-pd.isna(table[flags_col])) & (table[flags_col] != 'p') & (-pd.isna(table[notes_col]))][con_checks_id_col].values)
                        ones = ones - (ones & twos)
                else:
                    ones = set(table[table[flags_col] == 1][con_checks_id_col].values); twos = set(table[table[flags_col] == 2][con_checks_id_col].values)
                st.table(cr_metrics_table(flag_notes_on, ck_flags, ones, twos))

            st.table(pd.DataFrame(list_fin_res, index = [con_checks_feature + ' (' + str(cat) + ')' for cat in list_un_cat] + ['Total'], columns = list_countries + ['Total']))
            if cat_sel_col == '-':
                st.table(pd.DataFrame(list_prob_cases, columns = ['Variable', 'Country', '% Value', 'Absolute values']))
            else:
                st.table(pd.DataFrame(list_prob_cases, columns = ['Variable', 'Country', 'Category', '% Value', 'Absolute values']))
            
            dict_trend = {'Strong decrease': [], 'Weak decrease': [], 'Undetermined trend': [], 'Weak increase': [], 'Strong increase': []}; set_trend = set()
            for inst in df_fin.index:
                class_tr = int(table[table[con_checks_id_col] == inst]['Class trend'].unique()[0])
                if class_tr != 0:
                    dict_trend[list(dict_trend.keys())[class_tr-1]].append(inst)
                    if class_tr == 1 or class_tr == 3 or class_tr == 5:
                        set_trend.add(inst)
            st.table(pd.DataFrame([len(v) for v in dict_trend.values()], index = dict_trend.keys(), columns = ['Number of institutions']))
         
            if flag_radio == 'Yes':
                st.table(cr_metrics_table(flag_notes_on, set_trend, ones, twos))
            
            trend_type = st.selectbox('Institution trend type', list(dict_trend.keys()), 0)
            trend_inst = st.selectbox('Institution to vizualize', dict_trend[trend_type])
            try:
                line_trend_ch_inst = px.line(table[table[con_checks_id_col] == trend_inst][[con_checks_feature, time_col]], x = time_col, y = con_checks_feature)
                line_trend_ch_inst.update_yaxes(range = [0, max(table[table[con_checks_id_col] == trend_inst][con_checks_feature].values) + (.05 * max(table[table[con_checks_id_col] == trend_inst][con_checks_feature].values))])
                st.plotly_chart(line_trend_ch_inst, use_container_width=True)
            except:
                st.warning('To produce the plot select a different combination of trend type and institution')
            
            st.write('To download the results select a time variable and then click the Download data button')
            descr_col = st.multiselect("Select Descriptive columns to add to results (optional):", table.columns)

            st.download_button(label = "Download data with lables", file_name = 'result.csv', mime = 'text/csv',
                               data = cr_dnwl_tab_cc(table, con_checks_id_col, time_col, con_checks_feature, descr_col, flags_col))
           
