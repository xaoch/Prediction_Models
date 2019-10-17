#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 00:16:37 2019

@author: xavierochoa
"""

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, ClientsideFunction

import pandas as pd
import numpy
from sklearn import cluster
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pickle

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server
app.config.suppress_callback_exceptions = False

students = pd.read_csv("students.csv")
semesters = pd.read_csv("semesters.csv")
students_train = pd.read_csv("students_train.csv")
semesters_train = pd.read_csv("semesters_train.csv")
clus_students = pickle.load(open("student_model.sav", 'rb'))
clus_semesters = pickle.load(open("semester_model.sav", 'rb'))
rf_model=pickle.load(open("rf_model.sav",'rb'))

semesters_test=semesters[semesters['year'] > 2011]
students_test=students[students['student_id'].isin(semesters_test['student_id'].tolist())]


def get_student_data(student_id):
    student_data=students_test[students_test['student_id']==student_id]
    print(student_data)
    factor1=student_data['factor1'].values[0]
    factor2=student_data['factor2'].values[0]
    factor3=student_data['factor3'].values[0]
    factor4=student_data['factor4'].values[0]
    factor5=student_data['factor5'].values[0]
    gpa=student_data['gpa'].values[0]
    return factor1,factor2,factor3,factor4,factor5,gpa

def get_semester_data(student_id):
    semester_data=semesters_test[semesters_test['student_id']==student_id]
    order=semester_data['order'].values[0]
    beta_total=semester_data['beta_total'].values[0]
    num_classes=semester_data['num_classes'].values[0]
    return order,beta_total,num_classes

def get_new_risk_and_uncertainty(factor1,factor2,factor3,factor4,factor5,gpa,order, beta_total,num_classes):
    columns_student=["factor1","factor2","factor3","factor4","factor5","gpa"]
    columns_semester=['order', 'beta_total','num_classes']
    student_data = pd.DataFrame([[factor1,factor2,factor3,factor4,factor5,gpa]], columns=columns_student)
    semester_data = pd.DataFrame([[order,beta_total,num_classes]], columns=columns_semester)
    student_cluster=clus_students.predict(student_data)[0]
    semester_cluster=clus_semesters.predict(semester_data)[0]
    
    similar_students=students_train[students_train["cluster"]==student_cluster]
    similar_students_ids=similar_students["student_id"].tolist()
    
    selected_semesters=semesters_train[semesters_train["student_id"].isin(similar_students_ids)]
    selected_semesters=selected_semesters[selected_semesters['cluster']==semester_cluster]
    
    total_cases=len(selected_semesters)
    failed_cases=len(selected_semesters[selected_semesters['fail']==True])
    risk=failed_cases/total_cases
    return risk,total_cases

def get_forest_risk_and_uncertainty(factor1,factor2,factor3,factor4,factor5,gpa,order, beta_total,num_classes):
    df = pd.DataFrame([[factor1,factor2,factor3,factor4,factor5,gpa,order, beta_total,num_classes]], columns=['factor1','factor2','factor3','factor4','factor5','gpa','order','beta_total','num_classes'])
    prediction=rf_model.predict(df)[0]
    risk=0
    if (prediction):
        risk=1
    certainty=0.7355
    return risk,certainty
    
opt_st=[]
for student in students_test['student_id'].values:
    opt_st.append({'label': student, 'value': student})

navbar = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand("Predictor Dashboard", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Select Student"),
                        dcc.Dropdown(
                           id='student',
                           options=opt_st,
                           value=opt_st[0]['value'],
                           ),
                        html.Br(),
                        html.H2("Select Model"),
                        dcc.Dropdown(
                           id='model',
                           options=[{'label': 'Cluster', 'value': 1},{'label': 'Random Forest', 'value': 2}],
                           value=1,
                           ),
                       
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.H2("Student Data"),
                        html.H3("GPA"),
                        daq.GraduatedBar(id='student_gpa',
                                         color={"gradient":True,"ranges":{"red":[0,6],"yellow":[6,8],"green":[8,10]}},
                                         showCurrentValue=True,
                                         value=10
                                         ),
                        html.Br(),
                        html.H3("Factors"),
                        dcc.Graph(id='student_graph'),
                        html.H2("Semester Data"),
                        dbc.Container([dbc.Row([
                                dbc.Col([
                                        daq.LEDDisplay(id="order",
                                                       label="Semester Order",
                                                       value="0",
                                                       size=64,
                                                       color="#FF5E5E"
                                                       ),
                                          ]),
                                dbc.Col([
                                        daq.LEDDisplay(id="beta",
                                                       label="Total Beta",
                                                       value="0",
                                                       size=64,
                                                       color="#FF5E5E"
                                                       ),
                                          ]),
                                dbc.Col([
                                        daq.LEDDisplay(id="classes",
                                                       label="Number of Classes",
                                                       value="0",
                                                       size=64,
                                                       color="#FF5E5E"
                                                       ),
                                           ]),
                                ])]),
                        html.Br(),
                        html.H2("Prediction"),
                        dbc.Container([dbc.Row([
                                dbc.Col([
                                        daq.Gauge(id='risk-gauge',
                                                  showCurrentValue=True,
                                                  color={"gradient":True,"ranges":{"red":[0,0.4],"yellow":[0.4,0.7],"green":[0.7,1]}},
                                                  label="Risk",
                                                  max=1,
                                                  min=0,
                                                  value=1
                                                  ),
                                        ]),
                                dbc.Col([
                                        daq.Gauge(id='certainty-gauge',
                                                  showCurrentValue=True,
                                                  color={"gradient":True,"ranges":{"red":[0,200],"yellow":[200,500],"green":[500,1000]}},
                                                  label="Certainty",
                                                  max=1000,
                                                  min=0,
                                                  value=1
                                                  ),
                                        ]),
                             ])]),
                    ],
                ),
            ]
        )
    ],
    className="mt-4",
)

app.layout = html.Div(children=[navbar,body]
)


@app.callback(
    [Output("student_graph", "figure"),
     Output("student_gpa", "value"),
     Output("order", "value"),
     Output("beta", "value"),
     Output("classes", "value"),
     Output("risk-gauge", "value"),
     Output("certainty-gauge", "value")],
    [Input("student", "value"),
     Input("model", "value")],
)
def update_plots(student_value,model_value):
    factor1,factor2,factor3,factor4,factor5,gpa=get_student_data(student_value)
    order,beta_total,num_classes=get_semester_data(student_value)
    
    if(model_value==1):
        risk,certainty=get_new_risk_and_uncertainty(factor1,factor2,factor3,factor4,factor5,gpa,order,beta_total,num_classes)
    else:
        risk,certainty=get_forest_risk_and_uncertainty(factor1,factor2,factor3,factor4,factor5,gpa,order,beta_total,num_classes)
        
    data_semester = [
        {
            "x": ['Factor 1','Factor 2', 'Factor 3', 'Factor 4', 'Factor 5', 'GPA'],
            "y": [factor1,factor2,factor3,factor4,factor5,gpa],
            #"y": [10,5,8,4,3,2],
            "text": ['Factor 1','Factor 2', 'Factor 3', 'Factor 4', 'Factor 5', 'GPA'],
            "type": "bar",
            "name": student_value,
        }
    ]
    layout_semester = {
        "autosize": True,
        "xaxis": {"showticklabels": True},
    }
    
    data_student = [{
            "type": 'scatterpolar',
            "r": [factor1, factor2, factor3, factor4, factor5, factor1],
            "theta": ['Factor 1','Factor 2','Factor 3', 'Factor 4', 'Factor 5', 'Factor 1'],
            "fill": 'toself'
            }]

    layout_student = {
            "polar": {
                    "radialaxis": {
                            "visible": True,
                            "range": [0, 10]
                            }
            },
            "showlegend": False
            }

    return {"data": data_student, "layout": layout_student},gpa,order,round(beta_total,2),num_classes,risk,certainty


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)