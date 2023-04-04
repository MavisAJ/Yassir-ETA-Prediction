# Import the required Libraries
import gradio as gr
import numpy as np
import pandas as pd
import os, pickle
import re

def loaded_object(filepath= r'ml_items' ):
  with open(filepath,'rb') as file:
    loaded_object = pickle.load(file)
  return loaded_object
###### SETUP

###### instantiating loaded objects
loaded_object = loaded_object()
ml_model = loaded_object['model']
ml_scaler = loaded_object['scaler']
print(ml_model)
print(ml_scaler)
####loading model
inputs = ['Origin_lat','Origin_lon','Destination_lat','Destination_lon','Trip_distance','maximum_2m_air_temperature',
          'mean_2m_air_temperature','minimum_2m_air_temperature','times_encoded','cluster_id']
 #Defining the predict function
 
def predict(*args,scaler = ml_scaler, model =ml_model):
    # Creating a dataframe of inputs
    input_data = pd.DataFrame([args], columns=inputs)
    print(input_data)
    input_data= scaler.transform(input_data)
    # Modeling
    #with gr.Row():
    output_str = 'Hey there,Your ETA is' 
    dist = 'seconds'
    model_output = abs(int(model.predict(input_data)))
    #if model_output <0:
    #  model_output = 0
    return f"{output_str} {model_output} {dist}"

   
# Function to process inputs and return prediction
    # Creating a dataframe of inputs
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
  gr.Markdown("# YASSIR ETA PREDICTION")
  gr.Markdown("""This app uses a machine learning model to predict the ETA of trips on the Yassir Hailing App.Refer to the expander at the bottom for more information on the inputs.""")

  with gr.Row():
        origin_lat= gr.Slider(2.807,3.381,step = 0.01,interactive=True, value=2.807, label = 'origin_lat')
        origin_lon = gr.Slider(36.589,36.82,step =0.01,interactive=True, value=36.589,label = 'origin_lon')
        Destination_lat =gr.Slider(2.807,3.381,step = 0.1,interactive=True, value=2.81,label ='Destination_lat')
        Destination_lon =gr.Slider(36.596,36.819,step = 0.1,interactive=True, value=36.596,label ='Destination_lon')
        Trip_distance = gr.Slider(1,62028,step =100,interactive=True, value= 200,label = 'Trip_distance')
        
  with gr.Column():
    maximum_2m_air_temperature =gr.Slider(288.201, 294.411, step = 0.1,interactive=True, value=288.201,label ='maximum_2m_air_temperature')
    mean_2m_air_temperature =gr.Slider(285.203, 291.593,step = 0.1,interactive=True, value=285.203,label ='mean_2m_air_temperature')
    minimum_2m_air_temperature = gr.Slider( 282.348, 287.693,step = 0.1,interactive=True,value=282.348, label ='minimum_2m_air_temperature')
    times_encoded = gr.Dropdown([1,2,3],label="Time of the day",value= 3)
    cluster_id = gr.Dropdown([1,2,3,4,5,6,7],label="Cluster ID", value=4)
  with gr.Row():
    btn = gr.Button("Predict").style(full_width=True)
    output = gr.Textbox(label="Prediction")
    # Expander for more info on columns
  with gr.Accordion("Information on inputs"):
      gr.Markdown("""These are information on the inputs the app takes for predicting a rides ETA.
                    - origin_lat: Origin in degree latitude)
                    - origin_lon:  Origin in degree longitude
                    - Destination_lat: Destination latitude
                    - Destination_lon: Destination logitude
                    - Trip Distance : Distance in meters on a driving route
                    - Cluster ID : Select the cluster within which you started your trip
                    - Time of the day: What time in the day did your trip start, 1- morning(or daytime),2 - evening 3- midnight
                    """)
  btn.click(fn = predict,inputs= [origin_lat,origin_lon, Destination_lat, Destination_lat,Trip_distance,
            maximum_2m_air_temperature,mean_2m_air_temperature, minimum_2m_air_temperature,times_encoded,
            cluster_id], outputs = output)
  app.launch(share = True, debug =True,server_port= 6006)