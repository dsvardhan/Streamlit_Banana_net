import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Define the model class
class BananaNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, hiddenSize2, hiddenSize3, outputSize):
        super(BananaNet, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc4 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Load the trained model
@st.cache_resource
def load_model():
    inp = 7
    hid = 5
    hid2 = 3
    hid3 = 2
    out = 1
    model = BananaNet(inp, hid, hid2, hid3, out)
    model.load_state_dict(torch.load('banana_net.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Streamlit app
st.title('Banana Quality Prediction')

# Input form
st.header('Input Features')
input_data = []
input_labels = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']
for label in input_labels:
    input_data.append(st.number_input(label, value=0.0))

# Convert to tensor
input_tensor = torch.tensor([input_data], dtype=torch.float32)

# Make prediction
if st.button('Predict'):
    with torch.no_grad():
        prediction = model(input_tensor).item()
    st.write(f'Prediction: {"Good" if prediction > 0.5 else "Bad"} (Score: {prediction:.4f})')

# Display the model architecture
st.header('Model Architecture')
st.write(model)
