import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and tokenizer
def load_model():
    with open('GBRmodel.pkl', 'rb') as f:
        return pickle.load(f)

def load_scaler():
    with open('model_scaler.pkl', 'rb') as f:
        return pickle.load(f)

model_test = load_model()
scaler_test = load_scaler()

def main():
    st.title('Mortality Prediction App')
    st.write('Enter the values for the following columns:')

    # Input fields for user to enter data
    columns = ['HIV/AIDS', 'Income composition of resources', 'Adult Mortality', 'BMI', 'under-five deaths']
    input_data = {}
    for col in columns:
        input_data[col] = st.number_input(col, value=0.0)

    if st.button('Predict'):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        d=scaler_test.transform(input_df)

        predictions=model_test.predict(d)

        rounded_x_values = [round(x, 2) for x in predictions]

        st.write('Predicted values:')
        st.write(rounded_x_values)

if __name__ == '__main__':
    main()
