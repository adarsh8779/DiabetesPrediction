import streamlit as st
import pandas as pd
import pickle
lg_std_model = pickle.load(open("lg_std.sav", 'rb'))
std_scaler_model = pickle.load(open("std_scaler.sav", 'rb'))
def main():
    st.title("Slide to Input")
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    num_records = st.slider("Number of Records", min_value=1, max_value=10, step=1)
    inputs = []
    # Create sliders for each feature for each record
    for i in range(num_records):
        record = []
        for feature in features:
            if feature == 'Pregnancies' or feature == 'Age':
                value = st.slider(f"{feature} (Record {i+1})", min_value=0, max_value=30, step=1)
            else:
                value = st.slider(f"{feature} (Record {i+1})", min_value=0, max_value=300, step=1)
            record.append(value)
        inputs.append(record)
    df = pd.DataFrame(inputs, columns= ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    st.write(df)
    inputs_scaled = std_scaler_model.transform(df)
    df1 = pd.DataFrame(inputs_scaled)
    predictions = lg_std_model.predict(df1)
    # Display predictions
    if predictions==0:
        st.write("Not Diabetic")
    else:
        st.write("Diabetic")
if __name__ == "__main__":
    main()

def TakeInputText(rows, cols):
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
           element =  input("Enter Input for Model")
           row.append(element)
        matrix.append(row)
    return matrix
