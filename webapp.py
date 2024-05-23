import numpy as np
import pickle
import streamlit as st 

# Load the model
model_path = 'predictor.sav'  
# Update the path to your model file
loaded_model = pickle.load(open(model_path, 'rb'))

# Creating a function for Prediction
def completionrate_prediction(input_data):
    # Changing the input_data to numpy array and converting to float
    input_data_as_float = [float(x) for x in input_data]
    input_data_as_numpy_array = np.asarray(input_data_as_float)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    print('the predicted completion rate is {}'.format(prediction))

def main():
    # Giving a title
    st.title('Completion Rate Prediction Application')

    
    # Getting the input data from the user
    ptwmrseagpi = st.text_input('ptwmrseagpi', '0.0')
    ppels_2_cg = st.text_input('ppels_2_cg', '0.0')
    ofsussa = st.text_input('ofsussa', '0.0')
    earcuse_twnetyfive = st.text_input('earcuse_twnetyfive', '0.0')
    pseb_12lse = st.text_input('pseb_12lse', '0.0')
    naspi = st.text_input('naspi', '0.0')
    earclshp_twentyfive = st.text_input('earclshp_twentyfive', '0.0')
    pspmprbs = st.text_input('pspmprbs', '0.0')
    gils = st.text_input('gils', '0.0')
    ofspsa = st.text_input('ofspsa', '0.0')
    ofspsa = st.text_input('ofspsa', '0.0')
    
    
    # Code for Prediction
    if st.button('Predict school completion rate'):
        prediction = completionrate_prediction(['ptwmrseagpi', 'ppels_2_cg',
                                              'ofsussa','earcuse_twnetyfive','pseb_12lse',
                                        'naspi','earclshp_twentyfive','pspmprbs','gils',
                         'ofspsa', 'pcypesfm'])
        st.success(f'Predicted school completion rate is : {prediction}')

if __name__ == '__main__':
    main()
