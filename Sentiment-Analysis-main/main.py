import streamlit as st
import pandas as pd
import requests
from io import BytesIO


prediction_endpoint = "http://127.0.0.1:5000/predict"

st.set_page_config(page_title="Sentiment Analysis App")
st.title("Text Sentiment Predictor")

st.write("Upload a CSV file for bulk prediction or enter a single sentence below:")


uploaded_file = st.file_uploader(
    "Choose a CSV file (with a column named 'Sentence')", 
    type="csv"
)


user_input = st.text_input("Enter text for sentiment prediction:")


if st.button("Predict"):
  
    if uploaded_file is not None:
        try:
            file = {"file": uploaded_file}
            response = requests.post(prediction_endpoint, files=file)

            if response.status_code == 200:
                
                response_bytes = BytesIO(response.content)
                response_df = pd.read_csv(response_bytes)

                st.success("Bulk predictions completed successfully.")
                st.dataframe(response_df)

                
                st.download_button(
                    label="Download Predictions CSV",
                    data=response_bytes,
                    file_name="Predictions.csv",
                    mime="text/csv"
                )

                
                if "X-Graph-Data" in response.headers:
                    import base64
                    graph_data = base64.b64decode(response.headers["X-Graph-Data"])
                    st.image(graph_data, caption="Sentiment Distribution", use_container_width=True)

            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    
    else:
        if not user_input.strip():
            st.warning("Please enter text for prediction.")
        else:
            try:
                
                response = requests.post(prediction_endpoint, json={"text": user_input})

                if response.status_code == 200:
                    result = response.json()
                    if "prediction" in result:
                        st.success(f"Predicted sentiment: {result['prediction']}")
                    else:
                        st.error(f"Unexpected response format: {result}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Request failed: {e}")
