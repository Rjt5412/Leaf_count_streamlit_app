import streamlit as st

from get_prediction import get_prediction_count_DenseNet121_transfer_learning_model
 
from PIL import Image

PAGE_CONFIG = {"page_title":"Leaf Count App","page_icon":":seedling:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

def main():
    st.title('Leaf Count App')
    
    # Take the uploaded file
    uploaded_file = st.file_uploader("Too lazy to count? Upload a plant image and we'll count the leaves for you", type="png")	
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Show the uploaded file
        image_data = uploaded_file.getvalue()
        with st.spinner(text="Processing your image"):
          predicted_count, predicted_count_quantized, time_taken, time_taken_quantized = get_prediction_count_DenseNet121_transfer_learning_model(image_data)
          st.success("Count predicted successfully!")
        st.markdown("The predicted count of leaves by unquantized model is **{}**".format(predicted_count))
        st.markdown("The predicted count of leaves by quantized model is **{}**".format(predicted_count_quantized))
        st.image(image)
        st.subheader("Additional Info: ")
        st.markdown("Time taken by unquantized model to predict the leaf count: {:.2f} seconds".format(time_taken))
        st.markdown("Unquantized model size: 35mb")
        st.markdown("Time taken by quantized model to predict the leaf count: {:.2f} seconds".format(time_taken_quantized))
        st.markdown("Quantized model size: 7mb")

if __name__ == '__main__':
	main()
