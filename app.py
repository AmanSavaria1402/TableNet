import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from tqdm import tqdm
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from PIL import Image
import pytesseract
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Concatenate, Add, Activation, UpSampling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.applications import VGG19
from tensorflow.keras import regularizers
from tensorflow.keras.applications.densenet import DenseNet121
from PIL.Image import frombuffer
from numpy.core.fromnumeric import size
import streamlit as st

# Classes for the custom decoders
# table decoder
class table_decoder(tf.keras.layers.Layer):
    '''
        This is the custom layer defined for the table decoder following the above architecture.
    '''
    def __init__(self):
        super().__init__()
        # defining the layers
        self.conv7 = Conv2D(filters=128, kernel_size=(1,1), kernel_regularizer=regularizers.l2(0.001))
        self.upsamp_pool4 = UpSampling2D(size=(2,2), interpolation='bilinear')
        self.upsamp_pool3 = UpSampling2D(size=(2,2), interpolation='bilinear')
        self.upsamp_out = Conv2DTranspose(filters=2, kernel_size=(3,3), strides=2, padding='same', activation='softmax')
        
    def call(self, inp, pool3, pool4):
        # passing through convolution
        x = self.conv7(inp)
        # upsampling and concatenating with pool4
        x = self.upsamp_pool4(x)
        x = Concatenate()([x, pool4])
        
        # upsampling and concatenating with pool3
        x = self.upsamp_pool3(x)
        x = Concatenate()([x, pool3])
        
        # further upsampling and output
        x = UpSampling2D((2,2))(x)
        x = UpSampling2D((2,2))(x)
        fin = self.upsamp_out(x)
        
        return fin

# column decoder
class col_decoder(tf.keras.layers.Layer):
    '''
        This custom layer is defined for the Column deocder following the above column decoder architecture. 
    '''
    def __init__(self):
        super().__init__()
        # defining the layers
        self.conv7 = Conv2D(filters=128, kernel_size=(1,1), kernel_regularizer=regularizers.l2(0.001), activation='relu')
        self.drop = Dropout(rate=0.8)
        self.conv8 = Conv2D(filters=128, kernel_size=(1,1), kernel_regularizer=regularizers.l2(0.001))
        self.upsamp_pool4 = UpSampling2D((2,2), interpolation='bilinear')
        self.upsamp_pool3 = UpSampling2D((2,2), interpolation='bilinear')
        self.upsamp_out = Conv2DTranspose(filters=2, kernel_size=(3,3), strides=2, padding='same', activation='softmax')
        
    def call(self, inp, pool3, pool4):
        # passing through convolutions
        x = self.conv7(inp)
        x = self.drop(x)
        x = self.conv8(x)
        
        # upsampling and concatenating encoder pool outputs
        x = self.upsamp_pool4(x)
        x = Concatenate()([x, pool4])
        x = self.upsamp_pool3(x)
        x = Concatenate()([x, pool3])
        
        # final upsampling and outputs
        x = UpSampling2D((2,2))(x)
        x = UpSampling2D((2,2))(x)
        fin = self.upsamp_out(x)
        
        return fin

# making the model archtecture
def ModelConstructor():
    '''
        This function makes the tablenet architecture and returns the model object after loading the trained weights.
    '''
    tf.keras.backend.clear_session()
    # making the encoder architecture
    tf.keras.backend.clear_session()
    model_input = Input(shape=(1024,1024,3))
    encoder = DenseNet121(include_top=False, weights='imagenet', input_tensor=model_input)

    # for pool3 and pool4, we are going to use the outputs of the following layers
    # pool4 = pool4_relu
    # pool3 = pool3_relu
    pool3 = encoder.get_layer('pool3_relu').output
    pool4 = encoder.get_layer('pool4_relu').output

    # making all the layers of the encoder untrainable
    for layer in encoder.layers:
        layer.trainable = False

    # continuing the model architecture
    # convolution layers
    conv_6 = Conv2D(filters=512, kernel_size=(1,1), activation='relu', name='block6_conv1',
                    kernel_regularizer=regularizers.l2(0.001))(encoder.output)
    conv6_drop = Dropout(0.2)(conv_6) # this is the dropping probability and in the paper the keep_prop seems to be 0.8.

    conv_7 = Conv2D(filters=256, kernel_size=(1,1), activation='relu', name='block6_conv2',
                    kernel_regularizer=regularizers.l2(0.001))(conv6_drop)
    conv7_drop = Dropout(0.2)(conv_7) # this is the dropping probability and in the paper the keep_prop seems to be 0.8.
    # decoders
    table_mask = table_decoder()
    column_mask = col_decoder()

    table_out = table_mask(conv7_drop, pool3, pool4)
    column_out = column_mask(conv7_drop, pool3, pool4)

    # declaring the model

    tablenet = Model(inputs=model_input, outputs=[table_out, column_out])

    # loading the weights
    tablenet.load_weights('DenseNet-Tablenet.h5')

    return tablenet

# HELPER FUNCTIONS
def decode_image(uploader):
    '''
        This functions takes the uploader object and extracts the image out of it
        and then decodes the image into a numpy array to be used for the model.
        Note: This function does not check if the image is uploaded or not, thus, 
        a manual check is required for checking if the uploader actually contains an image.
    '''
    # getting the bytes from the uploader
    image_bytes = uploader.getvalue()
    # converting the bytes into a numpy array - https://stackoverflow.com/a/49517948/11881261
    image_decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)[:,:,:3]
    
    return image_decoded
    
def predict_masks(image, model):
    '''
        This function takes the image tensor, preprocesses it and predicts the table and column masks from the image.
    '''
    # preprocessing the image
    # resizing
    im = tf.image.resize(image, size=[1024,1024])
    im = tf.cast(im, dtype=tf.float32)/255

    # making a batch
    im = tf.expand_dims(im, axis=0)

    # making prediction using the model
    table_mask, col_mask = model.predict(im)

    return (im, table_mask, col_mask)

def get_mask_image(mask_pred):
    '''
        This function gets the predicted mask image from the masks predicted by the model
    '''
    # taking argmax from both the channels
    mask_pred = tf.argmax(mask_pred, axis=-1)
    # adding a channel axis
    mask_pred = mask_pred[..., tf.newaxis][0]
    
    return mask_pred

def filter_table(image, table_mask):
    '''
        This function turns the image from a matrix to actual image and then uses the table mask to filter out the table from the image.
    '''
    # converting image and mask from matrices to images
    im = tf.keras.preprocessing.image.array_to_img(image)
    mask = tf.keras.preprocessing.image.array_to_img(table_mask)
    # st.text()
    # converting mask to greyscale
    mask = mask.convert('L')

    # changing the alpha values of the image using the table mask
    im.putalpha(mask)
    
    return im

def OCR_Reader(image):
    '''
        This function takes an image as input and uses pytesseract to read and return the textual content in the image.
    '''
    text_data = pytesseract.image_to_string(image)
    return text_data

###############################################################################################################################################
                                                            # ACTUAL STREAMLIT PAGE #
###############################################################################################################################################
st.set_page_config(layout='wide') # set layout wide by default

# title
# set header font size in streamlit - https://discuss.streamlit.io/t/change-font-size-in-st-write/7606
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
    font-family: georgia;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Table Detection in Document Images Using Deep Learning</p>', unsafe_allow_html=True)
# image
st.image('https://images.unsplash.com/photo-1618044733300-9472054094ee?ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTl8fGRvY3VtZW50fGVufDB8fDB8fA%3D%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60',
    caption='Source: https://unsplash.com/photos/XrIfY_4cK1w',
    width=750)

# introduction
st.markdown("""
            In modern times, more and more number of people are sharing their documents as photos taken from smartphones. A lot of these documents contain lots of information in one or more tables. These tables often contain 
            very important information and extracting this information from the image is a task of utmost importance.
            In modern times, information extraction from these tables is done manually, which requires a lot of effort and time and hence is very inefficient. Therefore, having an end-to-end system that given only the document
            image, can recognize and localize the tabular region and also recognizing the table structure (columns) and then 
            extract the textual information from the tabular region automatically will be of great help since it will make our work easier and much faster.
            
            TableNet is just that. It is an end-to-end deep learning model that can localize the tabular region in a document image, understand the table structure and extract text data from it given only the document image.
""", unsafe_allow_html=True)

st.markdown('''<hr>''', unsafe_allow_html=True)

# layout modelling and prediction
# A dialog box/area that asks the user to upload the image
# then displays the original image, table mask, column mask and the extracted table
# at the end also displays text extracted using pytesseract.

# getting the model
tablenet = ModelConstructor()
print("Model loaded...")

# making the file uploader object
# https://docs.streamlit.io/en/stable/api.html#streamlit.file_uploader
st.markdown('''## Upload Image:''', unsafe_allow_html=True)
uploader = st.file_uploader(
    label='Upload the Document Image (Please make sure the image has .bmp format.)',
    type='bmp',
    accept_multiple_files=False,
)

if uploader is not None:
    # getting the image tensor from the uploader
    decoded_image  = decode_image(uploader)
    print('Shape', decoded_image.shape)

    # predicting the masks
    print('Making Predictions...')
    img, table_mask, col_mask = predict_masks(decoded_image, tablenet)
    print('Predictions done...')
    table_mask = get_mask_image(table_mask)
    col_mask = get_mask_image(col_mask)
    # filtering out the table using the table mask
    table = filter_table(img[0], table_mask)

    # plotting all the images
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.header('Actual Image')
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(decoded_image)
        ax.axis('off')
        st.pyplot(fig=fig)
    with col2:
        col2.header('Column Mask')
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(col_mask[:,:,0])
        ax.axis('off')
        st.pyplot(fig=fig)
    with col3:
        col3.header('Table Mask')
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(table_mask[:,:,0])
        ax.axis('off')
        st.pyplot(fig=fig)
    with col4:
        col4.header('Filtered table')
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(table)
        ax.axis('off')
        st.pyplot(fig=fig)

    # printing the textual content in the table
    text = OCR_Reader(table)
    st.markdown('''## Text extracted from the filtered table: ''', unsafe_allow_html=True)
    st.text(text)
