# TableNet: Deep Learning model for end-to-end Table Detection and Tabular data extraction from Scanned Data Images.
In modern times, more and more number of people are sharing their documents as photos taken from smartphones. A lot of these documents contain lots of information in one or more tables. These tables often contain very important information and extracting this information from the image is a task of utmost importance.
In modern times, information extraction from these tables is done manually, which requires a lot of effort and time and hence is very inefficient. Therefore, having an end-to-end system that given only the document image, can recognize and localize the tabular region and also recognizing the table structure (columns) and then extract the textual information from the tabular region automatically will be of great help since it will make our work easier and much faster.
TableNet is just that. It is an end-to-end deep learning model that can localize the tabular region in a document image, understand the table structure and extract text data from it given only the document image.
Earlier state-of-the-art deep learning methods took the two problems, that is, table detection and table structure recognition (recognizing rows and columns in the table) as separate and treated them separately. However, given the interdependence of the two tasks, TableNet considers them as two related sub-problems and solves them using a single neural network. Thus, also making it relatively lightweight and less compute intensive solution.<br>


## How to run the webapp:

This repository contains a webapp that is made using the [Streamlit](https://streamlit.io/) library. To run the app, go to the repo directory in the terminal and run the following line of code:

    streamlit run app.py
This will start the streamlit server in your local machine and you can then use it. All you need to do in the webapp  is *upload the image in the .bmp format* and the webapp will show you the images of the the table mask, the column masks and the extracted table from the image. Below that, it will also show you the text extracted from the table using OCR.<br> For a better understanding of how the webapp looks and works, please take a look at [this youtube video](https://youtu.be/mHdxzDvSxPQ).

## Prerequisites for running the webapp:

The code for this webapp was tested with `tensorflow 2.6.0, python 3.7.11 and pytesseract 0.3.8` and the webapp was made using `Streamlit 0.86.0`.<br> For the pre-trained model weights, download them from [here](https://drive.google.com/file/d/11cl-QP5xsYmuM-IwCtc1psMH14bb7kFx/view?usp=sharing) and rename the file to **DenseNet-Tablenet.h5** and keep it in the same directory as the webapp script. 
