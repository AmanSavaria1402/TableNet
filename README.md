TableNet: Deep Learning model for end-to-end Table Detection and Tabular data extraction from Scanned Data Images



In modern times, more and more number of people are sharing their documents as photos taken from smartphones. A lot of these documents contain lots of information in one or more tables. These tables often contain very important information and extracting this information from the image is a task of utmost importance.
In modern times, information extraction from these tables is done manually, which requires a lot of effort and time and hence is very inefficient. Therefore, having an end-to-end system that given only the document image, can recognize and localize the tabular region and also recognizing the table structure (columns) and then extract the textual information from the tabular region automatically will be of great help since it will make our work easier and much faster.
TableNet is just that. It is an end-to-end deep learning model that can localize the tabular region in a document image, understand the table structure and extract text data from it given only the document image.
Earlier state-of-the-art deep learning methods took the two problems, that is, table detection and table structure recognition (recognizing rows and columns in the table) as separate and treated them separately. However, given the interdependence of the two tasks, TableNet considers them as two related sub-problems and solves them using a single neural network. Thus, also making it relatively lightweight and less compute intensive solution.
