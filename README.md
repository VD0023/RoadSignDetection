# Road Sign Detection using Computer Vision - Python Project Documentation

### Introduction:
Road Sign Detection is an important application of computer vision that aims to identify and classify various road signs present in images. This project utilizes image processing techniques and deep learning to recognize road signs and display their corresponding labels. The user interacts with a graphical user interface (GUI) to load an image containing a road sign, and the system predicts and displays the recognized class of the road sign.

### Prerequisites:
- Python 3.x
- PyQt5
- Keras (with TensorFlow backend)
- scikit-learn
- numpy
- matplotlib
- PIL (Python Imaging Library)

### Project Structure:
The project consists of two main parts:
1. Model Training: This part involves loading the dataset of road sign images, splitting it into training and testing sets, and training a deep learning model to recognize the road signs.
2. GUI Application: The GUI application allows the user to interact with the trained model by providing an image containing a road sign, classifying the road sign, and viewing the training process.

### Model Training:
The **Model Training** code is responsible for:
1. Loading the road sign dataset: The dataset is organized in folders, each representing a specific class of road signs. Images are read from the respective folders, resized to a fixed size (30x30), and converted into NumPy arrays.
2. Preprocessing the data: The image data and corresponding labels are converted into NumPy arrays to be used for training and testing. The labels are also one-hot encoded to represent the classes.
3. Building the deep learning model: A Convolutional Neural Network (CNN) is constructed using the Keras library. The model consists of convolutional layers, pooling layers, and fully connected layers, followed by a softmax activation function to output the class probabilities.
4. Compiling and training the model: The model is compiled with a categorical cross-entropy loss function and the Adam optimizer. It is then trained on the training data and validated on the testing data for a fixed number of epochs.
5. Saving the trained model: The trained model and training history are saved to disk for later use.

### GUI Application:
The **GUI Application** code is responsible for:
1. Creating the graphical user interface (GUI): A PyQt5-based GUI is designed to allow the user to interact with the road sign detection system.
2. Loading an image: The user can browse and select an image containing a road sign. The image is then displayed on the GUI.
3. Classifying the road sign: By clicking the "Classify" button, the system loads the trained model and predicts the class of the road sign in the loaded image.
4. Training the model: The user can train the model by clicking the "Training" button. The model is trained using the same architecture and dataset as in the model training code. The training progress is displayed using loss and accuracy graphs.
5. Displaying the recognized class: The predicted class of the road sign is displayed in the GUI using the "Recognized Class" label.

### Usage:
To use the Road Sign Detection system:
1. Make sure all the prerequisites are installed.
2. Run the "Model Training" code to train the deep learning model on the road sign dataset.
3. Run the "GUI Application" code to open the graphical user interface.
4. Click the "Browse Image" button in the GUI to load an image containing a road sign.
5. Click the "Classify" button to predict and display the recognized class of the road sign in the loaded image.
6. Optionally, click the "Training" button to train the model again. Note that this step is optional if you already have a trained model.

### Conclusion:
The Road Sign Detection

 project demonstrates the use of computer vision and deep learning techniques to recognize and classify road signs. By training a model on a road sign dataset and building a GUI application, users can interact with the system to identify road signs in images. This project can be further enhanced by incorporating real-time video stream processing and integrating it into autonomous driving systems for enhanced road safety.
