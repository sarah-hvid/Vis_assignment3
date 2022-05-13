# Assignment 3 - Transfer learning + CNN classification
 
 Link to github of this assignment: 

## Assignment description
In this assignment transfer learning with a pretrained CNN should be used to build an image classifier. ```VGG16``` should be used for feature extraction, and classification should be done on the ```cifar10``` dataset.  
The full assignment description is available in the ```assignment3.md``` file. 

## Methods
This problem relates to classification of images. ```VGG16``` is used to perform feature extraction. The model is loaded without the top classification layer. The ```cifar10``` data is then loaded, the images are normalized and the labels are binarized. The new classification layer is created with a flattening layer, a dense relu layer and a dense softmax output layer. The new model is compiled with exponential decay and a high initial learning rate. It is then fitted to the data. 
As input, the user may specify the following parameters:
  - A name for the plot and classification report using flag -n 
  - How many epochs to run using flag -epochs
  - The batch size using flag -batch_size
If unspecified, standard values will be used with plot name ```plot_history``` and report name ```report```, 10 epochs and a batch size of 128. A plot of the loss and accuracy is saved in the ```output``` folder. The classification report is also saved in the ```output``` folder.


## Usage
In order to run the scripts, certain modules need to be installed. These can be found in the ```requirements.txt``` file. The folder structure must be the same as in this GitHub repository (ideally, clone the repository). The current working directory when running the script must be the one that contains the ```data```, ```output``` and ```src``` folder. Examples of how to run the scripts from the command line: 

Without any specifications::
- python src/transfer_learning.py
    
With all specifications:
- python src/transfer_learning.py -n 4_epochs_256_batch -epochs 4 -batch_size 256
  
Examples of the outputs of the scripts can be found in the ```output``` folder. 

## Results
Using the standard values of the script, the curves of the loss and accuracy plots are nice and even. The accuracy of the model reaches 0.49. Approximately half of the images are therefore classified correctly. Altering the number of epochs and batch size could improve the results. 
