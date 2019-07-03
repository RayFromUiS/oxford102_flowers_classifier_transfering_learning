# Project description
_Train an image classifier to recognize different species of flowers, 
similar to a phone app telling you the name of your flowers. 
It contains 3 main steps._
1. Load and preprocess the image dataset
2. Train the image classifier on your dataset
3. Use the trained classifier to predict image content

# Project flow
* Load and preprocess the image dataset
  * Training data augmentation
  * Data normalization
  * Data loading
  * Data batching
* Initiate the traing process
  * Import pretrained network
  * Feed forward classifier
  * Track Validation Loss and Accuracy
  * Measure the accuracy on test data
  * Saving the model
* Predict image name
  * Load checkpoint
  * Image processing
  * Class recoginization

# Implementation
**import relevant libraries**
1. Data loading and Preprocessing
  * Download data from [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
  * Using `torchvision.transforms` and `transform.Compose` to preprocessing image files
  * `DataLoader` for batching and shuffling
2. Trainging initiating
  * From `torchvision.models` import resnet101 as model architecture
  * Replace **fc** layer with own structure,and define **loss function** and **optimizer**
  * Feed **train data loader and validation data loader** to traing epoch,which gives a 90% accuracy.
  * Test on data loader for the test data as well, having accuracy of 88%
  * Save check point with `save_check_point(model,train_dataset,check_point)` function
 3. Image recognization
  * Load check point with `load_checkpoint(checkpoint)` return the model
  * Using `PIL.Image` and `numpy` to accomplish image data preprocessing
  * Feeding preprocessed image data to check poing model 
  
# Command line application
  * train.py
    * python train.py --epochs, define the training epoch
    * python train.py --learning_rate, define learning speeed
    * python train.py --gpu, enable user traing data on GPU
    * python train.py --hidden_units, allow user define their hidden units
  * predict.py
    * python predict.py --topk, allower model to show the number of top possibility 
 
# Software and libraries
1. Python 3.7
2. NumPy
3. pandas
4. pytorch
5. matplotlib
6. PIL
7. Json



