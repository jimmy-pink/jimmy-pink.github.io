

## M1 Introduction to DL, Neurons, and ANN

### Summary 
- Deep learning is one of the hottest subjects in data science.
    
- Color restoration applications can automatically convert a grayscale image into a colored image.
    
- Speech enactment applications can synthesize audio clips with lip movements in videos, extracting audio from one video and syncing its lip movements with the audio from another video.
    
- Handwriting generation applications can rewrite a provided message in highly realistic cursive handwriting in a wide variety of styles.
    
- Deep learning algorithms are largely inspired by the way neurons and neural networks function and process data in the brain.
    
- The main body of a neuron is the soma, and the extensive network of arms that stick out of the body are called dendrites. The long arm that sticks out of the soma in the other direction is called the axon. 
    
- Whiskers at the end of the axon are called the synapses. 
    
- Dendrites receive electrical impulses that carry information from synapses of other adjoining neurons. Dendrites carry the impulses to the soma. 
    
- In the nucleus, electrical impulses are processed by combining them, and then they are passed on to the axon. The axon carries the processed information to the synapses, and the output of this neuron becomes the input to thousands of other neurons.
    
- Learning in the brain occurs by repeatedly activating certain neural connections over others, and this reinforces those connections.
    
- An artificial neuron behaves in the same way as a biological neuron.
    
- The first layer that feeds input into the neural network is the input layer.
    
- The set of nodes that provide network output is the output layer.
    
- Any sets of nodes in between the input and output layers are the hidden layers.
    
- Forward propagation is the process through which data passes through layers of neurons in a neural network from the input layer to the output layer.
    
- Given a neural network with weights and biases, you can compute the network output for any given input.
    
## M2 Basics of DL

### Summary
- Gradient descent is an iterative optimization algorithm for finding the minimum of a function.
    
- A large learning rate can lead to big steps and miss the minimum point.
    
- A small learning rate can result in extremely small steps and cause the algorithm to take a long time to find the minimum point.
    
- Neural networks train and optimize their weights and biases by initializing them to random values. Subsequently, we repeat the following process in a loop.
    
- First, we calculate the network output using forward propagation. Second, we calculate the error between the ground truth and the estimated or predicted output of the network. Third, we update the weights and the biases through backpropagation. Last, we repeat the previous three steps until the number of iterations or epochs is reached or the error between the ground truth and the predicted output is below a predefined threshold.
    
- The vanishing gradient problem is caused by the problem with the sigmoid activation function, which prevents neural networks from booming sooner.
    
- In a very simple network of two neurons only, the gradients are small, but more importantly, the error gradient with respect to w1 is very small.
    
- When we do backpropagation, we keep multiplying factors that are less than one by each other, so their gradients tend to get smaller and smaller as we keep moving backward in the network.
    
- Neurons in the earlier layers of the network learn very slowly compared to the neurons in the later layers.
    
- As the earlier layers are the slowest to train, the training process takes too long, and prediction accuracy is compromised.
    
- We don't use the sigmoid or similar functions as activation functions since they are prone to vanishing gradient problems.
    
- Activation functions perform a major role in training a neural network.
    
- You can use seven activation functions to build a neural network.
    
- Sigmoid functions are one of the most widely used activation functions in the hidden layers of a neural network.
    
- Hyperbolic tangent function is a scaled version of the sigmoid function, but it is symmetric over the origin.
    
- The ReLU function is the most widely used activation function when designing networks today, and its main advantage is that it doesn’t activate all neurons at the same time.
    
- Softmax function is ideally used in the output layer of the classifier, where we are trying to get the probabilities to define the class of each input.

## M3 Keras & DL Libraries

### Summary
- TensorFlow, PyTorch, and Keras are the most popular deep learning libraries.
    
- TensorFlow is used in the production of deep learning models and has a very large community of users.
    
- PyTorch is based on the Torch framework in Lua and supports machine learning algorithms running on GPUs. It is preferred over TensorFlow in academic research settings.
    
- PyTorch and TensorFlow are not easy to use and have a steep learning curve. 
    
- Keras is a high-level API for building deep learning models. It has gained favor due to its ease of use and syntactic simplicity, facilitating fast development. 
    
- Keras can build a very complex deep learning network with only a few lines of code. It normally runs on top of a low-level library, such as TensorFlow. 
    
- Before using the Keras library, you need to prepare your data and organize it in the correct format.
    
- You can build and train a neural network with only a few lines of code in Keras. 
    
- A data set can be divided into predictors and a target. 
    
- When you’re using Keras to solve classification problems, you need to transform the target column into an array with binary values.
    
- You can use the 'to_categorical' function from the Keras utilities package to transform a data set column into an array of binary values. 
    
- You can use Keras code to build a classification model.

## M4 DL Models
### Summary
- A neural network with one hidden layer is considered a shallow neural network.
    
- A network with many hidden layers and a large number of neurons in each layer is considered a deep neural network.
    
- Shallow neural networks only take vectors as input.
    
- Deep neural networks can take raw data, such as images and text, as input.
    
- The sudden boom in the deep learning field can be attributed to three main factors: advancements in the field, data availability, and greater computational power.
    
- Convolutional neural networks make the explicit assumption that the inputs are images.
    
- Convolutional neural networks are best for solving problems related to image recognition, object detection, and other computer vision applications.
    
- The input to a convolutional neural network is mostly an (n x m x 1) for grayscale images or an (n x m x 3) for colored images.
    
- In the convolutional layer, we define filters and compute the convolution between the defined filters and each of the three images.
    
- A convolutional layer also consists of ReLUs, which filter the output of the convolutional step, passing only positive values and turning any negative values to 0.
    
- The pooling layer is added to reduce the spatial dimensions of the data propagating through the network.
    
- The two types of pooling widely used in convolutional neural networks are max pooling and average pooling.
    
- In the fully connected layer, we flatten the output of the last convolutional layer and connect every node of the current layer with every other node of the next layer.
    
- Neural networks and deep learning models see data points as independent instances.
    
- Recurrent Neural Networks, or RNNs, don't just take new input but also take the output from the previous data point as input.
    
- RNNs are good at modeling patterns and sequences of data, such as texts, genomes, handwriting, and stock markets.
    
- A popular type of RNN is the long short-term memory model (LSTM).
    
- LSTMs are used for applications such as image generation, handwriting generation, automatic image captioning, and automatic video descriptions.
    
- Autoencoding is a data compression algorithm where the compression and decompression functions are learned automatically from data.
    
- Autoencoders are data-specific.
    
- Applications of autoencoders include data denoising and dimensionality reduction for data visualization.
    
- Autoencoding can take an image as an input, use an encoder to find the optimal compressed representation of the input image, and then use a decoder to restore the original image.
    
- A popular type of autoencoder is restricted Boltzmann machines.
    
- Applications of restricted Boltzmann machines include fixing imbalanced data sets, estimating missing data set values, and automatic feature extraction.