
## M1 Advanced Keras Functionalities
### Summary
- Keras is a high-level neural networks API written in Python and capable of running on top of TensorFlow, Theano, and CNTK.
    
- Keras is widely used in industry and academia for various applications, from image and speech recognition to recommendation systems and natural language processing.
    
- Keras Functional API offers advantages like flexibility, clarity, and reusability.
    
- You can use Keras Functional API to develop models in diverse fields such as healthcare, finance, and autonomous driving.
    
- Keras Functional API enables you to define layers and connect them in a graph of layers.
    
- The Functional API can handle models with multiple inputs and outputs.
    
- Another powerful feature of the Functional API is shared layers, which are helpful when you want to apply the same transformation to multiple inputs.
    
- Creating custom layers in Keras allows you to tailor your models to specific needs, implement novel research ideas, and optimize performance for unique tasks. 
    
- By practicing and experimenting with custom layers, you’ll better understand how neural networks work and enhance your innovation ability. 
    
- TensorFlow 2.x is a powerful and flexible platform for machine learning with features such as eager execution, high-level APIs, and a rich ecosystem of tools. 
    
- Understanding these features and capabilities will help you build and deploy machine learning models more effectively, whether working on research, prototyping, or production applications.


## M2 Advanced CNNs in Keras

### Transfer Learning Implementaion

Transfer learning is a powerful technique that enables the use of pre-trained models on new tasks, significantly saving time and computational resources. Here are key tips for implementing transfer learning effectively:

1. **Choose the right pre-trained model:** Select a model trained on a dataset similar to your target task to enhance performance. Popular models like VGG16, ResNet, or InceptionV3 are particularly effective for image-related tasks. Ensure that the architecture aligns with your specific problem requirements.
    
2. **Freeze early layers:** In the initial training stages, freeze the early layers of the pre-trained model to preserve their learned features. This approach is beneficial when working with small datasets or datasets that closely resemble the original dataset the model was trained on.
    
3. **Fine-tune later layers:** As training progresses, gradually unfreeze the deeper layers and fine-tune them. These layers capture task-specific features, and fine-tuning allows the model to adapt better to the nuances of your new dataset.
    
4. **Adjust learning rates:** Use a lower learning rate for fine-tuning to prevent catastrophic forgetting of the pre-trained knowledge. High learning rates during this phase can disrupt the learned features and degrade model performance.
    
5. **Use data augmentation:** Implement data augmentation techniques, particularly for image tasks, to increase variability within the dataset. This practice helps prevent overfitting and enhances the model's ability to generalize.
    
6. **Consider domain adaptation:** If there is a significant disparity between the domain of the pre-trained model and your target task, consider applying domain adaptation techniques. These methods can help align the source and target datasets, improving the model's performance.
    
### Summary

- Using advanced techniques to develop convolutional neural networks (CNNs) using Keras can enhance deep learning models and significantly improve performance on complete tasks.
    
- Incorporating various data augmentation techniques using Keras can improve the performance and generalization ability of models.
    
- Transfer learning using pre-trained models in Keras improves training time and performance.
    
- Pre-trained models in Keras allow you to build high-performing models even with limited computational resources and data.
    
- Transfer learning involves fine tuning of pre-trained models when you do not have enough data to train a deep-learning model.
    
- Fine tuning pre-trained models allows you to adapt the model to a specific task, leading to even better performance.
    
- TensorFlow is a powerful library that enables image manipulation tasks, such as classification, data augmentations, and more advanced techniques.
    
- TensorFlow’s high-level APIs simplify the implementation of complete image-processing tasks.
    
- Transpose convolution is helpful in image generation, super-resolution, and semantic segmentation applications.
    
- It performs the inverse convolution operation, effectively up-sampling the input image to a larger higher resolution size.
    
- It works by inserting zeros between elements of the input feature map and then applying the convolution operation.

### Glossary

| Term                                       | Definition                                                                                                                                                              |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Activation function**                    | A mathematical function used in neural networks to determine the output of a neuron.                                                                                    |
| **Adam optimizer**                         | An optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iteratively based on training data. |
| **Augmentation**                           | A process of increasing the diversity of training data by applying various transformations like rotation, scaling, and so on.                                           |
| **Binary cross-entropy**                   | A loss function used for binary classification tasks, measuring the performance of a classification model whose output is a probability value between 0 and 1.          |
| **Convolution**                            | A mathematical operation used in deep learning, especially in convolutional neural networks (CNNs), for filtering data.                                                 |
| **Custom augmentation function**           | A user-defined function that applies specific transformations to images during data augmentation, providing full control over the augmentation process.                 |
| **Data augmentation**                      | Techniques used to increase the diversity of training data by applying random transformations such as rotation, translation, flipping, scaling, and adding noise.       |
| **Deconvolution**                          | Also known as transpose convolution, this is a technique used to up-sample an image, often used in generative models.                                                   |
| **Dense layer**                            | A fully connected neural network layer, where each input node is connected to each output node, commonly used in the final stages of a network.                         |
| **Feature map**                            | A set of features generated by applying a convolution operation to an image or data input.                                                                              |
| **Feature-wise normalization**             | A technique to set the mean of the data set to 0 and normalize it to have a standard deviation of 1.                                                                    |
| **Fine-tuning**                            | The process of unfreezing some of the top layers of a pre-trained model base and jointly training both the newly added layers and the base layers for a specific task.  |
| **Flatten layer**                          | A layer that converts the output of a convolutional layer to a 1D array, allowing it to be passed to a fully connected layer.                                           |
| **Generative adversarial networks (GANs)** | A class of machine learning frameworks where two neural networks compete with each other to create realistic data samples.                                              |
| **Height shift range**                     | A data augmentation parameter that randomly shifts an image vertically, altering its position to improve model robustness to vertical translations.                     |
| **TensorFlow Hub**                         | A repository of reusable machine learning modules, which can be easily integrated into TensorFlow applications to accelerate development.                               |
| **TensorFlow.js**                          | A library for training and deploying machine learning models in JavaScript environments, such as web browsers and Node.js.                                              |
| **Horizontal flip**                        | A data augmentation technique where the image is flipped horizontally, creating a mirror image to increase data diversity.                                              |
| **ImageDataGenerator**                     | A Keras class used for generating batches of tensor image data with real-time data augmentation.                                                                        |
| **ImageNet**                               | A large visual database designed for use in visual object recognition software research, often used as a data set for pre-training convolutional neural networks.       |
| **Image processing**                       | The manipulation of an image to improve its quality or extract information from it.                                                                                     |
| **Kernel**                                 | A small matrix used in convolution operations to detect features such as edges in images.                                                                               |
| **Latent vector**                          | A vector representing compressed data in a lower-dimensional space, often used in generative models.                                                                    |
| **Pre-trained model**                      | A model previously trained on a large data set, which can be used as a starting point for training on a new, related task.                                              |
| **Random noise**                           | A type of custom augmentation that adds random noise to images, simulating different lighting conditions and sensor noise to make models more robust.                   |
| **Rotation range**                         | A data augmentation parameter that randomly rotates an image within a specified range of degrees, enhancing model robustness to rotations.                              |
| **Sample-wise normalization**              | A technique to set the mean of each sample to 0 and normalize each sample to have a standard deviation of 1.                                                            |
| **Semantic segmentation**                  | A deep learning task that involves classifying each pixel in an image into a predefined class.                                                                          |
| **Shear range**                            | A data augmentation parameter that applies a shear transformation to an image, slanting it along one axis to simulate different perspectives.                           |
| **Stride**                                 | A parameter in convolution that determines the step size of the kernel when moving across the input data.                                                               |
| **TensorFlow**                             | An open-source machine learning library used for various tasks, including deep learning and image processing.                                                           |
| **Transfer learning**                      | A method where a pre-trained model is adapted to a new, related task by adjusting its weights, allowing it to perform well even with limited data for the new task.     |
| **Transpose convolution**                  | An operation that reverses the effects of convolution, often used for up-sampling in image processing.                                                                  |
| **VGG16**                                  | A convolutional neural network model pre-trained on the ImageNet data set, commonly used in transfer learning for tasks involving image classification.                 |
| **Width shift range**                      | A data augmentation parameter that randomly shifts an image horizontally, altering its position to improve model robustness to horizontal translations.                 |
| **Zoom range**                             | A data augmentation parameter that randomly zooms in or out on an image, altering its scale during training.                                                            |

## M3 Transformers in Keras

### Summary

- The transformer model consists of two main parts: the encoder and the decoder.
    
- Both the encoder and decoder are composed of layers that include self-attention mechanisms and feedforward neural networks.
    
- Transformers have become a cornerstone in deep learning, especially in natural language processing.
    
- Understanding and implementing transformers will enable you to build powerful models for various tasks.
    
- Sequential data is characterized by its order and the dependency of each element on previous elements.
    
- Transformers address the limitations of recurrent neural networks (RNNs) and long short-term memory networks (LSTMs) by using self-attention mechanisms, which allow the model to attend to all positions in the input sequence simultaneously.
    
- Transformers’ versatile architecture makes them applicable to a wide range of domains, including computer vision, speech recognition, and even reinforcement learning.
    
- Vision transformers have shown that self-attention mechanisms can be applied to image data.
    
- By converting audio signals into spectrograms, transformers can process the sequential nature of speech data.
    
- Transformers have found applications in reinforcement learning, where they can be used to model complex dependencies in sequences of states and actions.
    
- Time series data is a sequence of data points collected or recorded at successive points in time.
    
- By leveraging the self-attention mechanism, transformers can effectively capture long-term dependencies in time series data, making them a powerful tool for forecasting.
    
- The key components of the transformer model include an embedding layer, multiple transformer blocks, and a final dense layer for output prediction.
    
- Sequential data is characterized by its temporal or sequential nature, meaning that the order in which data points appear is important.
    
- TensorFlow provides several layers and tools specifically designed for sequential data. These include:
    
    - RNNs
        
    - LSTMs
        
    - Gated recurrent units
        
    - Convolutional layers for sequence data (Conv1D)
        
- Text data requires specific preprocessing steps, such as tokenization and padding.
    
- TensorFlow’s TextVectorization layer helps in converting text data into numerical format suitable for model training.


### Glossary
| Term                                                           | Definition                                                                                                                                                                                                                                        |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Autoregressive integrated moving average (ARIMA)               | A time series forecasting model that is used to predict future data points by combining three components: autoregression, differencing, and moving averages.                                                                                      |
| Bidirectional Encoder Representations from Transformers (BERT) | A deep learning model in which every output element is connected to every input element. The weightings between them are dynamically calculated based on their connection.                                                                        |
| Convolutional neural networks (CNNs)                           | A type of deep learning model designed to process and analyze visual data by automatically detecting patterns through convolutional layers.                                                                                                       |
| Computer vision                                                | A field of AI that enables machines to interpret and understand visual information from various content such as images and videos.                                                                                                                |
| Cross-attention mechanism                                      | A model that allows the model to focus on relevant parts of the input sequence while generating output.                                                                                                                                           |
| Decoder                                                        | A transformer decoder is a neural network architecture used in natural language processing tasks like machine translation and text generation. It generates output text by combining with an encoder to process input text.                       |
| Deep learning                                                  | A branch of artificial intelligence (AI). Deep learning is a subset of machine learning that uses large, multi-layered neural networks to automatically learn and make predictions from complex data.                                             |
| Embed                                                          | In transformers, embedding is the technique of converting input tokens into dense, continuous vectors that represent their semantic meaning within the model.                                                                                     |
| Encoder                                                        | Encoders are neural network layers that process the input sequence and produce a continuous representation of the input. The transformer encoder architecture is the basis for many state-of-the-art models in natural language processing tasks. |
| Generative pre-trained transformers (GPT)                      | The neural network-based language prediction models that are built on the Transformer architecture. They analyze natural language queries, known as prompts, and predict the best possible response based on their understanding of language.     |
| Image recognition                                              | A software's ability to identify and classify people, objects, places, writing, and actions in digital images and video.                                                                                                                          |
| Image processing                                               | A technique of manipulating and analyzing digital images to enhance, extract information, or convert them into a different format.                                                                                                                |
| Long short-term memory networks (LSTMs)                        | A type of recurrent neural network (RNN) designed to capture and maintain long-term dependencies in sequential data.                                                                                                                              |
| Layer normalization                                            | A technique used in Transformer neural networks to normalize the input values of all neurons in a layer for each data sample.                                                                                                                     |
| Natural language generation                                    | The use of artificial intelligence to produce spoken or written human-like text.                                                                                                                                                                  |
| Natural language processing (NLP)                              | Branch of artificial intelligence that enables computers to understand, manipulate, and generate human language (natural language).                                                                                                               |
| Parallelization                                                | In transformers, parallelization refers to the ability to process multiple elements of a sequence simultaneously, rather than sequentially, to speed up training and inference.                                                                   |
| Recurrent neural network (RNN)                                 | A deep learning model that is trained to process and convert a sequential data input into a specific sequential data output.                                                                                                                      |
| Reinforcement learning                                         | An area of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative rewards.                                                                                                           |
| Self-attention mechanisms                                      | Mechanisms in transformers that allow each element of a sequence to dynamically focus on and weigh the importance of other elements in the sequence to capture context and dependencies.                                                          |
| Sequence                                                       | In transformers, a sequence refers to an ordered set of tokens, such as words, that are processed together as input to capture dependencies and contextual information across the entire sequence.                                                |
| Sequential data                                                | Data that is ordered in a specific sequence, where the arrangement of elements matters, such as time series, audio, or text.                                                                                                                      |
| Speech recognition                                             | A technology that converts spoken language into text by analyzing and interpreting audio signals.                                                                                                                                                 |
| Vision transformers (ViTs)                                     | A type of neural network architecture that applies transformer models to image analysis, treating image patches as sequences to capture visual patterns.                                                                                          |
| Transformers                                                   | A technology that can leverage self-attention mechanisms to process input data in parallel, making them highly efficient and powerful.                                                                                                            |
## M4 Unsupervised Learning and Generative Models in Keras

### Summary and Highlights

- Unsupervised learning is a type of machine learning in which an algorithm finds patterns in data without labels or predefined outcomes.
    
- Unsupervised learning can be broadly categorized into two types: clustering and dimensionality reduction.
    
- Autoencoders consist of two main parts: encoder and decoder.
    
- Generative adversarial networks (GANs) consist of two networks, the generator and the discriminator, which compete against each other in a zero-sum game.
    
- Generator network generates new data instances that resemble the training data.
    
- Discriminator network evaluates the authenticity of the generated data.
    
- Autoencoders are versatile tools for various tasks, including data denoising, dimensionality reduction, and feature learning.
    
- The basic architecture of an autoencoder includes three main components: encoder, bottleneck, and decoder.
    
- There are different types of autoencoders: basic autoencoders, variational autoencoders (VAEs), and convolutional autoencoders.
    
- Diffusion models are powerful tools for generative tasks, capable of producing high-quality data samples and enhancing image quality.
    
- They are probabilistic models that generate data by iteratively refining a noisy initial sample.
    
- The process is akin to simulating the physical process of diffusion, where particles spread out from regions of high concentration to regions of low concentration.
    
- Diffusion models work by defining a forward process and a reverse process.
    
- GANs are a revolutionary type of neural network architecture designed for generating synthetic data that closely resembles real data.
    
- GANs consist of two main components: a generator and a discriminator.
    
- These two networks are trained simultaneously through a process of adversarial training.
    
- This adversarial training loop continues until the generator produces data that the discriminator can no longer distinguish from real data.
    
- Unsupervised learning is a powerful approach for discovering hidden patterns in data, and TensorFlow provides robust tools to facilitate these tasks.
    
- Common applications include clustering, dimensionality reduction, and anomaly detection.
    
- These applications are widely used in various domains such as customer segmentation, image compression, and fraud detection.

### Glossary

|Term|Definition|
|---|---|
|Anomaly detection|A type of unsupervised learning used to identify unusual data points that do not fit the general pattern in a dataset.|
|Adam Optimizer|An optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iteratively based on training data.|
|Adversarial training|A training method in GANs where the generator and discriminator networks are trained simultaneously, with the generator aiming to fool the discriminator, and the discriminator trying to accurately classify real and fake data.|
|Autoencoders|A type of neural network used to learn efficient representations of data, often for dimensionality reduction or feature learning.|
|Binary crossentropy loss|A loss function used in binary classification tasks, often utilized in training neural networks to measure the difference between predicted and actual outputs.|
|Bottleneck|The central, most compressed layer in an autoencoder that contains the most critical features of the input data.|
|Clustering|A method in unsupervised learning that involves grouping data points into clusters, where data points in the same cluster are more similar to each other.|
|Convolutional autoencoders|A type of autoencoder that uses convolutional layers, making it particularly effective for tasks involving image data.|
|Convolutional neural network (CNN)|A type of deep neural network commonly used in image processing tasks, known for its ability to capture spatial hierarchies in images.|
|Data augmentation|A technique used to increase the diversity of data available for training models by generating new synthetic data, often by applying transformations to existing data.|
|Decoder|The part of an autoencoder that reconstructs the input data from the compressed latent space representation.|
|Denoising|The process of removing noise from data, such as images, to improve their quality.|
|Diffusion|A physical process where particles spread from regions of high concentration to low concentration; in diffusion models, this concept is simulated to generate or enhance data.|
|Diffusion model|A type of probabilistic generative model that iteratively refines noisy data to produce high-quality samples, often used in image generation.|
|Dimensionality reduction|A process in unsupervised learning that reduces the number of random variables under consideration by obtaining a set of principal variables.|
|Discriminator network|In a generative adversarial network (GAN), this network evaluates the authenticity of the generated data, distinguishing between real and fake data.|
|Encoder|The part of an autoencoder that compresses the input data into a latent-space representation.|
|Epochs|In machine learning, an epoch refers to one complete pass of the training dataset through the learning algorithm.|
|Feature learning|A set of techniques that allow a machine to automatically discover the representations needed for feature detection or classification from raw data.|
|Forward process|In diffusion models, the process of gradually adding noise to data over a series of steps.|
|Functional API|A way to build neural networks in Keras that allows for more flexible model architectures than the Sequential API.|
|Generative adversarial networks (GANs)|A class of neural networks where two networks, the generator and the discriminator, compete against each other, leading to the generation of realistic data.|
|Generator|In GANs, the neural network creates synthetic data from random noise, aiming to produce data that closely resembles real data.|
|Image-to-image translation|A task in computer vision where an image from one domain is transformed into an image in another domain, such as converting a sketch into a photo.|
|K-means algorithm|A popular clustering technique that partitions a dataset into distinct groups based on the features of the data points.|
|Keras|An open-source software library that provides a Python interface for artificial neural networks and is used to create deep learning models.|
|Latent-space representation|The compressed version of input data generated by the encoder in an autoencoder.|
|MNIST dataset|A large database of handwritten digits that is commonly used for training image processing systems and machine learning models.|
|Modified National Institute of Standards and Technology (MNIST) Dataset|A large database of handwritten digits commonly used for training various image processing systems.|
|Neural network architecture|The structured layout of a neural network, including its layers, connections, and the flow of information within it.|
|Normalization|The process of scaling input features so they have a mean of zero and a standard deviation of one, often used to improve the performance of neural networks.|
|Principal component analysis (PCA)|A dimensionality reduction technique that transforms data into a set of linearly uncorrelated variables called principal components.|
|Probabilistic model|A model that incorporates randomness and uncertainty, often used to predict distributions or simulate processes that have inherent variability.|
|Reverse process|In diffusion models, the process of removing noise step by step to reconstruct the original data from a noisy sample.|
|Stochastic gradient descent (SGD)|An optimization method that adjusts weights iteratively based on a subset of training data, used in training neural networks.|
|Supervised learning|A type of machine learning where the algorithm is trained on labeled data, meaning the outcome or target variable is known during training.|
|t-distributed stochastic neighbor embedding (t-SNE)|A dimensionality reduction technique used for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map.|
|TensorFlow|An open-source machine learning library developed by Google, widely used for building and training machine learning models.|
|Text-to-image synthesis|A process where a model generates an image based on a textual description provided as input.|
|Training epoch|A single pass through the entire training dataset during the training process of a machine learning model.|
|Unsupervised learning|A type of machine learning that finds patterns in data without any labels or predefined outcomes.|
|Variational autoencoders (VAEs)|A type of autoencoder that introduces probabilistic elements to generate new data samples, often used in generative models.|
|Zero-Sum Game|A situation in competitive contexts where gain or loss of participants is exactly balanced by the losses or gains of another participant.|

## M5 Advanced Keras Techniques

### Summary and Highlights

- Advanced Keras techniques include custom training loops, specialized layers, advanced callback functions, and model optimization with TensorFlow.
    
- These techniques will help you create more flexible and efficient deep learning models.
    
- A custom training loop consists of a data set, model, optimizer, and the loss function.
    
- To implement the custom training loop, you iterate over the data set, compute the loss, and apply gradients to update the model’s weights.
    
- Some of the benefits of custom training loops include custom loss functions and metrics, advanced logging and monitoring, flexibility for research, and integration with custom operations and layers.
    
- Hyperparameters are the variables that govern the training process of a model.
    
- Examples include the learning rate, batch size, and the number of layers or units in a neural network.
    
- Keras Tuner is a library that helps automate the process of hyperparameter tuning.
    
- You can define a model with hyperparameters, configure the search, run the hyperparameter search, analyze the results, and train the optimized model.
    
- Various techniques for model optimization include weight initialization, learning rate scheduling, batch normalization, mixed precision training, model pruning, and quantization.
    
- These techniques can significantly improve the performance, efficiency, and scalability of your deep learning models.
    
- TensorFlow includes several optimization tools such as mixed precision training, model pruning, quantization, and the TensorFlow Model Optimization Toolkit.

### Glossary

| Term                                          | Definition                                                                                                                                                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Batch normalization                           | A technique that normalizes the inputs of each layer to improve the training stability and speed.                                                                                                             |
| Callback functions                            | A powerful tool to customize and extend the behavior of your model training process.                                                                                                                          |
| Custom dense layer                            | A standard fully connected neural network layer.                                                                                                                                                              |
| Custom training loops                         | Custom training loops are user-defined procedures in machine learning that give developers more control over the training process. They allow the developer to tailor the training process to specific needs. |
| Epoch                                         | Refers to one complete pass through the entire training dataset.                                                                                                                                              |
| fit method                                    | A high-level function commonly used in Keras to train a model.                                                                                                                                                |
| GradientTape                                  | A feature in TensorFlow that provides an easy way to record operations for automatic differentiation.                                                                                                         |
| hp.Float function                             | A function used for floating-point hyperparameters.                                                                                                                                                           |
| hp.Int function                               | Function used for integer hyperparameters                                                                                                                                                                     |
| Hyperparameter tuning                         | A crucial step in the machine learning pipeline that helps optimize model performance by finding the best set of hyperparameters.                                                                             |
| Hyperparameters                               | Variables that govern the training process of a model.                                                                                                                                                        |
| Keras Tuner                                   | A library that helps automate the process of hyperparameter tuning. It provides several search algorithms, including Random Search, Hyperband, and Bayesian Optimization.                                     |
| Lambda layer                                  | Used for creating custom operations that are simple enough to be defined using a lambda function or a small custom function.                                                                                  |
| Loss function                                 | Measures how well the predictions made by the model match the true labels.                                                                                                                                    |
| Mixed precision training                      | An optimization tool that leverages both 16-bit and 32-bit floating-point types to accelerate model training and reduce memory usage.                                                                         |
| MNIST database                                | A large database of handwritten digits that is commonly used for training various image processing systems.                                                                                                   |
| Model optimization                            | Refers to a set of techniques and processes that aim to improve the performance and efficiency of your neural networks.                                                                                       |
| Model pruning                                 | An optimization tool involves removing less important weights from the model, reducing the model size, and improving inference speed.                                                                         |
| Optimizer                                     | Updates the weights of the model to minimize the loss.                                                                                                                                                        |
| prune_low_magnitude                           | A function that prunes the weights based on their magnitude.                                                                                                                                                  |
| Quantization                                  | A technique that converts model weights and activations from floating-point to lower bit-width representations, such as 8-bit integers.                                                                       |
| ReLU                                          | An activation function that outputs the input directly if positive; otherwise, it outputs zero. Commonly used in hidden layers.                                                                               |
| TensorFlow Model Optimization Toolkit         | Provides easy-to-use APIs for pruning during training.                                                                                                                                                        |
| Xavier (Glorot) and He initialization methods | Methods that set the initial weights in a way that avoids issues like vanishing or exploding gradients.                                                                                                       |