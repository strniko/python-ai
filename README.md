# AIs in Python

Hi, this project of mine is to visualize the different types of AIs that you can use in Python and how to implement them in your projects. Here, I am providing examples for all sorts of AI structures from [GAs](#The-Genetic-Algorithm-(GA)) to [NEAT algorithms](#Neuroevolution-of-augmenting-topologies-(NEAT)). You can find information about the AI structure in this document and on the provided Wikipedia link.

AIs I will be taking about:
- [Genetic Algorithm (GA)](#genetic-algorithm-ga)
- [Feedforward Neural Network (FNN)](#feedforward-neural-network-fnn)
- [Neuroevolution of Augmenting Topologies (NEAT)](#neuroevolution-of-augmenting-topologies-neat)

## [The Genetic Algorithm (GA)](https://en.wikipedia.org/wiki/Genetic_algorithm)

Genetic Algorithms (GAs) are a class of optimization algorithms inspired by the process of natural selection and genetics. In the context of a list of instructions that gets changed over time, GAs work by evolving a population of candidate solutions (represented as individuals or genomes) over successive generations. Each individual in the population represents a potential solution to the problem at hand.

### The Operation

1. **Initialization:** The GA starts with an initial population of individuals, often randomly generated or based on some heuristic.

2. **Evaluation:** Each individual in the population is evaluated against a fitness function that measures its performance or suitability for the task. The fitness function quantifies how well an individual solves the problem.

3. **Selection:** Individuals are selected from the current population based on their fitness scores. The probability of selection is usually proportional to an individual's fitness, with fitter individuals having a higher chance of being selected.

4. **Crossover:** Selected individuals undergo crossover or recombination, where pairs of individuals exchange genetic information to produce offspring. This process mimics genetic recombination in biological organisms. This step is optional; in my example, I just changed the Instruction list of the best-performing genome.

5. **Mutation:** Random genetic changes (mutations) may occur in the offspring, introducing variation into the population. Mutation helps explore new areas of the solution space and can prevent premature convergence to suboptimal solutions.

6. **Replacement:** The offspring, along with some individuals from the previous generation, form the next generation population. The population size remains constant throughout the evolutionary process.

7. **Termination:** The GA continues to iterate through the selection, crossover, mutation, and replacement steps until a termination criterion is met. This criterion could be a maximum number of generations, reaching a satisfactory solution, or running out of computational resources.

### Use Cases

Genetic Algorithms are widely used in low-complexity game automation or similar tasks. The biggest constraint of the GA is that it completely stops working when any type of random inputs are happening because it is blindly following a list of instructions.

### Examples and Implementations

I have prepared a [folder](https://www.github.com/strniko/python-ai/tree/main/GA/) containing four Python scripts. These scripts are training a GA to find a string that you provide. It can contain letters from a to z, A to Z, and spaces. I am using a type of Levenshtein distance without adding and removing letters as my fitness function. The rate of randomness that I use is 

$$ \text{randomness} = \max\left(\left(0.9^{0.625 \times \text{generation}}\right), 0.1\right) $$

1. **rebirth.py:** This file contains an implementation of a Genetic Algorithm (GA) known as *rebirth* (*not an official name*). In this variant of the GA, if a generation produces a worse AI compared to the previous generation, the algorithm revives the better AI from the previous generation, hence the name *rebirth*. This mechanism allows for the retention of successful solutions across generations, potentially preventing the loss of beneficial traits. This has shown 101.785714% as effective as the *nature* algorithm in a test performed by me where both AIs were run 1000 times with the phrase *Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you*. The *rebirth* algorithm tends to follow the *greedy strategy*.

2. **nature.py:** This file contains another implementation of a Genetic Algorithm (GA) known as *nature*. In contrast to *rebirth*, the *nature* GA follows a more traditional approach mimicking the evolutionary selection found in nature, where each generation is independent, and only the best-performing AI from the current generation is retained for the next generation.

3. **rebirth_speed.py:** This file is a speed-optimized version of the *rebirth* Genetic Algorithm. It works by drastically reducing the output and using threading and multiprocessing. If you want to understand how the GA is working, I suggest using the non-speed-optimized version and a very small maximum number of generations and a small children count.

4. **nature_speed.py:** Similarly, this file is a speed-optimized version of the *nature* Genetic Algorithm.

## [Feedforward Neural Network (FNN)](https://en.wikipedia.org/wiki/Feedforward_neural_network).

Feedforward Neural Networks (FNNs) are a fundamental type of artificial neural network where connections between the nodes or neurons do not form any cycles. Information moves in only one direction, forward, from the input nodes through hidden nodes (if any) to the output nodes. FNNs are extensively used in various machine learning tasks due to their simplicity and effectiveness in approximating complex functions.

### Operation

1. **Initialization:** FNNs typically involve initializing the weights and biases of the network's connections randomly or using pre-trained weights.

2. **Forward Propagation:** Input data is fed forward through the network layer by layer, with each layer applying a linear transformation followed by a non-linear activation function.

3. **Training:** FNNs are trained using optimization algorithms like gradient descent and its variants. The training process involves adjusting the weights and biases iteratively to minimize a loss function, which measures the difference between the predicted outputs and the actual targets.

4. **Backpropagation:** During training, errors are propagated backward through the network using the chain rule of calculus to update the weights and biases efficiently.

### Use Cases

Feedforward Neural Networks find applications in a wide range of fields, including but not limited to:

- Image classification
- Speech recognition
- Natural language processing
- Financial forecasting
- Recommendation systems

### Examples and Implementations

I have **not** prepared a [folder](https://www.github.com/strniko/python-ai/tree/main/FNN/) containing example Python scripts showcasing the implementation of FNNs in various tasks. These scripts demonstrate how to build, train, and evaluate FNN models without using popular libraries such as TensorFlow, PyTorch, or Keras.

## [Neuroevolution of Augmenting Topologies (NEAT)](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)

PLACEHOLDER

# Other Important Aspects

In addition to the specific AI structures covered in this document, there are other important aspects to consider when working with artificial neural networks and evolutionary algorithms. Some of these include:

## Activation Functions

Activation functions play a crucial role in artificial neural networks by introducing non-linearity to the model, enabling it to learn complex patterns. Here, we'll delve deeper into some common activation functions:

### Sigmoid Activation Function

The sigmoid function, also known as the logistic function, is defined as:

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

where \(x\) is the input to the function.

It squashes the input values between 0 and 1, making it useful for binary classification problems. However, it suffers from the vanishing gradient problem and is not recommended for deep neural networks.

### ReLU Activation Function

ReLU, or Rectified Linear Unit, is defined as:

$$ f(x) = \max(0, x) $$

ReLU sets all negative values to zero and keeps positive values unchanged. It's computationally efficient and helps mitigate the vanishing gradient problem. However, it suffers from the problem of dying ReLU units where gradients become zero for negative inputs.

### Tanh Activation Function

The hyperbolic tangent function, tanh, is defined as:

$$ \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

It squashes input values between -1 and 1, making it suitable for outputs that may be negative or positive. Tanh addresses the vanishing gradient problem better than sigmoid but can still suffer from it in deep networks.

### Softmax Activation Function

Softmax function is often used in the output layer of a neural network for multi-class classification tasks. It converts raw scores into probabilities, ensuring that the sum of probabilities across classes is equal to one. Softmax is defined as:

$$ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} $$

where \(x_i\) is the raw score for class \(i\).

### Visualization

Let's visualize these activation functions:


```{r}
x <- seq(-5, 5, length.out=100)
sigmoid <- 1 / (1 + exp(-x))
relu <- pmax(0, x)
tanh <- tanh(x)

plot(x, sigmoid, type='l', col='blue', ylim=c(0,1), xlab='Input', ylab='Output', main='Activation Functions')
lines(x, relu, col='red')
lines(x, tanh, col='green')
legend('topleft', legend=c('Sigmoid', 'ReLU', 'Tanh'), col=c('blue', 'red', 'green'), lty=1)
grid()
```

## Training Techniques

Training artificial neural networks and evolutionary algorithms involves various techniques to improve convergence and generalization. Some popular techniques include:

- **Batch Normalization**: Normalizes the activations of each layer to stabilize training and improve convergence.
