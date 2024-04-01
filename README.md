# AIs in Python by me
Hi, this project of mine is to visualize the different types of AIs that you can use in Python and how to implement them in your projects. Here I am giving examples for all sorts of AI structures from [GAs](#The-Genetic-Algorithm-(GA)) to [NEAT algorithms](#Neuroevolution-of-augmenting-topologies-(NEAT)). You can find information about the AI structure in this doc and on the provided Wikipedia link.
## [The Genetic Algorithm (GA)](https://en.wikipedia.org/wiki/Genetic_algorithm)
Genetic Algorithms (GAs) are a class of optimization algorithms inspired by the process of natural selection and genetics. In the context of a list of instructions that gets changed over time, GAs work by evolving a population of candidate solutions (represented as individuals or genomes) over successive generations. Each individual in the population represents a potential solution to the problem at hand.

### The Operation

1. **Initialization:** The GA starts with an initial population of individuals, often randomly generated or based on some heuristic.

2. **Evaluation:** Each individual in the population is evaluated against a fitness function that measures its performance or suitability for the task. The fitness function quantifies how well an individual solves the problem.

3. **Selection:** Individuals are selected from the current population based on their fitness scores. The probability of selection is usually proportional to an individual's fitness, with fitter individuals having a higher chance of being selected.

4. **Crossover:** Selected individuals undergo crossover or recombination, where pairs of individuals exchange genetic information to produce offspring. This process mimics genetic recombination in biological organisms. This step is optional, in my exaple I just changed the Instruction list of the best performing gemone.

5. **Mutation:** Random genetic changes (mutations) may occur in the offspring, introducing variation into the population. Mutation helps explore new areas of the solution space and can prevent premature convergence to suboptimal solutions.

6. **Replacement:** The offspring, along with some individuals from the previous generation, form the next generation population. The population size remains constant throughout the evolutionary process.

7. **Termination:** The GA continues to iterate through the selection, crossover, mutation, and replacement steps until a termination criterion is met. This criterion could be a maximum number of generations, reaching a satisfactory solution, or running out of computational resources.

### Usecases

Genetic Algorithms are widely used in low complex game automation or similar. The biggest constraint of the GA is that it completly stops working when any type of random inputs are happening because it is blindly following a list of instructions.

### Examples and Implementations

I have prepared a [folder](https://www.github.com/strniko/python-ai/tree/main/GA/) containing four Python scripts. These scripts are training a GA to find a string that you provide. It can contain letters from a to z, A to Z and spaces. I am using a type of Levenshtein distance without adding and removing letters as my fitness function. The rate of randomness that I use is 

$$ \text{randomness} = \max\left(\left(0.9^{0.625 \times \text{generation}}\right), 0.1\right) $$

1. **rebirth.py:** This file contains an implementation of a Genetic Algorithm (GA) known as "rebirth" *not an official name*. In this variant of the GA, if a generation produces a worse AI compared to the previous generation, the algorithm revives the better AI from the previous generation, hence the name "rebirth". This mechanism allows for the retention of successful solutions across generations, potentially preventing the loss of beneficial traits. This has shown 187.5% as effective as the "nature" algorithm in a test performed by me where both AIs were run 150 times with the phrase "Never gonna give you up".

2. **nature.py:** This file contains another implementation of a Genetic Algorithm (GA) known as "nature". In contrast to "rebirth", the "nature" GA follows a more traditional approach mimicing the evolutional selection found in nature, where each generation is independent, and only the best-performing AI from the current generation is retained for the next generation.

3. **rebirth speed.py:** This file is a speed-optimized version of the "rebirth" Genetic Algorithm. It works by drasticly reducing the output and using threading and multiprocessing. if you want to understand how the GA is working I suggest using the non speed optimised version and verry little max generations and a small children count.

4. **nature speed.py:** Similarly, this file is a speed-optimized version of the "nature" Genetic Algorithm.

## [Neuroevolution of augmenting topologies (NEAT)](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)
PLACEHOLDER
