# Introduction to Neural Networks

This repository contains the code and Colab notebooks for the Epoch "Introduction to Neural Networks" intersession, led by Ethan Haque and Dev Singh. 

While these notebooks may run on environments outside of Colab, they do use some specific Colab libraries. Therefore, it is reccomended to use Colab to run these notebooks. Any notebooks that have architectures which greatly benefit from GPU acceleration have GPU acceleration enabled in the notebooks.

**Note about Calculus**

These notebooks assume that someone has *no Calculus experience*. While this means that we explain some things like backprop, gradient descent, etc. in easier-to-understand ways, it also means that we can't go as deep into ML math (like "Why is this activation function not a good idea for this purpose" or "What is the vanishing gradient problem?"). The curriculum is written assuming knowledge of MI4, so we can't explain those. For those who have a working knowledge of calculus, i.e. have taken at least BC1, preferably BC2 as well, and some intuition of how MVC might work, you may benefit from additional instruction in some ML/CS theory. The [ML Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/) is a good resource for those interested.

# Schedule

## Day 1 - Introduction

* [What is a Neural Network?](day-1/what_is_a_neural_network.ipynb)
  * NN vs. ML vs. AI vs. Deep Learning
  * Goes over some of the mathematical concepts behind neural networks
  * Explain in depth how linear/dense neurons work
  * Implement an MNIST-classifying NN using just numpy

## Day 2 - Implementing real-world neural networks

* Introduce [PyTorch](https://pytorch.org/)
    * Reduces a lot of the overhead of writing ML from "scratch"
    * Allows you to focus on ML concepts instead of the code itself
* [Implement MNIST classifier MLP in PyTorch](day-2/mlp-mnist-classifier.ipynb)
  * Lot faster and more convenient than writing in numpy
* [Introduce Convolutional Neural Networks (CNNs)](day-2/what-is-cnn.ipynb)
* [MNIST Classifier using CNNs](day-2/cnn-mnist-classifier.ipynb)

## Day 3 - A deeper dive

* Discuss various architectures
  * Discuss Bayes' theorem as a precursor to so much of ML
  * [Variational Autoencoders](day-3/archs/variational-ae.ipynb) for text classification
  * [GANs](day-3/archs/gan.ipynb) for creation of novel content

* The [ethics of AI](day-3/ethics-in-ai.ipynb)
  * Privacy & Surveillance
  * Manipulation of Behaviour
  * "Black Box" AI
  * Bias
  * Singularity
  
* [Machine learning Frameworks](day-3/machine-learning-frameworks.ipynb)

  
