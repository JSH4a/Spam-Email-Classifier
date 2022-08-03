# Spam Email Classifier

## Description
A simple Naieve Bayes classifier for identifying spam and non-spam (ham) emails.

#### Overview of project contents
The myclassifer.py file contains my solution and the spam classifier.py implements it using the testing and training files.

#### Features
- Classifier that can be instantiated and trained to linearly seperate between two classes with an approximate 88.6% accuracy (according to tests on hidden assement data)

- Used log of probabilities to maintain precision for small probabilities, and to increase performance by allowing for multiplication to be transformed into addition

- Used laplace smoothing to handle the problem of zero data producing zero probabilities

#### Background
This project was developed as a solution for a CM10310 Artificial Intelligence coursework, and has been my first attempt at using Naieve Bayes.

One limitation of this was that I was unaware of the hidden test data that would be used and so did not focus on optimising the data set, which may have improved performance.

The feedback on this coursework estimated the accuracy with extra hidden data to be around 88.6%.

## Installation
Python 3 is all that is required to run the code and there are no extrenal libraries used accept numpy which is part of the standard library.

## Usage
Given the accuracy of the classifier and simplistic nature of the test data provided, it is unlikely this project will prove useful as an actual spam email classifer.

However, the purpose of this project was to develop a Naieve Bayes Classifier from scratch with no external libraries and as such it may be helpful for understanding a simple implementation of Naieve Bayes.

For sufficiently simple problems for linear sepeartion, the code may be adapted to be helpful.

## Roadmap

- Explore parallelization of calculations

	This is a common improvement to Naieve Bayes to improve performance and may be an interesting challenge. 
## Authors and acknowledgment
This solution was developed for a coursework from the University of Bath. 

They developed the spam classifier.py and training files, as well as a small section of skeleton code.

