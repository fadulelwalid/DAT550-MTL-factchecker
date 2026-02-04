# Multi-Task Learner for Fact-Checking

Objective of this repo is to leverage MTL methodologies, as it involves concurrently training a model on multiple tasks. This enables it to aquire shared representations that offer utility across all designated tasks. Specifically, the aim was to create and train a model to recognize whether a claim warrants verification and to determine the accuracy of a given rumor or text, categorizing it as supported (true), refuted (false), or unverifiable (in instances where evidence is insufficient for verification). Furthermore, the Hugging Face Transformer library serves as the foundational framewok for implementing machine learning adaptations in this context. 

## Dataset:

The dataset comes in different format, each format for each type of task. For this multitask learning, there are two specific tasks that the model should undestand and process: stance detection and claim detection. 

For the stance detection task, there are several datasets available to use. Before any preprocessing, the features in these datasets are: "id", "rumor", "label", "timeline" and "evidence". The "id" corresponds to the id of the selected data entry. "Rumor" feature is a rumor derived from a tweet, which is the part that the model should be able to understand based on the other features, such as "timeline" and "evidence", to give a prediction ("label").

For the claim detection task, there are several datasets to choose from. The following format holds for these types of datasets. The features in these datasets are as follows: "Sentence_id", "Text", "class_label". The "Sentence_id" feature is the id of a given political debate. The "Text" feature is the actual text of the given political debate. The "class_label" is the feature telling us if the given text from the political debate to be either a claim or not. 

## Dependencies:

- pip install torch transformers pandas

## prepare_data.py:

- this cleans up and creates additional data sets, it needs to be run before all the other programs

## machine_learning_multitask.py:

- the program has five arguments: train, save, load, accuracy, interactive

  - train: train the model on the training sets
  - save: save the model to model.pt
  - load: load a previously saved model
  - accuracy: measure the models accuracy against the test sets, on each individual task
  - interactive: asks repeatedly for a prompt, which it will detect if the prompt is true or false, or neither
- Arguments train or load needs to be used in order for save, accuracy and interactive to function.
- The order of the arguments does not matter.
- examples:

  - python machine_learning_multitask.py train save accuracy interactive
  - python machine_learning_multitask.py load interactive
  - python machine_learning_multitask.py train save

## machine_learning_singletask.py:

- will measure the accuracies of the task as single task models, instead of multitask models.
- takes no arguments.
