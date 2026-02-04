Dependencies:

- pip install torch transformers pandas

prepare_data.py:

- this cleans up and creates additional data sets, it needs to be run before all the other programs

machine_learning_multitask.py:

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

machine_learning_singletask.py:

- will measure the accuracies of the task as single task models, instead of multitask models.
- takes no arguments.
