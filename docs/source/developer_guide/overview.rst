Introduction
===============

The "Developer Guide" section is for those who want to know more about 
the implementation details and develop new models. 

XGCN has four key modules: ``Model``, ``DataLoader``, ``Evaluator``, and ``Trainer``.  
An overview of their interactions is shown in the figure below:

.. image:: ../asset/overview.jpg
  :width: 600
  :alt: key modules (Model, DataLoader, Evaluator, and Trainer) and their interactions

``Trainer`` lies in the center of the control flow and is responsible for 
the whole model training process. 
``Model`` is in the center of the data flow and receives training/evaluation data. 
``DataLoader`` feeds batch training data to the ``Model``. 
``Evaluator`` sends batch evaluation data to the ``Model``, receives inference outputs, 
and calculates accuracy metrics. 
