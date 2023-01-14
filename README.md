# Fire Classification using Operational Neural Network (ONN)
This repository contains scripts for training and testing a fire classification model using an Operational Neural Network (ONN**). The model is able to detect fires in images with high accuracy compared to state-of-the-art methods.
<!-- "We utilized an Operational Neural Network (ONN) to successfully detect fires, which has proven to be more effective than other state-of-the-art methods. The ONN is a convolutional neural network with a reduced number of parameters, providing a balance of complexity and efficiency."  -->

** : http://selfonn.net/

# Requirements
    Matplotlib (for visualization)
    Numpy 1.23.5
    ONN github link: https://github.com/junaidmalik09/fastonn
    Pandas 1.2.4
    Python 3.6 or later
    PyTorch 1.9.1
    Scikit-learn 1.0.2
    [Dataset](http://www.nnmtl.cn/EFDNet/)
# Usage
## Training
    Download the dataset and extract it to the root directory of the project.
    Run the following command to train the model:

    python run_ONN_fire.py     
<!-- --data-dir path/to/dataset --model-dir path/to/save/model -->
The model will be saved in the specified directory after training is complete.

## Testing

    Run the following command to test the model on a single image:

## Evaluation

    Run the following command to evaluate the model on the test dataset:

This will produce an accuracy score of the model on the test dataset.

# Notes

    The training process may take several hours depending on the hardware.
    The provided scripts are for demonstration purposes and may need to be modified to suit your specific use case

# References
<!-- paper1
paper2
paper3 -->

Please cite our paper if you use this code.
Please let me know if you have any question or concern.