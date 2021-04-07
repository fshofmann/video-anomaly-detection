### Anomaly Detection through Video Generation Forecasting


### Contribution


#### Detection Approach
Based on the [IFTM Approach](https://ieeexplore.ieee.org/abstract/document/8456348) by the CIT/DOS research group, a new technique for detecting anomalies in video streams is designed:
- a behavioral model is trained based on a training dataset (offline)
    - using a two-stream DNN to model scene dynamics (background+foreground).
    - training is done by adversarial training, the behavioral model is the generator, while a second DNN is the discriminator.
- a static threshold is applied to the delta of the output to the actual values

How anomalies are detected depends on the result of the different approaches, because IFTM is not directly applicable to the state of the art video generation techniques, such as the video hallucination and generation found [here](http://www.cs.columbia.edu/~vondrick/tinyvideo/). There are however several ways to incorporate it into the framework:
- (C-VGAN) Video generation techniques to generate 1+K frames out of a single input frame. The predicted K frames is compared to the actual future to compute the reconstruction error (similar to IFTM). This can be done over the the N-1 frames leading to the Nth frame, resulting in N-1 plausible predictions for frame N. This could be utilized to compute an reconstruction error, with all N-1 plausible predictions (possibly differently weighted). Issue with this is, that predictions are (according to the state of the art papers) merely plausible, but there are other possible futures that might not get covered by the model. Issue with this idea is that video generation strives when using scene dynamics and two stream models. One has more information about the past than a single frame.
- (our approach) The two stream model is adjusted to accept N frames as input and generate N+K frames. Extend the information that can be accessed by the model, giving insight into more of the past and potentially allowing a greater vision into the future. The input can be further refined by untangling the input for the two stream network into a single vector for background and a matrix of vectors for the foreground. This way, one encoder for each stream encode the input frames into their respective latent spaces.


#### Dataset 
The dataset is created by the author themselves and details about it can be found in its respective folder.


#### Evaluation 

- (see thesis for more information)
- Video generation/prediction models:
    1. Losses during training are measured and evaluated based on whether the two models converge or not. This metric serves as an indicator whether training has collapsed and whether the models are indeed continuously improving in parallel.
    2. For qualitative evaluation of the predictions/generations, synthetic outputs are inspected.
- Evaluating the results for anomaly detection:
    1. The prediction error over time is evaluated and compared to the actual normal/anomalous events and their respective intervals in the video stream.
    2. A confusion matrix is generated based on true labels and the predictions by the model on the evaluation dataset. Then, anomaly detection evaluation metrics are derived from it (e.g. accuracy, precision, recall, f1).


### Structure

In the `thesis` directory, one will find the thesis, tex files, figures, everything in terms of the final work. In the `src` folder, you will find the actual source code of the thesis. `io_` features the preprocessing steps and some helper scripts that are used during training and evaluation. `models` contains all of the deep learning models and notebooks to run and evaluate them. `eval` contains the same models plus an evaluation script for IFTM, but all scripts will have a few more options for parameter tuning and will generate more data (and metrics) for eval purposes.

To run any of the code found in `src` you can simply start them. Code will process data found in the `data` folder and results will be put in the `output` directory. 


### Installation

The project was mostly maintained through conda and pip, so both requirement text files are available in the root directory. Note that pip requirements should be considered redundant if one already uses conda. Furthermore, the project was developed on a nvidia 3000-series graphics card, while the tensorflow version 2.5 (the only one being compatible for the latest gpu series) was still in development. The project should work with the current latest stable version (TF 2.4 at the time), but otherwise one should go with the development version 2.5.

### Notes 

Some of the notebooks' code was originally based on the [DCGAN example by TensorFlow](https://www.tensorflow.org/tutorials/generative/dcgan).

### License

This project is licensed under the terms of the MIT license.