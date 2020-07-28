# VideoQA using Deep Learning and Logic

A dataset and implementation of a 'hybrid' VideoQA model which extracts properties from objects using deep learning and reasons about the objects using Answer Set Programming.

## Requirements 

Most packages can be installed using pip by running `pip install -r requirements.txt`. However, the project also requires that Clingo is installed. It can be installed using Anaconda by running:
* `conda install -c potassco clingo`

It is therefore recommended that Anaconda is used as a package manager and all other requirements are installed with pip.

For some baseline models, installing spacy en-core-web-md is also required. This can be done using `python -m spacy download en_core_web_md`.

The project uses Clingo version 5.3.0.

## Running

Interaction with the project should be through python scripts, which are stored in their functional locations.

### Scripts

Run scripts by running `python3 -m hvqa.<name>`. The `-m` is required to add the current directory to the Python path variable. Note that `<name>` must be given without `.py` at the end.

The following scripts are contained in the project:
* '**train**' trains one of the H-PERL models
* '**evaluate**' evaluates the performance of a trained H-PERL model

There are also separate training and evaluation scripts for the object detector in the `hvqa.detection` folder.

The best way to understand the inputs to all of the above scripts is to look at the required arguments inside the scripts themselves.

## Models and Data

Full details of the project can be found in the final report of a sister repo called `hvqa-report`.

### Models

The `saved-models` folder contains pre-trained two H-PERL models: hardcoded and ind-trained models, as well as the pre-trained object detector used by these models.

### Data

The `compressed-data` folder contains the full OceanQA dataset for the project. It contains three subdatasets:
* `train` contains 1000 training videos.
* `test` contains 200 testing videos.
* `val` contains 200 validation videos (used for hyper-parameter searches).

### Results

In the results folder a number of png images showing printouts of each model's results can be found. The `standard.png` files are the evaluation results when the trained object detector is used. The `fail_<p>.png` files show the results when hardcoded object detection data is used with a `<p>` probability of missing each object.
