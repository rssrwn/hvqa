# VideoQA using Deep Learning and Logic

A dataset and implementation (ongoing) of a 'hybrid' VideoQA model which extracts properties from objects using deep learning and reasons about the objects using Answer Set Programming.

## Requirements 

Most packages can be installed using pip by running `pip install -r requirements.txt`. However, the project also requires that Clingo and Clorm are installed. These can be installed using Anaconda by running:
* `conda install -c potassco clingo`
* `conda install -c potassco clorm` 

The project uses Clingo version 5.3.0.

## Running

At the moment, interaction with the project should be through python scripts, which are stored in their functional locations.

### Scripts

Run scripts by running `python3 -m hvqa.<folder>.<name>`. The `-m` is required to add the current directory to the Python path variable. Note that `<name>` must be given without `.py` at the end.

The following scripts are contained in the project:
* '**build**' builds the dataset (there are options for building just the json files, just the frames, or everything)
* '**analyse**' runs analysis on the current dataset (see the file for full list of flags)
* '**train**' trains either the classifier (backbone) model or the detector model with a pre-trained classifier
* '**evaluate**' evaluates the performance of either the classifier or detector model

