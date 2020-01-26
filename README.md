# VideoQA using Deep Learning and Logic

A dataset and implementation (ongoing) of a 'hybrid' VideoQA model which extracts properties from objects using deep learning and reasons about the objects using Answer Set Programming.

## Running

### Scripts

Run scripts in hvqa/scripts/ by running `python3 -m hvqa.scripts.<name>`. The `-m` is required to add the current directory to the Python path variable. Note that `<name>` must be given without `.py` at the end.

The scripts are used as follows:
* _build_ Builds the dataset (there are options for building just the json files, just the frames, or everything)
* _analyse_ Runs analysis on the current dataset (see the file for full list of flags)
