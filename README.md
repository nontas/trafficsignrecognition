#Traffic Sign Detection and Recognition

Traffic sign detection and Recognition using a Filter-bank of Correlation Filters.

## Installation
This project depends on the [Menpo Project](http://www.menpo.org/), which is multi-platform (Windows, Lnux, OS X). As explained in [Menpo's installation isntructions](http://www.menpo.org/installation/), it is highly recommended to use [conda](http://conda.pydata.org/miniconda.html) as your Python distribution.

Once downloading and installing [conda](http://conda.pydata.org/miniconda.html), this project can be isntalled by:

**Step 1:** Create a new conda environment and activate it:
```console
conda create -n road_signs python=3.5
source activate road_signs
```

**Step 2:** Install [menpo](http://www.menpo.org/menpo/), [menpofit](http://www.menpo.org/menpofit/) and [menpowidgets](http://www.menpo.org/menpowidgets/) from the menpo channel: 
```console
conda install -c menpo menpofit menpowidgets
```

**Step 3:** Clone and install the `trafficsignrecognition` project
```console
cd ~/Documents
git clone git@github.com:nontas/trafficsignrecognition.git
pip install -e trafficsignrecognition/
```


## Get Started
Please see the [**Notebooks**](https://github.com/nontas/trafficsignrecognition/tree/master/trafficsignrecognition) for examples on how to train and use the model.
