import pathlib
import pickle
import os
from tensorflow.keras.models import load_model

model_output_dir = pathlib.Path('saved_models')
model_output_dir.mkdir(exist_ok=True, parents=True)

"""
:param modelpath:   string, path to the model
:return             bool, true if the file exist and false if it don't

Checks if there is a file in at the end of the modelpath
"""
def model_exist(modelpath):
    return os.path.isfile(modelpath)


"""
:param modelname:   string, name of the model
:param model:       sklearn class model

Saves the model with given modelname in the directory 'saved models'.
"""
def save_sklearn_model(modelname, model):
    if modelname.endswith('.sav'):
        modelpath = os.path.join(model_output_dir, modelname)
        os.makedirs(model_output_dir, exist_ok=True)
        print("Saving model to:", modelpath)
        pickle.dump(model, open(modelname, 'wb'))
    else:
        print(f"File extension unknown: {modelname.split('.')[-1]} \t-->\t should be .sav")


"""
:param modelname:   string, name of the model
:return             sklearn class model

Returns the model saved with 'modelname', if the model does not exist it will exit the program with exit with exit 
code 1.
"""
def load_sklearn_model(modelname):
    modelpath = os.path.join(model_output_dir, modelname)
    if model_exist(modelpath):
        return pickle.load(open(modelpath, 'rb'))
    else:
        print(f"Could not find the model: {modelpath}")
        exit(1)


def save_tf_model(modelname, model):
    modelpath = os.path.join(model_output_dir, modelname)
    os.makedirs(model_output_dir, exist_ok=True)
    print("Saving model to:", modelpath)
    model.save(modelpath)


def load_tf_model(modelname):
    modelpath = os.path.join(model_output_dir, modelname)
    if os.path.isdir(modelpath):
        return load_model(modelpath)
    else:
        print(f"Could not find the directory: {modelpath}")
        exit(1)
