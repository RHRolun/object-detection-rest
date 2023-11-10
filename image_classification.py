import numpy as np
from requests import post
import json
from huggingface_hub import hf_hub_download

REPO_ID = "dennisjooo/Birds-Classifier-EfficientNetB2"
CONFIG_FILENAME = "config.json"

def _get_classes():
    config_location = hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILENAME)
    f = open(config_location)
    config = json.load(f)
    return config


def postprocess(raw_result):
    classes = _get_classes()
    scores, detected_classes = [], []
    for sample in raw_result:
        ind = np.argpartition(sample, -min(sample.shape[0], 4))[-4:]
        ind = ind[np.argsort(-sample[ind])]
        ind = ind[sample[ind]>0]
        
        detected_classes.append([classes["id2label"][str(i)] for i in ind])
        scores.append(sample[ind]/(sample[ind].sum()))
    
    return scores, detected_classes