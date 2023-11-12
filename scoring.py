from csv import writer
from pickle import load
import json

import numpy as np
from onnxruntime import InferenceSession

from classes import classes

def predict(data_folder='./data'):
    print('Commencing offline scoring.')

    with open(f'{data_folder}/images.pickle', 'rb') as inputfile:
        image_names, images = load(inputfile)

    session = InferenceSession('model.onnx', providers=['CPUExecutionProvider'])

    raw_results = [
        session.run(
            [], {'pixel_values': image_data}
        )
        for image_data in images
    ]
    results = [
        postprocess(*raw_result) for raw_result in raw_results
    ]

    _to_csv(results, image_names, data_folder)

    print('Offline scoring complete.')
    
def _get_classes():
    f = open("config.json")
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


def _to_csv(results, image_names, data_folder):
    column_names = ['file id', 'object', 'score']

    with open(f'{data_folder}/results.csv', 'w', newline='') as outputfile:
        csv_writer = writer(outputfile, delimiter='|')
        csv_writer.writerow(column_names)
        for result_index, result in enumerate(results):
            image_name = image_names[result_index]
            scores, object_names = result
            for object_index, object_name in enumerate(object_names):
                csv_writer.writerow(
                    [image_name, object_name, scores[object_index]]
                )


if __name__ == '__main__':
    predict(data_folder='/data')
