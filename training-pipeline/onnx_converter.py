from fastai.vision.all import load_learner
import torch
import torchvision
import torchvision.transforms as transforms
import json

def convert_fastai_to_onnx(model_location="/train_data/birds-res34-fast.pkl"):
    learn = load_learner(model_location) #Load the fastai model
    labels = learn.dls.vocab #Load labels
    pytorch_model = _convert_fastai_to_pytorch(learn)
    _export_pytorch_as_onnx(pytorch_model)
    _export_labels_as_json(labels)
    
def _convert_fastai_to_pytorch(learn):
    pytorch_model = learn.model.eval() # gets the PyTorch model
    softmax_layer = torch.nn.Softmax(dim=1) # define softmax
    normalization_layer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization layer

    # assembling the final model
    final_model = torch.nn.Sequential(
        normalization_layer,
        pytorch_model,
        softmax_layer
    )
    return final_model

def _export_pytorch_as_onnx(pytorch_model):
    torch.onnx.export(
        pytorch_model, 
        torch.randn(1, 3, 260, 260),
        "/train_data/model.onnx",
        do_constant_folding=True,
        export_params=True, # if set to False exports untrained model
        input_names=["pixel_values"],
        output_names=["outputs"],
        opset_version=11
    )
    
def _export_labels_as_json(labels):
    json_labels = {"id2label":{i:label for i,label in enumerate(labels)}}
    with open('/train_data/labels.json', 'w') as f:
        json.dump(json_labels, f)
    
if __name__ == '__main__':
    convert_fastai_to_onnx()