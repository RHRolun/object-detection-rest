from fastai.vision.all import ImageDataLoaders, vision_learner, Interpretation, resnet34, Resize, error_rate

def train_model(data="./train_data/bird_data"):
    dls = ImageDataLoaders.from_folder(data, valid_pct=0.2, item_tfms = Resize(260))
    learn = vision_learner(dls, resnet34, pretrained=True, metrics=error_rate)
    learn.fine_tune(1)
    learn.export('../birds-res34-fast.pkl')
    
if __name__ == '__main__':
    train_model("/train_data/bird_data")