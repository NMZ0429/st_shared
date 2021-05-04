import json
import urllib, urllib.request

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from resnet_anime import resnet50

@st.cache
def load_resnet():
    # use resnet50
    model = resnet50()
    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()

    return model

@st.cache
def preprocess(img):
    preprocess = transforms.Compose([
        transforms.Resize(360),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
    ])
    batch = preprocess(img).unsqueeze(0)

    if torch.cuda.is_available():
        batch = batch.to('cuda')

    return batch

def predict_probs(batch, model):
    with torch.no_grad():
        output = model(batch)
        probs = torch.sigmoid(output[0])

    return probs

@st.cache
def get_label():
    with urllib.request.urlopen("https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json") as url:
        global class_names
        class_names = json.loads(url.read().decode())

    return class_names

class_names = get_label()

def calc_result(probs, thresh=0.2):
    tmp = probs[probs > thresh]
    inds = probs.argsort(descending=True)
    txt = ['Predictions with probabilities above ' + str(thresh) + ':\n']
    for i in inds[0:len(tmp)]:
        txt +=[ class_names[i] + ': {:.4f} \n'.format(probs[i].cpu().numpy())]

    return txt

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

model = load_resnet()

st.title("教師無し対照学習でオタクっぽいCNNを作ってみた")

st.write("Image understanding with respect to a specific field (Anime) using Inception trained by SimCLR")

imgfile = st.file_uploader("Upload Image: (must be at least 360^2)", type=["png", "jpg"], accept_multiple_files=False)

if imgfile:
    image = Image.open(imgfile)
    st.image(
        image, caption='upload images',
        use_column_width=True
    )
    x = preprocess(image)
    out = predict_probs(batch=x, model=model)
    result = calc_result(out)

    st.write(result)
