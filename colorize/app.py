import io
from typing import List, ByteString

import numpy as np
import requests
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import streamlit as st

def maybe_convert_to_grayscale(image: Image) -> np.ndarray:
    """
    Converts color images to grayscale using PIL.
    If image is already single-channel, does nothing.
    """
    pixel = image.getpixel((1,1))
    if isinstance(pixel, (int, float)):
        num_channels = 1
    else:
        num_channels = len(pixel)

    if num_channels == 1:
        return image

    return ImageOps.grayscale(ImageOps.grayscale(image))


# @st.cache
def load_image_from_url(url: str) -> Image:
    content = requests.get(url).content
    bytes_data = io.BytesIO(content)
    return bytes_to_image(bytes_data)


def load_image_from_upload():
    pass


def colorize_with_mpl(image: Image, colormap: str = 'hot') -> np.ndarray:
    data = np.array(list(image.getdata())).reshape(image.size[::-1])
    fig = save_image(data, colormap, 'test.png')
    print('saved')
    return fig


def save_image(data, cm, filename=None):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(data, cmap=cm)
    if filename:
        fig.savefig(filename, dpi=height) 
    plt.close()
    return fig

def colorize_with_nn():
    pass


def colorize_with_pil(image: Image) -> Image:
    pass


def header():
    st.markdown('# Colorize Photos')
    st.markdown('Upload or link to a black and white photo (or color...) and have it be colorized')
    st.markdown('TODO:')
    st.markdown('- add more ways of colorizing')
    st.markdown('- add more colormap options')
    st.markdown('***')


def get_image_with_st() -> Image:
    bytes_data = None
    input_mode = st.selectbox(label='Image Input Mode', options=['url', 'file'])
    if input_mode == 'url':
        data_url = st.text_input("URL of Image (Right Click > Copy Image Location)",
                                 value='')
        if data_url:
            st.text("image preview:")
            st.image(data_url)
            bytes_data = load_image_from_url(data_url)
    else:
        uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            bytes_data = uploaded_file
            st.image(bytes_data)
    
    image = bytes_to_image(bytes_data)
    return image


def bytes_to_image(bytes_data: ByteString) -> Image:
    if not bytes_data:
        return None
    image = Image.open(bytes_data, 'r')
    image = maybe_convert_to_grayscale(image)
    return image


def colorize_with_st(image: Image):
    if not image:
        return None
    method = st.selectbox(label='Select Method', options=['mpl'])
    all_colormaps = ['viridis', 'hot']  # FIXME get full list
    if method == 'mpl':
        colormap = st.selectbox(label='Select Color Map', options=all_colormaps)
        fig = colorize_with_mpl(image, colormap)
    else:
        fig = None
    
    st.pyplot(fig)


if __name__ == '__main__':
    # get url from user in cmd line
    import sys
    nargs = len(sys.argv)
    print("Testing syntax: `python app.py URL`")
    if nargs == 1:
        streamlit = True
    else:
        DATA_URL = sys.argv[1]
        streamlit = False

    if streamlit:
        header()
        image = get_image_with_st()
        colorize_with_st(image)
    else:
        image = load_image_from_url(DATA_URL)
        colorize_with_mpl(image, colormap='viridis')
