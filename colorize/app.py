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
    num_channels = len(pixel)
    if num_channels == 1:
        return image
    return ImageOps.grayscale(ImageOps.grayscale(image))


# @st.cache
def load_image_from_url(url: str) -> ByteString:
    content = requests.get(url).content
    return io.BytesIO(content)


def load_image_from_upload():
    pass


def colorize_with_mpl(image: Image, colormap: str = 'hot') -> np.ndarray:
    data = np.array(list(image.getdata())).reshape(image.size[::-1])
    # plt.savefig('test.png')
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=colormap)
    ax.axis('off')
    st.pyplot(fig)
    return data

def colorize_with_nn():
    pass

def colorize_with_pil(image: Image) -> Image:
    pass

def main(bytes_data: ByteString, colormap='viridis'):
    image = Image.open(bytes_data, 'r')
    image = maybe_convert_to_grayscale(image)
    data = colorize_with_mpl(image, colormap)
    return data

if __name__ == '__main__':
    # get url from user in cmd line
    import sys
    if len(sys.argv) > 1:
        DATA_URL = sys.argv[1]
    else:
        DATA_URL = 'https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/macbook-air-gold-select-201810?wid=1078&hei=624&fmt=jpeg&qlt=80&.v=1603332211000'

    ##### START OF APP #####
    st.write('# Colorize Photos')
    st.write('***')
    all_colormaps = ['viridis', 'hot']  # FIXME get full list
    bytes_data = None

    input_mode = st.selectbox(label='url', options=['url', 'file'])
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

    if bytes_data:
        colormap = st.selectbox(label='Select Color Map', options=all_colormaps)
        main(bytes_data, colormap) 
    ######### END ##########


