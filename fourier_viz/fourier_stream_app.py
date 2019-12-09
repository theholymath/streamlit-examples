import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from Equation import Expression

import matplotlib.pyplot as plt
sns.set(style="whitegrid")

st.title('Fourier Series Vizualization')

sentence = st.sidebar.text_input('Input your 1D equation here (x as variable):')

if sentence:
    fn = Expression(sentence, ["x"])
    t = np.arange(256)
    sp = np.fft.fft(fn(t))
    freq = np.fft.fftfreq(t.shape[-1])
    plt.plot(freq, sp.real, freq, sp.imag)
    plt.show()
    st.pyplot()