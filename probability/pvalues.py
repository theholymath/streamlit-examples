from scipy.stats import gaussian_kde as gkde
import scipy.stats.distributions as dist

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import streamlit as st


def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    
    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))
    
    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)
    
    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]
    
    return perm_sample_1, perm_sample_2


def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""
    
    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)
    
    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)
        
        # Compute the test statistics
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
        
    return perm_replicates


def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    
    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) -  np.mean(data_2)
    
    return diff


st.header("P-Values Demo")
st.write("We are testing the hypothesis that two distributions share the same mean.")

st.subheader("Define Class A")
st.write("Parameters for a Beta distribution $\\Beta (\\alpha, \\beta, 0,1)$.")
alpha_a = st.slider("Alpha A", min_value=0.01, max_value=10.0, step=0.01, value=1.0)
beta_a = st.slider("Beta A", min_value=0.01, max_value=10.0, step=0.01, value=1.0)

st.subheader("Define Class B")
st.write("Parameters for a Beta distribution $\\Beta (\\alpha, \\beta, 0,1)$.")
alpha_b = st.slider("Alpha B", min_value=0.01, max_value=10.0, step=0.01, value=1.0)
beta_b = st.slider("Beta B", min_value=0.01, max_value=10.0, step=0.01, value=1.0)

dist_a = dist.beta(alpha_a, beta_a)
dist_b = dist.beta(alpha_b, beta_b)

fig, ax = plt.subplots()
x = np.linspace(0.01, 0.99, 100)
pdf_a = dist_a.pdf(x)
pdf_b = dist_b.pdf(x)


num_a = st.sidebar.number_input("Num A", min_value=1, max_value=10000, value=100)
num_b = st.sidebar.number_input("Num B", min_value=1, max_value=10000, value=100)

sample_a = dist_a.rvs(num_a)
sample_b = dist_b.rvs(num_b)

# Compute difference of mean from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(sample_a, sample_b)

st.sidebar.subheader("Bootstrapping")
perm_reps = st.sidebar.number_input("Permutation Samples", min_value=10, value=100)

run = st.button("Run")

# Draw a number of permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(sample_a, sample_b, diff_of_means, size=perm_reps)

# Compute p-value: p
pval = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# ax.hist(perm_replicates, bins=100, density=True)
ax.plot(x, pdf_a, label='A')
ax.plot(x, pdf_b, label='B')
ax.legend()
if pval < 0.05:
    color = 'r'
else:
    color = 'k'
ax.set_title(f"p-value: {pval:1.2e}", fontsize=24, c=color)
ax.set_ylim([0, 1.05*np.max(np.concatenate([pdf_a, pdf_b]))])
st.pyplot(fig)



kde = gkde(perm_replicates)
x0 = np.linspace(min(perm_replicates), max(max(perm_replicates), pval), 100)
p_y = kde.pdf(x0)

fig_p, ax_p = plt.subplots()
ax_p.plot(x0, p_y, c='k', lw=2)
ax_p.axvline(pval, c='r', lw=2, ls='--')
section = np.linspace(pval, max(x0))
ax_p.fill_between(section, kde.pdf(section), color='r')
ax_p.set_title('Perumuted Samples: Density Estimate')
st.pyplot(fig_p)
