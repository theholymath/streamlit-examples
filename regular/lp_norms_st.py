import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


def lp_ball_pts(radius, p):
    """ For plotting - plots the LP-Ball of radius, radius

    Parameters:
    -----------
    radius (float): The "radius" of the LP-ball
    p (float): L(p) space
    """
    alpha = np.linspace(0, 2*np.pi, 2000, endpoint=True)
    x = np.cos(alpha)
    y = np.sin(alpha)

    vecs = np.array([x, y])

    #norms = np.sum(np.abs(vecs)**p, axis=0)**(1/p)
    norms = norm(vecs, p)
    norm_vecs = radius*vecs/norms

    return norm_vecs

st.title('Regression & Optimization')
st.markdown('Norm chosen $||\cdot||_p, \; 1 \leq p \leq \infty$ impacts conclusions made.')
st.markdown('$\hat{y} = \mathbf{w}^T \mathbf{x} + b$, where $b$ represents a bias term.')
st.markdown('We want to minimize the difference between observations $\mathbf{y}$ and predictions $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$.')
st.subheader('$\mathbf{w}^*, b^* = \\text{argmin}_{\mathbf{w}, b} L(\mathbf{w}, b)$')
st.markdown('$L(\mathbf{w}, b) := ||\mathbf{y} - \mathbf{X}\mathbf{w}||_2 + \\alpha||\mathbf{w}||_p$, $\\alpha > 0$ is a dial we can tune.')
#st.subheader('Minimizing $||x||_p$ given $ax + by + c = 0$')
p = st.slider('p:', min_value=1.0, max_value=10.0, value=2.0, step=0.25)

st.sidebar.markdown('$Ax + By + C = 0$')
#st.sidebar.markdown('$y = w_1 x + w_2 x + b$')

form = st.sidebar.form(key='my-form')
a = st.sidebar.slider("A", min_value=-2.0, max_value=1.0, value=-1.0)
b = st.sidebar.slider("B", min_value=2.0, max_value=3.0, value=2.0)
c = st.sidebar.slider("C", min_value=-2.0, max_value=1.0, value=-0.1)

st.sidebar.markdown('$y = (-A/B) x - C/B$')
st.sidebar.markdown('$(w, b) := (-A/B, -C/B)$')

def norm(x, p):
    return np.sum(np.abs(x)**p, axis=0)**(1/p)


def func_p(p=1): 
    def func_to_minimize(x):
        return norm(x, p)
    return func_to_minimize

def eqn_of_line_optim(x):
    return a*x[0] + b*x[1] + c

def eqn_line_y_equals(x):
    return (-c - a*x)/b

if p != 1.0:
    # solve for p = 1 to determine width of plot first.
    cons = ({'type': 'eq', 'fun': eqn_of_line_optim})
    width = minimize(func_p(1), (2, 0), method='SLSQP', constraints=cons).fun
    res = minimize(func_p(p), (2, 0), method='SLSQP', constraints=cons)
else:
    cons = ({'type': 'eq', 'fun': eqn_of_line_optim})
    res = minimize(func_p(p), (2, 0), method='SLSQP', constraints=cons)
    width = res.fun

#st.sidebar.subheader("Solution")
#st.sidebar.write(f"x = {res.x[0]:1.3f}, y = {res.x[1]:1.3f}")

lp_func = lp_ball_pts(res.fun, p)

sns.set(style='ticks')

x = np.linspace(-2*width,2*width,1000)
y = eqn_line_y_equals(x)


fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(x, y)
ax.scatter(res.x[0], res.x[1], c='red', label = 'min point for Lp norm')
ax.plot(lp_func[0], lp_func[1])
ax.set_aspect('equal')
ax.grid(True, which='both')

sns.despine(ax=ax, offset=0) # the important part here
ax.set_ylim([-2*width,2*width])
ax.set_xlim([-2*width,2*width])
ax.set_ylabel('y', fontsize=24)
ax.set_xlabel('x', fontsize=24)
ax.scatter([0], [0], c='k', s=20)
ax.set_title(f'y = {-a/b}x + {-c/b}, p={p}',fontsize=24)
st.sidebar.pyplot(fig)




##########

import numpy as np
import matplotlib.pyplot as plt

# x, y
data = np.array([
    [2.4, 1.7], 
    [2.8, 1.85], 
    [3.2, 1.79], 
    [3.6, 1.95], 
    [4.0, 2.1], 
    [4.2, 2.0], 
    [5.0, 2.7]
])

num_pts = st.sidebar.slider("Num Data", min_value=1, max_value=7, value=1)
data = data[:num_pts,:]
x, y = data[:,0], data[:,1]

def f(x):
    return -a/b * x - c/b

min_x, max_x = min(x), max(x)
fig, ax = plt.subplots()
ax.scatter(x, y) # original data points
ax.plot([min_x, max_x], [f(min_x), f(max_x)], 'k-') # line of f1
ax.scatter(x, f(x), color='black') # points predicted by f1
for x_, y_ in zip(x, y):
    ax.plot([x_, x_], [y_, f(x_)], '--', c='red') # error bars
ax.set_title("error bars: $y_i-f(x_i)$")
ax.set_xlabel('x', fontsize=24)
ax.set_ylabel('y', fontsize=24)

st.sidebar.pyplot(fig)

def cost(w, x, b, y_actual, p=2, a=0):
    y_pred = w*x + b
    residual = y_actual - y_pred
    return norm(residual, p=2) + a*norm( np.array([w,b]), p)

x, y = data[:,0], data[:,1]

alpha = st.slider('alpha', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

# check all combinations of m between [-2, 4] and b between [-6, 8], to precision of 0.1
W = np.arange(-2, 4, 0.1)
B = np.arange(-6, 8, 0.1)

# get MSE at every combination
J = np.zeros((len(W), len(B)))
for i, w_ in enumerate(W):
    for j, b_ in enumerate(B):
        J[i][j] = cost(w_, x, b_, y, p, alpha)

# plot loss surface
B, W = np.meshgrid(B, W)
ax.plot_surface(B, W, J, rstride=1, cstride=1, cmap=plt.cm.viridis, linewidth=0, antialiased=False, alpha=0.75)
ax.contour(B, W, J, 20, zdir='z', offset=0, linestyles='solid', cmap=plt.cm.viridis, linewidth=1, antialiased=True)
ax.set_title("cost for different w, b")
ax.set_xlabel("b")
ax.set_ylabel("w")
ax.set_zlim([0, 50])
st.write('---')
elev = st.slider('elevation', min_value=0, max_value=90, step=5, value=30)
azim = st.slider('azimuth', min_value=0, max_value=180, step=5, value=60)
ax.view_init(elev=elev, azim=azim)
st.pyplot(fig)
