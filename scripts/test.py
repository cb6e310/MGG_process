import numpy as np

from scipy import signal

import numpy as np

from scipy.signal import butter, lfilter, freqz

import matplotlib.pyplot as plt


from filters import butter_filter
import padasip as pa


def test_1():
    np.random.seed(0)

    n_samples = 2000

    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal

    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal

    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]

    S += 0.2 * np.random.normal(size=S.shape)  # Add noise

    S /= S.std(axis=0)  # Standardize data

    # Mix data

    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix

    X = np.dot(S, A.T)  # Generate observations

    print(X.shape)


def test_2():
    def butter_lowpass(cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype="low", analog=False)

    # def butter_lowpass_filter(data, cutoff, fs, order=5):

    #     b, a = butter_lowpass(cutoff, fs, order=order)

    #     y = lfilter(b, a, data)

    #     return y

    # Filter requirements.

    order = 6

    fs = 30.0  # sample rate, Hz

    cutoff = 3.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.

    b, a = butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.

    w, h = freqz(b, a, fs=fs, worN=8000)

    plt.subplot(2, 1, 1)

    plt.plot(w, np.abs(h), "b")

    plt.plot(cutoff, 0.5 * np.sqrt(2), "ko")

    plt.axvline(cutoff, color="k")

    plt.xlim(0, 0.5 * fs)

    plt.title("Lowpass Filter Frequency Response")

    plt.xlabel("Frequency [Hz]")

    plt.grid()

    # Demonstrate the use of the filter.

    # First make some data to be filtered.

    T = 5.0  # seconds

    n = int(T * fs)  # total number of samples

    t = np.linspace(0, T, n, endpoint=False)

    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.

    data = (
        np.sin(1.2 * 2 * np.pi * t)
        + 1.5 * np.cos(9 * 2 * np.pi * t)
        + 0.5 * np.sin(12.0 * 2 * np.pi * t)
    )

    # Filter the data, and plot both the original and filtered signals.

    y = butter_filter(data, cutoff, fs, order)

    plt.subplot(2, 1, 2)

    plt.plot(t, data, "b-", label="data")

    plt.plot(t, y, "g-", linewidth=2, label="filtered data")

    plt.xlabel("Time [sec]")

    plt.grid()

    plt.legend()

    plt.subplots_adjust(hspace=0.35)

    plt.show()


def test_3():
    # %% Simple example: generate signal with 3 components + noise
    import numpy as np
    import matplotlib.pyplot as plt
    from vmdpy import VMD

    # . Time Domain 0 to T
    T = 1000
    fs = 1 / T
    t = np.arange(1, T + 1) / T
    freqs = 2 * np.pi * (t - 0.5 - fs) / (fs)

    # . center frequencies of components
    f_1 = 2
    f_2 = 24
    f_3 = 288

    # . modes
    v_1 = np.cos(2 * np.pi * f_1 * t)
    v_2 = 1 / 4 * (np.cos(2 * np.pi * f_2 * t))
    v_3 = 1 / 16 * (np.cos(2 * np.pi * f_3 * t))

    f = v_1 + v_2 + v_3 + 0.1 * np.random.randn(v_1.size)

    # . some sample parameters for VMD
    alpha = 2000  # moderate bandwidth constraint
    tau = 0.0  # noise-tolerance (no strict fidelity enforcement)
    K = 5  # 3 modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7

    # . Run VMD
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

    # . Visualize decomposed modes
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(f)
    plt.title("Original signal")
    plt.xlabel("time (s)")
    plt.subplot(2, 1, 2)
    plt.plot(u.T)
    plt.title("Decomposed modes")
    plt.xlabel("time (s)")
    plt.legend(["Mode %d" % m_i for m_i in range(u.shape[0])])
    plt.tight_layout()

    plt.show()


def test_4():
    import numpy as np
    import matplotlib.pylab as plt
    import padasip as pa

    # creation of data
    N = 500
    x = np.random.normal(0, 1, (N, 4))  # input matrix
    v = np.random.normal(0, 0.1, N)  # noise
    d = 2 * x[:, 0] + 0.1 * x[:, 1] - 4 * x[:, 2] + 0.5 * x[:, 3] + v  # target

    # identification
    f = pa.filters.FilterLMS(n=4, mu=0.1, w="random")
    y, e, w = f.run(d, x)

    # show results
    plt.figure(figsize=(15, 9))
    plt.subplot(211)
    plt.title("Adaptation")
    plt.xlabel("samples - k")
    plt.plot(d, "b", label="d - target")
    plt.plot(y, "g", label="y - output")
    plt.legend()
    plt.subplot(212)
    plt.title("Filter error")
    plt.xlabel("samples - k")
    plt.plot(10 * np.log10(e**2), "r", label="e - error [dB]")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_5():
    import numpy as np
    from scipy import signal

    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise

    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations

    # %%
    # Fit ICA and PCA models
    # ----------------------

    from sklearn.decomposition import PCA, FastICA

    # Compute ICA
    ica = FastICA(n_components=8, whiten="arbitrary-variance")
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix

    # We can `prove` that the ICA model applies by reverting the unmixing.
    # assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

    # For comparison, compute PCA
    pca = PCA(n_components=3)
    H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

    # %%
    # Plot results
    # ------------

    import matplotlib.pyplot as plt

    plt.figure()

    models = [X, S, S_, H]
    names = [
        "Observations (mixed signal)",
        "True Sources",
        "ICA recovered signals",
        "PCA recovered signals",
    ]
    colors = ["red", "steelblue", "orange"]

    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.tight_layout()
    plt.show()


def test_6():
    # signals creation: u, v, d
    N = 5000
    n = 10
    u = np.sin(np.arange(0, N / 10.0, N / 50000.0))
    v = np.random.normal(0, 1, N)
    d = u + v

    # filtering
    x = pa.input_from_history(d, n)[:-1]
    d = d[n:]
    u = u[n:]
    f = pa.filters.FilterRLS(mu=0.9, n=n)
    y, e, w = f.run(d, x)

    # error estimation
    MSE_d = np.dot(u - d, u - d) / float(len(u))
    MSE_y = np.dot(u - y, u - y) / float(len(u))

    # results
    plt.figure(figsize=(12.5, 6))
    plt.plot(u, "r:", linewidth=4, label="original")
    plt.plot(d, "b", label="noisy, MSE: {}".format(MSE_d))
    plt.plot(y, "g", label="filtered, MSE: {}".format(MSE_y))
    plt.xlim(N - 100, N)
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_7():
    N = 5000
    n = 10
    u = np.sin(np.arange(0, N / 10.0, N / 50000.0))
    v = np.random.normal(0, 1, N)
    d = u + v

    # filtering
    x = pa.input_from_history(d, n)[:-1]
    d = d[n:]
    u = u[n:]
    f = pa.filters.FilterRLS(mu=0.9, n=n)
    y, e, w = f.run(d, x)

    # error estimation
    MSE_d = np.dot(u - d, u - d) / float(len(u))
    MSE_y = np.dot(u - y, u - y) / float(len(u))

    # results
    plt.figure(figsize=(12.5, 6))
    plt.plot(u, "r:", linewidth=4, label="original")
    plt.plot(d, "b", label="noisy, MSE: {}".format(MSE_d))
    plt.plot(y, "g", label="filtered, MSE: {}".format(MSE_y))
    plt.xlim(N - 100, N)
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_8():
    #!/usr/bin/env python
    # coding: utf-8

    # # PyDMD

    # ## Tutorial 6: Higher Order Dynamic Mode Decomposition on low dimensional snapshots

    # In this tutorial we will show the application of the higher order
    #  dynamic mode decomposition (for complete description read the 
    # original work by [Soledad Le Clainche and José M. Vega]
    # (https://epubs.siam.org/doi/10.1137/15M1054924)). 
    # This method allows to apply the DMD also when the dimension of 
    # the snapshots is less than the number of snapshots: 
    # following we will consider a limit case, dealing with 1D snapshots.

    # First of all we import the HODMD class from the pydmd package, 
    # we set matplotlib for the notebook and we import numpy.

    # In[1]:

    import matplotlib.pyplot as plt
    import numpy as np
    import time

    from pydmd import HODMD
    from pydmd.plotter import plot_eigs

    # Now, we create our toy dataset: we evaluate a nonlinear function 
    # in equispaced points, in order to simulate a temporal signal.
    #  Here our function:
    #
    # $
    # f(t) = \cos(t)\sin(\cos(t)) + \cos(\frac{t}{5})
    # $

    # In[2]:

    def myfunc(x):
        return np.cos(x) * np.sin(np.cos(x)) + np.cos(x * 0.2)

    # Because we trust in the DMD power, we add a bit of noise and 
    # we plot our function:

    # In[3]:

    x = np.linspace(0, 10, 64)
    y = myfunc(x)
    snapshots = y
    plt.plot(x, snapshots, ".")
    plt.show()

    # Now we create the `HODMD` object: in addition to the usual
    #  DMD parameters, we have the `d` parameter. Basically, 
    # using this method the initial snapshots matrix is rearranged 
    # in order to be able to extract the main structures using the 
    # singular value decomposition. This parameter `d` handle this 
    # arrangement (for further details, check the [HODMD]
    # (https://mathlab.github.io/PyDMD/hodmd.html) documentation).

    # In[4]:

    hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=30).fit(snapshots[None])

    # Despite the arrangement, the shape of the reconstructed data 
    # is the same of the original input.

    # In[5]:

    hodmd.reconstructed_data.shape

    # As always, we take a look at the eigenvalues to check the 
    # stability of the system.

    # In[6]:

    plot_eigs(hodmd)

    # Now we can compare the DMD output with the inital dataset:
    #  we plot the snapshots, the original function we are trying to 
    # reconstruct and the DMD reconstruction.

    # In[7]:

    hodmd.original_time["dt"] = hodmd.dmd_time["dt"] = x[1] - x[0]
    hodmd.original_time["t0"] = hodmd.dmd_time["t0"] = x[0]
    hodmd.original_time["tend"] = hodmd.dmd_time["tend"] = x[-1]

    plt.plot(hodmd.original_timesteps, snapshots, ".", label="snapshots")
    plt.plot(hodmd.original_timesteps, y, "-", label="original function")
    plt.plot(
        hodmd.dmd_timesteps,
        hodmd.reconstructed_data[0].real,
        "--",
        label="DMD output",
    )
    plt.legend()
    plt.show()

    # Quite good! Let see if we can see in the future.

    # In[8]:

    hodmd.dmd_time["tend"] = 50

    fig = plt.figure(figsize=(15, 5))
    plt.plot(hodmd.original_timesteps, snapshots, ".", label="snapshots")
    plt.plot(
        np.linspace(0, 50, 128),
        myfunc(np.linspace(0, 50, 128)),
        "-",
        label="original function",
    )
    plt.plot(
        hodmd.dmd_timesteps,
        hodmd.reconstructed_data[0].real,
        "--",
        label="DMD output",
    )
    plt.legend()
    plt.show()

    # The reconstruction is perfect. Also when time is far away from 
    # the snapshots.
    #
    # We can check what happens  if we add some noise.

    # In[10]:

    noise_range = [0.01, 0.05, 0.1, 0.2]
    fig = plt.figure(figsize=(15, 10))
    future = 20

    for id_plot, i in enumerate(noise_range, start=1):
        snapshots = y + np.random.uniform(-i, i, size=y.shape)
        hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=30).fit(snapshots[None])
        hodmd.original_time["dt"] = hodmd.dmd_time["dt"] = x[1] - x[0]
        hodmd.original_time["t0"] = hodmd.dmd_time["t0"] = x[0]
        hodmd.original_time["tend"] = hodmd.dmd_time["tend"] = x[-1]
        hodmd.dmd_time["tend"] = 20

        plt.subplot(2, 2, id_plot)
        plt.plot(hodmd.original_timesteps, snapshots, ".", label="snapshots")
        plt.plot(
            np.linspace(0, future, 128),
            myfunc(np.linspace(0, future, 128)),
            "-",
            label="original function",
        )
        plt.plot(
            hodmd.dmd_timesteps,
            hodmd.reconstructed_data[0].real,
            "--",
            label="DMD output",
        )
        plt.legend()
        plt.title("Noise [{} - {}]".format(-i, i))
    plt.show()

    # Results are obviously worst, but depending from the amount of 
    # noise the HODMD output matches the original function pretty well,
    #  at least in a short-time prevision. Within the temporal window, 
    # instead, it can reconstruct very well the trend and denoise the
    #  input data.

def test_10():
    #!/usr/bin/env python
# coding: utf-8

# # PyDMD

# ## Tutorial 1: Dynamic Mode Decomposition on a toy dataset

# In this tutorial we will show the typical use case, applying the 
# dynamic mode decomposition on the snapshots collected during the 
# evolution of a generic system. We present a very simple system since
#  the main purpose of this tutorial is to show the capabilities of the 
# algorithm and the package interface.

# First of all we import the DMD class from the pydmd package, 
# we set matplotlib for the notebook and we import numpy.

# In[1]:

    import matplotlib.pyplot as plt
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np

    from pydmd import DMD
    from pydmd.plotter import plot_eigs


    # We create the input data by summing two different functions:<br>
    # $f_1(x,t) = \text{sech}(x+3)\exp(i2.3t)$<br>
    # $f_2(x,t) = 2\text{sech}(x)\tanh(x)\exp(i2.8t)$.<br>

    # In[2]:


    def f1(x, t):
        return 1.0 / np.cosh(x + 3) * np.exp(2.3j * t)


    def f2(x, t):
        return 2.0 / np.cosh(x) * np.tanh(x) * np.exp(2.8j * t)


    x = np.linspace(-5, 5, 65)
    t = np.linspace(0, 4 * np.pi, 129)

    xgrid, tgrid = np.meshgrid(x, t)

    X1 = f1(xgrid, tgrid)
    X2 = f2(xgrid, tgrid)
    X = X1 + X2


    # The plots below represent these functions and the dataset.

    # In[3]:


    titles = ["$f_1(x,t)$", "$f_2(x,t)$", "$f$"]
    data = [X1, X2, X]

    fig = plt.figure(figsize=(17, 6))
    for n, title, d in zip(range(131, 134), titles, data):
        plt.subplot(n)
        plt.pcolor(xgrid, tgrid, d.real)
        plt.title(title)
    plt.colorbar()
    plt.show()


    # Now we have the temporal snapshots in the input matrix rows: 
    # we can easily create a new DMD instance and exploit it in order 
    # to compute the decomposition on the data. Since the snapshots 
    # must be arranged by columns, in this case we need to transpose the matrix.

    # In[4]:


    dmd = DMD(svd_rank=2)
    dmd.fit(X.T)


    # The `dmd` object contains the principal information about the decomposition:
    # - the attribute `modes` is a 2D numpy array where the columns are the 
    # low-rank structures individuated;
    # - the attribute `dynamics` is a 2D numpy array where the rows refer to 
    # the time evolution of each mode;
    # - the attribute `eigs` refers to the eigenvalues of the low dimensional operator;

    # - the attribute `reconstructed_data` refers to the approximated system evolution.

    #
    # Moreover, some helpful methods for the graphical representation are provided.

    # Thanks to the eigenvalues, we can check if the modes are stable or not: 
    # if an eigenvalue is on the unit circle, the corresponding mode will be stable; 
    # while if an eigenvalue is inside or outside the unit circle,
    #  the mode will converge or diverge, respectively. From the following plot, 
    # we can note that the two modes are stable.

    # In[5]:


    for eig in dmd.eigs:
        print(
            "Eigenvalue {}: distance from unit circle {}".format(
                eig, np.abs(np.sqrt(eig.imag**2 + eig.real**2) - 1)
            )
        )

    plot_eigs(dmd, show_axes=True, show_unit_circle=True)


    # We can plot the modes and the dynamics:

    # In[6]:


    for mode in dmd.modes.T:
        plt.plot(x, mode.real)
        plt.title("Modes")
    plt.show()

    for dynamic in dmd.dynamics:
        plt.plot(t, dynamic.real)
        plt.title("Dynamics")
    plt.show()


    # Finally, we can reconstruct the original dataset as the
    #  product of modes and dynamics. We plot the evolution of each 
    # mode to emphasize their similarity with the input functions and we plot 
    # the reconstructed data.

    # In[7]:


    fig = plt.figure(figsize=(17, 6))

    for n, mode, dynamic in zip(range(131, 133), dmd.modes.T, dmd.dynamics):
        plt.subplot(n)
        plt.pcolor(
            xgrid, tgrid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T
        )

    plt.subplot(133)
    plt.pcolor(xgrid, tgrid, dmd.reconstructed_data.T.real)
    plt.colorbar()

    plt.show()


    # We can also plot the absolute error between the approximated data 
    # and the original one.

    # In[8]:


    plt.pcolor(xgrid, tgrid, (X - dmd.reconstructed_data.T).real)
    fig = plt.colorbar()


# The reconstructed system looks almost equal to the original one: 
# the dynamic mode decomposition made possible the identification of the 
# meaningful structures and the complete reconstruction of the system using 
# only the collected snapshots.

if __name__ == "__main__":
    test_10()
