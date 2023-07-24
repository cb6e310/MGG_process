import numpy as np

from scipy import signal

import numpy as np

from scipy.signal import butter, lfilter, freqz

import matplotlib.pyplot as plt


from filters import butter_filter


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
    #%% Simple example: generate signal with 3 components + noise  
    import numpy as np  
    import matplotlib.pyplot as plt  
    from vmdpy import VMD  

    #. Time Domain 0 to T  
    T = 1000  
    fs = 1/T  
    t = np.arange(1,T+1)/T  
    freqs = 2*np.pi*(t-0.5-fs)/(fs)  

    #. center frequencies of components  
    f_1 = 2  
    f_2 = 24  
    f_3 = 288  

    #. modes  
    v_1 = (np.cos(2*np.pi*f_1*t))  
    v_2 = 1/4*(np.cos(2*np.pi*f_2*t))  
    v_3 = 1/16*(np.cos(2*np.pi*f_3*t))  

    f = v_1 + v_2 + v_3 + 0.1*np.random.randn(v_1.size)  

    #. some sample parameters for VMD  
    alpha = 2000       # moderate bandwidth constraint  
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
    K = 3              # 3 modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly  
    tol = 1e-7  


    #. Run VMD 
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)  

    #. Visualize decomposed modes
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(f)
    plt.title('Original signal')
    plt.xlabel('time (s)')
    plt.subplot(2,1,2)
    plt.plot(u.T)
    plt.title('Decomposed modes')
    plt.xlabel('time (s)')
    plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
    plt.tight_layout()

    plt.show()

def test_4():
    import numpy as np
import matplotlib.pylab as plt
import padasip as pa

# creation of data
N = 500
x = np.random.normal(0, 1, (N, 4)) # input matrix
v = np.random.normal(0, 0.1, N) # noise
d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # target

# identification
f = pa.filters.FilterLMS(n=4, mu=0.1, w="random")
y, e, w = f.run(d, x)

# show results
plt.figure(figsize=(15,9))
plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
plt.plot(d,"b", label="d - target")
plt.plot(y,"g", label="y - output");plt.legend()
plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
plt.tight_layout()
plt.show()

if __name__ == "__main__":
    test_4()
