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


if __name__ == "__main__":
    test_2()
