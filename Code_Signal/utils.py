import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.utils import check_random_state


def plot_data(data, match=None, axes=None, title=None):

    if axes is None:
        fig, axes = plt.subplots(5, 2, sharex=True, sharey=True)

    for ii, ax in enumerate(axes.ravel()):
        ax.cla()
        ax.plot(data[ii])
        if match is not None:
            ax.vlines(match[ii], ax.get_ylim()[0], ax.get_ylim()[1],
                      colors='g')

    if title is not None:
        plt.suptitle(title)

    plt.show()
    return axes


def add_atom(data, atom, low, high, random_state=None):
    rng = check_random_state(random_state)

    support = atom.shape[0]
    n_samples = data.shape[0]

    starts = rng.random_integers(low=low, high=high,
                                 size=(n_samples))
    for i in range(n_samples):
        start = starts[i]
        data[i, start: start + support] = atom


def mock_data(n_samples=100, support=20, noise_level=0.1, random_state=None):
    rng = check_random_state(random_state)

    n_times = 300
    data = np.zeros((n_samples, n_times))

    print('Computing morlet')
    # morl = signal.morlet(support).real
    # tri = np.hstack((np.linspace(0, 1, support), np.zeros(support)))
    tri = np.hstack((np.linspace(0, 1, support),
                     np.linspace(0, 1, support)[::-1]))
    tri -= np.mean(tri)

    # normalize data
    tri = tri / np.linalg.norm(tri)
    print('[Done]')

    square_u = np.ones((support))
    # square_d = -2 * np.ones((support / 2))
    square_d = -np.ones((support))
    square = np.hstack((square_u, square_d))

    # normalize data
    square = np.hstack((square[::2], square[::2]))
    square = square / np.linalg.norm(square)

    low = 0
    high = n_times // 2 - tri.shape[0]
    add_atom(data, atom=tri, low=low, high=high, random_state=random_state)
    low = n_times // 2
    high = n_times - square.shape[0]
    add_atom(data, atom=square, low=low, high=high, random_state=random_state)

    # add some random noise
    data = data + noise_level * rng.rand(*data.shape)
    print('[Done]')

    sfreq = 1

    return data, sfreq


def _get_somato_data():
    from mne.datasets import somato
    data_path = somato.data_path()
    raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'
    event_id, tmin, tmax = 1, -1., 3.

    # Setup for reading the raw data
    raw = io.Raw(raw_fname, preload=True)
    raw.filter(1, 40, n_jobs=-1)
    baseline = (None, 0)
    events = mne.find_events(raw, stim_channel='STI 014')

    # picks MEG gradiometers
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                           stim=False)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=baseline,
                        reject=dict(grad=4000e-13, eog=350e-6))

    return epochs, epochs.info['sfreq']


def load_data(dataset='somato', **kwargs):
    if dataset == 'simulated':
        return mock_data(**kwargs)
    elif dataset == 'somato':
        return _get_somato_data()
