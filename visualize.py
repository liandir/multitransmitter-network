import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_sequence(sequence, cmap="viridis", dt=0.01, vmin=-1.0, vmax=1.0, **kwargs):
    times = np.arange(sequence.shape[0]) * dt
    ns = np.arange(sequence.shape[1]) + 1
    fig, ax = plt.subplots(
        figsize=(14,6)
    )
    pcm = ax.pcolormesh(times, ns, sequence.T.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neuron #")
    ax.set(**kwargs)
    fig.colorbar(pcm, ax=ax)
    plt.show()
    
def plot_weights(weights, cmap="inferno", vmin=-1.0, vmax=1.0, **kwargs):
    ms = np.arange(weights.shape[0]) + 1
    ns = np.arange(weights.shape[1]) + 1
    norm = Normalize(vmin, vmax)
    fig, axs = plt.subplots(
        2,2,
        figsize=(12,10),
        sharex=True,
        sharey=True
    )
    for i, ax in enumerate(axs.flat):
        ax.pcolormesh(ms, ns, weights[:,:,i].T.cpu().numpy(), cmap=cmap, norm=norm, shading="auto")
        ax.set_title(f"Receptor {i+1}")
        ax.set(**kwargs)
        if i > 1:
            ax.set_xlabel("Output Neuron #")
        if i % 2 == 0:
            ax.set_ylabel("Input Neuron #")
        
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axs.ravel().tolist(), label="Synaptic Weight [Arb. Units]")
    plt.show()
    
def plot(lines, dt=0.01, **kwargs):
    fig, ax = plt.subplots(
        figsize=(12,6)
    )
    ax.plot(np.arange(len(lines))*dt, lines.cpu().numpy())
    ax.set(**kwargs)
    ax.grid(alpha=0.3)
    plt.show()
    
def plot_data(data, **kwargs):
    fig, ax = plt.subplots(
        figsize=(10,8)
    )
    ax.plot(data.T[0].cpu().numpy(), data.T[1], ".")
    ax.grid(alpha=0.3)
    ax.set(**kwargs)
    plt.show()