import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neo.core import SpikeTrain
from quantities import ms
from elephant.statistics import instantaneous_rate, isi
from elephant.kernels import GaussianKernel


def load_results(results_dir):
    """Load the saved NEST results.

    :return: ``(data, meta)`` where ``data[pop] = {"times", "senders",
        "n_neurons", "first_gid"}`` and ``meta`` holds duration/resolution/etc.
    """
    with open(os.path.join(results_dir, "meta.json")) as f:
        meta = json.load(f)
    data = {}
    for pop in meta["populations"]:
        npz = np.load(os.path.join(results_dir, f"{pop}.npz"))
        data[pop] = {
            "times": npz["times"],
            "senders": npz["senders"],
            "n_neurons": int(npz["n_neurons"]),
            "first_gid": int(npz["first_gid"]),
        }
    return data, meta


def _spike_train(times, duration):
    """Build a single neo SpikeTrain from an array of spike times [ms]."""
    times = np.asarray(times, dtype=float)
    # guard against numerical spikes landing exactly on / past t_stop
    times = times[times <= duration]
    return SpikeTrain(times * ms, t_stop=duration * ms)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def population_firing_rate(times, n_neurons, duration, resolution=0.1, kernel_sigma=10.0):
    """Mean per-neuron instantaneous firing rate of a population (Elephant).

    Concatenated spikes are convolved with a Gaussian kernel and divided by the
    number of neurons, as in the example ``plot_firing``.

    :return: ``(rate_times [ms], rate [Hz])`` numpy arrays.
    """
    if len(times) == 0:
        t = np.arange(0.0, duration, resolution)
        return t, np.zeros_like(t)
    st = _spike_train(times, duration)
    rate = instantaneous_rate(
        st,
        sampling_period=resolution * ms,
        kernel=GaussianKernel(kernel_sigma * ms),
        border_correction=True,
    )
    return rate.times.rescale(ms).magnitude.ravel(), rate.magnitude.ravel() / n_neurons


def firing_rate_per_neuron(senders, n_neurons, duration):
    """Per-neuron mean firing rate [Hz], INCLUDING non-spiking neurons.

    Counts spikes per neuron, then pads with zeros for the silent cells up to
    ``n_neurons`` and divides by the recording time — exactly as
    ``cerebellar_models`` ``SimResultsTable`` does.

    :return: array of length ``n_neurons`` with each neuron's rate in Hz.
    """
    T = duration / 1000.0  # s
    counts = np.unique(senders, return_counts=True)[1] if len(senders) else np.array([])
    counts = np.concatenate([counts, np.zeros(max(n_neurons - len(counts), 0))])
    return counts / T


def mean_firing_rates(data, meta):
    """Mean +/- std firing rate [Hz] per population over ALL neurons.

    Silent cells count as 0 Hz (as in the cerebellar_models analysis), so the
    average is over ``n_neurons`` and not only over the spiking ones.

    :return: dict ``{pop: (mean_Hz, std_Hz)}``.
    """
    stats = {}
    for pop, d in data.items():
        fr = firing_rate_per_neuron(d["senders"], d["n_neurons"], meta["duration"])
        stats[pop] = (float(np.mean(fr)), float(np.std(fr))) if fr.size else (0.0, 0.0)
    return stats


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_raster(times, senders, pop, duration, output_dir):
    """Raster plot of one population."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    if len(times) > 0:
        _, ids = np.unique(senders, return_inverse=True)
        plt.scatter(times, ids, marker="|", s=2, color="k")
    plt.title(f"Raster — {pop}")
    plt.xlabel("Time [ms]")
    plt.ylabel("Neuron index")
    plt.xlim(0, duration)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"raster_{pop}.png"), dpi=120)
    plt.close()


def plot_firing_rate(times, senders, n_neurons, pop, duration, output_dir,
                     resolution=0.1, kernel_sigma=10.0, print_rate=False):
    """Instantaneous mean firing-rate plot of one population.

    The curve is the population instantaneous rate (Elephant Gaussian kernel).
    The text box reports the per-neuron mean +/- std firing rate computed over
    ALL neurons (silent cells counted as 0 Hz), as in the cerebellar_models
    ``SimResultsTable``.
    """
    os.makedirs(output_dir, exist_ok=True)
    t, rate = population_firing_rate(times, n_neurons, duration, resolution, kernel_sigma)

    # per-neuron mean firing rate (silent cells included) -> reported on the plot
    per_neuron = firing_rate_per_neuron(senders, n_neurons, duration)
    fr_mean, fr_std = float(np.mean(per_neuron)), float(np.std(per_neuron))
    if print_rate:
        print(f"  {pop:<15s} FR = {fr_mean:.3f} +- {fr_std:.3f} Hz (all {n_neurons} cells)")

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.plot(t, rate, color="red", lw=1)
    ax.hlines(fr_mean, 0, duration, linestyles="--", colors="grey")
    ax.set_title(f"Mean firing rate — {pop} (Gaussian kernel {kernel_sigma} ms)")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Firing rate [Hz]")
    ax.set_xlim(0, duration)
    ax.text(
        0.01, 0.95,
        r"FR (all cells): {:.2f} $\pm$ {:.2f} Hz".format(fr_mean, fr_std),
        ha="left", va="top", transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"firing_rate_{pop}.png"), dpi=120)
    plt.close()
    return fr_mean, fr_std


def plot_psth(times, pop, duration, output_dir, bin_ms=20.0):
    """Peri-stimulus time histogram (population spike count) of one population."""
    os.makedirs(output_dir, exist_ok=True)
    bins = np.arange(0, duration + bin_ms, bin_ms)
    plt.figure(figsize=(10, 4))
    if len(times) > 0:
        plt.hist(times, bins=bins, color="steelblue")
    plt.title(f"PSTH — {pop} (bin {bin_ms} ms)")
    plt.xlabel("Time [ms]")
    plt.ylabel("Spike counts")
    plt.xlim(0, duration)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"psth_{pop}.png"), dpi=120)
    plt.close()


def plot_isi(times, senders, pop, duration, output_dir, nb_bins=50):
    """Inter-spike-interval histogram of one population (per-neuron ISIs pooled)."""
    os.makedirs(output_dir, exist_ok=True)
    all_isis = []
    if len(times) > 0:
        order = np.argsort(senders, kind="stable")
        s_sorted, t_sorted = senders[order], times[order]
        for neuron in np.unique(s_sorted):
            nt = np.sort(t_sorted[s_sorted == neuron])
            if nt.size > 1:
                all_isis.append(isi(nt * ms).magnitude)
    plt.figure(figsize=(8, 4))
    if all_isis:
        plt.hist(np.concatenate(all_isis), nb_bins, color="darkorange")
        plt.yscale("log")
    plt.title(f"ISI distribution — {pop}")
    plt.xlabel("ISI [ms]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"isi_{pop}.png"), dpi=120)
    plt.close()


def summary_table(data, meta):
    """Print and return mean +/- std firing rate [Hz] per population (silent cells included)."""
    stats = mean_firing_rates(data, meta)
    print("\nMean firing rate per population (silent cells included):")
    for pop in meta["populations"]:
        mean, std = stats[pop]
        print(f"  {pop:<15s}: {mean:8.3f} +- {std:6.3f} Hz   (n={meta['n_neurons'][pop]})")
    return stats


def analyze_all(results_dir, output_dir=None, kernel_sigma=10.0, bin_ms=20.0):
    """Load saved results and produce every plot for every population."""
    output_dir = output_dir or os.path.join(results_dir, "figs")
    data, meta = load_results(results_dir)
    duration = meta["duration"]
    resolution = meta["resolution"]

    for pop in meta["populations"]:
        d = data[pop]
        plot_raster(d["times"], d["senders"], pop, duration, output_dir)
        plot_firing_rate(d["times"], d["senders"], d["n_neurons"], pop, duration, output_dir,
                         resolution=resolution, kernel_sigma=kernel_sigma, print_rate=True)
        plot_psth(d["times"], pop, duration, output_dir, bin_ms=bin_ms)
        plot_isi(d["times"], d["senders"], pop, duration, output_dir)

    summary_table(data, meta)
    print(f"\nSaved figures to {output_dir}")


if __name__ == "__main__":
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nest_results")
    analyze_all(RESULTS_DIR)

