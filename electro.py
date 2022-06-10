"""
Analyzes physiology data and makes figures
"""
from pathlib import Path
from enum import Enum, auto
from typing import Optional
import shutil

import numpy as np
from pyabf import ABF
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from allensdk.ephys import ephys_features
from matplotlib.patches import Rectangle

from scalebar import AnchoredScaleBar, add_scalebar

DIR = Path("/storage/archive/sylwestrak/2022-05-24/")
PETER_DIR = Path("/storage/archive/sylwestrak/peter")
RTALK = Path().home() / "Documents/rtalk3"


def restrict_range(time: np.ndarray, y: np.ndarray, trange: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    "restricts the range, returning new time (starting at 0) and y"
    out_y = y[ephys_features.find_time_index(time, trange[0]): ephys_features.find_time_index(time, trange[1])]
    out_t = np.linspace(0, trange[1] - trange[0], len(out_y))
    assert (out_t[1] - out_t[0] - time[1] - time[0]) < 1e-8
    return out_t, out_y

class TrialType(Enum):
    "The kind of abf file"
    vc_gapless = auto()
    ic_gapless = auto()
    fi = auto()
    slow_ramp = auto()
    fast_pulse = auto()
    opto = auto()
    none = auto()

def sort_abfs(abfs: list[ABF]) -> list[ABF]:
    "returns the list of abfs sorted by number"
    def get_number(abf: ABF):
        return int(Path(abf.abfFilePath).with_suffix("").name[-4:])
    out = sorted(abfs, key=get_number)
    # print(*[Path(o.abfFilePath).name for o in out])
    return out

def get_dir_info(directory: Path, filt_string: Optional[str] = None) -> dict[Path, TrialType]:
    """
    returns the types of experiments in a directory
    filt_string only selects files with the contents in its name
    """
    # This code probably ought to contain the sort_abfs function, but this will break my exisiting code
    if filt_string is None:
        filt_string = ""

    out = {}
    for path in directory.glob("*.abf"):
        if filt_string not in path.name:
            continue
        abf = ABF(path, loadData=False)
        protocol = abf.protocol
        if "1s-stim" in protocol:
            out[path] = TrialType.fi
        elif "slow-ramp" in protocol:
            out[path] = TrialType.slow_ramp
        elif "fast-stim" in protocol:
            out[path] = TrialType.fast_pulse
        elif "img" in protocol:
            out[path] = TrialType.none
        elif "opto_stim" in protocol:
            out[path] = TrialType.opto
        elif "gapless" in protocol:
            if abf.adcUnits[0] == "mV":
                out[path] = TrialType.ic_gapless
            elif abf.adcUnits[0] == "pA":
                out[path] = TrialType.vc_gapless
            else:
                raise ValueError(f"{abf.adcUnits[0]} is not a valid unit for gapless")
        else:
            raise ValueError(f"{protocol} is not recognized")
    return out


def verify_fi_current(data: ABF, thresh=5.0, verbose=False) -> bool:
    "Verifies that the FI injected roughly the correct current"
    if thresh is None:
        thresh = 5.0
    for sweep in range(data.sweepCount):
        data.setSweep(sweep, 0)
        command = data.sweepC
        data.setSweep(sweep, 1)
        # it might be better to get a median instead of a max range
        error = (np.max(data.sweepY) - np.min(data.sweepY)) - (np.max(command) - np.min(command))
        if verbose:
            print(f"{error=}")
        if error > thresh:
            return False
    return True

def get_fi_currents(data: ABF) -> list[float]:
    "Gets all of the current injections from command"
    out = []
    for sweep in range(data.sweepCount):
        data.setSweep(sweep, 0)
        # handle positive steps
        sweep_max = data.sweepC.max()
        if sweep_max > 0:
            out.append(sweep_max)
        else:
            out.append(data.sweepC.min())
    return out


def get_fi_freqs(data: ABF, debug_plot=False, spike_detect_params: Optional[dict[str, float]] = None, 
                 inst=True) -> np.ndarray:
    "Gets the frequency at each current step"
    filter_freq = 8
    freq_means = np.zeros(data.sweepCount)
    if spike_detect_params is None:
        spike_detect_params = {"dv_cutoff": 20, "height": 0}
    # get spike window
    data.setSweep(0)
    window = data.sweepX[:-1][np.diff(data.sweepC) != 0]

    for sweep, _ in enumerate(freq_means):
        data.setSweep(sweep, 0)
        putitive_spikes = ephys_features.detect_putative_spikes(data.sweepY, data.sweepX, filter=filter_freq, 
                                                                dv_cutoff=spike_detect_params["dv_cutoff"])
        peaks = ephys_features.find_peak_indexes(data.sweepY, data.sweepX, putitive_spikes)
        # filter out low peaks
        high_peaks = data.sweepY[peaks.astype(int)] > spike_detect_params["height"]
        putitive_spikes = putitive_spikes[high_peaks]
        peaks = peaks[high_peaks]
        spike_inds, peaks = ephys_features.filter_putative_spikes(data.sweepY, data.sweepX,
                                                                  putitive_spikes, peaks, filter=filter_freq)
        # else:
            # putitive_spikes = signal.find_peaks(data.sweepX, height=spike_detect_params["height"])
            # putitive_spike_times = data.sweepX[spike_inds.astype(int)]
            # putitive_isi = np.diff(putitive_spike_times)

        if debug_plot:
            fig, axs = plt.subplots()
            axs.plot(data.sweepX, data.sweepY)
            axs.plot(data.sweepX[putitive_spikes.astype(int)], [-10]*len(putitive_spikes), "or")
            axs.plot(data.sweepX[putitive_spikes.astype(int)], [0]*len(putitive_spikes), "ow")
        # nan is no spikes or 1 spike
        spikes = data.sweepX[spike_inds.astype(int)]
        # filter out spikes outside window
        spikes = spikes[np.logical_and(
            spikes > window[0],
            spikes < window[1]
        )]

        if len(spikes) == 0:
            freq_means[sweep] = 0
            continue
        elif len(spikes) == 1:
            freq_means[sweep] = np.NAN
            continue
        if inst:
            # average 1/isi
            isi = np.diff(spikes)
            freqs = 1 / isi
            freq_means[sweep] = np.nanmean(freqs)
        else:
            freq_means[sweep] = len(spikes) / (window[1] - window[0])
    return freq_means
        

def plot_opto_on_axs(axs, data: ABF, sweep: Optional[int] = None, 
                     trange: Optional[tuple[float, float]] = None,
                     vclamp: Optional[np.ndarray] = None,
                     lblx=False, lbly=False, color="k", ocolor="b", ochan=3):
    "makes  a pretty trace for opto stim"
    if sweep is None:
        sweep = 0
    data.setSweep(sweep, channel=0)
    # filter out 60 hz
    b, a =  signal.iirnotch(60, 30, data.dataRate)
    filtered = signal.filtfilt(b, a, data.sweepY)
    if vclamp is not None:
        axs[1].plot(data.sweepX, vclamp, color=color, linewidth=1)
    else:
        axs[1].plot(data.sweepX, filtered, color=color, linewidth=1)
    data.setSweep(sweep, channel=ochan)
    # deteck edges to find the windows where the light was on
    edges = data.sweepX[:-1][np.abs(np.diff(data.sweepY) - 1) > 2]
    windows = edges.reshape(int(len(edges) / 2), 2)
    for window in windows:
        axs[0].axvspan(*window, color=ocolor)
    if lbly:
        axs[0].set_ylabel("Light")
        if vclamp is None:
            axs[1].set_ylabel("Vm (mV)")
        else:
            axs[1].set_ylabel("I (pA)")
    if lblx:
        axs[1].set_xlabel("Time (s)")
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axs[0].set_yticks([])


def plot_ic_on_axs(axs, data: ABF, sweep: Optional[int] = None, 
                   trange: Optional[tuple[float, float]] = None,
                   lblx=False, lbly=False, color="k", icolor="w", ichan=1):
    "makes a pretty trace for current clamp"
    if sweep is None:
        sweep = 0
    data.setSweep(sweep, channel=0)
    # filter out 60 hz
    b, a =  signal.iirnotch(60, 30, data.dataRate)
    filtered = signal.filtfilt(b, a, data.sweepY)
    axs[1].plot(data.sweepX*1000, filtered, color=color, linewidth=1)
    data.setSweep(sweep, channel=ichan)
    axs[0].plot(data.sweepX*1000, data.sweepY, color=icolor, linewidth=.7)
    if trange is not None:
        axs.set_xlim(trange)

    if lbly:
        axs[0].set_ylabel("Current (pA)")
        axs[1].set_ylabel("Vm (mV)")
    if lblx:
        axs[1].set_xlabel("Time (ms)")

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

def plot_psp_traces(data: ABF, wide_start: float, narrow_start: float, color="m", wide_duration=10, narrow_duration=.5):
    "makes the figure and axes for sample trace"
    axs_dict = plt.figure(constrained_layout=True).subplot_mosaic(
        """
        caaaaa.
        bbbbbbb
        """
    )
    axs = [axs_dict["a"], axs_dict["b"], axs_dict["c"]]
    axs[1].set_ylabel("Vm (mV)")
    axs[1].set_xlabel("Time (s)")
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    axs[2].axis("off")
    fig = axs[1].get_figure()
    wide_trange = wide_start, wide_start+wide_duration
    narrow_trange = narrow_start, narrow_start+narrow_duration
    data.setSweep(0)
    # plot wide
    wide_t, wide_vm = restrict_range(data.sweepX, data.sweepY, wide_trange)
    axs[1].plot(wide_t, wide_vm, color=color, linewidth=1, zorder=-1)
    # plot narrow
    narrow_t, narrow_vm = restrict_range(data.sweepX, data.sweepY, narrow_trange)
    axs[0].plot(narrow_t, narrow_vm, color=color, linewidth=1)
    # plot rectangle
    startx = narrow_start - wide_start
    width = axs[0].get_xlim()[1]
    starty, endy = axs[0].get_ylim()
    rectangle = Rectangle((startx, starty), width, endy-starty, edgecolor="w", facecolor=(1, 1, 1, 0))
    axs[1].add_patch(rectangle)
    # plot scalebar
    ytick = round(10*(endy - starty) / 4) / 10
    scalebar = AnchoredScaleBar(axs[0].transData, sizex=.1, labelx="100 ms",
                            sizey=ytick, labely=f"{ytick: .1f} mV", loc="center")
    axs[2].add_artist(scalebar)
    fig.set_size_inches([5.28, 4.32])
    return fig, axs


def inspect_fis(fis: list[ABF], current_example: Optional[int] = None, last_abf: Optional[int] = None,
                spike_detect_params: Optional[dict[str, float]] = None, thresh: Optional[float] = None):
    """
    some code for inspecting fi curves
    This code is designed to be run iteritivly adding optional arguments as 
    they become known
    """
    fis = sort_abfs(fis)
    for ind, fi in enumerate(fis):
        print(ind, get_fi_currents(fi))
    if current_example is None:
        return
    currents = get_fi_currents(fis[current_example])
    # filter out usable fis
    usable_fis = [fi for fi in fis if get_fi_currents(fi)==currents and verify_fi_current(fi, thresh=thresh, verbose=False)]
    print(f"{len(usable_fis) = }")
    # check maximum current injection
    if last_abf is None:
        for ind, fi in enumerate(usable_fis):
            fig, axs = plt.subplots(2,1)
            plot_ic_on_axs(axs, fi, 0, color="m")
            axs[0].set_title(f"{ind}")
        plt.show()
    # check spike detection
    if last_abf is not None:
        get_fi_freqs(usable_fis[last_abf-2], debug_plot=True, spike_detect_params=spike_detect_params)
    plt.show()

def plot_fi_on_ax(ax, fis: list[ABF], current_example: int, last_abf: Optional[int] = None,
                  spike_detect_params: Optional[dict[str, float]] = None, debug_plot=False,
                  thresh: Optional[float] = None, color="m", inst=True):
    "Plots an fi curve on axes"
    if last_abf is None:
        last_abf = len(fis) - 1
    fis = sort_abfs(fis)
    currents = get_fi_currents(fis[current_example])
    usable_fis = [fi for fi in fis if get_fi_currents(fi)==currents and verify_fi_current(fi, thresh=thresh)][:last_abf]
    freqs2d = np.zeros((len(usable_fis), len(currents)))
    for i, fi in enumerate(usable_fis):
       freqs2d[i, :] = get_fi_freqs(usable_fis[i], spike_detect_params=spike_detect_params, inst=inst)
    # # convert_ nans_to_zeros
    # freqs2d = np.nansum(np.stack((freqs2d, np.zeros(freqs2d.shape)), -1), 2)
    if debug_plot:
        ax.imshow(freqs2d, aspect="auto")
    # plt.show(); quit()
    freqs = np.nanmean(freqs2d, axis=0)
    ax.plot(currents, freqs, color)
    ax.set_xlabel("Current (pA)")
    ax.set_ylabel("Spike frequency (hz)")
    # pass through 0
    xlim = ax.get_xlim()
    ax.set_xlim([0, xlim[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_neg_inj_on_ax(ax, fis: list[ABF], current_example: int, current_ind=0, last_abf: Optional[int] = None, 
                       thresh: Optional[float] = None, color="m", alpha=.2, debug_plot=False):
    "plots average and example negitive current injection on axis"
    if last_abf is None:
        last_abf = len(fis) - 1
    fis = sort_abfs(fis)
    currents = get_fi_currents(fis[current_example])
    usable_fis = [fi for fi in fis if get_fi_currents(fi)==currents and verify_fi_current(fi, thresh=thresh)][:last_abf]
    current2d = np.zeros((len(usable_fis), len(usable_fis[0].sweepX)))
    for i, fi in enumerate(usable_fis):
        fi.setSweep(current_ind)
        current2d[i, :] = fi.sweepY
    if debug_plot:
        ax.imshow(current2d, aspect="auto"); plt.show(); plt.quit()
    for vm in current2d:
        ax.plot(fi.sweepX*1000, vm, color="m", alpha=alpha, linewidth=1)
    ax.plot(fi.sweepX*1000, current2d.mean(axis=0), "m", linewidth=3)
    ax.set_ylabel("Vm (mV)")
    ax.set_xlabel("Time (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{int(np.min(fi.sweepC))} pA current injection")

def plot_round_robin():
    # sample traces
    fig, axs = plt.subplots(2, 3, sharex=True, sharey="row", tight_layout=True, gridspec_kw={"height_ratios": [1, 5]})
    fis = [ABF(p) for p, t in get_dir_info(DIR, "cell2").items() if t == TrialType.fi]
    fi = fis[5]
    plot_ic_on_axs(axs[:, 0], fi, 0, lblx=True, lbly=True, color="r")
    plot_ic_on_axs(axs[:, 1], fi, 14, lblx=True, color="r")
    plot_ic_on_axs(axs[:, 2], fi, 22, lblx=True, color="r")
    # fi Curve
    currents = get_fi_currents(fis[5])
    full_fis = [fi for fi in fis if get_fi_currents(fi) == currents]
    freqs2d = np.zeros((len(full_fis), len(currents)))
    for i, fi in enumerate(full_fis):
       freqs2d[i, :] = get_fi_freqs(full_fis[-1])
    freqs = freqs2d.mean(axis=0)
    fig, ax = plt.subplots()
    ax.plot(currents, freqs, "r")
    ax.set_xlabel("Current (pA)")
    ax.set_ylabel("Spike frequency (hz)")
    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_waveform():
    "plots the waveform of a fi curve"
    fig, ax = plt.subplots()
    fis = [ABF(p) for p, t in get_dir_info(DIR, "cell2").items() if t == TrialType.fi]
    fi = fis[5]
    fi.setSweep(14)
    ax.plot(fi.sweepX, fi.sweepY, color="m", linewidth=1)
    ax.axis('off')
    fig.savefig(Path().home() / "Documents/rtalk3/img_src/spikes.svg")
    
    
def plot_j():
    "plots the j sample traces and data"
    jdir = PETER_DIR / "2022-05-27"
    # make fi
    fis = [ABF(p) for p, t in get_dir_info(jdir, "cell3").items() if t == TrialType.fi]
    # inspect_fis(fis, 4, 15, {"dv_cutoff": 3, "height": -10})
    fig1, ax = plt.subplots()#1, 2, tight_layout=True)
    # plot_neg_inj_on_ax(ax[0], fis, 4, 2, 15)
    plot_fi_on_ax(ax, fis, 4, 15, {"dv_cutoff": 3, "height": -10}, inst=True)
    # plot_fi_on_ax(ax[1], fis, 4, 15, {"dv_cutoff": 3, "height": -10}, inst=False, color="b")
    # ax[1].legend(["Instantaneous", "Average"])
    fig1.set_size_inches(5.47, 6.36)
    fig1.savefig(RTALK / "j" / "fi.svg")
    # plt.show(); quit()
    # opto = [ABF(p) for p, t in get_dir_info(jdir, "cell3").items() if t == TrialType.opto]
    # fig, axs = plt.subplots(2, 1, sharex=True, sharey="row", 
                            # tight_layout=True, gridspec_kw={"height_ratios": [1, 5]})
    # plot_ic_on_axs(axs, opto[-1], 1, lblx=True, lbly=True, color="m", icolor="b", ichan=3)
    # axs[0].set_ylabel("Light")
    # axs[0].set_yticks([0, 5])
    # axs[0].set_yticklabels(["off", "on"])
    # fig.set_size_inches(6, 4)
    # fig.savefig(RTALK / "j" / "opto.svg")
    # plot fi data
    p = next(jdir.glob("*cell3*30.abf"))
    fi = ABF(p)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey="row", 
                            tight_layout=True, gridspec_kw={"height_ratios": [1, 5]})
    plot_ic_on_axs(axs[:, 0], fi, 2, lblx=True, lbly=True, color="m")
    plot_ic_on_axs(axs[:, 1], fi, 7, lblx=True, lbly=False, color="m")
    plot_ic_on_axs(axs[:, 2], fi, 9, lblx=True, lbly=False, color="m")
    fig.set_size_inches(6, 6.36)
    fig.savefig(RTALK / "j" / "traces.svg")
    data = ABF(next(jdir.glob("*cell3*00.abf")))
    # plt.show()
    # print(fig1.get_size_inches())
    # print(fig.get_size_inches())
    # quit()
    fig, axs = plt.subplots(2, 1)
    plot_ic_on_axs(axs, data, 0, color="m")
    fig, axs = plot_psp_traces(data, 15.44, 16.4, wide_duration=6.47)
    fig.savefig(RTALK / "j" / "nothing.svg")
    
def plot_d():
    "plots the d sample traces and data"
    ddir = PETER_DIR / "2022-05-12"
    # make fi
    fis = [ABF(p) for p, t in get_dir_info(ddir, "cell1").items() if t == TrialType.fi]
    # inspect_fis(fis, 4, 14)
    fig1, ax = plt.subplots()#1, 2, tight_layout=True)
    # plot_neg_inj_on_ax(ax[0], fis, 4, 0, 12)
    plot_fi_on_ax(ax, fis, 4, 13)
    # plot_fi_on_ax(ax[1], fis, 4, 13, inst=False, color="b")
    # ax[1].legend(["Instantaneous", "Average"])
    fig1.set_size_inches(5.47, 6.36)
    fig1.savefig(RTALK / "d" / "fi.svg")
    # plot fi data
    p = next(ddir.glob("*cell1*22.abf"))
    fi = ABF(p)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey="row", 
                            tight_layout=True, gridspec_kw={"height_ratios": [1, 5]})
    plot_ic_on_axs(axs[:, 0], fi, 0, lblx=True, lbly=True, color="m")
    plot_ic_on_axs(axs[:, 1], fi, 20, lblx=True, lbly=False, color="m")
    plot_ic_on_axs(axs[:, 2], fi, 25, lblx=True, lbly=False, color="m")
    fig.set_size_inches(6, 6.36)
    fig.savefig(RTALK / "d" / "traces.svg")
    fig, axs = plt.subplots()
    ic_gapless = [ABF(p) for p, t in get_dir_info(ddir, "cell1").items() if t == TrialType.ic_gapless]
    print(len(ic_gapless))
    fig, axs = plt.subplots(2, 1)
    data = ic_gapless[0]
    plot_ic_on_axs(axs, data, 0, color="m")
    fig, axs = plot_psp_traces(data, 2, 5.25)
    fig.savefig(RTALK / "d" / "nothing.svg")

def plot_e():
    "plots the d sample traces and data"
    edir = PETER_DIR / "2022-05-19"
    # make fi
    fis = [ABF(p) for p, t in get_dir_info(edir, "cell1").items() if t == TrialType.fi]
    # inspect_fis(fis, 0, 5, thresh=7, spike_detect_params={"dv_cutoff": 10, "height": -10})
    # plt.show(); quit()
    fig1, axs = plt.subplots()#1, 2, tight_layout=True)
    # plot_neg_inj_on_ax(axs[0], fis, 0, 0, thresh=7, last_abf=5)
    plot_fi_on_ax(axs, fis, 0, 5, thresh=7, spike_detect_params={"dv_cutoff": 15, "height": -0})
    # plot_fi_on_ax(axs[1], fis, 0, 5, thresh=7, spike_detect_params={"dv_cutoff": 15, "height": -0}, inst=False, color="b")
    # axs[1].legend(["Instantaneous", "Average"])
    fig1.set_size_inches(5.47, 6.36)
    fig1.savefig(RTALK / "e" / "fi.svg")
    # plt.show(); quit()
    # plot fi data
    p = next(edir.glob("*cell1*27.abf"))
    fi = ABF(p)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey="row", 
                            tight_layout=True, gridspec_kw={"height_ratios": [1, 5]})
    plot_ic_on_axs(axs[:, 0], fi, 0, lblx=True, lbly=True, color="m")
    plot_ic_on_axs(axs[:, 1], fi, 4, lblx=True, lbly=False, color="m")
    plot_ic_on_axs(axs[:, 2], fi, 7, lblx=True, lbly=False, color="m")
    fig.set_size_inches(6, 6.36)
    fig.savefig(RTALK / "e" / "traces.svg")
    fig, axs = plt.subplots()
    ic_gapless = []
    for p, t in get_dir_info(edir, "cell1").items():
        if t == TrialType.ic_gapless:
            try:
                ic_gapless.append(ABF(p))
            except ValueError:
                print(f"cannot load {p} :(")

    print(len(ic_gapless))
    data=ic_gapless[1]
    fig, axs = plt.subplots(2, 1)
    plot_ic_on_axs(axs, ic_gapless[1], 0, color="m")
    fig, axs = plot_psp_traces(data, 10, 16.2)
    fig.savefig(RTALK / "e" / "nothing.svg")
    
def plot_h():
    "plots the d sample traces and data"
    hdir = PETER_DIR / "2022-05-27"
    jdir = PETER_DIR / "2022-05-27"
    # make fi
    fis = [ABF(p) for p, t in get_dir_info(hdir, "cell1").items() if t == TrialType.fi]
    # inspect_fis(fis, 0, 4)
    # plt.show(); quit()
    fig1, axs = plt.subplots()#1, 2, tight_layout=True)
    # plot_neg_inj_on_ax(axs[0], fis, 0, 6, thresh=7, last_abf=3, debug_plot=False)
    plot_fi_on_ax(axs, fis, 0, 3, thresh=7, spike_detect_params={"dv_cutoff": 15, "height": -0}, debug_plot=False)
    # plot_fi_on_ax(axs[1], fis, 0, 3, thresh=7, spike_detect_params={"dv_cutoff": 15, "height": -0},
                  # debug_plot=False, inst=False, color="b")
    # axs[1].legend(["Instantaneous", "Average"])
    fig1.set_size_inches(5.47, 6.36)
    fig1.savefig(RTALK / "h" / "fi.svg")

    ic_gapless = []
    for p, t in get_dir_info(hdir, "cell1").items():
        if t == TrialType.ic_gapless:
            try:
                ic_gapless.append(ABF(p))
            except ValueError:
                print(f"cannot load {p} :(")

    print(len(ic_gapless))
    data=ic_gapless[2]
    fig, axs = plot_psp_traces(data, 45, 46.4)
    fig.savefig(RTALK / "h" / "nothing.svg")
    fig, axs = plt.subplots(3, 1, sharex=True, sharey="row", 
                            tight_layout=True, gridspec_kw={"height_ratios": [1, 5, 5]})
    # share Ys
    axs[1].get_shared_y_axes().join(axs[1], axs[2])
    jopto = [ABF(p) for p, t in get_dir_info(jdir, "cell3").items() if t == TrialType.opto]
    plot_opto_on_axs(axs, jopto[-1], 1, lbly=True, color="m")
    hopto = [ABF(p) for p, t in get_dir_info(hdir, "cell1").items() if t == TrialType.opto]
    print(len(hopto))
    hopto[2].setSweep(3)
    plot_opto_on_axs([axs[0], axs[2]], hopto[2], 0, lblx=True, lbly=True, color="m")
    fig.savefig(RTALK / "h" / "opto.svg")

    # plot fi data
    p = next(hdir.glob("*cell1*15.abf"))
    fi = ABF(p)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey="row", 
                            tight_layout=True, gridspec_kw={"height_ratios": [1, 5]})
    plot_ic_on_axs(axs[:, 0], fi, 6, lblx=True, lbly=True, color="m")
    plot_ic_on_axs(axs[:, 1], fi, 14, lblx=True, lbly=False, color="m")
    plot_ic_on_axs(axs[:, 2], fi, 18, lblx=True, lbly=False, color="m")
    fig.set_size_inches(6, 6.36)
    fig.savefig(RTALK / "h" / "traces.svg")
    fig, axs = plt.subplots()

def plot_f():
    "plots data for cell F"
    fdir = PETER_DIR / "2022-05-24"
    fcell = "cell2"
    ic_gapless = []
    for p, t in get_dir_info(fdir, fcell).items():
        if t == TrialType.ic_gapless:
            try:
                ic_gapless.append(ABF(p))
            except ValueError:
                print(f"cannot load {p} :(")
    ic_gapless = sort_abfs(ic_gapless)
    print(len(ic_gapless))
    data = ic_gapless[0]
    fig, axs = plt.subplots(2, 1, sharex=True)
    # plot_ic_on_axs(axs, data, 0, color="m")
    # plt.show(); quit()
    fig, axs = plot_psp_traces(data, 25, 29.6)
    fig.savefig(RTALK / "f" / "nothing.svg")
    # plt.show(); quit()

    fis = [ABF(p) for p, t in get_dir_info(fdir, fcell).items() if t == TrialType.fi]
    fis = sort_abfs(fis)
    print(*enumerate(Path(f.abfFilePath).name for f in fis), sep="\n")

    # plot sample_trace
    fi = fis[15]
    fig, axs = plt.subplots(2, 3, sharex=True, sharey="row", 
                            tight_layout=True, gridspec_kw={"height_ratios": [1, 5]})
    plot_ic_on_axs(axs[:, 0], fi, 2, lblx=True, lbly=True, color="m")
    plot_ic_on_axs(axs[:, 1], fi, 14, lblx=True, lbly=False, color="m")
    plot_ic_on_axs(axs[:, 2], fi, 18, lblx=True, lbly=False, color="m")
    fig.set_size_inches(6, 6.36)
    fig.savefig(RTALK / "f" / "traces.svg")

    # make fi
    # inspect_fis(fis, 4, 12)
    # plt.show(); quit()
    fig1, axs = plt.subplots()#1, 2, tight_layout=True)
    # plot_neg_inj_on_ax(axs[0], fis, current_example=4, current_ind=2, last_abf=12, debug_plot=False)
    plot_fi_on_ax(axs, fis, current_example=4, last_abf=12, debug_plot=False)
    # plot_fi_on_ax(axs[1], fis, current_example=4, last_abf=12, debug_plot=False, inst=False, color="b")
    # axs[1].legend(["Instantaneous", "Average"])
    fig1.set_size_inches(5.47, 6.36)
    fig1.savefig(RTALK / "f" / "fi.svg")

def plot_hvc():
    "plots h in voltage clamp"
    hdir = PETER_DIR / "2022-05-27"
    hopto = [ABF(p) for p, t in get_dir_info(hdir, "cell1").items() if t == TrialType.opto]
    fig, axs = plt.subplots(2, 1, sharex=True, sharey="row", 
                            tight_layout=True, gridspec_kw={"height_ratios": [1, 5]})
    vclamp = hopto[-2]
    data = vclamp.data
    light = data[3]
    current = data[0]
    t = np.linspace(0, light.size / vclamp.dataRate, light.size)
    rising_inds = np.nonzero(np.diff(light) > 2)[0]
    min_ind_period = int(np.min(np.diff(rising_inds)))
    # rising_edges = t[:-1][np.diff(light) >2]
    # pulse_per_trial = 10
    # rising_edges_2d = rising_edges.reshape(pulse_per_trial, rising_edges.size//pulse_per_trial)
    # periods = np.diff(rising_edges_2d[0, :])
    # i2d = np.zeros((periods.size, int(np.max(periods) * vclamp.dataRate)))
    # l2d = np.zeros((periods.size, int(np.max(periods) * vclamp.dataRate)))
    # curser = 0
    # for i, period in enumerate(periods):
        # period_index = int(period*vclamp.dataRate)
        # i2d[i, 0:period_index] = current[curser: curser+period_index]
        # l2d[i, 0:period_index] = light[curser: curser+period_index]
        # curser += period_index
    bl = rising_inds[0]
    i2d = np.zeros((rising_inds.size, min_ind_period+bl))
    l2d = np.zeros((rising_inds.size, min_ind_period+bl))
    assert bl < min_ind_period
    for i, rising_ind in enumerate(rising_inds):
        i2d[i, :] = current[rising_ind-bl:rising_ind+min_ind_period]
        l2d[i, :] = light[rising_ind-bl:rising_ind+min_ind_period]
    # axs[1].imshow(l2d, aspect="auto")
    # plt.show(); quit()
    mean_current = i2d.mean(axis=0)
    mean_light = l2d.mean(axis=0)
    t = np.linspace(0, mean_current.size / vclamp.dataRate, mean_current.size)
    # axs[0].plot(t*1000, l2d.mean(axis=0))
    
    edges = (t*1000)[:-1][np.abs(np.diff(mean_light) - 1) > 2]
    windows = edges.reshape(int(len(edges) / 2), 2)
    for window in windows:
        axs[0].axvspan(*window, color="b")
    axs[1].plot(t*1000, mean_current, linewidth=1, color="m")
    axs[0].set_ylabel("Light")
    axs[1].set_ylabel("Membrane current")
    add_scalebar(axs[1], hidey=False, matchx=False, matchy=False, sizex=20, labelx="20 ms", sizey=5, labely="5 pA")
    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


    # for i in range(vclamp.sweepCount):
        # vclamp.setSweep(i)
        # i2d[i] = vclamp.sweepY
    plt.show()

def copy_gapless():
    "copys all of the gapless files into a new folder"
    new_folder = PETER_DIR.parent / "gapless"
    h = PETER_DIR / "2022-05-27", "cell1"
    j = PETER_DIR / "2022-05-27", "cell3"
    e = PETER_DIR / "2022-05-19", "cell1"
    d = PETER_DIR / "2022-05-12", "cell1"
    ic_gapless = []
    for directory, cell in [d]:#[h, j, e, d]:
        for p, t in get_dir_info(directory, cell).items():
            if t == TrialType.ic_gapless:
                try:
                    gapless = (ABF(p))
                except ValueError:
                    print(f"cannot load {p} :(")
                dst = new_folder / f"{p.parent.name}-{cell}" / p.name
                dst.parent.mkdir(exist_ok=True)
                shutil.copy(p, dst)
                fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True, gridspec_kw={"height_ratios": [1, 5]})
                plot_ic_on_axs(axs, gapless, 0, lblx=True, lbly=True, color="m")
                axs[0].set_title(p.name)
                plt.show()

def save_all_sample_traces():
    "saves lowest current injection, first spike, and final current injection for all fi curves"
    cells = ["cell1", "cell2", "cell3", "cell4"]
    directories = [d for d in PETER_DIR.iterdir() if d.is_dir()]
    for directory in directories:
        for cell in cells:
            cell_dir = PETER_DIR / "imgs" / f"{directory.name}_{cell}"
            fis = [ABF(p) for p, t in get_dir_info(directory, cell).items() if t == TrialType.fi]
            for fi in fis:
                if verify_fi_current(fi):
                    print(f"{fi.abfFilePath} failed thresh")
                    continue
                fig, axs = plt.subplots(2, 3, sharex=True, sharey="row", 
                                        tight_layout=True, gridspec_kw={"height_ratios": [1, 5]})
                plot_ic_on_axs(axs[:, 0], fi, 0, lblx=True, lbly=True, color="r")
                freqs = get_fi_freqs(fi)
                first_sweep = int(np.searchsorted(np.logical_not(np.isnan(freqs)), 0, "right"))
                if first_sweep == fi.sweepCount:
                    first_sweep = fi.sweepCount-1
                "the first sweep with spikes"
                plot_ic_on_axs(axs[:, 1], fi, first_sweep, lblx=True, color="r")
                plot_ic_on_axs(axs[:, 2], fi, fi.sweepCount-1, lblx=True, color="r")
                cell_dir.mkdir(exist_ok=True, parents=True)
                plt.savefig(cell_dir / Path(fi.abfFilePath).with_suffix(".png").name)
                # save some ram
                fig.clear()
                plt.close(fig)

def main():
    with plt.style.context("dark_background"):
        plot_hvc()
        plot_f()
        plot_d()
        plot_h()
        plot_e()
        plot_j()
        plt.show()

if __name__ == '__main__':
    main()

