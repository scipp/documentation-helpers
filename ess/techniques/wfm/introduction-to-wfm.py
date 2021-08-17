import numpy as np
import scipp as sc
import ess.wfm as wfm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import inspect
import io


class Figure:
    def __init__(self, func=None, fig=None):
        self._func = func
        if fig is None:
            self._fig = self._func()
        else:
            self._fig = fig
        self._fig.set_size_inches(6.5, 4.75)

    def _ipython_display_(self):
        return self._fig.canvas._ipython_display_()

    def show_source(self):
        print(inspect.getsource(self._func))

    def _repr_png_(self):
        buf = io.BytesIO()
        self._fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(self._fig)
        buf.seek(0)
        return buf.getvalue()


def figure1():
    t_P = sc.scalar(2.860e+03, unit='us')
    detector_position = sc.vector(value=[0., 0., 60.0], unit='m')
    z_det = sc.norm(detector_position).value

    fig, ax = plt.subplots()
    ax.add_patch(
        Rectangle((0, 0), t_P.value, -0.05 * z_det, lw=1, fc='grey', ec='k', zorder=10))
    # Indicate source pulse and add the duration.
    ax.text(0,
            -0.05 * z_det,
            "Source pulse ({} {})".format(t_P.value, t_P.unit),
            ha="left",
            va="top",
            fontsize=8)
    ax.plot([0, 1.0e4], [z_det] * 2, lw=3, color='grey')
    ax.text(0., z_det, 'Detector', ha='left', va='top')
    # Draw 2 neutron paths
    ax.plot([0.02 * t_P.value, 4.0e3], [0, z_det], lw=2, color='r')
    ax.plot([t_P.value, 4.0e3], [0, z_det], lw=2, color='b')
    ax.text(3.7e3, 0.5 * z_det, r'$\lambda_{1}$', ha='left', va='center', color='b')
    ax.text(1.5e3, 0.5 * z_det, r'$\lambda_{2}$', ha='left', va='center', color='r')
    ax.set_xlabel("Time [microseconds]")
    ax.set_ylabel("Distance [m]")
    ax.set_title('Figure 1')
    return fig


def figure2():
    t_P = sc.scalar(2.860e+03, unit='us')
    detector_position = sc.vector(value=[0., 0., 60.0], unit='m')
    t_0 = sc.scalar(5.0e+02, unit='us')
    t_A = (t_0 + t_P).value
    z_det = sc.norm(detector_position).value
    fig, ax = plt.subplots()
    ax.add_patch(
        Rectangle((0, 0), (t_P + t_0).value,
                  -0.05 * z_det,
                  lw=1,
                  fc='grey',
                  ec='k',
                  zorder=10))

    # Indicate source pulse and add the duration.
    ax.text(0, -0.05 * z_det, "Source pulse", ha="left", va="top", fontsize=8)
    ax.plot([0, 3.1e4], [z_det] * 2, lw=3, color='grey')
    ax.text(0., z_det, 'Detector', ha='left', va='top')

    dt = 1000.0
    z_wfm = 15.0
    xmin = 0.0
    for i in range(3):
        xmax = 3000.0 + (i * 2.0 * dt)
        ax.plot([xmin, xmax], [z_wfm] * 2, color='k')
        xmin = xmax + dt
    ax.plot([xmin, 3.1e4], [z_wfm] * 2, color='k')
    ax.text(25000.0, z_wfm, "WFM", ha='left', va='top')
    ax.plot([t_A, 5000.0], [0, z_det], color='r')
    ax.plot([2600.0, 5000.0], [0, z_det], color='b')
    ax.plot([t_A, 13000.0], [0, z_det], color='r')
    ax.plot([2400.0, 13000.0], [0, z_det], color='b')
    ax.plot([t_A, 21500.0], [0, z_det], color='r')
    ax.plot([2200.0, 21500.0], [0, z_det], color='b')

    ax.set_xlabel("Time [microseconds]")
    ax.set_ylabel("Distance [m]")
    ax.set_title('Figure 2')
    return fig


def figure3():
    x = np.linspace(0, 5.0, 100)
    a = 4.0
    b = 0.0
    c = 1.5e10
    d = 3.0
    e = 3.0
    y = 2.0 * c / (np.exp(-a * (x - b)) + 1.0) - c
    n = 60
    y2 = c * np.exp(-e * (x - d))
    y[n:] = y2[n:]
    fig, ax = plt.subplots()
    ax.plot(x, y, lw=2, color='k')

    i1 = 5
    i2 = 65
    ax.fill([x[i1]] + x[i1:i2].tolist() + [x[i2 - 1]], [0] + y[i1:i2].tolist() + [0],
            alpha=0.3)

    fs = 15

    ax.axvline(x[i1], color='k')
    ax.axvline(x[i2 - 1], color='k')
    ax.text(x[i1], 1.58e10, r' $t_{0}$', ha='left', va='top', fontsize=fs)
    ax.text(x[i2 - 1], 1.58e10, r' $t_{\rm A}$', ha='left', va='top', fontsize=fs)
    ax.plot([4.5] * 2, [0, 0.3e10], color='k')
    ax.text(4.5, 0.3e10, r'$t_{\rm B}$', ha='center', va='bottom', fontsize=fs)
    ax.annotate(text='',
                xy=(x[i1], 0.7e10),
                xytext=(x[i2 - 1], 0.7e10),
                arrowprops=dict(arrowstyle='<->'))
    ax.text(0.5 * (x[i1] + x[i2 - 1]),
            0.7e10,
            'utilised\n pulse length',
            ha='center',
            va='bottom',
            fontsize=fs)
    ax.text(0.5 * (x[i1] + x[i2 - 1]),
            0.7e10,
            r'$t_{\rm P}$',
            ha='center',
            va='top',
            fontsize=fs)

    ax.set_ylim(0., 1.6e10)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(r"Flux density $[{\rm n/s/cm}^2]$")
    ax.set_title('Figure 3')
    return fig


def figure4():
    coords = wfm.make_fake_beamline(nframes=1,
                                    chopper_positions={
                                        "WFMC1": sc.vector(value=[0.0, 0.0, 4.5],
                                                           unit='m'),
                                        "WFMC2": sc.vector(value=[0.0, 0.0, 5.5],
                                                           unit='m')
                                    })
    coords['position'] = sc.vector(value=[0., 0., 15.], unit='m')
    ds = sc.Dataset(coords=coords)

    z_det = sc.norm(ds.coords["position"]).value
    t_0 = ds.coords["source_pulse_t_0"].value
    t_A = (ds.coords["source_pulse_length"] + ds.coords["source_pulse_t_0"]).value
    t_B = (ds.coords["source_pulse_length"] + 2.0 * ds.coords["source_pulse_t_0"]).value
    z_foc = 12.0
    tmax_glob = 1.4e4
    height = 0.02

    chopper_wfm1 = coords["choppers"].value["WFMC1"]
    chopper_wfm2 = coords["choppers"].value["WFMC2"]

    fig, ax = plt.subplots()
    ax.add_patch(
        Rectangle((0, 0), t_B, -height * z_det, lw=1, fc='lightgrey', ec='k',
                  zorder=10))
    ax.add_patch(
        Rectangle((ds.coords["source_pulse_t_0"].value, 0),
                  ds.coords["source_pulse_length"].value,
                  -height * z_det,
                  lw=1,
                  fc='grey',
                  ec='k',
                  zorder=10))

    # Indicate source pulse and add the duration.
    ax.text(ds.coords["source_pulse_t_0"].value,
            -height * z_det,
            r"$t_{0}$",
            ha="center",
            va="top",
            fontsize=8)
    ax.text(t_A, -height * z_det, r"$t_{A}$", ha="center", va="top", fontsize=8)
    ax.text(t_B, -height * z_det, r"$t_{B}$", ha="center", va="top", fontsize=8)

    z_wfm = sc.norm(0.5 * (chopper_wfm1.position + chopper_wfm2.position)).value
    xmin = chopper_wfm1.time_open.values
    xmax = chopper_wfm1.time_close.values
    dt = xmax - xmin

    ax.plot([0, xmin], [z_wfm] * 2, color='k')
    ax.plot([xmax, tmax_glob], [z_wfm] * 2, color='k')
    ax.text(tmax_glob, z_wfm, "WFMC", ha='right', va='top')

    slope_lambda_max = z_wfm / (xmin - t_0)
    slope_lambda_min = z_wfm / (xmax - t_A)
    int_lambda_max = z_wfm - slope_lambda_max * xmin
    int_lambda_min = z_wfm - slope_lambda_min * xmax
    x_lambda_max = (z_det - int_lambda_max) / slope_lambda_max
    x_lambda_min = (z_det - int_lambda_min) / slope_lambda_min

    ax.plot([t_0, x_lambda_max, x_lambda_max + dt, t_0 + dt], [0.0, z_det, z_det, 0],
            color='C0')
    ax.plot([t_A, x_lambda_min, x_lambda_min - dt, t_A - dt], [0.0, z_det, z_det, 0],
            color='C2')

    x_lambda_max_foc = (z_foc - int_lambda_max) / slope_lambda_max + dt
    x_lambda_min_foc = (z_foc - int_lambda_min) / slope_lambda_min - dt
    ax.plot([0, x_lambda_min_foc], [z_foc] * 2, color='k')
    ax.plot([x_lambda_max_foc, tmax_glob], [z_foc] * 2, color='k')
    ax.text(0.0, z_foc, "FOC", ha='left', va='top')

    slope_lambda_min_prime = z_wfm / (xmin - t_B)
    slope_lambda_max_prime = z_wfm / xmax
    int_lambda_max_prime = z_wfm - slope_lambda_max_prime * xmax
    int_lambda_min_prime = z_wfm - slope_lambda_min_prime * xmin
    x_lambda_max_prime = (z_foc - int_lambda_max_prime) / slope_lambda_max_prime
    x_lambda_min_prime = (z_foc - int_lambda_min_prime) / slope_lambda_min_prime

    ax.plot([t_B, x_lambda_min_prime], [0.0, z_foc], color='k', ls='dashed', lw=1)
    ax.plot([0, x_lambda_max_prime], [0.0, z_foc], color='k', ls='dashed', lw=1)

    ax.text(x_lambda_min - dt,
            z_det,
            r'$\lambda_{\rm min}$',
            ha='right',
            va='top',
            color='C2')
    ax.text(x_lambda_max + dt,
            z_det,
            r'$\lambda_{\rm max}$',
            ha='left',
            va='top',
            color='C0')
    ax.text(x_lambda_min_prime,
            z_foc,
            r"$\lambda_{\rm min}^{'}$",
            ha='left',
            va='top',
            color='k')
    ax.text(x_lambda_max_prime,
            z_foc,
            r"$\lambda_{\rm max}^{'}$",
            ha='left',
            va='top',
            color='k')

    ax.plot([xmin] * 2, [z_wfm, z_det + 1.0], lw=1, color='k')
    ax.plot([x_lambda_min - dt] * 2, [z_det, z_det + 1.0], lw=1, color='k')
    ax.plot([x_lambda_min] * 2, [z_det, z_det + 1.0], lw=1, color='k')
    ax.plot([x_lambda_max + dt] * 2, [z_det, z_det + 1.0], lw=1, color='k')
    ax.plot([x_lambda_max] * 2, [z_det, z_det + 1.0], lw=1, color='k')

    ax.fill([t_0 + dt, t_A - dt, 3013.10], [0, 0, 3.2963],
            color='mediumpurple',
            alpha=0.3,
            zorder=-2)
    ax.fill([x_lambda_min, x_lambda_max, 4515.077], [z_det, z_det, 6.704],
            color='mediumpurple',
            alpha=0.3,
            zorder=-2)

    ax.annotate(text='',
                xy=(xmin, z_det + 0.7),
                xytext=(x_lambda_min - dt, z_det + 0.7),
                arrowprops=dict(arrowstyle='<->'))
    ax.text(0.5 * (xmin + x_lambda_min - dt),
            z_det + 0.7,
            r'$t(\lambda_{\rm min})$',
            va='bottom',
            ha='center')
    ax.text(x_lambda_min - 0.5 * dt,
            z_det + 0.7,
            r'$\Delta t$',
            va='bottom',
            ha='center')
    ax.text(x_lambda_max + 0.5 * dt,
            z_det + 0.7,
            r'$\Delta t$',
            va='bottom',
            ha='center')

    ax.plot([0, tmax_glob], [z_det] * 2, lw=3, color='grey')
    ax.text(0., z_det, 'Detector', ha='left', va='top')

    ax.grid(True, color='lightgray', linestyle="dotted")
    ax.set_axisbelow(True)
    ax.set_xlabel("Time [microseconds]")
    ax.set_ylabel("Distance [m]")
    ax.set_title('Figure 4')
    return fig


def figure5():
    coords = wfm.make_fake_beamline(nframes=1,
                                    chopper_positions={
                                        "WFMC1": sc.vector(value=[0.0, 0.0, 4.5],
                                                           unit='m'),
                                        "WFMC2": sc.vector(value=[0.0, 0.0, 5.5],
                                                           unit='m')
                                    })
    coords['position'] = sc.vector(value=[0., 0., 15.], unit='m')
    ds = sc.Dataset(coords=coords)
    z_det = sc.norm(ds.coords["position"]).value
    frames = wfm.get_frames(ds)
    fig = wfm.plot.time_distance_diagram(ds)
    ax = fig.get_axes()[0]

    chopper_wfm1 = coords["choppers"].value["WFMC1"]
    chopper_wfm2 = coords["choppers"].value["WFMC2"]
    z_wfm = sc.norm(0.5 * (chopper_wfm1.position + chopper_wfm2.position)).value
    xmax = chopper_wfm1.time_close.values
    z_foc = 12.0

    ax.plot([xmax] * 2, [z_wfm, z_det + 1.0], lw=1, color='k')
    ax.plot([0, frames["time_max"].values], [z_wfm] * 2, lw=1, color='k', ls='dotted')
    ax.text(frames["time_max"].values, z_wfm, r'$z_{\rm WFM}$', ha='left', va='center')

    ax.plot([0, 5770.5], [z_foc] * 2, color='k')
    ax.plot([9578.9, frames["time_max"].values], [z_foc] * 2, color='k')
    ax.text(frames["time_max"].values, z_foc, 'FOC', ha='right', va='bottom')

    ax.plot([(frames["time_min"] + frames["delta_time_min"]).values] * 2,
            [z_det, z_det + 1.0],
            lw=1,
            color='k')
    ax.plot([frames["time_min"].values] * 2, [z_det, z_det + 1.0], lw=1, color='k')
    ax.plot([(frames["time_max"] - frames["delta_time_max"]).values] * 2,
            [z_det, z_det + 1.0],
            lw=1,
            color='k')
    ax.plot([frames["time_max"].values] * 2, [z_det, z_det + 1.0], lw=1, color='k')

    xmid = (0.5 * ((frames["time_min"] + frames["time_min"] +
                    frames["delta_time_min"]).data)).values
    ax.plot([xmid] * 2, [z_det, z_det + 0.5], lw=1, color='k')

    ax.annotate(text='',
                xy=(xmax, z_det + 0.4),
                xytext=(xmid, z_det + 0.4),
                arrowprops=dict(arrowstyle='<->'))
    ax.text(0.5 * (xmax + frames["time_min"].values),
            z_det + 0.4,
            r'$t(\lambda_{N})$',
            va='bottom',
            ha='center')
    ax.text(xmid,
            z_det + 1.0,
            r'$\Delta t(\lambda_{N})$',
            va='bottom',
            ha='center',
            color='C2')
    ax.text((0.5 * ((frames["time_max"] + frames["time_max"] -
                     frames["delta_time_max"]).data)).values,
            z_det + 1.0,
            r'$\Delta t(\lambda_{N+1})$',
            va='bottom',
            ha='center',
            color='C0')
    ax.plot([xmax, xmid], [z_wfm, z_det], lw=1, ls='dashed', color='k')
    ax.text(frames['time_min'].values,
            z_det,
            r'$\lambda_{N}$   ',
            ha='right',
            va='top',
            color='C2')
    ax.text(frames['time_max'].values,
            z_det,
            r'$\lambda_{N+1}$',
            ha='left',
            va='top',
            color='C0')

    ax.lines[4].set_color('C2')
    ax.patches[2].set_color('mediumpurple')
    ax.set_xlim(-400, 12500)
    ax.set_title('Figure 5')
    return fig


def figure6():
    coords = wfm.make_fake_beamline(nframes=2,
                                    chopper_positions={
                                        "WFMC1": sc.vector(value=[0.0, 0.0, 4.5],
                                                           unit='m'),
                                        "WFMC2": sc.vector(value=[0.0, 0.0, 5.5],
                                                           unit='m')
                                    })
    coords['position'] = sc.vector(value=[0., 0., 15.], unit='m')
    ds = sc.Dataset(coords=coords)
    frames = wfm.get_frames(ds)
    fig = wfm.plot.time_distance_diagram(ds)
    ax = fig.get_axes()[0]

    chopper_wfm1 = coords["choppers"].value["WFMC1"]
    chopper_wfm2 = coords["choppers"].value["WFMC2"]
    z_wfm = sc.norm(0.5 * (chopper_wfm1.position + chopper_wfm2.position)).value
    z_det = sc.norm(ds.coords["position"]).value

    ax.plot([0, frames["time_max"].values[-1]], [z_wfm] * 2,
            lw=1,
            color='k',
            ls='dotted')
    ax.text(frames["time_max"].values[-1],
            z_wfm,
            r'$z_{\rm WFM}$',
            ha='left',
            va='center')

    ax.plot([(frames["time_min"] + frames["delta_time_min"]).values] * 2,
            [z_det, z_det + 1.0],
            lw=1,
            color='k')
    ax.plot([frames["time_min"].values] * 2, [z_det, z_det + 1.0], lw=1, color='k')
    ax.plot([(frames["time_max"] - frames["delta_time_max"]).values] * 2,
            [z_det, z_det + 1.0],
            lw=1,
            color='k')
    ax.plot([frames["time_max"].values] * 2, [z_det, z_det + 1.0], lw=1, color='k')

    xmid_min = (0.5 * ((frames["time_min"] + frames["time_min"] +
                        frames["delta_time_min"]).data)).values
    xmid_max = (0.5 * ((frames["time_max"] + frames["time_max"] -
                        frames["delta_time_max"]).data)).values

    ax.text(xmid_min[0],
            z_det + 1.0,
            r'$\lambda_{N=1}$',
            va='bottom',
            ha='center',
            color='C2')
    ax.text(xmid_max[0],
            z_det + 1.0,
            r'$\lambda_{2}$',
            va='bottom',
            ha='center',
            color='C0')
    ax.text(xmid_min[1],
            z_det + 1.0,
            r'$\lambda_{2}$',
            va='bottom',
            ha='center',
            color='C0')
    ax.text(xmid_max[1],
            z_det + 1.0,
            r'$\lambda_{3}$',
            va='bottom',
            ha='center',
            color='C1')

    ax.lines[6].set_color('C2')
    ax.lines[8].set_color('C0')
    ax.patches[2].set_color('mediumpurple')
    ax.patches[5].set_color('grey')
    return fig


Figure1 = Figure(func=figure1)
Figure2 = Figure(func=figure2)
Figure3 = Figure(func=figure3)
Figure4 = Figure(func=figure4)
Figure5 = Figure(func=figure5)
Figure6 = Figure(func=figure6)

if sc.plotting.is_doc_build:
    setattr(Figure1, "_ipython_display_", None)
    setattr(Figure2, "_ipython_display_", None)
    setattr(Figure3, "_ipython_display_", None)
    setattr(Figure4, "_ipython_display_", None)
    setattr(Figure5, "_ipython_display_", None)
    setattr(Figure6, "_ipython_display_", None)
