import numpy as np
import scipp as sc
import scippneutron as scn
import ess.wfm as wfm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import inspect


class Figure:
    def __init__(self, func):
        self._func = func
        self._fig = self._func()
        self._fig.set_size_inches(6.5, 4.75)

    def _ipython_display_(self):
        return self._fig.canvas._ipython_display_()

    def show_source(self):
        print(inspect.getsource(self._func))


def figure1():
    t_P = sc.scalar(2.860e+03, unit='us')
    detector_position = sc.vector(value=[0., 0., 60.0], unit='m')
    z_det = sc.norm(detector_position).value

    fig0, ax0 = plt.subplots()
    ax0.add_patch(
        Rectangle((0, 0), t_P.value, -0.05 * z_det, lw=1, fc='grey', ec='k', zorder=10))
    # Indicate source pulse and add the duration.
    ax0.text(0,
             -0.05 * z_det,
             "Source pulse ({} {})".format(t_P.value, t_P.unit),
             ha="left",
             va="top",
             fontsize=8)
    ax0.plot([0, 1.0e4], [z_det] * 2, lw=3, color='grey')
    ax0.text(0., z_det, 'Detector', ha='left', va='top')
    # Draw 2 neutron paths
    ax0.plot([0.02 * t_P.value, 4.0e3], [0, z_det], lw=2, color='r')
    ax0.plot([t_P.value, 4.0e3], [0, z_det], lw=2, color='b')
    ax0.text(3.7e3, 0.5 * z_det, r'$\lambda_{1}$', ha='left', va='center', color='b')
    ax0.text(1.5e3, 0.5 * z_det, r'$\lambda_{2}$', ha='left', va='center', color='r')
    ax0.set_xlabel("Time [microseconds]")
    ax0.set_ylabel("Distance [m]")
    ax0.set_title('Figure 1')
    return fig0


Figure1 = Figure(figure1)
