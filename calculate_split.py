from matplotlib import pylab as plt
from numpy import arange, cos, sin

def calculate_split(seisR, seisT, azimuth, plot=False, ax=None):
    from numpy import zeros, pi
    from numpy.linalg import eig
    from scipy.signal import tukey

    C = zeros(4).reshape(2, 2)

    phis = arange(-pi / 2., pi / 2., 0.05)
    dts = arange(0.0, 4, 0.05)

    lams = zeros(len(phis) * len(dts)).reshape(len(dts), len(phis))

    minlam = 1E25

    delta=seisR.stats.delta

    for ii, dt in enumerate(dts):
        for jj, phi in enumerate(phis):
            nsamp = int(dt / delta)

            #phi is angle clockwise of seisE

            phir = azimuth - phi

            assert abs(azimuth) < 10

            seis1 =  cos(phir) * seisR.data + sin(phir) * seisT.data
            seis2 = -sin(phir) * seisR.data + cos(phir) * seisT.data

            seis2 = seis2 * tukey(len(seis2), alpha=0.3)
            seis1 = seis1 * tukey(len(seis1), alpha=0.3)

            u2 = seis2[nsamp:]
            u1 = seis1[:len(u2)]

            # u1=u1*tukey(len(u1),alpha=1.0)
            # u1=u1*tukey(len(u1),alpha=1.0)

            # u1=u1*tukey(len(u2))
            # u2=u2*tukey(len(u2))

            c12 = sum(u1 * u2) * delta
            c21 = sum(u2 * u1) * delta
            c11 = sum(u1 * u1) * delta
            c22 = sum(u2 * u2) * delta

            C[0, 0] = c11
            C[1, 0] = c21
            C[0, 1] = c12
            C[1, 1] = c22

            lam, v = eig(C)

            # Get minimum eigenvalue, lambda2 in Silver and Chan
            lams[ii, jj] = min(lam)

            minlam = min(minlam, lams[ii, jj])
            if minlam == lams[ii, jj]:
                iimin = ii
                jjmin = jj

    tmp = sum(lams.T - minlam * 1.50 < 0)
    dtmin = max(dts)
    dtmax = min(dts)
    for ii, each in enumerate(tmp):
        if each > 0:
            dtmin = min([dtmin, dts[ii]])
            dtmax = max([dtmax, dts[ii]])
            continue

    if plot:
        ax.contourf(dts, phis * 180 / pi, lams.T, 25)
        ax.plot(dts[iimin], phis[jjmin] * 180 / pi, '+', color='white', markersize='10', zorder=500000)
        # plt.colorbar()
        if minlam > 0:
            print('min. lambda 2 = %e' % minlam)
            h = plt.contour(dts, phis * 180 / pi, lams.T, levels=[minlam * 1.00, minlam * 2.00], colors=('w'))
        #ax.set_ylabel('Angle (degrees)')
        #ax.set_xlabel('Split Time (s)')
        ax.set_ylabel(r'$\phi^S$',rotation='horizontal')
        ax.set_xlabel(r'$\delta t$')
        ax.tick_params(labelsize=6)
        ax.yaxis.set_label_coords(-0.05, 1.02)
        ax.xaxis.set_label_coords(1.02, -0.05)
        plt.xticks(rotation=90)
        plt.yticks(rotation=90)
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_horizontalalignment("right")
        # plt.title('Split Range: %.1f - %.1f s' % (dtmin, dtmax))
        # plt.show()

    return dts[iimin], phis[jjmin]