import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
from scipy import signal
from scipy.optimize import curve_fit 
from matplotlib import rc
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 12})
fontSize=12
rc('xtick', labelsize=12) 
rc('ytick', labelsize=12) 

def plotProjections2D(samples, angles,x, y, bins, convFunction=None, fileName = None, microCharge=None):
    f =plt.figure('projections', figsize=(18,5))
    f.clf()
    plotRange=15
    
    gs = GridSpec(3,5)
#    ax0 = f2.add_subplot(121)
    ax00 = f.add_subplot(gs[:, 3:])
#    plt.setp(ax00.get_yticklabels(), visible=False)
    #cax = f.add_subplot(gs[0:3,-1])
    ax11 = f.add_subplot(gs[0, 0])
    ax12 = f.add_subplot(gs[0, 1])
    ax13 = f.add_subplot(gs[0, 2])
    ax21 = f.add_subplot(gs[1, 0])
    ax22 = f.add_subplot(gs[1, 1])
    ax23 = f.add_subplot(gs[1, 2])
    ax31 = f.add_subplot(gs[2, 0])
    ax32 = f.add_subplot(gs[2, 1])
    ax33 = f.add_subplot(gs[2, 2])
                 
    for ax in [ax31, ax32, ax33]:
        ax.set(xLabel = r"$\xi$ (μm)")
    for ax in [ax11,ax12,ax13, ax21, ax22, ax23]:
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in [ax11, ax21, ax31]:
        ax.set(yLabel = r"(arb. unit)")
    for ax in [ax11, ax21, ax31, ax12, ax22, ax32, ax13, ax23, ax33]:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xticks([-10,0,10])
    
    axList = [ax11,ax12,ax13, ax21, ax22, ax23, ax31, ax32, ax33]
    cList = ["#e50000", "#15b01a", "#0343df", "#06c2ac", "#aaff32", "#f97306", "#7e1e9c", "#ff81c0", "#929591"]
             
    for i in range(len(angles)):
       
        axList[i].scatter(x[i], y[i], marker = 'x', color = cList[i])
        cos = np.cos(np.pi*angles[i]/180)
        sin = -np.sin(np.pi*angles[i]/180)
        histConvIList = []
        histIList = []
        for j in range(len(samples)):
            X = samples[j]
            Xrot = sin*X[0]+cos*X[1]
        #    Xrot = X
            hist, b = np.histogram(Xrot, bins=bins[i])
            if microCharge == None:
                hist = hist/np.sum(hist)/(bins[1]-bins[0])
            else:
                hist = np.array(hist, dtype = float)*microCharge/np.average(np.diff(bins[i]))
            
            if not(convFunction is None):
                if sum(convFunction)>0:
                    histConv = signal.convolve(hist, convFunction, mode = "same")
        #            axList[i].plot(x[i]-0.5*(x[i][1]-x[i][0]), histConv, label="Rec. + Conv", ls="--")
#                    histConvI = np.interp(x[i], bins[i][:-1]+0.5*(bins[i][1]-bins[i][0]), histConv)
#                    histConvIList.append(histConvI/np.sum(histConvI))
                    histIList.append(histConv)
            
#            histI = np.interp(x[i], bins[i][:-1]+0.5*(bins[i][1]-bins[i][0]), hist)
#            histIList.append(histI/np.sum(histI))
           
        
#        plotListError(axList[i], histConvIList, x[i], "Rec. + Conv", "--")
        plotListError(axList[i], histIList, bins[i][:-1], r"$\theta = $" + str(int(angles[i])) + "°", "-", color = cList[i])
        axList[i].set_xlim(-plotRange,plotRange)
        axList[i].set_ylim(-0.02,1.1*np.max(np.array(y)))
        axList[i].grid()
#        axList[i].set_xticks([])
#        axList[i].set_yticks([])
        ax00.plot(bins[i][:-1],bins[i][:-1]*np.tan(-np.deg2rad(angles[i])), color = cList[i], linestyle = ":")
        axList[i].legend(loc=1)
    
#    ax00 = plt.subplot2grid((12, 3), (4,0), rowspan=9, colspan=3)
    ax00.hist2d(-X[0], X[1], bins=256, cmap="hot")
#    axDist.scatter(*X[:,:600], s=1, c="g")
    ax00.set_xlim(axList[0].get_xlim())
    ax00.set_ylim(axList[0].get_xlim())
    ax00.set_facecolor('k')
    ax00.set_xlabel("x (um)")
    ax00.set_ylabel("y (um)")
    ax00.set_xlim(-plotRange,plotRange)
    ax00.set_ylim(-plotRange,plotRange)
    sigX = np.std(X[0])
    sigY = np.std(X[1])

    bins = np.linspace(-plotRange, plotRange, 200)
    xHist, bins = np.histogram(X[0], bins=bins)
    yHist, bins = np.histogram(X[1], bins=bins)
    profile = xHist/np.max(xHist)*plotRange/2
    profile-=plotRange
#    ax00.plot(bins[:-1], profile, color='1', lw=1.5)
#    sigX, sigErrX = fitGauss(bins[:-1], profile, ax00)
    
    profile = yHist/np.max(yHist)*plotRange/2
    profile-=plotRange
#    ax00.plot(profile, bins[:-1], color='1', lw=1.5)
#    sigY, sigErrY = fitGauss(bins[:-1], profile, ax00, True)
    
#    ax00.set_title(r'$\sigma_x = $'+str(np.round(sigX, 2))+r'$\pm$'+str(np.round(sigErrX, 2))+' μm, '+
#             r'$\sigma_y = $'+str(np.round(sigY, 2))+r'$\pm$'+str(np.round(sigErrY, 2))+' μm'
#             )
#    plt.pause(0.01)
    ax00.set_aspect(1)
    f.tight_layout()
#    if not(fileName is None):
#        plt.savefig("plot20191214/" + str(fileName) + ".png", dpi=600)#

def gauss(x, a, x0, sigma, offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+offset

def fitGauss(x,y, ax, plotVertical=False):
    a = np.max(y)-np.min(y)
    x0 = x[np.argmax(y)]
    sig = (np.max(x)-np.min(x))/10
    offset = np.min(y)
    p0 = [a, x0, sig, offset]
    popt, perr = curve_fit(gauss, x, y, p0=p0)
    xFit = np.linspace(min(x), max(x), 300)  
    yFit = gauss(xFit, *popt)
    if plotVertical:
        ax.plot(yFit, xFit, color='c', ls='--', lw=1.5)
    else:        
        ax.plot(xFit, yFit, color='c', ls='--', lw=1.5)
    return popt[2], perr[2,2]**0.5  
    

def plotListError(ax, samples, x, label, ls, color = "k"):
    historyArray = np.array(samples)  
    avg = np.average(historyArray, axis=0)
#    errMax = np.max(historyArray, axis = 0)-avg
#    errMin = avg-np.min(historyArray, axis = 0)
    errMax = np.std(historyArray, axis = 0)
    errMin = np.std(historyArray, axis = 0)
#    ax.errorbar(x, avg, [errMin, errMax], label = label, ls = ls)
    ax.plot(x, avg, label = label, ls = ls, color=color)
    ax.fill_between(x, avg-errMin, avg+errMax, facecolor = "grey", alpha=0.5, zorder=-1)
    
#    print(np.min(avg))
#    print(errMax[np.argmin(avg)])
    
def plotError2D(X, samples, fileName = None):
    fig =plt.figure('error2D', figsize=(10,5), dpi=100)
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    hist = []
    for Xs in samples:
        h, xedges, yedges = np.histogram2d(*Xs, bins = 128)
        hist.append(h.T)
    
    XHist, xedges, yedges = np.histogram2d(*X, bins = 128)
    XHist = XHist.T
    ax1.imshow(XHist, origin='lower', cmap="nipy_spectral", vmax=np.max(XHist), vmin=0)
    ax2.imshow(np.std(hist, axis=0), origin='lower', cmap="nipy_spectral",vmax=np.max(XHist), vmin=0)
    return 
    

def propagateX(X, s):
    Xnew = X.copy()
    Xnew[0] = X[0]+s*X[2]
    Xnew[1] = X[1]+s*X[3]
    return Xnew
    
def plotProjections4D(samples, Z, Angles,D, s, bins, convFunction=None, fileName = None, microCharge=None):
    fig =plt.figure('projections', figsize=(15,10), dpi=150)
    fig.clf()
#    nZ, nAngles, nD = D.shape
    nZ = len(Z)
    nAngles = len(Angles)
    gs = mgs.GridSpec(nAngles, nZ)
    
    projectionAxs = []
    for i in range(nZ):
        axs = []
        for j in range(nAngles):
            axs.append(fig.add_subplot(gs[j, i]))
        
        projectionAxs.append(axs)
    projectionAxs = np.array(projectionAxs)

#    Xprop = propagateX(X, Z[-1])
   
    for i in range(nZ):
        
        projectionAxs[i,0].set_title(r"$z$ = "+str(np.round(Z[i]*100,1))+' cm', fontsize=fontSize)
        projectionAxs[i,-1].set_xlabel(r'$\xi$ (μm)', fontsize=fontSize)
        
        for j in range(nAngles):
            projectionAxs[0,j].set_ylabel(r"$\theta$ = "+str(int(Angles[j]))+r'$^\circ$', fontsize=fontSize)
        
            projectionAxs[i,j].plot(D[i][j], s[i][j], label="Meas.", marker = 'x')
            cos = np.cos(np.pi*Angles[j]/180)
            sin = -np.sin(np.pi*Angles[j]/180)
            
            histIList = []
            for k in range(len(samples)):
                X = samples[k]
                Xprop = propagateX(X, Z[i])
                Xrot = sin*Xprop[0]+cos*Xprop[1]

                hist, b = np.histogram(Xrot, bins=bins, density=False)
                if microCharge == None:
                    hist = hist/np.sum(hist)/(bins[1]-bins[0])
                else:
                    hist = np.array(hist, dtype = float)*microCharge/np.average(np.diff(bins))


                if not(convFunction is None) and sum(convFunction)>0:
                    histConv = signal.convolve(hist, convFunction, mode = "same")
        #            axs[i].plot(x[i]-0.5*(x[i][1]-x[i][0]), histConv, label="Rec. + Conv", ls="--")
                    histConvI = np.interp(D[i][j], bins[:-1]+0.5*(bins[1]-bins[0]), histConv)
    #                histConvI= histConvI/np.sum(histConvI)            
#                    projectionAxs[i,j].plot(D[i][j], histConvI, label="Rec. + Conv", ls="--")
                    histIList.append(histConv)
            
                histI = np.interp(D[i][j], bins[:-1]+0.5*(bins[1]-bins[0]), hist)
    #            histI = histI/np.sum(histI)
                if convFunction is None:
                    histIList.append(hist)
#            projectionAxs[i,j].plot(D[i][j], histI, label="Rec.")
            plotListError(projectionAxs[i,j], histIList, bins[:-1]+0.5*(bins[1]-bins[0]), "Rec.", "--")
            projectionAxs[i,j].set_xlim(-10,10)
            if j<nAngles-1:
                projectionAxs[i,j].set_xticklabels([])
            projectionAxs[i,j].set_yticks([])
            maxS = np.max([np.max(amp) for amp in s])
            projectionAxs[i,j].set_ylim(0,maxS*1.05)
            projectionAxs[i,j].xaxis.grid(True)
            

#    projectionAxs[int(nZ/2),0].set_title(r'$\leftarrow z \rightarrow$')
#    projectionAxs[0,int(nAngles/2)].set_ylabel(r'$\leftarrow \theta \rightarrow$')
          
    fig.tight_layout()
    plt.pause(0.01)


def plotPhaseSpace4D(X, Z):
    fig =plt.figure('phaseSpace', figsize=(15,9), dpi=150)
    fig.clf()
#    nZ, nAngles, nD = D.shape
    nZ = len(Z)

    gs = mgs.GridSpec(3, nZ)
    
    projectionAxs = []
    for i in range(nZ):
        axs = []
        for j in range(3):
            axs.append(fig.add_subplot(gs[j, i]))
        
        projectionAxs.append(axs)
    projectionAxs = np.array(projectionAxs)
        
    for i,z in enumerate(Z):
        Xprop = propagateX(X, z)
        projectionAxs[i,0].hist2d(Xprop[0], Xprop[2], bins=256, cmap='hot')
        twissGauss(Xprop[0]*1e-6, Xprop[2]*1e-6, projectionAxs[i,0])
        
        projectionAxs[i,1].hist2d(Xprop[1], Xprop[3], bins=256, cmap='hot')
        twissGauss(Xprop[1]*1e-6, Xprop[3]*1e-6, projectionAxs[i,1])
        
        projectionAxs[i,2].hist2d(Xprop[0], Xprop[1], bins=256, cmap='hot')
        twissGauss(Xprop[0]*1e-6, Xprop[1]*1e-6, projectionAxs[i,2])
        print(np.corrcoef(Xprop[0], Xprop[1]))
        
    for i in range(nZ):
        for j in range(3):
            projectionAxs[i,j].set_xlim(-10,10)
            projectionAxs[i,j].set_facecolor('k')
        
        projectionAxs[i,0].set_title(r"$z$ = "+str(np.round(Z[i]*100,1))+' cm', fontsize=fontSize)
        projectionAxs[i,0].set_xlabel(r"$x$ (μm)", fontsize=fontSize)
        if i==0:
            projectionAxs[i,0].set_ylabel(r"$x´$ (μrad)", fontsize=fontSize)
            projectionAxs[i,1].set_ylabel(r"$y´$ (μrad)", fontsize=fontSize)
            projectionAxs[i,2].set_ylabel(r"$y$ (μm)", fontsize=fontSize)
        else:
            projectionAxs[i,0].set_yticks([])
            projectionAxs[i,1].set_yticks([])
            projectionAxs[i,2].set_yticks([])
        projectionAxs[i,1].set_xlabel(r"$y$ (μm)", fontsize=fontSize)
        projectionAxs[i,2].set_xlabel(r"$x$ (μm)", fontsize=fontSize)
        projectionAxs[i,2].set_ylim(-10,10)
              
    fig.tight_layout()
    plt.pause(0.01)

def plotTwissEvolution(samples, Z, zMeas):
    fig =plt.figure('twissEvolution', figsize=(6,8), dpi=150)
    fig.clf()
    fig2 =plt.figure('twissEvolution2', figsize=(6,4), dpi=150)
    fig2.clf()
    
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax4 = fig2.add_subplot(111)
    
    nZ = len(Z)
    
    p = 6400
    EmittanceX, EmittanceY, BetaX, BetaY = [], [], [], []
    for i,z in enumerate(Z):
        emittanceX, emittanceY, betaX, betaY = [], [], [], []
        for j, X in enumerate(samples):
            print(i, j)
            Xprop = propagateX(samples[j], z)
            
            eX, bX = twissGauss(Xprop[0]*1e-6, Xprop[2]*1e-6, None)
            eY, bY = twissGauss(Xprop[1]*1e-6, Xprop[3]*1e-6, None)
            emittanceX.append(eX)
            emittanceY.append(eY)
            betaX.append(bX)
            betaY.append(bY)
        EmittanceX.append(emittanceX)
        EmittanceY.append(emittanceY)
        BetaX.append(betaX)
        BetaY.append(betaY)
    
    EmittanceX = np.array(EmittanceX).T
    EmittanceY = np.array(EmittanceY).T
    BetaX = np.array(BetaX).T
    BetaY = np.array(BetaY).T

#    ax1.plot(Z, 1e9*np.array(emittanceX)*p, label = r'$\varepsilon_{n,x}$')
#    ax2.plot(Z, np.array(betaX), label = r'$\beta_{x}$')
#    ax3.plot(Z, 1e6*np.sqrt(np.array(betaX)*np.array(emittanceX)), label = r'$\sigma_{x}$')
#    
#    ax1.plot(Z, 1e9*np.array(emittanceY)*p, label = r'$\varepsilon_{n,x}$')
#    ax2.plot(Z, np.array(betaY), label = r'$\beta_{y}$')
#    ax3.plot(Z, 1e6*np.sqrt(np.array(betaY)*np.array(emittanceY)), label = r'$\sigma_{y}$')
    plotListError(ax1, 1e9*EmittanceX*p, Z*1e2, r'$\varepsilon_{n,x}$', '-')
    plotListError(ax1, 1e9*EmittanceY*p, Z*1e2, r'$\varepsilon_{n,y}$', '-')
    plotListError(ax2, BetaX, Z*1e2, r'$\beta_{x}$', '-')
    plotListError(ax2, BetaY, Z*1e2, r'$\beta_{y}$', '-')
    for ax in [ax3,ax4]:
        plotListError(ax, 1e6*np.sqrt(np.array(BetaX)*np.array(EmittanceX)), Z*1e2, r'$\sigma_{x}$', '-')
        plotListError(ax, 1e6*np.sqrt(np.array(BetaY)*np.array(EmittanceY)), Z*1e2, r'$\sigma_{y}$', '-')
        
    ax1.set_title('Normalized Emittance', fontsize=fontSize)
    ax1.set_ylabel('(nm rad)', fontsize=fontSize)
    ax2.set_title(r'Twiss', fontsize=fontSize)
    ax2.set_ylabel('(m)', fontsize=fontSize)
    for ax in [ax3,ax4]:
        ax.set_title(r'Beam Size', fontsize=fontSize)
        ax.set_ylabel('(μm)', fontsize=fontSize)
    ax4.set_title('')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(loc = 9, fontsize=fontSize)
        ax.set_xlabel(r'$z$ (cm)', fontsize=fontSize)
        for z in zMeas:
            ax.axvline(z*1e2, color = 'k', ls = '--')
#        for z in [0., 2., 4., 6., 8.,]:
#            ax.axvline(z, color = 'k', ls = '-')
            
    fig.tight_layout()
    fig2.tight_layout()
    plt.pause(0.01)


def twiss(X):
    x = X[0]*1e-6
    xp = X[2]*1e-6
    y = X[1]*1e-6
    yp = X[3]*1e-6
    
    p = 6400
    
    xR = np.average(x**2)
    xpR = np.average(xp**2)
    xxpR = np.average(x*xp)
    emiX = np.sqrt(xR*xpR-xxpR**2)
    emiXN = emiX*np.average(p)
    
    yR = np.average(y**2)
    ypR = np.average(yp**2)
    yypR = np.average(y*yp)
    emiY = np.sqrt(yR*ypR-yypR**2)
    emiYN = emiY*np.average(p)
    
    betaX = xR/emiX
    alphaX = -xxpR/emiX
    gammaX = xpR/emiX
    
    betaY = yR/emiY
    alphaY = -yypR/emiY
    gammaY = ypR/emiY
#    print('emitX, emitY')
#    print(emiXN, emiYN)
#    print('alphaX, betaX, gammaX')
#    print(alphaX, betaX, gammaX)
#    print('alphaY, betaY, gammaY')
#    print(alphaY, betaY, gammaY)
#    print(betaX*gammaX-alphaX**2)    
#    print(betaY*gammaY-alphaY**2)
#    print('p avg. ', np.average(p))
    return emiXN, emiYN, betaX, betaY


def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x,y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def twissGauss(x, xp, ax):
  
    p = 6400
    nBins=128
    H, xedges, yedges = np.histogram2d(x, xp, bins=nBins)
    ext = np.array([xedges[0], xedges[-1], yedges[0], yedges[-1]])*1e6
    X,XP = np.meshgrid(xedges[:-1], yedges[:-1])
    H = H.T

    initial_guess = (np.max(H),0,0,np.std(x),np.std(xp),0,0)
    popt, pcov = curve_fit(twoD_Gaussian, (X, XP), H.flatten(), p0=initial_guess)

    t = np.linspace(0, 2*np.pi)
    xEllipse = popt[3]*np.cos(t)
    yEllipse = popt[4]*np.sin(t)
    xEllipseRot = np.cos(popt[5])*xEllipse + np.sin(popt[5])*yEllipse
    yEllipseRot = -np.sin(popt[5])*xEllipse + np.cos(popt[5])*yEllipse
    Hfit = twoD_Gaussian((X,XP), *popt).reshape(nBins, nBins)
#    ax.imshow(H, origin="lower", extent=ext, aspect = "auto")
#    plt.figure()
#    plt.imshow(Hfit, origin="lower", extent=ext, aspect = "auto")
    if ax != None:
        ax.plot(xEllipseRot*1e6, yEllipseRot*1e6, color = 'blue')
    emittance = popt[3]*popt[4]
#    print(emittance*p)
    beta = ((np.max(xEllipseRot)-np.min(xEllipseRot))*0.5)**2/emittance

    return emittance, beta

#twissGauss(rm.X[0]*1e-6, rm.X[2]*1e-6)
#Xprop = propagateX(rm.X, 0.06)
#twissGauss(Xprop[0]*1e-6, Xprop[1]*1e-6)

#plotTwissEvolution(rm.X, np.linspace(0, 1e-1, 30))