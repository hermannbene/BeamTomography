def genPoly(nWires,radius,scanLength,offAngle,center):
    '''
    this function generates a polygon path for wire scan tomography. 
    Input
    center: np.array([x,y,z,rx,ry,rz]) in um,deg
    nWires: int 
    radius: flaot um
    scanLength: flaot um
    offAngle: offset angle of the wire star, 0 means wire 0 is horizontal (aligned with pos. x axis). in deg

    Output
    A list of length 2*nWires is returned. Points in the same format as center are returned in the following order:
    start wire 0, end wire 0, start wire 1, end wire 1, ... 
    '''
    scanPointsPolygon = []
    if normalAxis == 'x':
        coordinate1 = 1
        coordinate2 = 2

    elif normalAxis == 'y':
        coordinate1 = 0
        coordinate2 = 2

    if normalAxis == 'z':
        coordinate1 = 0
        coordinate2 = 1

    for i in range(nWires):
        beta = 2.*np.pi/nWires*i + offAngle/180*np.pi # angle of i-th wire in rad 
        for j in range(2): # for start and end point
            offset = np.zeros(6)
            offset[coordinate1] = radius*np.cos(beta) + (-1)**(j+1)*scanLength/2.*np.sin(-beta)
            offset[coordinate2] = radius*np.sin(beta) + (-1)**(j+1)*scanLength/2.*np.cos(-beta)
            scanPointsPolygon.append(center+offset)
    return scanPointsPolygon


    

def plotScanPoints(scanPointsPolygon, idX, idY):
    '''
    this function plots a polygon path for wire scan tomography. 
    Input
    (output from genPoly) + The idX changes the mode of the genPoly, plotScanPoints functions (0 for SwissFEL operation, 2 for laser testing)
    A list of scan Points.
    start wire 0, end wire 0, start wire 1, end wire 1, ... 
    '''
    fig = plt.figure('Visualize Scan Path')
    ax = fig.add_subplot(111)
    for i,p in enumerate(scanPointsPolygon):
        ax.plot(p[idX], p[idY], marker='+', color='r')
        ax.annotate(i, (p[idX], p[idY]))
    
    xlabel = 'x'*(idX==0)+'y'*(idX==1)+'z'*(idX==2)+'rx'*(idX==3)+'ry'*(idX==4)+'rz'*(idX==5)+'arb.'*(idX==-1)
    ylabel = 'x'*(idY==0)+'y'*(idY==1)+'z'*(idY==2)+'rx'*(idY==3)+'ry'*(idY==4)+'rz'*(idY==5)+'arb.'*(idY==-1)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig('temp_ScanPoints.png')