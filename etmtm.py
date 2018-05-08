def ETMTM(P,D,TB,NT,dt,tag='data',nalphas=20, Poverlap=0.9, win_len=100):
    """
    Extended time multi-taper deconvolution.
    Originally written for Malab by Ved Lekic.
    Translated to Python by Nick Mancinelli.

    For deconvolving S-to-P receiver functions.

    Inputs:
            P
            D
            TB
            NT
            t0
            t1
            dt
            tag

    Returns: RF_Time

    Findings: when P arrival is not in the center of the window, the
    amplitudes are not unity at the beginning and decreasing from there on.
    Instead they peak at the time shift which corresponds to the middle index
    in the P time window.

    As your TB
    increases, the frequency smearing gets worse, which means that the RFs
    degrate at shorter and shorter lag times. Therefore, as you increase TB,
    you should also increase Poverlap.

    TB = 4; NT = 7; %choise of TB = 4, NT = 3 is supposed to be optimal
    t0 = -5; t1 = max(time);
        function [RF_Time] = MTMDecon_VedOptimized(P,D,TB,NT,t0,t1,Faza)
    Ved wrote MTM for MATLAB, which has the added advantage of
    finding the optimal damping parameter.
    TB  = time bandwidth product (usually between 2 and 4)
    NT  = number of tapers to use, has to be <= 2*TB-1
    """

    # Flip time axis in case of Sp
    from numpy import fliplr, round, zeros, concatenate, arange, where
    from numpy import logspace, var, std, median
    from numpy.fft import fft, ifft, fftshift
    from numpy import shape
    from numpy import hanning
    from scipy.interpolate import interp1d
    from nitime.algorithms.spectral import dpss_windows
    from numpy import array, conj, real
    from scipy.signal import convolve
    from numpy import nanmean, nansum
    from numpy import std, argmin, flipud


    D,P = D[::-1], P[::-1]

    # Length of moving time window in seconds
    Nwin = int(round(win_len/dt))
    
    # Fraction of overlap overlap between moving time windows. As your TB
    # increases, the frequency smearing gets worse, which means that the RFs
    # degrate at shorter and shorter lag times. Therefore, as you increase TB,
    # you should also increase Poverlap.
    nso=(1-Poverlap)*Nwin
    nso=int(round(nso))

    npad=zeros(nso)
    D=concatenate((npad,D,npad))
    P=concatenate((P,npad,npad))

    #MATLAB time = 0:dt:dt*(length(P)-1);
    time = arange(0,len(P))*dt
    t1 = max(time)
    t0 = -5

    # Create moving time windows and daughter/parent snippets
    #  MLAB starts = 1:round((1-Poverlap)*Nwin):length(P)-Nwin+1
    tmp=int(round((1-Poverlap)*Nwin))
    starts = range(1,len(P)-Nwin+1,tmp)
    nd=0

    #init tmp_times
    tmp_times=zeros(len(starts)*Nwin).reshape(len(starts),Nwin)
    Dwin=zeros(Nwin*len(starts)).reshape(len(starts),Nwin)

    for j in range(len(starts)):
        tmp_times[j,:(Nwin-1)] = time[starts[j]:starts[j]+Nwin-1]
        if j==0: # ASSUME THAT PARENT PHASE IS CENTERED IN FIRST WINDOW!
            #Pwin = interp1(double(time),double(P),double(tmp_times(j,:)),'linear',0)';
            Pwin = interp1d(time,P)(tmp_times[j,:])


        #print(j, tmp_times[j,:(Nwin-1)][0], tmp_times[j,:(Nwin-1)][-1])
        #print(len(time), len(D), len(tmp_times[j,:]))
        #Dwin(1:Nwin,j) = interp1(double(time),double(D),double(tmp_times(j,:)),'linear',0);
        Dwin[j,:Nwin] = interp1d(time,D)(tmp_times[j,:])

        ltp=win_len/5 # taper before deconvolving (important for synthetics)
        #NICK FIND A TAPER SUBROUTINE TO DEAL WITH NEXT 3 LINES
        #Dwin[1:Nwin,j]=taper(Dwin(1:Nwin,j)',tmp_times(j,1:Nwin),ltp,dt,tmp_times(j,1+round((ltp+dt)/dt)),tmp_times(j,Nwin-round((ltp+dt)/dt)));
        #

        #TheTaper=hanning(ltp*3/dt-2)
        #ind1=round((ltp+dt)/dt)
        #ind2=Nwin-round((ltp+dt)/dt)
        #Dwin[j,ind1:ind2]=Dwin[j,ind1:ind2] * TheTaper

        #from matplotlib import pylab as plt
        #plt.plot(Dwin[ind1:ind2,j])
        #plt.plot(TheTaper)
        #plt.show()

        #nd1=length(find(Dwin(:,j)>1e-2))

        #print '*** Warning no taper!!!'
        #Dwin[:Nwin,j]=taper(Dwin[:Nwin,j],tmp_times[j,:Nwin],ltp,dt,tmp_times[j,1+round((ltp+dt)/dt)],tmp_times[j,Nwin-round((ltp+dt)/dt)])

        nd1=len(where(Dwin[:,j]>1e-2))
        if nd1>nd:
            nd=nd1

    # Search range for optimal damping parameter alpha
    if tag=='data':
        tmpind=len(D)
        alphas = logspace(-2,2,nalphas)*var(D[int(round(tmpind/4)):3*int(round(tmpind/4))])*len(P)
    elif tag =='synth':
        alphas = logspace(-2,2,nalphas)*var(D)*len(P)

    # Figure out average times for each moving window
    t0_times = median(tmp_times,1);

    # Construct Slepians
    #[E,~] = dpss(length(Pwin),TB);

    E,dummy = dpss_windows(len(Pwin),TB,NT)
    E=E


    # Length of waveforms;
    nh = len(Pwin)

    #
    tmpint=shape(tmp_times)[0]

    tmpint2=len(arange(t0,t1+dt,dt))

    RF_Time_win=zeros(tmpint2*tmpint).reshape(tmpint2,tmpint)

    misfit = zeros(len(alphas))
    magntd = zeros(len(alphas))
    # Now, calculate misfit and RF size for each alpha

    for kj in range(len(alphas)):
        # Now loop through different time windows for the daughter component
         #number of elements along axis 0
        for k in range(tmpint): #= 1:size(tmp_times,1)
            # Create multitaper estimates
            for j in range(NT):  #= 1:NT
                tmp1 = fft(E[j,:] *Pwin,nh)
                tmp2 = fft(E[j,:] *Dwin[k,:],nh)
                if j==0:
                    NUM = conj(tmp1) *tmp2
                    DEN = conj(tmp1) *tmp1
                else:
                    NUM = NUM + conj(tmp1) *tmp2
                    DEN = DEN + conj(tmp1) *tmp1

        
            # Calculate optimal RF
            tmp = real(ifft(NUM/(DEN + alphas[kj])))
            
            # Filter and Normalize optimal RF
            nrm = max(real(ifft(DEN/(DEN + alphas[kj]))))
            
            #tmp = reshape(fftshift(tmp./nrm),[1 nh])
            tmp = fftshift(tmp/nrm).reshape(nh)
            
            # Time vector
            
            #vrijeme = dt*[-0.5*(nh-1):1:0.5*(nh-1)]+t0_times(k)-t0_times(1) -dt.*length(npad)
            vrijeme = dt* arange(-0.5*(nh-1),0.5*(nh-1)+1) +t0_times[k]-t0_times[1]-dt*len(npad)
            
            #RF_Time_win(:,k) = interp1(double(vrijeme),double(tmp),double(t0:dt:t1),'linear',NaN);
            RF_Time_win[:,k] = interp1d(vrijeme,tmp,kind='linear',bounds_error=False)(arange(t0,t1+dt,dt))
                                
            #length_Dwin=time(Nwin)-time(0)   

        
        tmp = convolve(nanmean(RF_Time_win,axis=1),P)
        t0d = int(round(t0/dt))        
        
        mfD=D
        
        misfit[kj] = nansum(abs(mfD - tmp[1-t0d:len(D)-t0d+1]))
        magntd[kj] = nansum(abs(tmp[1-t0d:len(D)-t0d]))
    
    # Find optimal alpha
    #[~,j2] = min((misfit./std(misfit)).^2+(magntd./std(magntd)).^2)
    j2 =  argmin((misfit/std(misfit))**2+(magntd/std(magntd))**2)

    # Now loop through different time windows for the daughter component
    for k in range(len(tmp_times[:,0])): #= 1:size(tmp_times,1)
        # Create multitaper estimates
        for j in range(NT): #= 1:NT
            tmp1 = fft(E[j,:]*Pwin,nh)
            tmp2 = fft(E[j,:]*Dwin[k,:],nh)
            if j==0:
                NUM = conj(tmp1)*tmp2;
                DEN = conj(tmp1)*tmp1;
            else:
                NUM = NUM + conj(tmp1)*tmp2
                DEN = DEN + conj(tmp1)*tmp1
        
        # Calculate optimal RF
        tmp = real(ifft(NUM/(DEN + alphas[j2])))
        
        # Filter and Normalize optimal RF
        nrm = max(real(ifft(DEN/(DEN + alphas[j2]))))
        
        tmp = fftshift(tmp/nrm).reshape(nh)
        
        # Time vector
        #vrijeme = dt*[-0.5*(nh-1):1:0.5*(nh-1)]+t0_times(k)-t0_times(1) ...
        #        -dt.*length(npad);
        vrijeme = dt* arange(-0.5*(nh-1),0.5*(nh-1)+1) +t0_times[k]-t0_times[1]-dt*len(npad)
        
        #RF_Time_win(:,k) = interp1(double(vrijeme),double(tmp),double(t0:dt:t1),'linear',0);
        RF_Time_win[:,k] = interp1d(vrijeme,tmp,kind='linear',bounds_error=False)(arange(t0,t1+dt,dt))
    
    RF_Time = nanmean(RF_Time_win,axis=1)
    
    Time = -1*arange(t0,t1+dt,dt)
    
    return Time, RF_Time

def taper(x,time,tpr,dt,t1,t2):
    """ 
        % ********* Function Description *********
        %
        % TAPER  Taper a time series.
        %
        % TAPER(X,TIME,TPR,DT,T1,T2) takes time
        % series sampled at DT and tapers it with
        % a cosine taper TPR seconds long from
        % beginning point T1-TPR and with reverse
        % cosine taper from point T2 to point T2+
        % TPR. Points outside the range (T1-TPR,
        % T2+TPR) are zeroed. If T1/T2 is negative
        % then taper is not implemented at the
        % beginning/end. If X is an array of
        % seismograms, then the taper is applied
        % to each row of X.
        %
        %
        % ****************************************
        % *                                      *
        % *  Modified from Kate Rychert's        *
        % *  receiver function code - May 2008   *
        % *                                      *
        % *  Email: David_Abt@brown.edu          *
        % *                                      *
        % ****************************************
        % Emily, 6th May 2013: problem with start time of phase being <100s, so
        % taper subscripts are negative.
        % Temporary workaround, set taper length to be shorter.
        % N.B. Only one event so far has had this issue!
    """
    from numpy import ones, cos, pi , floor, ceil, arange, zeros
    nn      = len(x)
    nx      = len(x)
    taper   = ones(nn)
    #it  = [0:fix(tpr/dt)]*dt/tpr;
    it      = arange(floor(tpr/dt))*dt/tpr
    ct      = 0.5-0.5*cos(pi*it)
    T1      = floor(time[0]/dt)             # Absolute sample point of first time step
    it1     = floor(t1/dt+1)-T1
    it2     = floor(t2/dt+1)-T1

    if t1>0:
        if it1>floor(tpr/dt):
            taper[(it1-floor(tpr/dt)):it1]	= ct
            taper[:(it1-floor(tpr/dt))]    = zeros(len(arange(it1-floor(tpr/dt))))
        else:
            taper[it1] = ct[floor(tpr/dt)-it1+1:-1]
            taper[0] = 0
            print('Bizarre taper!')

    if t2>0:
        if t2>time[-1]-tpr:
            t2  = time[-1]-tpr
            it2 = floor(t2/dt)-T1

        npts=len(ct[::-1])

        tmp1=it2
        tmp2=it2+npts+1
        #print ct[::-1], npts,tmp1,tmp2,t1,t2
        taper[it2:(it2+npts)] = ct[::-1]

        taper[it2+ceil(tpr/dt):nn]   = zeros(len(taper[it2+floor(tpr/dt):nn]))

    y = zeros(nx);  #modified to 1-D arrays only
    #for ix in range(nx):
        #print 'ix = ', ix, len(y), len(x)
    y[:] = x[:]*taper
