import pyfits
import pylab as py
import math
import numpy as np
from numpy import asarray
from numpy import interp
from matplotlib import colors
import os
import os.path
from scipy import stats
from scipy import interpolate
from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.signal import savgol_filter
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.mlab import griddata
import fileinput
import multiprocessing as mp
from multiprocessing import Pool
import time
from functools import partial
from iminuit import Minuit, describe
from matplotlib import rcParams
# this is an absurd command to make room for xlabel
rcParams.update({'figure.autolayout': True})


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++ some global declarations +++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#global file1, file2, plsfile
# Fermi files
file1='../mlfermidwarfs/data/wavelet_0.00_0.00_P8R2_SOURCE_V6_evtype3_nxpix3600_nypix1800_binsz0.1_Elo500_Ehi500000_ebins24.fits'
file2='../mlfermidwarfs/data/expmap_0.00_0.00_P8R2_SOURCE_V6_evtype3_nxpix3600_nypix1800_binsz0.1_Elo500_Ehi500000_ebins24.fits'
plsfile='../gll_psc_v16.fit'



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++ declaration of functions        ++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def importdata(f1,f2,f3):
    # f1,f2 are the  counts and exposure files from Fermi
    # f3 is the point-like sources
    hdulist1 = pyfits.open(f1)
    hdulist2 = pyfits.open(f2)
    hdulist3 = pyfits.open(f3)
    cdata=hdulist1[0].data
    edata=hdulist2[0].data
    plsdata=hdulist3[1].data
    plslat=plsdata.field('GLAT')
    plslon=plsdata.field('GLON')
    return cdata, edata, plslat, plslon

def drawcounts(f):
    # f: counts file
    img=pyfits.getdata(f)
    py.figure(1)
    py.clf()
    py.imshow(img[0],cmap=py.cm.Greys,vmin=20,vmax=2250,norm=colors.LogNorm() )
    py.clim(1.,10000.)
    py.colorbar()
    py.savefig('counts_source_B.png')

def calcflux(cdata,edata):
    # edata has 25 bin entries, while cdata has 24
    #so removing last row from edata
    edata = np.delete(edata,-1,0)
    flux = cdata
    return flux

def fluxtofits(fl):
    hduflux = pyfits.PrimaryHDU()
    hduflux.data = fl
    os.system('rm flux2.fits')
    hduflux.writeto('flux2.fits')
    

def getflux(file):
    hdu = pyfits.open(file)
    flux = hdu[0].data
    return flux


def getdSphs(file,all='yes'):
    # data from  1611.03184
    # all = 'no' : remove ultrafaint dSphs
    wfile = fileinput.input(file)
    blist = []
    llist = []
    for linea in wfile:
        lat, lon = linea.split()
        blist.append([float(lat)])
        llist.append([float(lon)])
    #print('get..blist = ',blist)
    if(all=='no'):
        blist = blist[0:31]
        llist = llist[0:31]
        #print('get..blist = ',blist)
        blist = np.delete(blist,[1,2,8,22,28,29])
        llist = np.delete(llist,[1,2,8,22,28,29])
        #print('get..blist = ',blist)
    return blist, llist

def kde_skl(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def optbandwidth(lista):
    grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 20.0, 200)},cv=10)
    grid.fit(lista)
    string=str(grid.best_params_)
    trash,val = string.split()
    return val[:-1]

def buildPDF(lista,grid,bw):
    # lista: array from which PDF will be built
    # bw: optimal bandwidth
    pdf = kde_skl((lista),grid,bandwidth=bw)
    pdf_norm = pdf/sum(pdf)
    return pdf_norm

def ppf(name,lista,grid,bw,y):
    lista=np.array(lista).reshape((-1,1))
    if(name=='lon'):
        RVdiscrete_l = stats.rv_discrete(values=(grid,buildPDF(lista,grid,bw)),
                                         name='RVdiscrete')
        return RVdiscrete_l.ppf(y)
    if(name=='lat'):
        RVdiscrete_b = stats.rv_discrete(values=(grid,buildPDF(lista,grid,bw)),
                                         name='RVdiscrete')
        return RVdiscrete_b.ppf(y)
    if(name=='bckg'):
        RVdiscrete_bckg = stats.rv_discrete(values=(grid,buildPDF(lista,grid,bw)),
                                            name='RVdiscrete')
        return RVdiscrete_bckg.ppf(y)

def voidgen(Nvoid,bwb,bwl,plslat,plslon):
    listb,listl=getdSphs('dSphs_positions.dat',all='yes')
    listb_red,listl_red=getdSphs('dSphs_positions.dat',all='no')
    #gridl = np.linspace(min(listl_red),max(listl_red),1000)
    #gridb = np.linspace(min(listb_red),max(listb_red),1000)
    gridl = np.linspace(1.,359.,1000)
    gridb = np.linspace(-89.,89.,1000)
    # including extended sources to the lists of dwarfs
    l_ext1 = [302.14498901]
    l_ext2 = [278.83999634]
    b_ext1 = [-44.41999817]
    b_ext2 = [-32.84999847]
    listb.append(b_ext1)
    listb.append(b_ext2)
    listl.append(l_ext1)
    listl.append(l_ext2)
    # generating a preliminary list of candidates
    np.random.seed(1111) # (1982) # (123) random seed for reproducibility
    lpdfval=np.random.random(Nvoid)
    bpdfval=np.random.random(Nvoid)
    l_voidpre = ppf('lon',listl,gridl,bwl,lpdfval) #listl_red
    b_voidpre = ppf('lat',listb,gridb,bwb,bpdfval) #listb_red
    xi = np.hstack((l_voidpre.reshape((-1,1)),b_voidpre.reshape((-1,1))))
    #print('len x_voidpre=',len(x_voidpre),x_voidpre[0])
    #quit()
    # excluding those overlapping with dwarfs, extended sources, disk,
    # and point-like sources
    rad = 1.0 # impose this angular separation between centroids
    rdisk = 20.0 # angular extension of galactic disk
    l_void = []
    b_void = []
    print('Nvoids = ',Nvoid)

    for i in range(len(l_voidpre)):
        delta_r = [] # array of delta_r between every empty region and all the dwarfs
        for dwarf in range(len(listl)):
            distancia = np.sqrt((l_voidpre[i]-listl[dwarf])**2 + (b_voidpre[i]-listb[dwarf])**2)
            delta_r.append(distancia[0])
        if min(delta_r) > rad and abs(b_voidpre[i]) > rdisk:
            # removing point-like sources
            delta_rp = []
            for pls in range(len(plslat)):
                delta_rp.append(math.sqrt((l_voidpre[i]-plslon[pls])**2 +
                                          (b_voidpre[i]-plslat[pls])**2))
            if min(delta_rp) > 2.5:
                #l_void.append(l_voidpre[i])
                #b_void.append(b_voidpre[i])
                
                if i < 10: 
                    l_void.append(l_voidpre[i])
                    b_void.append(b_voidpre[i])
                delta_ri = [] # delta_r between candidate empty regions and the stored ones
                for j in range(len(l_void)):
                    #print('len l_void=',len(l_void))
                    dist = np.sqrt((l_voidpre[i]-l_void[j])**2+(b_voidpre[i]-b_void[j])**2)
                    #print('dist=',dist)
                    delta_ri.append(dist)
                if min(delta_ri) > rad:
                    l_void.append(l_voidpre[i])
                    b_void.append(b_voidpre[i])
                
        if i % 100 == 0:
            print(i,'-th sample, len(array) = ',len(b_void))
    return l_void, b_void


def voidstofile(llist,blist):
    os.system('rm voids_all_45dSphs.dat')
    fileout = open('voids_all_45dSphs.dat',"a")
    palline=np.zeros((len(llist),2))
    for i in range(len(llist)):
        palline[i] = np.array([llist[i],blist[i]])
        fileout.write(" ".join([str(palline[i,0]),str(palline[i,1]),"\n"]))
    fileout.close()


def fill_regions(llist,blist,cdata,edata,sample='dSphs'):
    Nsample = len(llist)
    Nen = len(cdata)
    #print('llist[134],blist[134]=',llist[134],blist[134])
    #print('len cdata: ',len(cdata[0]),len(cdata[0,0]))
    count_sample = np.zeros((Nsample,Nen))
    exp_sample = np.zeros((Nsample,Nen))
    rae = 5 #extendend radius of region, in units of 0.1 deg
    for i in range(Nsample):
        #print('what i=',i)
        # converting real coordinates to countdata coordinates
        bt0 = int(900. + 10.*blist[i])
        if(llist[i]<=180):
            lt0 = int(1800 - 10*llist[i])
        else:
            lt0 = int(1800 + 3600 - 10*llist[i])
        for en in range(Nen):
            #print('what en: ',en)
            nb = 0 # number of pixels inside region
            # sum all contributions inside a square of side 2 deg.
            for lat in range(bt0-rae,bt0+rae+1):
                for lon in range(lt0-rae,lt0+rae+1):
                    nb = nb + 1
                    count_sample[i,en] = count_sample[i,en] + cdata[en,lat,lon]
                    exp_sample[i,en] = exp_sample[i,en] + edata[en,lat,lon]
                    # remove the corners outside the circle of radius 2 deg
                    r = math.sqrt((lat-bt0)**2 + (lon-lt0)**2) # distance to centroid
                    if(r > rae):
                        nb = nb - 1
                        count_sample[i,en] = count_sample[i,en] - cdata[en,lat,lon]
                        exp_sample[i,en] = exp_sample[i,en] - edata[en,lat,lon]
            exp_sample[i,en] = exp_sample[i,en] / nb # average exposure
        if(i % 500 == 0):
            print('filling region # ',i,' out of ',Nsample)
    os.system('rm counts_'+sample+'.dat')
    os.system('rm exp_'+sample+'.dat')
    fileout0 = open('counts_'+sample+'.dat',"a")
    fileout1 = open('exp_'+sample+'.dat',"a")
    for i in range(Nsample):
        string0 = str(count_sample[i])
        substr0 = string0[1:-1].split()
        string1 = str(exp_sample[i])
        substr1 = string1[1:-1].split()
        fileout0.write(" ".join([substr0[j] for j in range(len(substr0))]))
        fileout0.write("\n")
        fileout1.write(" ".join([substr1[j] for j in range(len(substr1))]))
        fileout1.write("\n")
    fileout0.close()
    fileout1.close()

def importcounts(infile):
    countarr = []
    exparr = []
    lonarr = []
    latarr = []
    for line in infile:
        lon, lat, count, exp  = line.split()
        countarr.append(float(count))
        exparr.append(float(exp))
        lonarr.append(float(lon))
        latarr.append(float(lat))
    return lonarr,latarr,countarr,exparr

'''
def readdata(infile):
    arr = []
    for line in infile:
        values = line.split()
        arr.append(values)
    return arr
'''

def KStest(fluxD,fluxV):
    #fluxD : dwarfs, fluxV: voids
    #++++++++++++++++++++++++++++
    # performing test statistics
    mystr = str(stats.ks_2samp(fluxV, fluxD))
    mystr = mystr.replace(",","  ")
    KS_dist = mystr[25:30]
    str1,str2 = mystr.split()
    pval = float(str2[7:-1])
    return KS_dist, pval


def histo_fig(fluxD,fluxV,p_val):
    # exporting figure with histogOBrams
    lenD=str(len(fluxD))
    lenV=str(len(fluxV))     
    Vweights = np.ones_like(fluxV)/len(fluxV)
    Dweights = np.ones_like(fluxD)/len(fluxD)
    fig4, ax4 = plt.subplots()
    ax4.hist(fluxD,5,facecolor='red',alpha=0.5,weights=Dweights)
    ax4.hist(fluxV,40,facecolor='black',alpha=0.5,weights=Vweights)
    ax4.set_xlabel('flux [$10^{-10}cm^{-1}s^{-1}$]')
    ax4.set_ylabel('fraction')
    plt.xlim(-1,100)
    ax4.text(70,0.35,'1: dSphs',fontsize=10,color='red')
    ax4.text(70,0.33,'2: voids',fontsize=10,color='black')
    ax4.text(70,0.29,'$n_1$='+lenD,fontsize=10,color='red')
    ax4.text(70,0.27,'$n_2$='+lenV,fontsize=10,color='black')
    ax4.text(70,0.25,'p-value= {0:5.3f}'.format(p_val),fontsize=10,color='black')
    plt.savefig('KS_method3.pdf')
    



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++ functions for computation of DM spectrum ++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Interp:
    """
    A Interp object is a interpolator for the production fluxes for a specific
    annhilation channel. Given dark matter masses and photon energies it will
    give you the fluxes. The tables being interpolated is from PPPC 4 DM ID.

    The interpolation algorithm used is bilinear. The main work is done in C.
    """

    def __init__(self, ch):
        from ctypes import cdll, c_double, c_int, POINTER

        lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),
                                            '../spectro_from_PPPC/cinterp/libpppc.so'))

        msize = c_int.in_dll(lib, 'msize_%s' % ch).value
        xsize = c_int.in_dll(lib, 'xsize_%s' % ch).value

        self.m = np.array((c_double*msize).in_dll(lib, 'mass_%s' % ch))
        self.log10x = np.array((c_double*xsize).in_dll(lib, 'logx_%s' % ch))

        if ch == 'tt':
            self.cdNdE = lib.dNdE_tt
            self.cinterp = lib.interp_tt
        if ch == 'bb':
            self.cdNdE = lib.dNdE_bb
            self.cinterp = lib.interp_bb
        if ch == 'cc':
            self.cdNdE = lib.dNdE_cc
            self.cinterp = lib.interp_cc
        if ch == 'qq':
            self.cdNdE = lib.dNdE_qq
            self.cinterp = lib.interp_qq
        if ch == 'gg':
            self.cdNdE = lib.dNdE_gg
            self.cinterp = lib.interp_gg

        self.cdNdE.restype = c_double
        self.cinterp.restype = c_double

        #c_double_p = lambda x : POINTER(c_double(x))
        c_double_p = POINTER(c_double)
        self.interp = np.vectorize(lambda m, lx: self.cinterp(c_double(m), c_double(lx)))
        self.dNdE = np.vectorize(lambda m, e: self.cdNdE(c_double_p(c_double(m)),
                                                         c_double_p(c_double(e))))

    def __call__(self, masses, energies):
        """ Returns dNdE for given arrays of masses and energies.
        """
        return self.dNdE(masses, energies)

# Supported channels.
channels = ['gg', 'tt', 'bb', 'cc', 'qq']

def loadInterpolators(channels=channels):
    d = {}
    for ch in channels:
        d[ch] = Interp(ch)

    return d


def integ_flux(mass, channel, emin, emax):
    spectra = loadInterpolators()
    enum = 1000
    e = np.logspace(np.log10(emin), np.log10(emax), enum)
    spec = spectra[channel](mass, e)

    integ = np.trapz(spec, e)
    return integ


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def DMflux(sv,mx,i,DOmega,emin,emax):
    # sv: sigma*v,  mx: DM mass, i: dwarf index, with associated J-factor
    eta = 0.5 #  Dirac: 0.25, Majorana: 0.5
    log10_Jfac=[18.2,17.4,17.6,17.9,19.0,18.8,17.8,16.9,18.64,16.56,17.8,18.0,16.3,
                16.4,17.90,18.9,18.5,19.4,17.5,18.7,17.9,19.4,18.9,19.29,17.96]
    flux = 1/(4.*np.pi)*eta*sv/(mx**2)*(10**log10_Jfac[i])*integ_flux(mx,'bb',emin,emax)
    DOmega05 = np.pi*(0.5)**2*(np.pi/180)**2 
    flux = flux * DOmega/DOmega05
    return flux

def DMcount(sv,mx,DOmega,exp):
    Ndwarf = len(exp)
    Nen = len(exp[0])
    count = np.zeros((Ndwarf,Nen))
    Elist=np.logspace(np.log10(0.5), np.log10(500.), Nen+1)
    for i in range(Ndwarf):
        for en in range(Nen):
            count[i,en] = float(DMflux(sv,mx,i,DOmega,Elist[en],Elist[en+1]))*float(exp[i][en])
    return count
            

def countBCKG_dSphs(sv,mx,DOmega,exp,countD):
    meas = np.array(countD)
    meas = meas.astype(np.float)
    DMcon = np.array(DMcount(sv,mx,DOmega,exp))
    countB = meas - DMcon
    return countB


def svmass_scan(func,mxlist,svlist):
    Flist=[]
    mxarr=[]
    svarr=[]
    for i in range(len(mxlist)):
        print('mx=',mxlist[i])
        for j in range(len(svlist)):
            print('sv=',svlist[j])
            mxarr.append(mxlist[i])
            svarr.append(svlist[j])
            Flist.append(func(svlist[j],mxlist[i]))
    return mxarr,svarr,Flist

def PDFvect2(yi,xi,sigma,sigmaY):
    Nchunk=421
    xi=xi
    yi=yi
    sz=round(len(xi)/Nchunk)
    n=len(xi)
    sum_arr=[]
    for p in range(Nchunk):
        XM = (np.repeat(xi[sz*p:sz*(p+1)],n,axis=0).reshape((sz,n,2)))
        YM = np.repeat(yi[sz*p:sz*(p+1)],n).reshape((sz,n))
        AuxD = XM - xi
        XM = None
        D = AuxD[:,:,0]**2 + AuxD[:,:,1]**2
        D = np.ma.masked_where(D==0,D)
        AuxD = None
        expX = np.float128(np.exp(-D/(2*sigma**2)))
        D = None
        Dy = (np.log(YM) - np.log(yi))**2
        expY = np.float128( np.exp(-Dy/(2*sigmaY**2)))
        Dy = None
        sum_arr.append( np.sum(expX*expY,axis=1)/YM[:,0]/((2*np.pi)**(3/2)*(sigma**2)*sigmaY*n))
        YM = None        
        expX = None
        expY = None
    sum_arr=np.array(sum_arr).flatten()
    return np.sum(np.log(sum_arr))



def Fermibound(infile):
    mxFlist = []
    svFlist = []
    for linea in infile:
        mdm,sv = linea.split()
        mxFlist.append(float(mdm))
        svFlist.append(float(sv))
    return mxFlist,svFlist



def sigmaOUT(index):
    lons = lon_voids
    lats = lat_voids
    fluxes = flux_voids
    Nvoids = len(lons)
    auxDi = np.zeros((Nvoids-1,2))
    Di = np.zeros((Nvoids-1))
    num = 0.
    den = 0.
    '''
    # normalizing longitude and latitude
    lons = lons - np.mean(lons)
    lons = lons / np.std(lons)
    lats = lats - np.mean(lats)
    lats = lats / np.std(lats)
    '''
    # removing i coordinates from sample
    lonALLbut = np.delete(lons,index)
    latALLbut = np.delete(lats,index)
    fluxALLbut = np.delete(fluxes,index)
    for i in range(Nvoids-1):
        auxDi[i,0] = lons[index] - lonALLbut[i]
        auxDi[i,1] = lats[index] - latALLbut[i]
        Di[i] = auxDi[i].T.dot(auxDi[i])
        expi = np.float64(np.exp(-Di[i]/(2*sigma**2)))
        contrib = fluxALLbut[i]*expi
        den = den + expi
        num = num + contrib
    Yhat = num/den
    return  (Yhat - fluxes[index])**2



def SErr(k,sigma,sz,lons,lats,fluxes):
    print('training # ',k)
    # extracting training subsample
    lons_tr = np.concatenate([ lons[:k*sz], lons[(k*sz+sz):] ])
    lats_tr = np.concatenate([ lats[:k*sz], lats[(k*sz+sz):] ])
    fluxes_tr = np.concatenate([ fluxes[:k*sz], fluxes[(k*sz+sz):] ])
    N_tr = len(lons_tr)
    # extracting validation subsample
    lons_val = lons[k*sz:(k*sz+sz)]
    lats_val = lats[k*sz:(k*sz+sz)]
    fluxes_val = fluxes[k*sz:(k*sz+sz)]
    N_val = len(lons_val)
    # computing error function
    auxDi = np.zeros(( len(lons_tr), 2 ))
    Di = np.zeros(( len(lons_tr) ))
    num = 0.
    den = 0.
    SE = []
    for j in range(N_val):
        for i in range(N_tr):
            auxDi[i,0] = lons_val[j] - lons_tr[i]
            auxDi[i,1] = lats_val[j] - lats_tr[i]
            Di[i] = auxDi[i].T.dot(auxDi[i])
            expi = np.float128(np.exp(-Di[i]/(2*sigma**2)))
            contrib = fluxes_tr[i]*expi
            den = den + expi
            num = num + contrib
        Yhat = num/den
        SE.append((Yhat - fluxes_val[j])**2 )
    return SE

def crossValidation(lon, lat, flux, sig, Kfold=5):
    # size of validation sample
    size = int(len(lon)/Kfold)
    # parallelizing the computation
    SErr_1arg = partial(SErr,sigma=sig,sz=size,lons=lon,lats=lat,fluxes=flux)
    p = Pool(4)
    SE = p.map(SErr_1arg,range(Kfold))
    return SE

def yhat(x,yi,xi,sigma):
    n = len(xi)
    X = np.array([x]*n).reshape((-1,2))
    AuxD = X-xi
    D =  AuxD[:,0]**2 + AuxD[:,1]**2
    expX = np.float128(np.exp(-D/(2*sigma**2)))
    num = expX*yi
    return np.sum(expX*yi)/ np.sum(expX)
    
def yhat_arr(d,xD,yi,xi,sigma):
    # first removing points with zero counts
    argdel=np.where(yi==0.)[0]
    yi=np.delete(yi,argdel)
    xi=np.delete(xi,argdel,axis=0)
    x=xD[d]
    n = len(xi)
    X = np.array([x]*n).reshape((-1,2))
    AuxD = X-xi
    D =  AuxD[:,0]**2 + AuxD[:,1]**2
    expX = np.float128(np.exp(-D/(2*sigma**2)))
    return np.sum(expX*np.log(yi))/ np.sum(expX)

def yhat2_arr(d,xD,yi,xi,sigma):
    x=xD[d]
    n = len(xi)
    X = np.array([x]*n).reshape((-1,2))
    AuxD = X-xi
    D =  AuxD[:,0]**2 + AuxD[:,1]**2
    expX = np.float128(np.exp(-D/(2*sigma**2)))
    return np.sum(expX*(np.log(yi))**2)/ np.sum(expX)

def diff_yhat_VOID(i,yi,xi,sigma,varsigma):
    x=xi[i]
    y=yi[i]
    xi=np.delete(xi,i,axis=0)
    yi=np.delete(yi,i)
    n = len(xi)
    X = np.array([x]*n).reshape((-1,2))
    AuxD = X-xi
    D =  AuxD[:,0]**2 + AuxD[:,1]**2
    expX = np.float128(np.exp(-D/(2*sigma**2)))
    lnyhat = np.sum(expX*np.log(yi))/ np.sum(expX)
    lnyhat2 = np.sum(expX*(np.log(yi))**2)/ np.sum(expX)
    std = np.sqrt(varsigma**2 + lnyhat2 - lnyhat**2)
    return abs(np.log(y)-lnyhat)/std


def Ypred(t,lonT,latT,fluxT,lonV,latV,fluxV,sig,sample='test'):
    #print("sample point ",t," out of ",len(lonT))
    Nvoids = len(lonV)
    auxDi = np.zeros((Nvoids,2))
    Di = np.zeros((Nvoids))
    num = 0.
    den = 0.
    exparr = []
    for i in range(Nvoids):
        auxDi[i,0] = lonT[t] - lonV[i]
        auxDi[i,1] = latT[t] - latV[i]
        Di[i] = auxDi[i].T.dot(auxDi[i])
        expi = np.float128(np.exp(-Di[i]/(2*sig**2)))
        exparr.append(expi)
        den = den + expi
        num = num + fluxV[i]*expi
    Yhat = num/den
    #print('lonT,latT = ',lonT[t],latT[t])
    #print('exparr[3988] = ',exparr[3988])
    #print('max exparr = ',np.max(exparr),np.argmax(exparr))
    SE = (Yhat - fluxT[t])**2
    if(sample=='test'):
        return  SE
    else:
        return Yhat


'''
def Ypred(t,lonT,latT,fluxT,lonV,latV,fluxV,sig,sample='test'):
    print("sample point ",t," out of ",len(lonT))
    Nvoids = len(lonV)
    auxDi = np.zeros((Nvoids,2))
    Di = np.zeros((Nvoids))
    num = 0.
    den = 0.
    for i in range(Nvoids):
        auxDi[i,0] = lonT[t] - lonV[i]
        auxDi[i,1] = latT[t] - latV[i]
        Di[i] = auxDi[i].T.dot(auxDi[i])
        expi = np.float128(np.exp(-Di[i]/(2*sig**2)))
        den = den + expi
        num = num + np.log(fluxV[i])*expi
    Yhat = num/den
    SE = (Yhat - np.log(fluxT[t]))**2
    if(sample=='test'):
        return  SE
    else:
        return Yhat
'''


def f_allbut1(index,lons,lats,yvals,sigma):
    Nvoids = len(lons)
    print('region # ',index," out of ",Nvoids)
    auxDi = np.zeros((Nvoids-1,2))
    Di = np.zeros((Nvoids-1))
    num = 0.
    den = 0.
    # removing i coordinates from sample
    lonALLbut = np.delete(lons,index)
    latALLbut = np.delete(lats,index)
    yvalsALLbut = np.delete(yvals,index)
    for i in range(Nvoids-1):
        auxDi[i,0] = lons[index] - lonALLbut[i]
        auxDi[i,1] = lats[index] - latALLbut[i]
        Di[i] = auxDi[i].T.dot(auxDi[i])
        expi = np.float64(np.exp(-Di[i]/(2*sigma**2)))
        contrib = yvalsALLbut[i]*expi
        den = den + expi
        num = num + contrib
    Yhat = num/den
    SE = (Yhat - yvals[index])**2
    SRE = SE/yvals[index]**2
    return SE, SRE



def Err_pred(index,lons,lats,fluxes,sigma):
    Nvoids = len(lons)
    print('region # ',index," out of ",Nvoids)
    auxDi = np.zeros((Nvoids-1,2))
    Di = np.zeros((Nvoids-1))
    num = 0.
    den = 0.
    # removing i coordinates from sample
    lonALLbut = np.delete(lons,index)
    latALLbut = np.delete(lats,index)
    fluxALLbut = np.delete(fluxes,index)
    for i in range(Nvoids-1):
        auxDi[i,0] = lons[index] - lonALLbut[i]
        auxDi[i,1] = lats[index] - latALLbut[i]
        Di[i] = auxDi[i].T.dot(auxDi[i])
        expi = np.float64(np.exp(-Di[i]/(2*sigma**2)))
        contrib = fluxALLbut[i]*expi
        den = den + expi
        num = num + contrib
    Yhat = num/den
    return  (Yhat - fluxes[index])**2


def Err_dSphs(t,lonT,latT,lonV,latV,errV,sig):
    print("sample point ",t," out of ",len(lonT))
    Nvoids = len(lonV)
    auxDi = np.zeros((Nvoids,2))
    Di = np.zeros((Nvoids))
    num = 0.
    den = 0.
    for i in range(Nvoids):
        auxDi[i,0] = lonT[t] - lonV[i]
        auxDi[i,1] = latT[t] - latV[i]
        Di[i] = auxDi[i].T.dot(auxDi[i])
        expi = np.float128(np.exp(-Di[i]/(2*sig**2)))
        den = den + expi
        num = num + errV[i]*expi
    errhat = num/den
    return errhat

    
def countexpo_dSphs(lonD,latD,cdata,edata):
    edata = np.delete(edata,-1,0)
    Nsample = len(lonD)
    Nen = len(cdata)
    rae=10.
    count_sample = np.zeros((Nsample,Nen))
    exp_sample = np.zeros((Nsample,Nen))
    countexp_tot = np.zeros((Nsample,2))
    for i in range(Nsample):
        for en in range(Nen):
            # sum all contributions inside a square of side 2 deg.
            for lat in range(int(latD[i]-rae),int(latD[i]+rae)):
                for lon in range(int(lonD[i]-rae),int(lonD[i]+rae)):
                    count_sample[i,en] = count_sample[i,en] + cdata[en,lat,lon]
                    exp_sample[i,en] = exp_sample[i,en] + edata[en,lat,lon]
                    # remove the corners outside the circle of radius 2 deg
                    r = math.sqrt((lat-latD[i])**2 + (lon-lonD[i])**2) # distance to centroid
                    if(r >= rae):
                        count_sample[i,en] = count_sample[i,en] - cdata[en,lat,lon]
                        exp_sample[i,en] = exp_sample[i,en] - edata[en,lat,lon]
        countexp_tot[i] = np.array([count_sample[i].sum(),exp_sample[i].sum()])
        print('dwarf # ',i+1,' count = ',count_sample[i].sum(),' expo = ',exp_sample[i].sum())
    print('exp_sample = ',exp_sample)
    return countexp_tot


def GNNbckg(lonD,latD,fluxD,lonV,latV,fluxV):
    # lonD,latD,fluxD: longitude, latitude and flux arrays for dSphs
    # lonV,latV,fluxV: longitude, latitude and flux arrays for voids
    NdSphs = len(lonD)
    Nvoids = len(lonV)
    Ypred = []
    SSE = []
    for d in range(NdSphs):
        auxDi = np.zeros((Nvoids,2))
        Di = np.zeros((Nvoids))
        num = 0.
        den = 0.
        for i in range(Nvoids):
            auxDi[i,0] = lonD[d] - lonV[i]
            auxDi[i,1] = latD[d] - latV[i]
            Di[i] = auxDi[i].T.dot(auxDi[i])
            expi = np.float64(np.exp(-Di[i]/(2*sigma**2)))
            den = den + expi
            num = num + fluxV[i]*expi
        Ypred.append(num/den)
        SSE.append((num/den - fluxD[d])**2)
    return  Ypred, SSE


def PDFbckg(x,xmean,lista):
    n,b=np.histogram(lista,200)    
    xaxis= np.array([(b[i]+b[i+1])/2 for i in range(len(b)-1)])
    yaxis=n/len(lista) # this is the PDF
    ysmooth = savgol_filter(yaxis,51,5)
    print('mean  =',np.sum(xaxis*yaxis))
    #plt.plot(xaxis,yaxis,'-',lw=2.0)
    plt.plot(xaxis,ysmooth,'-',color="black")
    plt.plot([617,617],[0.,0.09],'--',color='blue')
    plt.xlim(0,2000)
    plt.ylim(0,0.08)
    plt.xlabel("counts",fontsize=18)
    plt.text(617,0.025,'mean',fontsize=12,color='blue',rotation=90)
    plt.title('PDF of counts for background',fontsize=12)
    #plt.show()
    plt.savefig('pdf_bckg.pdf')
    
    xaxist = xaxis - np.sum(xaxis*yaxis) + xmean  # shift the mean of the PDF
    diffs = np.array([abs(x-xaxist[i]) for i in range(len(xaxist))])
    argmin=np.where(diffs == diffs.min())
    print('coordinates = ',xaxist[argmin], yaxis[argmin] )
    return yaxis[argmin]
    

# piece of likelihood for Poisson+jacobian
def logL_P(logJ,B,sv,mx,i,b,edata,n,bckg,log10Jmean,histoV):
    n = np.round(n)
    jacob = (10**logJ/10**log10Jmean[i])*DMcount(sv,mx,DOmega,edata,i,b)/sv
    lamb = (10**logJ/10**log10Jmean[i])*DMcount(sv,mx,DOmega,edata,i,b) + B
    res = n[i]*math.log(lamb) - lamb - math.log(math.factorial(n[i]))
    res = res + np.log(jacob)
    return -2*res

# piece of likelihood on J
def logL_J(logJ,i,log10Jmean,dJ):
    denJ = np.log(10)*np.sqrt(2.*np.pi)*dJ[i]
    res = - (logJ-log10Jmean[i])**2/(2*dJ[i]**2) - math.log(denJ)
    return -2*res
            
def PDF_B(lny,x,lnyi,xi,sigma,sigmaY):
        N = len(xi)
        X = np.array(x*len(xi)).reshape((-1,2))
        lnY = np.array([lny]*len(xi))
        AuxD = X-xi
        D =  AuxD[:,0]**2 + AuxD[:,1]**2
        Dy = (lnY - lnyi)**2
        expX = np.float128(np.exp(-D/(2*sigma**2)))
        expY = np.float128( np.exp(-Dy/(2*sigmaY**2)))
        res = np.sum(expX*expY) /( (2*np.pi)**(3/2)*(sigma**2)*sigmaY*N  )
        return res 
                                        


def main():
    task1 = False # importing data+computing flux 
    task2 = False  # generating centroids for voids
    task2a = False # plotting spatial PDF of dwarfs
    task2b = False
    task3 = False # filling voids and dwarfs with count data
    task3a = False # plotting sky with all regions
    task4 = False # CDF dSphs-voids compatibility, DM bounds from KS p-value
    task5 = False # DM-bound from global BCKG estimation
    task6 = False  # finding optimum smoothing parameters
    task7 = False # plotting PDF
    task8 = False  # estimation of bckg at dSph positions
    task8post = False # standard deviations from estimation to measured
    task8a = True
    task8b = False
    task91 = False # computing limits on sv_vs_mx
    task92 = False # computing limits on sv_vs_mx
    task93 = False
    task94 = False
    task95 = False
    task96 = False
    task97 = False
    task920 = False
    figPDF = False
    figMoney = False
    fig1dSph = False
    task101 = False
    task102 = False # plotting limits separatedly
    task1023 = False
    task1045 = False
    taskL = False
    task11 = False # combined limits
    task12 = False # plot combined limits

    if(task1):
        #+++++++++++++++++++++++++++++++++++++++++
        print('++++++++++++importing data ++++++++++++++')
        #++++++++++++++++++++++++++++++++++++++++++
        countdata,expdata,plslat,plslon = importdata(file1,file2,plsfile)
        # ++++ figure with counts ++++++++++++++++
        #drawcounts(file1)
        # +++++ computing flux +++++++++++++++++++++
        flux = calcflux(countdata,expdata)
        print('+++++ exporting flux to fits file ++++++')
        fluxtofits(flux)

    if(task2):
        countdata,expdata,plslat,plslon = importdata(file1,file2,plsfile)
        #+++++++++++++++++++++++++++++++++++++++++
        print('++++++++++++ generating void centroids ++++++++++++++')
        #++++++++++++++++++++++++++++++++++++++++++
        # +++++ getting dwarfs coordinates ++++++++++++
        blist, llist = getdSphs('dSphs_positions.dat',all='no') #'no'
        llist = np.array(llist).reshape(-1,1)
        blist = np.array(blist).reshape(-1,1)
        #print('min lon dSphs = ',min(llist))
        #print('lon mi max, lat mi max: ',min(llist),max(llist),min(blist),max(blist))
        # +++++ computing optimum bandwidth for KDE
        bandwlon=float(optbandwidth(llist))
        bandwlat=float(optbandwidth(blist))
        print('optimum bandwidth for lon and lat: ',bandwlon,bandwlat)
        quit()
        # +++ max number of void regions of area = pi deg^2 (r=1deg)
        rad = 0.5 # deg
        Area = np.pi*rad**2 
        Nvoid = int(4*np.pi/(Area*(np.pi/180)**2))
        print('generating ',Nvoid,' regions')
        l_void, b_void = voidgen(Nvoid,bandwlat,bandwlon,plslat,plslon)
        print('final number of void regions = ',len(l_void))
        print('+++++++exporting void coordinates to data file +++++')
        voidstofile(l_void,b_void)
        
    if(task2a):
        countdata,expdata,plslat,plslon = importdata(file1,file2,plsfile)
        #++++++++++++++++++++++++++++++++++++++++++
        # +++++ getting dwarfs coordinates ++++++++++++
        blist, llist = getdSphs('dSphs_positions.dat',all='no')
        # +++++ computing optimum bandwidth for KDE
        llist = np.array(llist).reshape(-1,1)
        blist = np.array(blist).reshape(-1,1)
        bandwlon=float(optbandwidth(llist))
        bandwlat=float(optbandwidth(blist))
        print('opt bws: ',bandwlon,bandwlat)
        l_grid = np.linspace(1.,360.,1000)
        b_grid = np.linspace(-90.,90.,1000)
        #l_grid = np.linspace(min(llist),max(llist),1000)
        #b_grid = np.linspace(min(blist),max(blist),1000)
        lpdf = kde_skl((llist),l_grid,bandwidth=bandwlon)
        bpdf = kde_skl((blist),b_grid,bandwidth=bandwlat)
        print('lpdf=',lpdf)
        fig1, ax1 = plt.subplots(2)
        ax1[0].plot(l_grid, lpdf, color='blue', alpha=0.5, lw=3)
        ax1[0].set_title("longitude PDF")
        ax1[0].set_xlim(0,360)
        ax1[1].plot(b_grid, bpdf, color='red', alpha=0.5, lw=3)
        ax1[1].set_title("latitude PDF")
        ax1[1].set_xlim(-90,90)
        fig1.tight_layout()
        plt.savefig('PDFs_dwarfs_25_extended.pdf')

    if(task2b):
        '''
        lonV,latV = np.loadtxt('voids_all_05.dat',unpack=True)
        fig1, ax1 = plt.subplots(2)
        LON_weights=np.ones_like(lonV)/len(lonV)
        LAT_weights=np.ones_like(latV)/len(latV)
        ax1[0].hist(lonV,100,facecolor='red',label="lon",weights=LON_weights)
        ax1[0].set_title('longitude')
        ax1[1].hist(latV,100,facecolor='blue',label="lat",weights=LAT_weights)
        ax1[1].set_title('latitude')
        plt.savefig('a_posteriori_PDF.pdf')
        '''
        D_blist, D_llist = getdSphs('dSphs_positions.dat',all='no')
        fig1, ax1 = plt.subplots(2)
        LON_weights=np.ones_like(D_llist)/len(D_llist)
        LAT_weights=np.ones_like(D_blist)/len(D_blist)
        ax1[0].hist(D_llist,100,facecolor='red',label="lon",weights=LON_weights)
        ax1[0].set_title('dSphs longitude')
        ax1[1].hist(D_blist,100,facecolor='blue',label="lat",weights=LAT_weights)
        ax1[1].set_title('dSphs latitude')
        plt.savefig('sample_PDF_25dSphs.pdf')

        
        
    if(task3):
        print('+++++ getting counts from fits file ++++++')
        countdata,expdata,plslat,plslon = importdata(file1,file2,plsfile)
        print('+++++ computing total count (sum over E bins)')
        countstot = countdata.sum(axis=0)
        print('len cdata=',len(countdata),len(countdata[0]),len(countdata[0,0]))
        # adding 5 degrees to the right of last long bin, for cyclic computation
        countdata=np.dstack((countdata,countdata[:,:,:6]))
        expdata=np.dstack((expdata,expdata[:,:,:6]))
        print('new len cdata=',len(countdata),len(countdata[0]),len(countdata[0,0]))
        #print('cyclic: ',countdata[0,900,5],countdata[0,900,3605])
        # +++ excluding dSphs from list ++++
        D_blist, D_llist = getdSphs('dSphs_positions.dat',all='all')
        D_blist=np.array(D_blist).flatten()
        D_llist=np.array(D_llist).flatten()
        '''
        print('+++++ filling dSphs positions with counts ++++')
        fill_regions(D_llist,D_blist,countdata,expdata,sample='dSphs_05_all')
        print('+++++ filling voids positions with counts ++++')
        quit()
        '''
        l_void = []
        b_void = []
        vfile = fileinput.input('voids_all_45dSphs.dat')
        for linea in vfile:
            lon, lat = linea.split()
            l_void.append(float(lon))
            b_void.append(float(lat))
        vfile.close()
        fill_regions(l_void,b_void,countdata,expdata,sample='voids_all_45dSphs')

    if(task3a):
        count_voids = np.loadtxt('counts_voids_all_05.dat')
        counttotV = np.sum(count_voids,axis=1)
        print('min max countot=',np.min(counttotV),np.max(counttotV))
        coords = np.loadtxt('voids_all_05.dat')
        lonV = coords[:,0]
        latV = coords[:,1]
        #print('min lonV =',min(lonV),max(lonV),min(latV),max(latV))
        latD, lonD = getdSphs('dSphs_positions.dat',all='no')
        latD,lonD=np.array(latD),np.array(lonD)
        for i in range(len(lonV)):
            if(lonV[i]<=180):
                lonV[i] = 180. - lonV[i]
            else:
                lonV[i] = 360. - lonV[i] + 180.
        for j in range(len(lonD)):
            if(lonD[j]<=180):
                lonD[j] = 180. - lonD[j]
            else:
                lonD[j] = 360. - lonD[j] + 180.
        
        plt.figure()
        fig, ax = plt.subplots()
        sc=plt.scatter(lonV,latV,c=np.log10(counttotV),
                       vmin=np.log10(1.), vmax=np.log10(max(counttotV)),s=5)
        plt.scatter(lonD,latD,c='red',s=10)
        plt.xlabel('longitude [deg]')
        plt.ylabel('latitude [deg]')
        plt.xlim(0,360)
        plt.xticks([0,60,120,180,240,300,360],
                   ['180','120','60','0/360','300','240','180'])
        plt.ylim(-90,90)
        cb=plt.colorbar(sc)
        cb.set_label('log (total counts)')
        plt.savefig('sky_with_dSphs_GCoord.pdf')                

        
    if(task4):
        print('++++++ importing counts from dSphs +++++')
        count_dSphs = np.loadtxt('counts_dSphs_05.dat')
        exp_dSphs = np.loadtxt('exp_dSphs_05.dat')
        Ndwarfs=len(count_dSphs)
        print('++++++ importing counts from voids +++++')
        count_voids = np.loadtxt('counts_voids_all_05.dat')
        Nvoids=len(count_voids)
        print('+++++++ KS test, no DM +++++++++++++++++++++')
        counttotD = []
        for i in range(len(count_dSphs)):
            row = [float(count_dSphs[i][j]) for j in range(len(count_dSphs[i]))]
            counttotD.append(np.array(row).sum())
        print('counttotD=',np.round(counttotD))
        #quit()
        counttotV = []
        for i in range(len(count_voids)):
            row = [float(count_voids[i][j]) for j in range(len(count_voids[i]))]
            counttotV.append(np.array(row).sum())
        KS, pval = KStest(counttotD, counttotV)
        print('KS, pval = ',KS,pval)
        Dweights = np.ones_like(counttotD)/len(counttotD)
        Vweights = np.ones_like(counttotV)/len(counttotV)
        plt.hist(counttotD,100,facecolor='red',alpha=0.5,weights=Dweights,
                 cumulative=True,label='dSphs',
                 range=(np.min(counttotV),np.max(counttotD)))
        plt.hist(counttotV,100,facecolor='black',alpha=0.5,weights=Vweights,
                 cumulative=True,label='bckg',
                 range=(np.min(counttotV),np.max(counttotD)))
        plt.xlim(0,np.max(counttotD))
        plt.xlabel('counts',fontsize=18)
        plt.ylabel('weighted cumulative',fontsize=18)
        plt.text(100,0.8, '$N_{dSphs}$='+str(Ndwarfs),fontsize=10,color='black')
        plt.text(100,0.75, '$N_{bckg}$='+str(Nvoids),fontsize=10,color='black')
        plt.text(100,0.7, 'p-value = '+str(round(pval,2)),fontsize=10,color='black')
        plt.legend(bbox_to_anchor=(0.2,1.0))
        plt.savefig('KS.pdf')
        quit()
        print('+++++++ KS test, with DM in scan ++++++++')
        rad = 0.5 # deg
        Area = np.pi*rad**2
        DOmega = Area*(np.pi/180.)**2 # Delta Omega of regions
        countB = lambda sv,mx : countBCKG_dSphs(sv,mx,DOmega,exp_dSphs,count_dSphs)
        countBtot = lambda sv,mx : countB(sv,mx).sum(axis=1)
        mxgrid=[10,50,100,500]
        svgrid=np.logspace(np.log10(1.E-27),np.log10(1.E-23),20)
        mxlist,svlist,Flist=svmass_scan(countBtot,mxgrid,svgrid)

        yesnos=[]
        for i in range(len(Flist)):
            mystr=str(stats.ks_2samp(counttotV,Flist[i]))
            str1,str2=mystr.split()
            pval=str2[7:-1]
            print('mx,sv,pval: ',mxlist[i],svlist[i],pval)
            yesno=np.sign(float(pval)-0.05)
            yesnos.append(yesno)
        for i in range(len(mxlist)):
            print('mx,sv,yesno=',mxlist[i],svlist[i],yesnos[i])
        np.savetxt('KS_bound.dat',np.transpose([mxlist,svlist,yesnos]))
        quit()
        mxlist, svlist, yesnos = np.loadtxt('KS_bound.dat',unpack=True)
        xi=np.logspace(np.log10(min(mxlist)),np.log10(max(mxlist)),num=400)
        yi=np.logspace(np.log10(min(svlist)),np.log10(max(svlist)),num=400)
        diffgrid=griddata(mxlist,svlist,yesnos,xi,yi,interp='nn')
        fig1,ax1 = plt.subplots()
        plt.contour(xi,yi,diffgrid,levels=[0.],colors='black')        
        plt.ylim(1.E-25,1.E-24)
        plt.xlim(10,100)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$m_{DM}$ [GeV]',fontsize=18)
        plt.ylabel('$<\sigma v>$ [$cm^3/s$]',fontsize=18)
        plt.text(20.,4.5E-25,'KS 95%CL',fontsize=10,rotation=42)
        plt.savefig('sv_mx_KS.pdf')

        
    if(task5):
        print('++++++ importing counts from dSphs +++++')
        filedwarfs0 = fileinput.input('counts_dSphs.dat')
        filedwarfs1 = fileinput.input('exp_dSphs.dat')
        count_dSphs = readdata(filedwarfs0)
        exp_dSphs = readdata(filedwarfs1)
        filedwarfs0.close()
        filedwarfs1.close()
        counttotD = []
        for i in range(len(count_dSphs)):
            row = [float(count_dSphs[i][j]) for j in range(len(count_dSphs[i]))]
            counttotD.append(np.array(row).sum())
        print('++++++ importing counts from voids +++++')
        filevoids = fileinput.input('counts_voids_all2.dat')
        count_voids = readdata(filevoids)
        filevoids.close()
        counttotV = []
        for i in range(len(count_voids)):
            row = [float(count_voids[i][j]) for j in range(len(count_voids[i]))]
            counttotV.append(np.array(row).sum())
        print('++++++  BCKG flux at 5% ++++++++++++++++++++++++++++++')
        count_voids_ord = np.sort(counttotV)
        lenvoids = len(count_voids_ord)
        count_void_05 = count_voids_ord[round(0.05*lenvoids)]
        print(' count_voids at 5%: ',count_void_05)
        lowlist = []
        for i in range(len(counttotD)):
            pos = np.searchsorted(count_voids_ord,counttotD[i])/lenvoids
            print('dwarf #',i+1,' position: ',pos)
            if(pos < 0.05):
                #observed flux for these dwarfs is already below global 5%
                lowlist.append(i)
        #excluding those dwarfs from the subsequent analysis
        count_dSphs_red = np.delete(count_dSphs,lowlist,axis=0)
        exp_dSphs_red = np.delete(exp_dSphs,lowlist,axis=0)
        print('len of reduced sample = ',len(count_dSphs_red))
        rad = 0.5 # deg
        Area = np.pi*rad**2
        DOmega = Area*(np.pi/180.)**2 # Delta Omega of regions
        countB = lambda sv,mx : countBCKG_dSphs(sv,mx,DOmega,exp_dSphs_red,count_dSphs_red)
        countBtot = lambda sv,mx : countB(sv,mx).sum(axis=1)
        mxlist,svlist,Flist=svmass_scan(countBtot,10,1000,1.E-27,1.E-24,20,20)
        print('+++++++ obtaining contours ++++++++++++++++++++++++++++')
        yesnos=[]
        yesnosCons=[]
        for i in range(len(Flist)):
            yesnos.append(np.sign(min(Flist[i])-count_void_05))
            yesnosCons.append(np.sign( min(countBtot(svlist[i],mxlist[i])) ))
        xi=np.logspace(np.log10(min(mxlist)),np.log10(max(mxlist)),num=500)
        yi=np.logspace(np.log10(min(svlist)),np.log10(max(svlist)),num=500)
        diffgrid=griddata(mxlist,svlist,yesnos,xi,yi,interp='nn')
        diffgridCons=griddata(mxlist,svlist,yesnosCons,xi,yi,interp='nn')
        fig1,ax1 = plt.subplots()
        plt.contour(xi,yi,diffgrid,levels=[0.],colors='black')
        plt.contour(xi,yi,diffgridCons,levels=[0.],colors='gray')
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('sv_mx_05_2.pdf')


    if(task6):
        '''
        x_voids = np.loadtxt('voids_all_clean.dat')
        count_voids = np.loadtxt('counts_voids_all_clean.dat')
        counttotV = count_voids[:,0]
        print('PDF=',PDFvect2(counttotV,x_voids,0.64,0.08))
        quit()
        sigma_list = np.logspace(np.log10(0.2),np.log10(5),10)
        sigmaY_list = np.logspace(np.log10(0.03),np.log10(0.5),10)
        sigL = []
        varsigL = []
        os.system('rm pdf_grid_bin1.dat')
        for i in range(len(sigma_list)):
            for j in range(len(sigmaY_list)):
                fileout = open('pdf_grid_bin1.dat',"a")
                res = PDFvect2(counttotV,x_voids,sigma_list[i],sigmaY_list[j])
                fileout.write(" ".join([str(sigma_list[i]),str(sigmaY_list[j]),
                                        str(res),"\n"]))
                fileout.close()
        '''
        # ++++++ next is for plotting ++++++++++++++++++++++++++++++++++++++++++++
        sigma,varsigma,lnPDF=np.loadtxt('pdf_grid_integ_05.dat',unpack=True)
        sigma=np.array(sigma).reshape((-1,1))
        varsigma=np.array(varsigma).reshape((-1,1))
        points=np.hstack((sigma,varsigma))
        lnPDF=np.array(lnPDF)
        X=np.logspace(np.log10(min(sigma)),np.log10(max(sigma)),100)
        Y=np.logspace(np.log10(min(varsigma)),np.log10(max(varsigma)),100)
        func=lambda x,y: interpolate.griddata(points,lnPDF,(x,y),method='cubic',rescale=True)
        S,V=np.meshgrid(X,Y)
        Z=func(S,V)
        Xstar=np.round(X[np.argmax(func(X,Y))],2)
        Ystar=np.round(Y[np.argmax(func(X,Y))],2)
        Zmax=np.max(func(X,Y))
        plt.scatter(S,V,c=Z)
        plt.plot([Xstar],[Ystar],marker='x',color='black')
        rcParams['contour.negative_linestyle'] = 'solid'
        CP=plt.contour(S,V,Z,levels=[1.03*Zmax,1.01*Zmax,1.005*Zmax],
                       colors='k',ls=['solid','dashed','dotted'])
        plt.text(1.5,0.14,'('+str(Xstar)+','+str(Ystar)+')')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(min(sigma),max(sigma))
        plt.ylim(min(varsigma),max(varsigma))
        plt.xlabel('$\sigma$',fontsize=18)
        plt.ylabel(r'$\varsigma$',fontsize=18)
        plt.title('Ln(PDF)')
        plt.savefig('PDFgrid_integ_05.pdf')
        
    if(task7):
        x_voids = np.loadtxt('voids_all_clean.dat')
        count_voids = np.loadtxt('counts_voids_all_clean.dat')
        counttotV = count_voids[:,0]
        i=5
        x = x_voids[i]
        y = counttotV[i]
        print('x,y = ',x,y)
        coords =np.delete(x_voids,i,axis=0)
        counts =np.delete(counttotV,i)
        flag1=True  # plot PDF vs sigma
        flag2=False # plot PDF vs counts
        if(flag1):
            sigmalist = np.logspace(np.log10(0.2),np.log10(5),20)
            pdf_arr1=[]
            pdf_arr2=[]
            pdf_arr3=[]
            [pdf_arr1.append(PDF(y,x,counts,coords,sigma,0.03)) for sigma in sigmalist]
            [pdf_arr2.append(PDF(y,x,counts,coords,sigma,0.08)) for sigma in sigmalist]
            [pdf_arr3.append(PDF(y,x,counts,coords,sigma,0.2)) for sigma in sigmalist]
            plt.loglog(sigmalist,pdf_arr1,c='blue',label=r'$\varsigma=0.03$')
            plt.loglog(sigmalist,pdf_arr2,c='red',label=r'$\varsigma=0.08$')
            plt.loglog(sigmalist,pdf_arr3,c='green',label=r'$\varsigma=0.2$')
            plt.legend(loc=2,borderaxespad=0.,fontsize=12)
            plt.ylim(1.E-7,1.E-5)
            plt.xlabel(r'$\sigma$',fontsize=18)
            plt.ylabel(r'$PDF_i$',fontsize=18)
            plt.tight_layout()
            plt.tick_params(right=True,top=True,labelsize=12)
            plt.tick_params(right=True,top=True,which='minor')
            plt.savefig('PDFi_vs_sigma_sevvarsigma_after.pdf')
        if(flag2):
            ylist = np.linspace(50,300,40)
            pdf1_vsy = []
            pdf2_vsy = []
            pdf3_vsy = []
            [pdf1_vsy.append(PDF(c,x,counts,coords,0.64,0.03)) for c in ylist]
            [pdf2_vsy.append(PDF(c,x,counts,coords,0.64,0.08)) for c in ylist]
            [pdf3_vsy.append(PDF(c,x,counts,coords,0.64,0.2)) for c in ylist]
            plt.semilogy(ylist,pdf1_vsy,color='blue',label=r'$\varsigma=0.03$')
            plt.semilogy(ylist,pdf2_vsy,color='red',label=r'$\varsigma=0.08$')
            plt.semilogy(ylist,pdf3_vsy,color='green',label=r'$\varsigma=0.2$')
            plt.xlabel('counts',fontsize=18)
            plt.ylabel('$PDF_i$',fontsize=14)
            plt.ylim(1.E-9,1.E-5)
            plt.legend(loc=2,borderaxespad=0.,fontsize=12)
            plt.tick_params(right=True,top=True,labelsize=12)
            plt.tick_params(right=True,top=True,which='minor')
            plt.tight_layout()
            plt.text(250,4.E-6,r'$\sigma=0.64$',fontsize=12)
            plt.savefig('PDFi_vs_y.pdf')

        
    if(task8):
        x_voids = np.loadtxt('voids_all_05.dat')
        #lon_voids,lat_voids = np.loadtxt('voids_all4.dat',unpack=True)
        lat_dSphs, lon_dSphs = getdSphs('dSphs_positions.dat',all='no')
        count_voids = np.loadtxt('counts_voids_all_05.dat')
        counttotV=np.sum(count_voids,axis=1)
        argdel=np.where(counttotV==0.)[0]
        counttotV=np.delete(counttotV,argdel)
        count_voids=np.delete(count_voids,argdel,axis=0)
        x_voids=np.delete(x_voids,argdel,axis=0)
        count_dSphs = np.loadtxt('counts_dSphs_05.dat')
        counttotD=np.sum(count_dSphs,axis=1)
        x_dSphs = np.hstack((lon_dSphs.reshape((-1,1)),lat_dSphs.reshape((-1,1))))
        # ++++ reagrupping in 6 bins; can't have c=0 for log reasons ++++++++
        b = [0,1,2,3,4,6,len(count_voids[0])]
        bincountV = count_voids[:,b[0]].reshape((-1,1))
        bincountD = count_dSphs[:,b[0]].reshape((-1,1))
        for i in range(1,len(b)-1):
            if(b[i]==b[i+1]-1):
                bincountV=np.concatenate((bincountV,count_voids[:,b[i]].reshape((-1,1))),axis=1)
                bincountD=np.concatenate((bincountD,count_dSphs[:,b[i]].reshape((-1,1))),axis=1)
            else:
                BinV=np.sum(count_voids[:,b[i]:b[i+1]],axis=1)
                BinD=np.sum(count_dSphs[:,b[i]:b[i+1]],axis=1)
                bincountV=np.concatenate((bincountV,BinV.reshape((-1,1))), axis=1)
                bincountD=np.concatenate((bincountD,BinD.reshape((-1,1))), axis=1)
        #print('zeros=',np.where(bincountV==0))
        #print('min bincountv=',np.min(bincountV))
        #bincountV=np.ma.masked_where(bincountV==0,bincountV)
        
        # ++++++++++ computing prediction ++++++++++++++++++++++++++++++++++
        sigma=1.58
        Nbins=len(bincountV[0])
        lnbckgest=[]
        for i in range(Nbins):
            print('+++++++++ this is bin # ',i)
            countV = bincountV[:,i]
            countD = bincountD[:,i]
            yhat_1arg=partial(yhat_arr,xD=x_dSphs,yi=countV,xi=x_voids,sigma=sigma)
            p=Pool(4)
            yhat_list=p.map(yhat_1arg,range(len(x_dSphs)))
            lnbckgest.append(yhat_list)
            os.system('rm lnbckg_dSphs_bin_'+str(i)+'.dat')
            fileout = open('lnbckg_dSphs_bin_'+str(i)+'.dat',"a")
            for j in range(len(x_dSphs)):
                fileout.write(" ".join([str(np.log(countD[j])),str(yhat_list[j]),"\n"]))
            fileout.close()
        bckgest =  np.sum(np.exp(lnbckgest),axis=0)
        print('sum bckg_est = ',bckgest)
        print('ln(sum bckg est) = ',np.log(bckgest))
        print('Poisson = ',np.sqrt(counttotD))
        print('++++++++++++++++++computing variance++++++++++++++++++++++++++')
        #sigma=2.54
        yhat_1arg=partial(yhat_arr,xD=x_dSphs,yi=counttotV,xi=x_voids,sigma=sigma)
        p=Pool(4)
        yhat_list=p.map(yhat_1arg,range(len(x_dSphs)))
        w2_2=np.array(yhat_list)**2
        yhat2_1arg=partial(yhat2_arr,xD=x_dSphs,yi=counttotV,xi=x_voids,sigma=sigma)
        p=Pool(4)
        w2_1=p.map(yhat2_1arg,range(len(x_dSphs)))
        varsigma=0.16
        Var = varsigma**2 + np.array(w2_1) - w2_2
        DeltaB = bckgest*np.sqrt(Var)
        print('Delta ln = ',np.sqrt(Var))
        print('DeltaB = ',DeltaB)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        for i in range(len(bckgest)):
            print(counttotD[i],' $\pm$ ',np.round(np.sqrt(counttotD[i])),' & ',
                  np.round(bckgest[i],2),' & ',
                  np.round(np.log(bckgest[i]),2),' $\pm$ ',np.round(np.sqrt(Var[i]),2))
        


        
    if(task8post):
        x_voids =  np.loadtxt('voids_all4.dat')
        count_voids = np.loadtxt('counts_voids_all4.dat')
        # ++++ reagrupping in 6 bins; can't have c=0 for log reasons ++++++++
        b = [0,1,2,3,4,6,len(count_voids[0])]
        bincountV = count_voids[:,b[0]].reshape((-1,1))
        for i in range(1,len(b)-1):
            if(b[i]==b[i+1]-1):
                bincountV=np.concatenate((bincountV,count_voids[:,b[i]].reshape((-1,1))),axis=1)
            else:
                BinV=np.sum(count_voids[:,b[i]:b[i+1]],axis=1)
                bincountV=np.concatenate((bincountV,BinV.reshape((-1,1))), axis=1)
        # +++++ computing no. of sigmas +++++++++++++++++++
        sigma=0.64
        varsigma=0.08
        Nvoids = len(x_voids)
        Nbins=len(bincountV[0])
        counttotV=np.sum(count_voids,axis=1)
        '''
        diff_1arg=partial(diff_yhat_VOID,yi=counttotV,xi=x_voids,sigma=sigma,varsigma=varsigma)
        p=Pool(4)
        diff_list=p.map(diff_1arg,range(Nvoids))
        fileout = open('Nsigmas_total.dat',"a")
        fileout.write(" ".join([str(diff_list),"\n"]))
        quit()
        '''
        '''
        Nsigs=[]
        for i in range(Nbins):
            print('+++++++++ this is bin # ',i)
            fileout = open('Nsigmas_bin_'+str(i)+'.dat',"a") 
            countV = bincountV[:,i]
            diff_1arg=partial(diff_yhat_VOID,yi=countV,xi=x_voids,sigma=sigma,varsigma=varsigma)
            p=Pool(4)
            diff_list=p.map(diff_1arg,range(Nvoids))
            Nsigs.append(diff_list)
            fileout.write(" ".join([str(Nsigs[i]),"\n"]))
        print('lens Nsigs = ',len(Nsigs),len(Nsigs[0]))
        print('Nsigs = ',Nsigs[0,:100])
        '''
        #Nsig0=np.loadtxt('Nsigmas_bin_0.dat',dtype='str')
        Nsig0 = np.loadtxt('Nsigmas_total.dat',dtype='str')
        Nsigmas=[]
        Nsigmas.append(np.float(Nsig0[0][0][3:-2]))
        for i in range(1,len(Nsig0[0])-1):
            Nsigmas.append(np.float(Nsig0[0][i][2:-2]))
        Nsigmas.append(np.float(Nsig0[0][-1][2:-3]))
        #print('Nsigmas = ',len(Nsigmas),Nsigmas[0],Nsigmas[-1],Nsigmas[1],np.max(Nsigmas))
        print('Nsigmas=',Nsigmas[:100])
        argarr=np.where(np.array(Nsigmas)>=8.0)
        pval = len(argarr[0])/Nvoids
        print('p-val =',pval)
        plt.hist(Nsigmas,20)
        plt.yscale('log')
        plt.xlabel('# of "sigmas" = $|\ln b_i-\widehat{\ln b_i}|/\Delta(\ln b_i)$')
        plt.text(10,1.E4,'$p_{>5.0}=$'+str(round(pval,5)))
        plt.title('integrated counts')
        plt.show()
        quit()
        plt.savefig('std_distribution.pdf')


    if(task8a):
        x_voids = np.loadtxt('voids_all_05.dat')
        count_voids = np.loadtxt('counts_voids_all_05.dat')
        countB0 = count_voids[:,0]
        argdel=np.where(countB0==0.)[0]
        countB0=np.delete(countB0,argdel)
        x_voids=np.delete(x_voids,argdel,axis=0)
        lncountB0 = np.log(countB0)
        blist, llist = getdSphs('dSphs_positions.dat',all='no')
        i=0 #cerchietto id
        lny=lncountB0[i]
        x=[x_voids[i,0],x_voids[i,1]]
        print('lny,x=',lny,x)
        
        x_voids=np.delete(x_voids,i,axis=0)
        lncountB0=np.delete(lncountB0,i)
        print('PDF_B=',PDF_B(lny,x,lncountB0,x_voids,1.5,0.16))
        sigmalist=np.logspace(np.log10(0.5),np.log10(5.0),50)
        print('sigmalist=',sigmalist)
        '''
        pdflist1=[]
        pdflist2=[]
        pdflist3=[]
        for sigma in sigmalist:
            pdflist1.append(PDF_B(lny,x,lncountB0,x_voids,sigma,0.08))
            pdflist2.append(PDF_B(lny,x,lncountB0,x_voids,sigma,0.16))
            pdflist3.append(PDF_B(lny,x,lncountB0,x_voids,sigma,0.24))
        plt.loglog(sigmalist,pdflist1,color='blue',label=r'$\varsigma=0.08$')
        plt.loglog(sigmalist,pdflist2,color='red',label=r'$\varsigma=0.16$')
        plt.loglog(sigmalist,pdflist3,color='green',label=r'$\varsigma=0.24$')
        plt.ylim(1.e-6,1.e-4)
        plt.ylabel('$PDF_i$',size=18)
        plt.xlabel('$\sigma$',size=18)
        plt.tick_params(right=True,top=True,labelsize=12)
        plt.tick_params(right=True,top=True,which='minor')
        plt.legend(bbox_to_anchor=(0.,1.0),loc=2,borderaxespad=0.)
        plt.savefig('PDF_vs_sigma_after.pdf')
        
        quit()
        '''
        lncountlist=np.linspace(lny-2.,lny+2,50)
        pdflist1=[]
        pdflist2=[]
        pdflist3=[]
        for lnc in lncountlist:
            pdflist1.append(PDF_B(lnc,x,lncountB0,x_voids,1.58,0.08))
            pdflist2.append(PDF_B(lnc,x,lncountB0,x_voids,1.58,0.16))
            pdflist3.append(PDF_B(lnc,x,lncountB0,x_voids,1.58,0.24))
        plt.loglog(np.exp(lncountlist),pdflist1,color='blue',label=r'$\varsigma=0.08$')
        plt.loglog(np.exp(lncountlist),pdflist2,color='red',label=r'$\varsigma=0.16$')
        plt.loglog(np.exp(lncountlist),pdflist3,color='green',label=r'$\varsigma=0.24$')
        plt.ylim(1.e-6,1.e-4)
        plt.xlim(10,100.)
        plt.xlabel('counts',size=18)
        plt.ylabel('$PDF_i$',size=18)
        #plt.savefig('PDF_vs_sigma.pdf')
        plt.text(11.,3.e-5,'$\sigma=1.58^\circ$',size=12)
        plt.legend(bbox_to_anchor=(0.,1.0),loc=2,borderaxespad=0.)
        plt.tick_params(right=True,top=True,labelsize=12)
        plt.tick_params(right=True,top=True,which='minor')
        plt.savefig('PDF_vs_counts_after.pdf')
        
        

    if(task8b):
        count_voids = np.loadtxt('counts_voids_all_05.dat')
        b1=count_voids[:,0]
        argdel1=np.where(b1<=1.)
        count_voids=np.delete(count_voids,argdel1,axis=0)
        b2=count_voids[:,1]
        argdel2=np.where(b2<=1.)
        count_voids=np.delete(count_voids,argdel2,axis=0)
        lncounts=np.log(count_voids)
        lnb1 = lncounts[:,0]
        lnb2 =  lncounts[:,1]
        lnb2ob1 = lnb2/lnb1
        f,ax=plt.subplots(1,3)
        ax[0].hist(lnb2ob1,bins=20,normed=True,label='ln($b_2$)/ln($b_1$)')
        ax[0].legend(bbox_to_anchor=(0.0,1.0),loc=2,borderaxespad=0.)
        ax[1].hist(lnb2,normed=True,label='ln($b_2$)')
        ax[2].hist(lnb1,normed=True,label='ln($b_1$)')
        ax[1].legend(bbox_to_anchor=(0.,1.0),loc=2,borderaxespad=0.)
        ax[2].legend(bbox_to_anchor=(0.,1.0),loc=2,borderaxespad=0.)
        plt.savefig('ratio_21.pdf')

    if(task91):
        os.system('ls -v profiling_case1N/ > lsout1.dat')
        filesP = np.loadtxt('lsout1.dat',dtype='str')
        logLarrAll = []
        for f in filesP:
            arrays = np.loadtxt('profiling_case1N/'+f[2:-1])
            mxarr = arrays[:,0]
            svarr = arrays[:,1]
            logL = arrays[:,2]
            logLarrAll.append(logL)            
        mxarr = np.reshape( np.array(mxarr) ,(-1,1))
        svarr = np.reshape( np.array(svarr) ,(-1,1))
        points=np.hstack((mxarr,svarr))
        mxgrid = np.logspace(math.log10(min(mxarr)),math.log10(max(mxarr)),20)
        svgrid = np.logspace(math.log10(min(svarr)),math.log10(max(svarr)),100)
        
        Ndwarf = len(filesP)
        deltaEXCL = 3.84
        for i in range(Ndwarf):
            print("+++++++++++++++  this is dwarf ",i," ++++++++++++++++++++++")
            os.system('rm excl_limits_case1N/sv_mx_dwarf'+str(i)+'.dat')
            fileout = open('excl_limits_case1N/sv_mx_dwarf'+str(i)+'.dat',"a")
            m2logPL = lambda mx, sv: interpolate.griddata(points,logLarrAll[i],
                                                          (mx,sv),method='cubic',rescale=True)
        
            #plt.plot(np.log10(svarr[:20]),logLarrAll[0][:20],color='black')
            #plt.plot(np.log10(svgrid),m2logPL(6,svgrid),color='blue')
            #plt.show()
            #quit()
            
            for j in range(len(mxgrid)):
                print("mDM = ",mxgrid[j])
                m2logPL_arr = []
                for k in range(len(svgrid)):
                    m2logPL_arr.append(m2logPL(mxgrid[j],svgrid[k]))
                    #print('  log sv = ',svgrid[k],' m2logPL = ',m2logPL_arr[k])
                m2LLmin = min(m2logPL_arr)
                kmin = np.argmin(m2logPL_arr)
                #print('  m2logPL_min = ',m2LLmin,' at k = ',kmin)
                for k in range(kmin,len(svgrid)):
                    if(m2logPL_arr[k] - m2LLmin > deltaEXCL):
                        svEXCL = svgrid[k]
                        print('    sv excl = ',svEXCL)
                        break
                fileout.write(" ".join([str(mxgrid[j]),str(svEXCL),"\n"]) )
        '''
        quit()
        mxlim,svlim=np.loadtxt('excl_limits_case1/sv_mx_dwarf0.dat',unpack=True)
        svfunc = savgol_filter(svlim,51,3)
        plt.plot(mxlim,svlim)
        plt.plot(mxlim,svfunc)
        plt.show()
        '''

    if(task920):
        os.system('ls -v profiling_case20/ > lsout1.dat')
        filesP = np.loadtxt('lsout1.dat',dtype='str')
        logLarrAll = []
        for f in filesP:
            arrays = np.loadtxt('profiling_case20/'+f[2:-1])
            mxarr = arrays[:,0]
            svarr = arrays[:,1]
            logL = arrays[:,2]
            Jarr = arrays[:,3]
            Jerr = arrays[:,4]
            lB0arr = arrays[:,5]
            lB0err = arrays[:,6]
            logLarrAll.append(logL)            
        mxarr = np.reshape( np.array(mxarr) ,(-1,1))
        svarr = np.reshape( np.array(svarr) ,(-1,1))
        points=np.hstack((mxarr,svarr))
        print('points = ',points[:4])
        mxgrid = np.logspace(math.log10(min(mxarr)),math.log10(max(mxarr)),20)
        svgrid = np.logspace(math.log10(min(svarr)),math.log10(max(svarr)),20)
        '''
        plt.plot(np.log10(svarr[:20]),logLarrAll[17][:20])
        plt.plot(np.log10(svarr[20:40]),logLarrAll[17][20:40])
        plt.plot(np.log10(svarr[40:60]),logLarrAll[17][40:60])
        plt.plot(np.log10(svarr[60:80]),logLarrAll[17][60:80])
        plt.show()
        quit()
        '''
        deltaEXCL = 3.84
        for i in range(19):
            print("+++++++++++++++  this is dwarf ",i," ++++++++++++++++++++++")
            os.system('rm excl_limits_case20/sv_mx_dwarf'+str(i)+'.dat')
            fileout = open('excl_limits_case20/sv_mx_dwarf'+str(i)+'.dat',"a")
            m2logPL = lambda mx, sv: interpolate.griddata(points,logLarrAll[i],
                                                          (mx,sv),method='cubic',rescale=True)
        
            #plt.plot(np.log10(svarr[:20]),logLarrAll[0][:20],color='black')
            #plt.plot(np.log10(svgrid),m2logPL(6,svgrid),color='blue')
            #plt.show()
            #quit()
            
            for j in range(len(mxgrid)):
                print("mDM = ",mxgrid[j])
                m2logPL_arr = []
                for k in range(len(svgrid)):
                    m2logPL_arr.append(m2logPL(mxgrid[j],svgrid[k]))
                    #print('  log sv = ',svgrid[k],' m2logPL = ',m2logPL_arr[k])
                m2LLmin = min(m2logPL_arr)
                kmin = np.argmin(m2logPL_arr)
                #print('  m2logPL_min = ',m2LLmin,' at k = ',kmin)
                for k in range(kmin,len(svgrid)):
                    if(m2logPL_arr[k] - m2LLmin > deltaEXCL):
                        svEXCL = svgrid[k]
                        print('    sv excl = ',svEXCL)
                        break
                fileout.write(" ".join([str(mxgrid[j]),str(svEXCL),"\n"]) )


    if(task92):
        os.system('ls -v profiling_case2N/ > lsout1.dat')
        filesP = np.loadtxt('lsout1.dat',dtype='str')
        logLarrAll = []
        for f in filesP:
            arrays = np.loadtxt('profiling_case2N/'+f[2:-1])
            mxarr = arrays[:,0]
            svarr = arrays[:,1]
            Jarr = arrays[:,2]
            Jerr = arrays[:,3]
            logL = arrays[:,4]            
            logLarrAll.append(logL)            
        mxarr = np.reshape( np.array(mxarr) ,(-1,1))
        svarr = np.reshape( np.array(svarr) ,(-1,1))
        points=np.hstack((mxarr,svarr))
        mxgrid = np.logspace(math.log10(min(mxarr)),math.log10(max(mxarr)),20)
        svgrid = np.logspace(math.log10(min(svarr)),math.log10(max(svarr)),100)
        '''
        plt.plot(np.log10(svarr[:20]),logLarrAll[17][:20])
        plt.plot(np.log10(svarr[20:40]),logLarrAll[17][20:40])
        plt.plot(np.log10(svarr[40:60]),logLarrAll[17][40:60])
        plt.plot(np.log10(svarr[60:80]),logLarrAll[17][60:80])
        plt.plot(np.log10(svarr[380:400]),logLarrAll[17][380:400])
        plt.show()
        quit()
        '''
        '''
        #print('mxarr avant:',mxarr[:10])
        mxarr,svarr,a3,a4,logLarrAll[17]=np.loadtxt('profiling_case2N/sv_mx_dwarf17.dat',unpack=True)
        '''
        mxarr=mxarr.reshape((-1,1))
        svarr=svarr.reshape((-1,1))
        #print('mxarr après:',mxarr[:10])
        points=np.hstack((mxarr,svarr))
        print('lens :',len(mxarr),len(svarr),len(logLarrAll[17]),len(points))
        
        Ndwarf = len(filesP)
        deltaEXCL = 3.84
        for i in range(Ndwarf):
            print("+++++++++++++++  this is dwarf ",i," ++++++++++++++++++++++")
            os.system('rm excl_limits_case2N/sv_mx_dwarf'+str(i)+'.dat')
            fileout = open('excl_limits_case2N/sv_mx_dwarf'+str(i)+'.dat',"a")
            m2logPL = lambda mx, sv: interpolate.griddata(points,logLarrAll[i],
                                                          (mx,sv),method='cubic',rescale=True)
            '''
            plt.plot(np.log10(svarr[380:400]),logLarrAll[17][380:400],color='black')
            plt.plot(np.log10(svgrid),m2logPL(1000,svgrid),color='blue')
            plt.show()
            quit()
            print('m2logPL(1000,1.E-24) = ',m2logPL(1000,1.4366554802837756e-24))
            quit()
            '''
            for j in range(len(mxgrid)):
                print("mDM = ",mxgrid[j])
                m2logPL_arr = []
                for k in range(len(svgrid)):
                    m2logPL_arr.append(m2logPL(mxgrid[j],svgrid[k]))
                    #print('  log sv = ',svgrid[k],' m2logPL = ',m2logPL_arr[k])
                m2LLmin = min(m2logPL_arr)
                kmin = np.argmin(m2logPL_arr)
                #print('  m2logPL_min = ',m2LLmin,' at k = ',kmin)
                for k in range(kmin,len(svgrid)):
                    if(m2logPL_arr[k] - m2LLmin > deltaEXCL):
                        svEXCL = svgrid[k]
                        print('    sv excl = ',svEXCL)
                        break
                fileout.write(" ".join([str(mxgrid[j]),str(svEXCL),"\n"]) )


    if(task93):
        mxarr,svarr,logL,J1,J1e,J2,J2e,J3,J3e,J4,J4e,J5,J5e,J6,J6e=np.loadtxt('profiling_case3N/sv_mx.dat',unpack=True)
        
        #print('mxarr=',mxarr[0:20])
        #print('svarr=',svarr[0:20])
        #print('logL=',logL[0:20])
        '''
        argdel=[]
        for i in range(20):
            argdel.append(20*i+1)
            argdel.append(20*i+2)
            argdel.append(20*i+3)
        mxarr=np.delete(mxarr,argdel)
        svarr=np.delete(svarr,argdel)
        logL=np.delete(logL,argdel)
        print('svarr after =',svarr[0:20])
        print('logL after =',logL[0:20])
        '''
        #mxarrComa,svarrComa,jarr,jerr,logLComa=np.loadtxt('profiling_case2N/sv_mx_dwarf22.dat',
        #                                                  unpack=True)
        #print('svarrComa',svarrComa[0:20])
        #logL = logL - logLComa
        #print('logL=',logL[0:20])
        #quit()
        mxarr = np.reshape( np.array(mxarr) ,(-1,1))
        svarr = np.reshape( np.array(svarr) ,(-1,1))
        points=np.hstack((mxarr,svarr))
        mxgrid = np.logspace(math.log10(min(mxarr)),math.log10(max(mxarr)),20)
        svgrid = np.logspace(math.log10(min(svarr)),math.log10(max(svarr)),100)
        os.system('rm excl_limits_case3N/sv_mx.dat')
        fileout = open('excl_limits_case3N/sv_mx.dat',"a")
        m2logPL = lambda mx, sv: interpolate.griddata(points,logL,
                                                      (mx,sv),method='cubic',rescale=True)
        deltaEXCL=3.84
        for j in range(len(mxgrid)):
            print("mDM = ",mxgrid[j])
            m2logPL_arr = []
            for k in range(len(svgrid)):
                m2logPL_arr.append(m2logPL(mxgrid[j],svgrid[k]))
            m2LLmin = min(m2logPL_arr)
            kmin = np.argmin(m2logPL_arr)
            for k in range(kmin,len(svgrid)):
                if(m2logPL_arr[k] - m2LLmin > deltaEXCL):
                    svEXCL = svgrid[k]
                    print('    sv excl = ',svEXCL)
                    break
            fileout.write(" ".join([str(mxgrid[j]),str(svEXCL),"\n"]) )
        

    if(task94):
        idwarf=11
        mxarr,svarr,logL,Jarr,Jerr,Barr,Berr = np.loadtxt('profiling_case4N/sv_mx_dwarf'+str(idwarf)+'.dat',unpack=True)
        nanlist=[]
        for i in range(len(mxarr)):
            if(np.isnan(Jerr[i])):
                nanlist.append(i)
        mxarr=np.delete(mxarr,nanlist)
        svarr=np.delete(svarr,nanlist)
        logL=np.delete(logL,nanlist)
        #print('logL ',logL)
        #quit()
        mxarr = np.reshape( np.array(mxarr) ,(-1,1))
        svarr = np.reshape( np.array(svarr) ,(-1,1))
        logsvarr = np.log10(svarr)
        points=np.hstack((mxarr,svarr))
        mxgrid = np.logspace(math.log10(min(mxarr)),math.log10(max(mxarr)),20)
        svgrid = np.logspace(np.log10(min(svarr)),np.log10(max(svarr)),100)
        os.system('rm excl_limits_case4N/sv_mx_dwarf'+str(idwarf)+'.dat')
        fileout = open('excl_limits_case4N/sv_mx_dwarf'+str(idwarf)+'.dat',"a")
        m2logPL = lambda mx, sv: interpolate.griddata(points,logL,
                                                      (mx,sv),method='cubic',rescale=True)
        #print('svgrid=',svgrid[:79])
        #print('logL,mx,sv = ',logL[:19],svarr[:19])
        #print('m2logPL = ',m2logPL(6,svgrid[:19]),svgrid[:19])
        #quit()
        #plt.semilogx(svgrid[:99],m2logPL(6,svgrid[:99]))
        #plt.semilogx(svarr[:19],logL[:19])
        #plt.show()
        #quit()
        deltaEXCL = 3.84
        for j in range(len(mxgrid)):
            print("mDM = ",mxgrid[j])
            m2logPL_arr = []
            #for k in range(len(svgrid[:79])):
            for k in range(len(svgrid)):
                m2logPL_arr.append(m2logPL(mxgrid[j],svgrid[k]))
                #print('m2logL, sv =',m2logPL_arr[-1],svgrid[k])
            m2LLmin = min(m2logPL_arr)
            kmin = np.argmin(m2logPL_arr)
            #print('len m2logPL, svgrid: ',len(m2logPL_arr),len(svgrid))
            for k in range(kmin,len(svgrid)):
                if(m2logPL_arr[k] - m2LLmin > deltaEXCL):
                    svEXCL = svgrid[k]
                    print('    sv excl = ',svEXCL)
                    break
            fileout.write(" ".join([str(mxgrid[j]),str(svEXCL),"\n"]) )


    if(task95):
        os.system('cd profiling_case5N && ls sv_mx*.dat > ../mxfiles.dat')
        mxfiles = np.loadtxt('mxfiles.dat',dtype='str')
        mvals = [float(mxfiles[i][7:-5]) for i in range(len(mxfiles))]
        perm = np.argsort(mvals)
        print('mxfiles sorted = ',mxfiles[perm])
        mxfiles = mxfiles[perm]
        mxarr = []
        svarr = []
        logL = []
        for i in range(len(mxfiles)):
            entries=np.loadtxt('profiling_case5N/'+mxfiles[i][2:-1])
            #print('entries: ',entries[i,0])
            mx=entries[:,0]
            sv=entries[:,1]
            Like=entries[:,2]
            mxarr.append(mx)
            svarr.append(sv)
            logL.append(Like)
        mxarr=np.array(mxarr).flatten()
        svarr=np.array(svarr).flatten()
        logL=np.array(logL).flatten()
        #print('mx,sv,logL')
        #for i in range(len(mxarr)):
        #    print(mxarr[i],svarr[i],logL[i])
        #quit()
        mxarr = np.reshape( np.array(mxarr) ,(-1,1))
        svarr = np.reshape( np.array(svarr) ,(-1,1))
        points=np.hstack((mxarr,svarr))
        mxgrid = np.logspace(math.log10(min(mxarr)),math.log10(max(mxarr)),20)
        svgrid = np.logspace(math.log10(min(svarr)),math.log10(max(svarr)),100)
        os.system('rm excl_limits_case5N/sv_mx.dat')
        fileout = open('excl_limits_case5N/sv_mx.dat',"a")
        m2logPL = lambda mx, sv: interpolate.griddata(points,logL,
                                                      (mx,sv),method='cubic',rescale=True)
        deltaEXCL=3.84
        for j in range(len(mxgrid)):
            print("mDM = ",mxgrid[j])
            m2logPL_arr = []
            for k in range(len(svgrid)):
                m2logPL_arr.append(m2logPL(mxgrid[j],svgrid[k]))
            m2LLmin = min(m2logPL_arr)
            kmin = np.argmin(m2logPL_arr)
            for k in range(kmin,len(svgrid)):
                if(m2logPL_arr[k] - m2LLmin > deltaEXCL):
                    svEXCL = svgrid[k]
                    print('    sv excl = ',svEXCL)
                    break
            fileout.write(" ".join([str(mxgrid[j]),str(svEXCL),"\n"]) )

            
    if(taskL):
        mxarr1,svlim1=np.loadtxt('excl_limits_case1_5bins/sv_mx_dwarf14.dat',unpack=True)
        mxarr2,svlim2=np.loadtxt('excl_limits_case2/sv_mx_dwarf14.dat',unpack=True)
        svfunc1 = savgol_filter(svlim1,11,3)
        svfunc2 = savgol_filter(svlim2,11,3)
        plt.loglog(mxarr1,svfunc1,label='prof. J',c='blue')
        plt.loglog(mxarr2,svfunc2,label='prof. J & B (2 dSphs)',c='red')        
        plt.legend(bbox_to_anchor=(1.02,1.0),loc=2,borderaxespad=0.)
        plt.xlabel('DM mass [GeV]',fontsize=18)
        plt.ylabel('$\sigma v$ [cm$^3/s$',fontsize=18)
        plt.title('$b\bar{b}$-channel, Segue I')
        plt.savefig('sv_mx_SegueI_case2and4.pdf',bbox_inches='tight')


    if(figPDF):
        counts,pdf=np.loadtxt('PDF_Segue.dat',unpack=True)
        plt.loglog(counts,pdf)
        plt.ylim(1.E-51,1.E-4)
        plt.savefig('PDF_Segue.pdf')
        

    if(figMoney):
        mxarr,svlim=np.loadtxt('../Steigman.dat',unpack=True)
        mxarr0,svlim0=np.loadtxt('../Mazziotta.dat',unpack=True)
        mxarr1,svlim1=np.loadtxt('../Fermi-bound_red.dat',unpack=True)
        mxarr2,svlim2=np.loadtxt('excl_limits_case3N/sv_mx.dat',unpack=True)
        mxarr3,svlim3=np.loadtxt('excl_limits_case5N/sv_mx.dat',unpack=True)
        mxarr4,svlim4=np.loadtxt('../contour_100BR_bb_2s.dat',unpack=True)
        svfunc1 = savgol_filter(svlim1,11,3)
        svfunc2 = savgol_filter(svlim2,11,3)
        svfunc3 = savgol_filter(svlim3,9,3)
        plt.loglog(mxarr3,svlim3,c='red',label='prof. J & B (this work)',lw=2.0)
        plt.loglog(mxarr2,svlim2,c='blue',label='prof. J (this work)')
        plt.loglog(mxarr1,svlim1,c='black',label='Fermi-LAT, measured J-factors (2016)')
        plt.loglog(mxarr0,svlim0,c='gray',label='Mazziotta et al. (combined)')
        plt.loglog(mxarr4,svlim4,c='violet',label='Calore et al. 2015 (2$\sigma$)')

        plt.loglog(mxarr,svlim,ls='--',c='gray')
        plt.text(205,1.6E-26,'Thermal Relic cross section',color='gray',fontsize=8)
        plt.text(300,1.2E-26,'Steigman et al. 2012',color='gray',fontsize=8)
        plt.text(750,2E-27,r'$b\bar{b}$',fontsize=12)
        plt.legend(loc=2,borderaxespad=0.)
        plt.xlim(8,1000)
        plt.xlabel('DM mass [GeV]',fontsize=18)
        plt.ylabel('$\sigma v$ [cm$^3/s$]',fontsize=18)
        plt.title('Summary plot')
        plt.tick_params(right=True,top=True,labelsize=12)
        plt.tick_params(right=True,top=True,which='minor')
        plt.savefig('sv_mx_stacked.pdf',bbox_inches='tight')


    if(fig1dSph):
        dwarf='16'
        mxarr,svlim=np.loadtxt('excl_limits_case1N/sv_mx_dwarf'+dwarf+'.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        plt.loglog(mxarr,svfunc,label='J & B fixed',c='green')
        mxarr,svlim=np.loadtxt('excl_limits_case2N/sv_mx_dwarf'+dwarf+'.dat',unpack=True)
        svfunc = savgol_filter(svlim,9,3)
        plt.loglog(mxarr,svfunc,label='prof. J, B fixed',c='red')
        mxarr,svlim=np.loadtxt('excl_limits_case4N/sv_mx_dwarf'+dwarf+'.dat',unpack=True)
        svfunc = savgol_filter(svlim,9,3)
        plt.loglog(mxarr,svfunc,label='prof. J & B',c='blue')
        mxarr,svlim=np.loadtxt('../Steigman.dat',unpack=True)
        plt.plot(mxarr,svlim,ls='--',color='gray')
        plt.text(205,1.6E-26,'Thermal Relic cross section',color='gray',fontsize=8)
        plt.text(300,1.1E-26,'Steigman et al. 2012',color='gray',fontsize=8)
        plt.text(750,1.5E-27,r'$b\bar{b}$',fontsize=12)
        plt.xlabel('DM mass [GeV]',fontsize=18)
        plt.ylabel('$\sigma v$ [cm$^3$/s]',fontsize=18)
        plt.xlim(8,1000)
        plt.ylim(1.E-27,1.E-23)
        plt.legend(loc=2,borderaxespad=0.)
        plt.title('Sculptor')
        plt.tick_params(right=True,top=True,labelsize=12)
        plt.tick_params(right=True,top=True,which='minor')
        plt.savefig('sv_mx_Sculptor_allcases.pdf',bbox_inches="tight")
        
        
    if(task101):
        # import results of exclusion for all dwarfs ++++++++++++++++++
        svarrAll = []
        lista = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        #lista = [5,16,4,17,22,11,24]
        colors = plt.cm.hsv(np.linspace(0,1.0,len(lista)) )
        for j in range(len(lista)):
            i = lista[j]
            mxarr,svlim=np.loadtxt('excl_limits_case1N/sv_mx_dwarf'+str(i)+'.dat',unpack=True)
            svfunc = savgol_filter(svlim,11,3)
            # following is just for plotting
            plt.loglog(mxarr,svfunc,label='dwarf#'+str(i+1),c=colors[j])
            #plt.loglog(mxarr,svfunc,label='dwarf#'+str(i+1))
            #plt.loglog(mxarr,svlim,label='dwarf#'+str(i+1),color='black')
        mxarr,svlim=np.loadtxt('excl_limits_case1N/sv_mx_dwarf23.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        #plt.loglog(mxarr,svfunc,label='dwarf#',c='black')
        plt.legend(bbox_to_anchor=(1.02,1.0),loc=2,borderaxespad=0.)
        plt.xlabel('DM mass [GeV]',fontsize=18)
        plt.ylabel('$\log_{10}(\sigma v/cm^3s^{-1})$',fontsize=18)
        #plt.xticks(np.arange(10,1000,100))
        #plt.grid()
        plt.title('case#1(no profiling), $N_{bins}=24$')
        plt.savefig('sv_mx_everyone_case1N.pdf',bbox_inches="tight")


    if(task102):
        # import results of exclusion for all dwarfs ++++++++++++++++++
        svarrAll = []
        #lista = [0]
        lista = [1,2,3,4,5,6,11,15,16,17,22,23]
        colors = plt.cm.hsv(np.linspace(0,1,len(lista)) )
        for j in range(len(lista)):
            i = lista[j]
            mxarr,svlim=np.loadtxt('excl_limits_case2N/sv_mx_dwarf'+str(i)+'.dat',unpack=True)
            svfunc = savgol_filter(svlim,11,3)
            plt.loglog(mxarr,svfunc,label=str(i+1),c=colors[j])
        
        mxarr,svlim=np.loadtxt('excl_limits_case2N/sv_mx_dwarf22.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        #plt.loglog(mxarr,svfunc,label='dwarf#23',c='black')
        
        plt.legend(bbox_to_anchor=(1.02,1.0),loc=2,borderaxespad=0.)
        plt.ylabel('$\sigma v$ [cm$^3$/s]',fontsize=18)
        plt.xlabel('DM mass [GeV]',fontsize=18)
        #plt.xticks(np.arange(10,1000,100))
        #plt.grid()
        plt.xlim(8,1000)
        plt.title('profiling J (B fixed)')
        plt.savefig('sv_mx_everyone_case2N.pdf',bbox_inches="tight")

    if(task1023):
        # import results of exclusion for all dwarfs ++++++++++++++++++
        svarrAll = []
        lista = [5,16,4,17,22,11]
        names = ["Draco","Sculptor","Coma Berenices","Segue I","Ursa Minor","Leo II"]
        #colors = plt.cm.hsv(np.linspace(0,0.5,len(lista)) )
        colors = ["red","green","blue","orange","lime","cyan"]

        fig, ax = plt.subplots()
        for j in range(len(lista)):
            i = lista[j]
            mxarr,svlim=np.loadtxt('excl_limits_case2N/sv_mx_dwarf'+str(i)+'.dat',unpack=True)
            svfunc = savgol_filter(svlim,9,3)
            ax.loglog(mxarr,svfunc,label=names[j],c=colors[j])
        #mxarr,svlim=np.loadtxt('excl_limits_case2N/sv_mx_dwarf17.dat',unpack=True)
        #svfunc = savgol_filter(svlim,9,3)
        #ax.loglog(mxarr,svfunc,label='SegueT',c='blue',lw=2.0)
        
        mxarr,svlim=np.loadtxt('excl_limits_case3N/sv_mx.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        ax.loglog(mxarr,svfunc,label='Combined',c='black',lw=2.0)
        
        '''
        mxarr,svlim=np.loadtxt('excl_limits_case3N/sv_mx_stack2dSphs.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        plt.loglog(mxarr,svfunc,label='Combined_2dSphs',c='brown')
        mxarr,svlim=np.loadtxt('excl_limits_case3N/sv_mx_stack3dSphs.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        plt.loglog(mxarr,svfunc,label='Combined_3dSphs',c='violet')
        '''
        mxarr,svlim=np.loadtxt('../Steigman.dat',unpack=True)
        ax.plot(mxarr,svlim,ls='--',color='gray')
        #plt.legend(bbox_to_anchor=(1.02,1.0),loc=2,borderaxespad=0.)
        plt.legend(loc=2,borderaxespad=0.,ncol=2)
        plt.text(205,1.6E-26,'Thermal Relic cross section',color='gray',fontsize=8)
        plt.text(300,1.E-26,'Steigman et al. 2012',color='gray',fontsize=8)
        plt.text(750,1.5E-27,r'$b\bar{b}$',fontsize=12)
        plt.xlabel('DM mass [GeV]',fontsize=18)
        plt.ylabel('$\sigma v$ [cm$^3$/s]',fontsize=18)
        plt.xlim(8,1000)
        plt.ylim(1.E-27,1.E-22)
        #plt.xticks(np.linspace(10,1000,10))
        #plt.grid()
        #plt.tick_params(labeltop=True,labelright=True)
        ax.tick_params(right=True,top=True,labelsize=12)
        ax.tick_params(right=True,top=True,which='minor')
        plt.title('profiling $J$ ($B$ fixed)')
        plt.savefig('sv_mx_everyone_cases2et3_6bins.pdf',bbox_inches="tight")

    if(task1045):
        # import results of exclusion for all dwarfs ++++++++++++++++++
        svarrAll = []
        lista = [5,16,4,17,22,11]
        names = ["Draco","Sculptor","Coma Berenices","Segue I","Usar Minor","Leo II"]
        colors = plt.cm.hsv(np.linspace(0,0.5,len(lista)) )
        colors = ["red","green","blue","orange","lime","cyan"]
        for j in range(len(lista)):
            i = lista[j]
            mxarr,svlim=np.loadtxt('excl_limits_case4N/sv_mx_dwarf'+str(i)+'.dat',unpack=True)
            svfunc = savgol_filter(svlim,11,3)
            plt.loglog(mxarr,svfunc,label=names[j],c=colors[j])
        mxarr,svlim=np.loadtxt('excl_limits_case5N/sv_mx.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        plt.loglog(mxarr,svfunc,label='Combined all',c='black',lw=2.0)
        #plt.loglog(mxarr,svlim,label='comb 17&16',c='black',lw=2.0)
        #mxarr,svlim=np.loadtxt('excl_limits_case5Ntest/sv_mx_17et4.dat',unpack=True)
        #svfunc = savgol_filter(svlim,11,3)
        #plt.loglog(mxarr,svlim,label='comb 17&4',c='gray',lw=2.0)
        #mxarr,svlim=np.loadtxt('excl_limits_case5Ntest/sv_mx_17.dat',unpack=True)
        #svfunc = savgol_filter(svlim,11,3)
        #plt.loglog(mxarr,svlim,label='comb 17',c='violet',lw=2.0)

        '''
        mxarr,svlim=np.loadtxt('excl_limits_case5N/sv_mx_stack2dSphs.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        plt.loglog(mxarr,svfunc,label='Combined(2)',c='black',ls='--')
        mxarr,svlim=np.loadtxt('excl_limits_case5N/sv_mx_stack3dSphs.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        plt.loglog(mxarr,svfunc,label='Combined(3)',c='black',ls='-.')
        mxarr,svlim=np.loadtxt('excl_limits_case5N/sv_mx_stack4dSphs.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        plt.loglog(mxarr,svfunc,label='Combined(4)',c='black',ls=':')
        mxarr,svlim=np.loadtxt('excl_limits_case5N/sv_mx_stack7dSphs.dat',unpack=True)
        svfunc = savgol_filter(svlim,11,3)
        plt.loglog(mxarr,svfunc,label='Combined(7)',c='brown',ls='-')
        '''
        
        mxarr,svlim=np.loadtxt('../Steigman.dat',unpack=True)
        plt.plot(mxarr,svlim,ls='--',color='gray')
        #plt.legend(bbox_to_anchor=(1.02,1.0),loc=2,borderaxespad=0.)
        plt.text(205,1.6E-26,'Thermal Relic cross section',color='gray',fontsize=8)
        plt.text(300,1.E-26,'Steigman et al. 2012',color='gray',fontsize=8)
        plt.text(750,1.5E-27,r'$b\bar{b}$',fontsize=12)
        plt.xlabel('DM mass [GeV]',fontsize=18)
        plt.ylabel('$\sigma v$ [cm$^3$/s]',fontsize=18)
        plt.xlim(8,1000)
        plt.ylim(1.E-27,1.E-22)
        #plt.xticks(np.linspace(10,1000,10))
        #plt.grid()
        plt.tick_params(right=True,top=True,labelsize=12)
        plt.tick_params(right=True,top=True,which='minor')
        plt.title('profiling $J$ & $B$')
        plt.savefig('sv_mx_everyone_cases4et5.pdf',bbox_inches="tight")


        
    if(task11):
        # +++++++  importing data  +++++++++++++++++++++++++++++++++
        filedwarfs1 = fileinput.input('exp_dSphs.dat')
        exp_dSphs = readdata(filedwarfs1)
        filedwarfs1.close()
        infile = fileinput.input('lnbckg_dSphs_2.dat')
        counts_meas = []
        bckg_est = []
        err_tot = []
        for line in infile:
            meas,est = line.split()
            counts_meas.append(float(meas))
            bckg_est.append(float(est))
        infile.close()
        NdSphs = len(counts_meas)
        err_tot = np.sqrt(0.026)*np.ones(NdSphs)
        # +++ importing J-factors with errors
        Jfile = fileinput.input('Jfactors.dat')
        log10Jfacs = []
        deltaJ = []
        for line in Jfile:
            J,d = line.split()
            log10Jfacs.append(float(J))
            deltaJ.append(float(d))
                
        # ++++++++ extracting J & B optimal +++++++++++++++++++++++++++++
        JarrAll = []
        BarrAll = []
        mxarr = []
        svarr = []
        for i in range(19):
            Jarr = []
            Barr = []
            fileU = fileinput.input("sv_mx_JBopt/sv_mx_logJopt_lnBopt_dwarf#"+str(i)+".dat")
            if(i == 0):
                for linea in fileU :
                    values = linea.split()
                    mxarr.append([float(values[0])])
                    svarr.append([float(values[1])])
                    Jarr.append(float(values[2]))
                    Barr.append(float(values[4]))
            else :
                for linea in fileU :
                    values = linea.split()
                    Jarr.append(float(values[2]))
                    Barr.append(float(values[4]))
            fileU.close()
            JarrAll.append(np.array(Jarr))
            BarrAll.append(np.array(Barr))
                    
        mxarr=np.array(mxarr)
        svarr=np.array(svarr)
        points=np.hstack((mxarr,svarr))
        # +++++++ building profiled likelihood  ++++++++++++++++++++++
        #global DOmega
        rad = 0.5 # deg
        Area = np.pi*rad**2
        DOmega = Area*(np.pi/180.)**2 # Delta Omega of regions
        deltaEXCL = 3.84

        mxgrid = np.logspace(math.log10(min(mxarr)),math.log10(max(mxarr)),20)
        svgrid = np.arange(math.log10(min(svarr)),math.log10(max(svarr)),0.05)

        os.system('rm excl_limits/sv_mx_all_dwarfs.dat')
        m2logPL_farr = [None]*NdSphs
        for i in range(NdSphs):
            print('++building likelihood for dwarf #',i)
            Jfunc = lambda mx,sv: interpolate.griddata(points,JarrAll[i],
                                                       (mx,sv),method='cubic',rescale=True)
            Bfunc = lambda mx,sv: interpolate.griddata(points,BarrAll[i],
                                                       (mx,sv),method='cubic',rescale=True)
            m2logPL_farr[i] = lambda mx, sv: logL_JB(Jfunc(mx,sv),Bfunc(mx,sv),sv,mx,i,
                                                     exp_dSphs,counts_meas,bckg_est,log10Jfacs,
                                                     deltaJ,err_tot)
        for j in range(len(mxgrid)):
            fileout = open('excl_limits/sv_mx_all_dwarfs.dat',"a")
            print("mDM = ",mxgrid[j])
            totL_arr = []
            for k in range(len(svgrid)):
                totL = 0.
                for d in range(NdSphs):
                    totL = totL + m2logPL_farr[d](mxgrid[j],10**svgrid[k])
                totL_arr.append(totL)
                print('  log sv = ',svgrid[k],' totL = ',totL_arr[k])
            totLmin = min(totL_arr)
            kmin = np.argmin(totL_arr)
            for k in range(kmin,len(svgrid)):
                if(totL_arr[k] - totLmin > deltaEXCL):
                    svEXCL = svgrid[k]
                    print('    sv excl = ',svEXCL)
                    break
            fileout.write(" ".join([str(mxgrid[j]),str(svEXCL),"\n"]) )

        quit()        
        for i in range(1):
            print("+++++++++++++++  this is dwarf ",i," ++++++++++++++++++++++")
            os.system('rm excl_limits/sv_mx_dwarf#'+str(i)+'.dat')
            Jfunc = lambda mx,sv: interpolate.griddata(points,JarrAll[i],
                                                       (mx,sv),method='cubic',rescale=True)
            Bfunc = lambda mx,sv: interpolate.griddata(points,BarrAll[i],
                                                       (mx,sv),method='cubic',rescale=True)
            m2logPL = lambda mx, sv: logL_JB(Jfunc(mx,sv),Bfunc(mx,sv),sv,mx,i,
                                             exp_dSphs,counts_meas,bckg_est,log10Jfacs,
                                             deltaJ,err_tot)
            for j in range(len(mxgrid)):
                fileout = open('excl_limits/sv_mx_dwarf#'+str(i)+'.dat',"a")
                print("mDM = ",mxgrid[j])
                m2logPL_arr = []
                for k in range(len(svgrid)):
                    m2logPL_arr.append(m2logPL(mxgrid[j],10**svgrid[k]))
                    print('  log sv = ',svgrid[k],' m2logPL = ',m2logPL_arr[k])
                m2LLmin = min(m2logPL_arr)
                kmin = np.argmin(m2logPL_arr)
                print('  m2logPL_min = ',m2LLmin,' at k = ',kmin)
                for k in range(kmin,len(svgrid)):
                    if(m2logPL_arr[k] - m2LLmin > deltaEXCL):
                        svEXCL = svgrid[k]
                        print('    sv excl = ',svEXCL)
                        break
                fileout.write(" ".join([str(mxgrid[j]),str(svEXCL),"\n"]) )

        
    if(task12):
        os.system('ls excl_limits/all_dwarfs_mx_*.dat > ls.out')
        files = np.loadtxt('ls.out',dtype='str')
        for i in range(len(files)):
            files[i] = files[i][2:-1]
        mxarr = []
        svarr = []
        for f in files:
            vals = np.loadtxt(f)
            mxarr.append(vals[0])
            svarr.append(vals[1])
        mxarr = np.array(mxarr)
        svarr = np.array(svarr)
        sortind = np.argsort(mxarr)
        mxarr = mxarr[sortind]
        svarr = svarr[sortind]
        print('mxarr = ',mxarr)
        print('red = ',np.delete(mxarr,[8,9,10]))
        mxarr = np.delete(mxarr,[8,9,10])
        svarr = np.delete(svarr,[8,9,10])
        
        plt.plot(mxarr,10.**svarr,'-')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('sv_mx_combined.pdf')

        
if __name__ == '__main__':
    main()


# ++++++++++++++++++++++++++ trash   +++++++++++++++++++++++++++
'''

        diff = []
        for i in range(len(mxarr)):
            diff.append(Jarr[i] - Jfunc(mxarr[i],svarr[i] ))         
        #print('max(diff)  = ',min(diff))
        plt.scatter(np.log10(mxarr),np.log10(svarr),c=diff)
        plt.scatter(np.log10(mxarr),np.log10(svarr),c=np.log10(Jarr))
        plt.colorbar()
        plt.show()
       

'''




