import numpy as np
from scipy.optimize import curve_fit
import LCmods
import sys
import string
from numpy import vectorize
import matplotlib.pyplot as plt
import warnings

#Welcome message
print ('*******************************************************************************************************')
print ('WELCOME TO TIGERFIT: THE COMPLETE CHI-SQUARE MINIMIZATION FITTER FOR SUPERNOVA LIGHTCURVES')
print ('USE TIGERFIT TO FIT A VARIETY OF POWER INPUT MODELS TO OBSERVED PSEUDOBOLOMETRIC LIGHT CURVE')
print ('DATA OF SUPERNOVA EXPLOSIONS. CONSULT THE README FILE FOR INSTRUCTIONS.')
print ('COPYRIGHT 2016: DR. MANOS CHATZOPOULOS (DEPARTMENT OF PHYSICS & ASTRONOMY, LOUISIANA STATE UNIVERSITY)')
print ('*******************************************************************************************************')

# Read bolometric LC data to be fit. 1st column is rest-frame days, 2nd: luminosity and 3rd: luminosity error,
# and define if errors are used for fitting
if sys.argv[3] == 'err':
    xdata, ydata, dydata = np.loadtxt(sys.argv[1], usecols=(0,1,2), unpack=True)
    sig = dydata
elif sys.argv[3] == 'nerr':
    xdata, ydata = np.loadtxt(sys.argv[1], usecols=(0,1), unpack=True)
    sig = None
    dydata = 1.

# Define if model is printed out on standard output (time,Lumunosity)
if sys.argv[4] == 'show':
    show_mod = True
elif sys.argv[4] == 'hide':
    show_mod = False

# Define if a MatPlot plot of the output fit to the data is desired
if sys.argv[5] == 'plot':
    plot = True
elif sys.argv[5] == 'noplot':
    plot = False

# Vectorize LC models computed in the LCmods module
lumrad_v     = np.vectorize(LCmods.Lum_rad)          # General radioactive decay (RD) model
lumradr0_v   = np.vectorize(LCmods.Lum_rad_r0)       # Radioactive decay model for small initial radius
lummag_v     = np.vectorize(LCmods.Lum_mag)          # General magnetar (MAG) spin-down model
lummagr0_v   = np.vectorize(LCmods.Lum_mag_r0)       # Magnetar spin-down model for small initial radius
lumcsm_v     = np.vectorize(LCmods.Lum_csm)          # General hybrid CSM interaction (CSI) plus RD/MAG model
lumcsmrads2  = np.vectorize(LCmods.Lum_csmrad_s2)    # Hybrid CSI+RAD model for collision with an 1/r^2 wind
lumcsmmags2  = np.vectorize(LCmods.Lum_csmmag_s2)    # Hybrid CSI+MAG model for collision with an 1/r^2 wind
lumcsmonlys2 = np.vectorize(LCmods.Lum_csmonly_s2)   # General CSI model for collision with an 1/r^2 wind
lumcsmrads0  = np.vectorize(LCmods.Lum_csmrad_s0)    # Hybrid CSI+RAD model for collision with a constant-density shell
lumcsmmags0  = np.vectorize(LCmods.Lum_csmmag_s0)    # Hybrid CSI+MAG model for collision with a constant-density shell
lumcsmonlys0 = np.vectorize(LCmods.Lum_csmonly_s0)   # General CSI model for collision with a constant-density shell
lumfb_v      = np.vectorize(LCmods.Lum_fb)           # Fall-back Accretion Model (Dexter & Kasen 2013)

#List models in an array
model = [lumrad_v,lumradr0_v,lummag_v,lummagr0_v,lumcsm_v,lumcsmrads2,lumcsmmags2,lumcsmonlys2, \
         lumcsmrads0,lumcsmmags0,lumcsmonlys0,lumfb_v]

#Definitions of initial guesses and parameter boundaries for all models

if sys.argv[2] == 'rad':
    model = lumrad_v
    guess = (0.1,10.,0.1,10000.,30.)
    bound = (((0.01,5.,0.0001,5000.,7.),(20.0,100.,1.0,20000.,100.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lumrad_v(xdata,p[0],p[1],p[2],p[3],p[4]))**2/dydata**2)
    Mej1 = ((3./10.)*(13.8*LCmods.c/0.05)*p[3]*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    Mej2 = ((3./10.)*(13.8*LCmods.c/0.2)*p[3]*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    Mej3 = ((3./10.)*(13.8*LCmods.c/0.33)*p[3]*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    DMej1 = (Mej1*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)+(p[3]/pcov.diagonal()[3])**2))
    DMej2 = (Mej2*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)+(p[3]/pcov.diagonal()[3])**2))
    DMej3 = (Mej3*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)+(p[3]/pcov.diagonal()[3])**2))
    E1 = (3./10.)*Mej1*LCmods.Msun*(p[3]*LCmods.kms2cms)**2
    E2 = (3./10.)*Mej2*LCmods.Msun*(p[3]*LCmods.kms2cms)**2
    E3 = (3./10.)*Mej3*LCmods.Msun*(p[3]*LCmods.kms2cms)**2
    DE1 = E1*np.sqrt((DMej1/Mej1)**2+4.*(p[3]/pcov.diagonal()[3])**2)
    DE2 = E2*np.sqrt((DMej2/Mej2)**2+4.*(p[3]/pcov.diagonal()[3])**2)
    DE3 = E3*np.sqrt((DMej3/Mej3)**2+4.*(p[3]/pcov.diagonal()[3])**2)
    print ('List of Derived Parameters for the General Radioactive Decay Model')
    print ('------------------------------------------------------------------')
    print ('M_Ni = ',p[0],' +/- ',pcov.diagonal()[0],' M_sun')
    print ('M_ej = ',Mej1,' +/- ',DMej1,' M_sun for opacity kappa = 0.05 cm^2/g')
    print ('M_ej = ',Mej2,' +/- ',DMej2,' M_sun for opacity kappa = 0.20 cm^2/g')
    print ('M_ej = ',Mej3,' +/- ',DMej3,' M_sun for opacity kappa = 0.33 cm^2/g')
    print ('E_SN = ',E1,' +/- ',DE1, 'erg for opacity kappa = 0.05 cm^2/g')
    print ('E_SN = ',E2,' +/- ',DE2, 'erg for opacity kappa = 0.20 cm^2/g')
    print ('E_SN = ',E3,' +/- ',DE3, 'erg for opacity kappa = 0.33 cm^2/g')
    print ('The gamma-ray trapping parameter A = ',p[4]*LCmods.day**2/LCmods.A0,' +/- ',\
      pcov.diagonal()[4]*LCmods.day**2/LCmods.A0, 's^2')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_rad(float(i),p[0],p[1],p[2],p[3],p[4]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_rad(float(i),p[0],p[1],p[2],p[3],p[4])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('General radioactive decay model fit')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()


elif sys.argv[2] == 'rad0':
    model = lumradr0_v
    guess = (0.02,10.)
    bound = (((0.01,2.),(25.0,150.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lumradr0_v(xdata,p[0],p[1]))**2/dydata**2)
    Mej1 = ((3./10.)*(13.8*LCmods.c/0.05)*10000.*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    Mej2 = ((3./10.)*(13.8*LCmods.c/0.2)*10000.*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    Mej3 = ((3./10.)*(13.8*LCmods.c/0.33)*10000.*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    DMej1 = (Mej1*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)))
    DMej2 = (Mej2*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)))
    DMej3 = (Mej3*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)))
    E1 = (3./10.)*Mej1*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    E2 = (3./10.)*Mej2*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    E3 = (3./10.)*Mej3*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    DE1 = E1*np.sqrt((DMej1/Mej1)**2)
    DE2 = E2*np.sqrt((DMej2/Mej2)**2)
    DE3 = E3*np.sqrt((DMej3/Mej3)**2)
    print ('List of Derived Parameters for the Basic (R0 = 0, vej = 10,000 km/s) Radioactive Decay Model')
    print ('--------------------------------------------------------------------------------------------')
    print ('M_Ni = ',p[0],' +/- ',pcov.diagonal()[0],' M_sun')
    print ('M_ej = ',Mej1,' +/- ',DMej1,' M_sun for opacity kappa = 0.05 cm^2/g')
    print ('M_ej = ',Mej2,' +/- ',DMej2,' M_sun for opacity kappa = 0.20 cm^2/g')
    print ('M_ej = ',Mej3,' +/- ',DMej3,' M_sun for opacity kappa = 0.33 cm^2/g')
    print ('E_SN = ',E1,' +/- ',DE1, 'erg for opacity kappa = 0.05 cm^2/g')
    print ('E_SN = ',E2,' +/- ',DE2, 'erg for opacity kappa = 0.20 cm^2/g')
    print ('E_SN = ',E3,' +/- ',DE3, 'erg for opacity kappa = 0.33 cm^2/g')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_rad_r0(float(i),p[0],p[1]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_rad_r0(float(i),p[0],p[1])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('Basic radioactive decay model fit')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'mag':
    model = lummag_v
    guess = (1.0,20.,10.,0.001,20000.)
    bound = (((0.1,1.,2.,0.0001,10000.),(10.0,100.,50.,0.01,30000.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lummag_v(xdata,p[0],p[1],p[2],p[3],p[4]))**2/dydata**2)
    B14 = 10.**14 # Magnetic field-strength in 10^14 Gauss
    e0 = 2.e50
    Pmag = 10.* np.sqrt(e0/(p[0]*LCmods.E51))
    DPmag = 0.5 * Pmag * (pcov.diagonal()[0]/p[0])
    Bmag = np.sqrt(130.*Pmag**2/(365.*p[2])) * B14
    DBmag = Bmag * np.sqrt((DPmag/Pmag)**2+(0.5*pcov.diagonal()[2]/p[2])**2)
    Mej1 = ((3./10.)*(13.8*LCmods.c/0.05)*p[4]*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    Mej2 = ((3./10.)*(13.8*LCmods.c/0.2)*p[4]*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    Mej3 = ((3./10.)*(13.8*LCmods.c/0.33)*p[4]*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    DMej1 = (Mej1*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)+(p[4]/pcov.diagonal()[4])**2))
    DMej2 = (Mej2*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)+(p[4]/pcov.diagonal()[4])**2))
    DMej3 = (Mej3*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)+(p[4]/pcov.diagonal()[4])**2))
    E1 = (3./10.)*Mej1*LCmods.Msun*(p[4]*LCmods.kms2cms)**2
    E2 = (3./10.)*Mej2*LCmods.Msun*(p[4]*LCmods.kms2cms)**2
    E3 = (3./10.)*Mej3*LCmods.Msun*(p[4]*LCmods.kms2cms)**2
    DE1 = E1*np.sqrt((DMej1/Mej1)**2+4.*(p[4]/pcov.diagonal()[4])**2)
    DE2 = E2*np.sqrt((DMej2/Mej2)**2+4.*(p[4]/pcov.diagonal()[4])**2)
    DE3 = E3*np.sqrt((DMej3/Mej3)**2+4.*(p[4]/pcov.diagonal()[4])**2)    
    print ('List of Derived Parameters for the General Magnetar Spin-down Model')
    print ('-------------------------------------------------------------------')
    print ('Pmag = ',Pmag,' +/- ',DPmag,' milliseconds')
    print ('Bmag = ',Bmag/B14,' +/- ',DBmag/B14,' x 10^14 Gauss')
    print ('M_ej = ',Mej1,' +/- ',DMej1,' M_sun for opacity kappa = 0.05 cm^2/g')
    print ('M_ej = ',Mej2,' +/- ',DMej2,' M_sun for opacity kappa = 0.20 cm^2/g')
    print ('M_ej = ',Mej3,' +/- ',DMej3,' M_sun for opacity kappa = 0.33 cm^2/g')
    print ('E_SN = ',E1,' +/- ',DE1, 'erg for opacity kappa = 0.05 cm^2/g')
    print ('E_SN = ',E2,' +/- ',DE2, 'erg for opacity kappa = 0.20 cm^2/g')
    print ('E_SN = ',E3,' +/- ',DE3, 'erg for opacity kappa = 0.33 cm^2/g')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)


    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_mag(float(i),p[0],p[1],p[2],p[3],p[4]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_mag(float(i),p[0],p[1],p[2],p[3],p[4])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('General magnetar spin-down model fit')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'mag0':
    model = lummagr0_v
    guess = (1.0,20.,10.)
    bound = (((0.1,1.,2.),(10.0,100.,50.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lummagr0_v(xdata,p[0],p[1],p[2]))**2/dydata**2)
    B14 = 10.**14 # Magnetic field-strength in 10^14 Gauss
    e0 = 2.e50
    Pmag = 10.* np.sqrt(e0/(p[0]*LCmods.E51))
    DPmag = 0.5 * Pmag * (pcov.diagonal()[0]/p[0])
    Bmag = np.sqrt(130.*Pmag**2/(365.*p[1])) * B14
    DBmag = Bmag * np.sqrt((DPmag/Pmag)**2+(0.5*pcov.diagonal()[2]/p[2])**2)
    Mej1 = ((3./10.)*(13.8*LCmods.c/0.05)*10000.*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    Mej2 = ((3./10.)*(13.8*LCmods.c/0.2)*10000.*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    Mej3 = ((3./10.)*(13.8*LCmods.c/0.33)*10000.*LCmods.kms2cms*(p[1]*LCmods.day)**2)/LCmods.Msun
    DMej1 = (Mej1*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)))
    DMej2 = (Mej2*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)))
    DMej3 = (Mej3*np.sqrt(4.*((pcov.diagonal()[1]/p[1])**2)))
    E1 = (3./10.)*Mej1*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    E2 = (3./10.)*Mej2*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    E3 = (3./10.)*Mej3*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    DE1 = E1*np.sqrt((DMej1/Mej1)**2)
    DE2 = E2*np.sqrt((DMej2/Mej2)**2)
    DE3 = E3*np.sqrt((DMej3/Mej3)**2)    
    print ('List of Derived Parameters for the Basic (R0 = 0, vej = 10,000 km/s) Magnetar Spin-down Model')
    print ('----------------------------------------------------------------------------------------------')
    print ('Pmag = ',Pmag,' +/- ',DPmag,' milliseconds')
    print ('Bmag = ',Bmag/B14,' +/- ',DBmag/B14,' x 10^14 Gauss')
    print ('M_ej = ',Mej1,' +/- ',DMej1,' M_sun for opacity kappa = 0.05 cm^2/g')
    print ('M_ej = ',Mej2,' +/- ',DMej2,' M_sun for opacity kappa = 0.20 cm^2/g')
    print ('M_ej = ',Mej3,' +/- ',DMej3,' M_sun for opacity kappa = 0.33 cm^2/g')
    print ('E_SN = ',E1,' +/- ',DE1, 'erg for opacity kappa = 0.05 cm^2/g')
    print ('E_SN = ',E2,' +/- ',DE2, 'erg for opacity kappa = 0.20 cm^2/g')
    print ('E_SN = ',E3,' +/- ',DE3, 'erg for opacity kappa = 0.33 cm^2/g')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_mag_r0(float(i),p[0],p[1],p[2]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_mag_r0(float(i),p[0],p[1],p[2])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('Basic magnetar spin-down model fit')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'csm':
    model = lumcsm_v
    guess = (0.01,0.1,10.,1.,0.1,10.,0.33,9,0,1.,0.01,0,100.)
    bound = (((0.0001,0.01,1.,0.05,0.0001,3.,0.2,8,0,0.01,0.00001,0,50.),(20.,5.,70.,10.,1.0,50.,0.4,12,2,100.,1.0,2,2000.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lumcsm_v(xdata,p[0],p[1],p[2],p[3],p[4],p[5],p[6],\
                                                                   p[7],p[8],p[9],p[10],p[11],p[12]))**2/dydata**2)
    B14 = 10.**14 # Magnetic field-strength in 10^14 Gauss
    e0 = 2.e50
    Pmag = 10.* np.sqrt(e0/(p[1]*LCmods.E51))
    DPmag = 0.5 * Pmag * (pcov.diagonal()[1]/p[1])
    Bmag = np.sqrt(130.*Pmag**2/(365.*p[1])) * B14
    DBmag = Bmag * np.sqrt((DPmag/Pmag)**2+(0.5*pcov.diagonal()[2]/p[2])**2)    
    print ('List of Derived Parameters for the General Hybrid CSM+RD+MAG Model')
    print ('------------------------------------------------------------------')
    print ('M_Ni = ',p[0],' +/- ',pcov.diagonal()[0],' M_sun')
    print ('Pmag = ',Pmag,' +/- ',DPmag,' milliseconds')
    print ('Bmag = ',Bmag/B14,' +/- ',DBmag/B14,' x 10^14 Gauss')
    print ('E_SN = ',p[3],' +/- ',pcov.diagonal()[3],' x 10^51 erg')
    print ('R_progenitor = ',p[4]/10.,' +/- ',pcov.diagonal()[4]/10.,' x 10^14 cm')
    print ('M_ej = ',p[5],' +/- ',pcov.diagonal()[5],' M_sun')
    print ('Kappa_ej = ',round(p[6],2),' SN ejecta opacity in cm^2/g')
    print ('n_ej = ',int(round(p[7],0)),' outer slope of SN ejecta density profile')
    print ('M_CSM = ',p[9],' +/- ',pcov.diagonal()[9],' M_sun')
    print ('Mdot = ',p[10],' +/- ',pcov.diagonal()[10],' M_sun/year')
    print ('s_CSM = ',int(round(p[11],0)), 'slope of CSM density profile')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_csm(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_csm(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('General hybrid CSM+RD+MAG model fit')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'csmrads2':
    model = lumcsmrads2
    guess = (0.01,1.0,0.01,8.,0.33,11,1.0,0.1,200.)
    bound = (((0.0001,0.1,0.0001,2.,0.2,8,0.01,0.00001,50.),(20.,10.,10.,80.,0.4,12,50.,1.0,2000.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lumcsmrads2(xdata,p[0],p[1],p[2],p[3],p[4],p[5],p[6],\
                                                                 p[7],p[8]))**2/dydata**2)
    print ('List of Derived Parameters for the Hybrid CSM+RD Model for 1/r^2 (s_CSM = 2) wind-like CSM')
    print ('------------------------------------------------------------------------------------------')
    print ('M_Ni = ',p[0],' +/- ',pcov.diagonal()[0],' M_sun')
    print ('E_SN = ',p[1],' +/- ',pcov.diagonal()[1],' x 10^51 erg')
    print ('R_progenitor = ',p[2]/10.,' +/- ',pcov.diagonal()[2]/10.,' x 10^14 cm')
    print ('M_ej = ',p[3],' +/- ',pcov.diagonal()[3],' M_sun')
    print ('Kappa_ej = ',round(p[4],2),' SN ejecta opacity in cm^2/g')
    print ('n_ej = ',int(round(p[5],0)),' outer slope of SN ejecta density profile')
    print ('M_CSM = ',p[6],' +/- ',pcov.diagonal()[6],' M_sun')
    print ('Mdot = ',p[7],' +/- ',pcov.diagonal()[7],' M_sun/year')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_csmrad_s2(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_csmrad_s2(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('General hybrid CSM+RD model fit (s=2)')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'csmmags2':
    model = lumcsmmags2
    guess = (1.0,10.,1.0,0.001,5.,0.33,11,5.0,0.1,1000.)
    bound = (((0.1,1.,0.1,0.0001,3.,0.2,8,0.01,0.01,50.),(5.,50.,10.,10.,80.,0.4,12,50.,1.0,2000.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lumcsmmags2(xdata,p[0],p[1],p[2],p[3],p[4],p[5]\
                                                                      ,p[6],p[7],p[8],p[9]))**2/dydata**2)
    B14 = 10.**14 # Magnetic field-strength in 10^14 Gauss
    e0 = 2.e50
    Pmag = 10.* np.sqrt(e0/(p[0]*LCmods.E51))
    DPmag = 0.5 * Pmag * (pcov.diagonal()[0]/p[0])
    Bmag = np.sqrt(130.*Pmag**2/(365.*p[1])) * B14
    DBmag = Bmag * np.sqrt((DPmag/Pmag)**2+(0.5*pcov.diagonal()[1]/p[1])**2)    
    print ('List of Derived Parameters for the Hybrid CSM+MAG Model for 1/r^2 (s_CSM = 2) wind-like CSM')
    print ('-------------------------------------------------------------------------------------------')
    print ('Pmag = ',Pmag,' +/- ',DPmag,' milliseconds')
    print ('Bmag = ',Bmag/B14,' +/- ',DBmag/B14,' x 10^14 Gauss')
    print ('E_SN = ',p[2],' +/- ',pcov.diagonal()[2],' x 10^51 erg')
    print ('R_progenitor = ',p[3]/10.,' +/- ',pcov.diagonal()[3]/10.,' x 10^14 cm')
    print ('M_ej = ',p[4],' +/- ',pcov.diagonal()[4],' M_sun')
    print ('Kappa_ej = ',round(p[5],2),' SN ejecta opacity in cm^2/g')
    print ('n_ej = ',int(round(p[6],0)),' outer slope of SN ejecta density profile')
    print ('M_CSM = ',p[7],' +/- ',pcov.diagonal()[7],' M_sun')
    print ('Mdot = ',p[8],' +/- ',pcov.diagonal()[8],' M_sun/year')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_csmmag_s2(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_csmmag_s2(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('General hybrid CSM+MAG model fit (s=2)')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'csm2':
    model = lumcsmonlys2
    guess = (1.0,0.01,10.,0.33,12,5.0,0.2,1000.)
    bound = (((0.1,0.0001,3.,0.2,8,0.01,0.0001,50.),(10.,1.0,80.,0.4,12,50.,1.0,2000.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lumcsmonlys2(xdata,p[0],p[1],p[2],p[3],p[4]\
                                                                      ,p[5],p[6],p[7]))**2/dydata**2)
    print ('List of Derived Parameters for the CSM Model for 1/r^2 (s_CSM = 2) wind-like CSM')
    print ('--------------------------------------------------------------------------------')
    print ('E_SN = ',p[0],' +/- ',pcov.diagonal()[0],' x 10^51 erg')
    print ('R_progenitor = ',p[1]/10.,' +/- ',pcov.diagonal()[1]/10.,' x 10^14 cm')
    print ('M_ej = ',p[2],' +/- ',pcov.diagonal()[2],' M_sun')
    print ('Kappa_ej = ',round(p[3],2),' SN ejecta opacity in cm^2/g')
    print ('n_ej = ',int(round(p[4],0)),' outer slope of SN ejecta density profile')
    print ('M_CSM = ',p[5],' +/- ',pcov.diagonal()[5],' M_sun')
    print ('Mdot = ',p[6],' +/- ',pcov.diagonal()[6],' M_sun/year')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_csmonly_s2(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_csmonly_s2(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('Basic CSM model fit (s=2)')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'csmrads0':
    model = lumcsmrads0
    guess = (0.1,1.0,0.01,10.,0.33,12,1.0,0.1,1000.)
    bound = (((0.01,0.1,0.0001,3.,0.2,8,0.01,0.0001,50.),(20.,20.,1.,80.,0.4,12,50.,1.0,2000.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lumcsmrads0(xdata,p[0],p[1],p[2],p[3],p[4],p[5],\
                                                                      p[6],p[7],p[8]))**2/dydata**2)
    print ('List of Derived Parameters for the Hybrid CSM+RD Model for constant density CSM shell (s_CSM = 0)')
    print ('-------------------------------------------------------------------------------------------------')
    print ('M_Ni = ',p[0],' +/- ',pcov.diagonal()[0],' M_sun')
    print ('E_SN = ',p[1],' +/- ',pcov.diagonal()[1],' x 10^51 erg')
    print ('R_progenitor = ',p[2]/10.,' +/- ',pcov.diagonal()[2]/10.,' x 10^14 cm')
    print ('M_ej = ',p[3],' +/- ',pcov.diagonal()[3],' M_sun')
    print ('Kappa_ej = ',round(p[4],2),' SN ejecta opacity in cm^2/g')
    print ('n_ej = ',int(round(p[5],0)),' outer slope of SN ejecta density profile')
    print ('M_CSM = ',p[6],' +/- ',pcov.diagonal()[6],' M_sun')
    print ('Mdot = ',p[7],' +/- ',pcov.diagonal()[7],' M_sun/year')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_csmrad_s0(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_csmrad_s0(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('General hybrid CSM+RAD model fit (s=0)')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'csmmags0':
    model = lumcsmmags0
    guess = (1.0,10.,1.0,0.01,10.,0.33,12,1.0,0.1,800.)
    bound = (((0.01,1.,0.1,0.0001,3.,0.2,8,0.01,0.0001,50.),(10.,50.,10.,1.,80.,0.4,12,50.,1.0,2000.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lumcsmmags0(xdata,p[0],p[1],p[2],p[3],p[4]\
                                                                      ,p[5],p[6],p[7],p[8],p[9]))**2/dydata**2)
    B14 = 10.**14 # Magnetic field-strength in 10^14 Gauss
    e0 = 2.e50
    Pmag = 10.* np.sqrt(e0/(p[0]*LCmods.E51))
    DPmag = 0.5 * Pmag * (pcov.diagonal()[0]/p[0])
    Bmag = np.sqrt(130.*Pmag**2/(365.*p[1])) * B14
    DBmag = Bmag * np.sqrt((DPmag/Pmag)**2+(0.5*pcov.diagonal()[1]/p[1])**2)    
    print ('List of Derived Parameters for the Hybrid CSM+MAG Model for constant density CSM shell (s_CSM = 0)')
    print ('--------------------------------------------------------------------------------------------------')
    print ('Pmag = ',Pmag,' +/- ',DPmag,' milliseconds')
    print ('Bmag = ',Bmag/B14,' +/- ',DBmag/B14,' x 10^14 Gauss')
    print ('E_SN = ',p[2],' +/- ',pcov.diagonal()[2],' x 10^51 erg')
    print ('R_progenitor = ',p[3]/10.,' +/- ',pcov.diagonal()[3]/10.,' x 10^14 cm')
    print ('M_ej = ',p[4],' +/- ',pcov.diagonal()[4],' M_sun')
    print ('Kappa_ej = ',round(p[5],2),' SN ejecta opacity in cm^2/g')
    print ('n_ej = ',int(round(p[6],0)),' outer slope of SN ejecta density profile')
    print ('M_CSM = ',p[7],' +/- ',pcov.diagonal()[7],' M_sun')
    print ('Mdot = ',p[8],' +/- ',pcov.diagonal()[8],' M_sun/year')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_csmmag_s0(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_csmmag_s0(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('General hybrid CSM+MAG model fit (s=0)')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'csm0':
    model = lumcsmonlys0
    guess = (1.0,0.01,8.,0.33,12,5.,0.1,1000.)
    bound = (((0.1,0.0001,3.,0.2,8,0.01,0.00001,50.),(20.,1.0,80.,0.4,12,50.,1.0,2000.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lumcsmonlys0(xdata,p[0],p[1],p[2],p[3],\
                                                                     p[4],p[5],p[6],p[7]))**2/dydata**2)
    print ('List of Derived Parameters for the CSM Model for constant density CSM shell (s_CSM = 0)')
    print ('---------------------------------------------------------------------------------------')
    print ('E_SN = ',p[0],' +/- ',pcov.diagonal()[0],' x 10^51 erg')
    print ('R_progenitor = ',p[1]/10.,' +/- ',pcov.diagonal()[1]/10.,' x 10^14 cm')
    print ('M_ej = ',p[2],' +/- ',pcov.diagonal()[2],' M_sun')
    print ('Kappa_ej = ',round(p[3],2),' SN ejecta opacity in cm^2/g')
    print ('n_ej = ',int(round(p[4],0)),' outer slope of SN ejecta density profile')
    print ('M_CSM = ',p[5],' +/- ',pcov.diagonal()[5],' M_sun')
    print ('Mdot = ',p[6],' +/- ',pcov.diagonal()[6],' M_sun/year')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_csmonly_s0(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_csmonly_s0(float(i),p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('Basic CSM model fit (s=0)')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'fbaccr':
    model = lumfb_v
    guess = (0.01,0.1,5000.,1.,40.)
    bound = (((0.001,0.001,1000.,0,5.),(1.,1.,10000.,3.,100.)))
    p, pcov = curve_fit(model, xdata, ydata, sigma = sig, p0=guess, bounds=bound)
    chi_red = (1./(len(xdata)-len(guess))) * sum((ydata-lumfb_v(xdata,p[0],p[1],p[2],p[3],p[4]))**2/dydata**2)
    Mej1 = ((3./10.)*(13.8*LCmods.c/0.05)*p[2]*LCmods.kms2cms*(p[4]*LCmods.day)**2)/LCmods.Msun
    Mej2 = ((3./10.)*(13.8*LCmods.c/0.2)*p[2]*LCmods.kms2cms*(p[4]*LCmods.day)**2)/LCmods.Msun
    Mej3 = ((3./10.)*(13.8*LCmods.c/0.33)*p[2]*LCmods.kms2cms*(p[4]*LCmods.day)**2)/LCmods.Msun
    DMej1 = (Mej1*np.sqrt(4.*((pcov.diagonal()[4]/p[4])**2)+(p[2]/pcov.diagonal()[2])**2))
    DMej2 = (Mej2*np.sqrt(4.*((pcov.diagonal()[4]/p[4])**2)+(p[2]/pcov.diagonal()[2])**2))
    DMej3 = (Mej3*np.sqrt(4.*((pcov.diagonal()[4]/p[4])**2)+(p[2]/pcov.diagonal()[2])**2))
    print ('List of Derived Parameters for the Fall-back Accretion Model (Dexter & Kasen 2013)')
    print ('----------------------------------------------------------------------------------')
    print ('L0 = ',p[0],' +/- ',pcov.diagonal()[0],' x 10^45 erg')
    print ('R0 = ',p[1],' +/- ',pcov.diagonal()[1],' x 10^15 cm')
    print ('Accreting mass M_accr = ',Mej1,' +/- ',DMej1,' M_sun for opacity kappa = 0.05 cm^2/g')
    print ('Accreting mass M_accr = ',Mej2,' +/- ',DMej2,' M_sun for opacity kappa = 0.20 cm^2/g')
    print ('Accreting mass M_accr = ',Mej3,' +/- ',DMej3,' M_sun for opacity kappa = 0.33 cm^2/g')
    print ('Power-law slope of mass accretion ',p[3], ' :(1.666 corresponds to the standard for t^-5/3 dependence).')
    print ('The reduced chi-square value for the best-fit model is ',chi_red)

    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_fb(float(i),p[0],p[1],p[2],p[3],p[4]))

    if plot:
        mod = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod.append(i)
            tt.append(i)
            mod[i] = LCmods.Lum_fb(float(i),p[0],p[1],p[2],p[3],p[4])
            tt[i] = i
        mod = np.asarray(mod)
        tt = np.asarray(tt)
        plt.title('Fallback accretion model Fit (s=0)')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod, 'k', linewidth=2.0, label='Model')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()
    
elif sys.argv[2] == 'basic':
    model1 = lumradr0_v
    guess_r = (5.0,30.)
    bound_r = (((0.1,5.),(25.0,150.)))
    p_r, pcov_r = curve_fit(model1, xdata, ydata, sigma = dydata, p0=guess_r, bounds=bound_r)
    chi_red_r = (1./(len(xdata)-len(guess_r))) * sum((ydata-lumradr0_v(xdata,p_r[0],p_r[1]))**2/dydata**2)
    Mej1 = ((3./10.)*(13.8*LCmods.c/0.05)*10000.*LCmods.kms2cms*(p_r[1]*LCmods.day)**2)/LCmods.Msun
    Mej2 = ((3./10.)*(13.8*LCmods.c/0.2)*10000.*LCmods.kms2cms*(p_r[1]*LCmods.day)**2)/LCmods.Msun
    Mej3 = ((3./10.)*(13.8*LCmods.c/0.33)*10000.*LCmods.kms2cms*(p_r[1]*LCmods.day)**2)/LCmods.Msun
    DMej1 = (Mej1*np.sqrt(4.*((pcov_r.diagonal()[1]/p_r[1])**2)))
    DMej2 = (Mej2*np.sqrt(4.*((pcov_r.diagonal()[1]/p_r[1])**2)))
    DMej3 = (Mej3*np.sqrt(4.*((pcov_r.diagonal()[1]/p_r[1])**2)))
    E1 = (3./10.)*Mej1*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    E2 = (3./10.)*Mej2*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    E3 = (3./10.)*Mej3*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    DE1 = E1*np.sqrt((DMej1/Mej1)**2)
    DE2 = E2*np.sqrt((DMej2/Mej2)**2)
    DE3 = E3*np.sqrt((DMej3/Mej3)**2)
    print ('List of Derived Parameters for the Basic (R0 = 0, vej = 10,000 km/s) Radioactive Decay Model')
    print ('--------------------------------------------------------------------------------------------')
    print ('M_Ni = ',p_r[0],' +/- ',pcov_r.diagonal()[0],' M_sun')
    print ('M_ej = ',Mej1,' +/- ',DMej1,' M_sun for opacity kappa = 0.05 cm^2/g')
    print ('M_ej = ',Mej2,' +/- ',DMej2,' M_sun for opacity kappa = 0.20 cm^2/g')
    print ('M_ej = ',Mej3,' +/- ',DMej3,' M_sun for opacity kappa = 0.33 cm^2/g')
    print ('E_SN = ',E1,' +/- ',DE1, 'erg for opacity kappa = 0.05 cm^2/g')
    print ('E_SN = ',E2,' +/- ',DE2, 'erg/s for opacity kappa = 0.20 cm^2/g')
    print ('E_SN = ',E3,' +/- ',DE3, 'erg for opacity kappa = 0.33 cm^2/g')
    print ('The reduced chi-square value for the best-fit model is ',chi_red_r)
    print ('********************************************************************************************')
    print ('********************************************************************************************')
    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_rad_r0(float(i),p_r[0],p_r[1]))
        print ('********************************************************************************************')
        print ('********************************************************************************************')

    model2 = lummagr0_v
    guess_m  = (0.5,35.,20.)
    bound_m  = (((0.1,1.,2.),(10.0,100.,50.)))
    p_m, pcov_m = curve_fit(model2, xdata, ydata, sigma = dydata, p0=guess_m, bounds=bound_m)
    chi_red_m = (1./(len(xdata)-len(guess_m))) * sum((ydata-lummagr0_v(xdata,p_m[0],p_m[1],p_m[2]))**2/dydata**2)
    B14 = 10.**14 # Magnetic field-strength in 10^14 Gauss
    e0 = 2.e50
    Pmag = 10.* np.sqrt(e0/(p_m[0]*LCmods.E51))
    DPmag = 0.5 * Pmag * (pcov_m.diagonal()[0]/p_m[0])
    Bmag = np.sqrt(130.*Pmag**2/(365.*p_m[1])) * B14
    DBmag = Bmag * np.sqrt((DPmag/Pmag)**2+(0.5*pcov_m.diagonal()[2]/p_m[2])**2)
    Mej1 = ((3./10.)*(13.8*LCmods.c/0.05)*10000.*LCmods.kms2cms*(p_m[1]*LCmods.day)**2)/LCmods.Msun
    Mej2 = ((3./10.)*(13.8*LCmods.c/0.2)*10000.*LCmods.kms2cms*(p_m[1]*LCmods.day)**2)/LCmods.Msun
    Mej3 = ((3./10.)*(13.8*LCmods.c/0.33)*10000.*LCmods.kms2cms*(p_m[1]*LCmods.day)**2)/LCmods.Msun
    DMej1 = (Mej1*np.sqrt(4.*((pcov_m.diagonal()[1]/p_m[1])**2)))
    DMej2 = (Mej2*np.sqrt(4.*((pcov_m.diagonal()[1]/p_m[1])**2)))
    DMej3 = (Mej3*np.sqrt(4.*((pcov_m.diagonal()[1]/p_m[1])**2)))
    E1 = (3./10.)*Mej1*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    E2 = (3./10.)*Mej2*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    E3 = (3./10.)*Mej3*LCmods.Msun*(10000.*LCmods.kms2cms)**2
    DE1 = E1*np.sqrt((DMej1/Mej1)**2)
    DE2 = E2*np.sqrt((DMej2/Mej2)**2)
    DE3 = E3*np.sqrt((DMej3/Mej3)**2)    
    print ('List of Derived Parameters for the Basic (R0 = 0, vej = 10,000 km/s) Magnetar Spin-down Model')
    print ('----------------------------------------------------------------------------------------------')
    print ('Pmag = ',Pmag,' +/- ',DPmag,' milliseconds')
    print ('Bmag = ',Bmag/B14,' +/- ',DBmag/B14,' x 10^14 Gauss')
    print ('M_ej = ',Mej1,' +/- ',DMej1,' M_sun for opacity kappa = 0.05 cm^2/g')
    print ('M_ej = ',Mej2,' +/- ',DMej2,' M_sun for opacity kappa = 0.20 cm^2/g')
    print ('M_ej = ',Mej3,' +/- ',DMej3,' M_sun for opacity kappa = 0.33 cm^2/g')
    print ('E_SN = ',E1,' +/- ',DE1, 'erg for opacity kappa = 0.05 cm^2/g')
    print ('E_SN = ',E2,' +/- ',DE2, 'erg for opacity kappa = 0.20 cm^2/g')
    print ('E_SN = ',E3,' +/- ',DE3, 'erg for opacity kappa = 0.33 cm^2/g')
    print ('The reduced chi-square value for the best-fit model is ',chi_red_m)
    print ('*********************************************************************************************')
    print ('*********************************************************************************************')
    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_mag_r0(float(i),p_m[0],p_m[1],p_m[2]))
        print ('*********************************************************************************************')
        print ('*********************************************************************************************')

    model3 = lumcsmonlys0
    guess_c0  = (1.0,0.01,8.,0.33,12,5.,0.1,1000.)
    bound_c0  = (((0.1,0.0001,3.,0.2,8,0.01,0.00001,50.),(20.,1.0,80.,0.4,12,50.,1.0,2000.)))
    p_c0, pcov_c0 = curve_fit(model3, xdata, ydata, sigma = dydata, p0=guess_c0, bounds=bound_c0)
    chi_red_c0 = (1./(len(xdata)-len(guess_c0))) * sum((ydata-lumcsmonlys0(xdata,p_c0[0],p_c0[1],p_c0[2]\
                                            ,p_c0[3],p_c0[4],p_c0[5],p_c0[6],p_c0[7]))**2/dydata**2)
    print ('List of Derived Parameters for the CSM Model for constant density CSM shell (s_CSM = 0)')
    print ('---------------------------------------------------------------------------------------')
    print ('E_SN = ',p_c0[0],' +/- ',pcov_c0.diagonal()[0],' x 10^51 erg')
    print ('R_progenitor = ',p_c0[1]/10.,' +/- ',pcov_c0.diagonal()[1]/10.,' x 10^14 cm')
    print ('M_ej = ',p_c0[2],' +/- ',pcov_c0.diagonal()[2],' M_sun')
    print ('Kappa_ej = ',round(p_c0[3],2),' SN ejecta opacity in cm^2/g')
    print ('n_ej = ',int(round(p_c0[4],0)),' outer slope of SN ejecta density profile')
    print ('M_CSM = ',p_c0[5],' +/- ',pcov_c0.diagonal()[5],' M_sun')
    print ('Mdot = ',p_c0[6],' +/- ',pcov_c0.diagonal()[6],' M_sun/year')
    print ('The reduced chi-square value for the best-fit model is ',chi_red_c0)
    print ('***************************************************************************************')
    print ('***************************************************************************************')
    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_csmonly_s0(float(i),p_c0[0],p_c0[1],p_c0[2],p_c0[3],p_c0[4],p_c0[5],p_c0[6],p_c0[7]))
        print ('***************************************************************************************')
        print ('***************************************************************************************')

    model4 = lumcsmonlys2
    guess_c2  = (1.0,0.01,10.,0.33,12,5.0,0.2,1000.)
    bound_c2  = (((0.1,0.0001,3.,0.2,8,0.01,0.0001,50.),(10.,1.0,80.,0.4,12,50.,1.0,2000.)))
    p_c2, pcov_c2 = curve_fit(model4, xdata, ydata, sigma = dydata, p0=guess_c2, bounds=bound_c2)
    chi_red_c2 = (1./(len(xdata)-len(guess_c2))) * sum((ydata-lumcsmonlys2(xdata,p_c2[0],p_c2[1],\
                                p_c2[2],p_c2[3],p_c2[4],p_c2[5],p_c2[6],p_c2[7]))**2/dydata**2)
    print ('List of Derived Parameters for the CSM Model for 1/r^2 (s_CSM = 2) wind-like CSM')
    print ('--------------------------------------------------------------------------------')
    print ('E_SN = ',p_c2[0],' +/- ',pcov_c2.diagonal()[0],' x 10^51 erg')
    print ('R_progenitor = ',p_c2[1]/10.,' +/- ',pcov_c2.diagonal()[1]/10.,' x 10^14 cm')
    print ('M_ej = ',p_c2[2],' +/- ',pcov_c2.diagonal()[2],' M_sun')
    print ('Kappa_ej = ',round(p_c2[3],2),' SN ejecta opacity in cm^2/g')
    print ('n_ej = ',int(round(p_c2[4],0)),' outer slope of SN ejecta density profile')
    print ('M_CSM = ',p_c2[5],' +/- ',pcov_c2.diagonal()[5],' M_sun')
    print ('Mdot = ',p_c2[6],' +/- ',pcov_c2.diagonal()[6],' M_sun/year')
    print ('The reduced chi-square value for the best-fit model is ',chi_red_c2)
    print ('********************************************************************************')
    print ('********************************************************************************')
    if show_mod:
        for i in range(1,300):
            print (float(i),LCmods.Lum_csmonly_s2(float(i),p_c2[0],p_c2[1],p_c2[2],p_c2[3],p_c2[4],p_c2[5],p_c2[6],p_c2[7]))
        print ('********************************************************************************')
        print ('********************************************************************************')

    chi_ar = np.array([chi_red_r,chi_red_m,chi_red_c0,chi_red_c2])
    mod_ar = np.array(['Radioactive decay', 'Magnetar spin-down', 'CSM interaction, s=0', 'CSM interaction, s=2'])
    print ('----------------------------------------------------------------------------------------------------------------------')
    print ('The lowest reduced chi-square value is ',min(chi_ar),' for the',mod_ar[np.where(chi_ar == min(chi_ar))[0][0]],' model.')
    print ('----------------------------------------------------------------------------------------------------------------------')
    
    if plot:
        mod1 = [0.,]
        mod2 = [0.,]
        mod3 = [0.,]
        mod4 = [0.,]
        tt = [0.,]
        for i in range(1,int(xdata[len(xdata)-1])):
            mod1.append(i)
            mod2.append(i)
            mod3.append(i)
            mod4.append(i)
            tt.append(i)
            mod1[i] = LCmods.Lum_rad_r0(float(i),p_r[0],p_r[1])
            mod2[i] = LCmods.Lum_mag_r0(float(i),p_m[0],p_m[1],p_m[2])
            mod3[i] = LCmods.Lum_csmonly_s0(float(i),p_c0[0],p_c0[1],p_c0[2],p_c0[3],p_c0[4],p_c0[5],p_c0[6],p_c0[7])
            mod4[i] = LCmods.Lum_csmonly_s2(float(i),p_c2[0],p_c2[1],p_c2[2],p_c2[3],p_c2[4],p_c2[5],p_c2[6],p_c2[7])
            tt[i] = i
        mod1 = np.asarray(mod1)
        mod2 = np.asarray(mod2)
        mod3 = np.asarray(mod3)
        mod4 = np.asarray(mod4)
        tt = np.asarray(tt)
        plt.title('Basic SN LC model fits')
        plt.xlabel('Rest-frame time [days]',fontsize=20)
        plt.ylabel('Luminosity [erg/s]',fontsize=20)
        plt.rcParams.update({'font.size': 15})
        plt.semilogy(tt, mod1, 'k', linewidth=2.0, label='RD')
        plt.semilogy(tt, mod2, 'r', linewidth=2.0, label='MAG')
        plt.semilogy(tt, mod3, 'g', linewidth=2.0, label='CSM0')
        plt.semilogy(tt, mod4, 'c', linewidth=2.0, label='CSM2')
        plt.semilogy(xdata, ydata, 'b', linewidth=2.0, label= str(sys.argv[1]))
        plt.errorbar(xdata, ydata, yerr=dydata, fmt="none", color='b', linewidth=2.0)
        legend = plt.legend(loc='upper right', shadow=True)
        plt.xlim([0,int(xdata[len(xdata)-1])+10])
        plt.ylim(0.2*min(ydata),2.0*max(ydata))
        plt.minorticks_on()
        plt.show()    
        
else:
    print ('This is not a valid model entry. For a list of available models please read intro comments above.')

warnings.filterwarnings("ignore")

#Goodbye message
print ('*************************************************************************************************')
print ('BOOM! HOPE YOU ENJOYED YOUR SUPERNOVA FITS! THANK YOU FOR USING TIGERFIT.')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#END OF TIGERFIT - USE SPACE BELOW TO ADD NEW MODELS AND PRINT ANY MODEL TO STANDARD OUTPUT

#for i in range(1,300):
#    print float(i),LCmods.Lum_mag_r0(float(i),1.0,50.,20.)
#    print float(i),LCmods.Lum_rad(float(i),popt[0],popt[1],popt[2],popt[3],popt[4])
#    print float(i),LCmods.Lum_mag(float(i),popt[0],popt[1],popt[2],popt[3],popt[4])
#     print float(i),LCmods.Lum_csm(float(i),6.63013941e-02,1.53349274e-01,6.94636374e+01,1.99176537e+00 \
#                                   ,5.85160431e-02,1.45320793e+01,3.43945790e-01,9.89028515e+00 \
#                                   ,1.90279842e+00,4.43890926e+00,1.93411596e-03,1.00000000e-10 \
#                                   ,1.03223107e+02)
#     print float(i),LCmods.Lum_csmrad_s2(float(i),0.01,1.0,0.01,8.,0.33,11,1.0,0.1,1000.)
#     print float(i), LCmods.Lum_csm(float(i),0.01,1.e-20,5.,2.0,0.1,18.,0.33,12,0,10.,0.1,2,1000.)
