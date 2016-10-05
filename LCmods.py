import numpy as np
from scipy.integrate import quad, nquad, odeint, simps
from scipy.special import gammainc, gamma
import math
import os
import sys
import contextlib
import warnings

#Global model constants, all units are CGI
day=86400.     #seconds in a day
year=3.15436e7 #seconds in a year
Msun=1.99e33   #solar mass in grams
c=2.99792458e10#speed of light
sb=5.67051e-5  #Stefan-Boltzmann constant
kms2cms=1.e5   #km/s in cm/s
r15=1.e15      #radii in units of 10^15cm
tni=8.8        #Ni-56 decay time-scale
tco=111.3      #Co-56 decay time-scale
eni=3.9e10     #Ni-56 decay specific energy generation rate
eco=6.8e9      #Co-56 decay specific energy generation rate
TH=5500.       #Hydrogen ionization temperature in K
A0=1.e12       #Radioactive decay model gamma-ray leakage parameter in 10^14 units (sec^2)
E51=1.e51      #Energy in units of 1 F.O.E. (10^51 erg)
L45=1.e45      #Luminosity in units of 10^45 erg/s
nmax=1000000   #Grid resolution for fallback accretion model integration

###FUNCTIONS TO PREVENT WARNING OUTPUT FOR TDE MODEL####

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    http://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
            
#####################################
#SUPERNOVA LIGHT CURVE MODELS FOLLOW
#####################################

#1.RADIOACTIVE DECAY DIFFUSION MODEL (Arnett 1980,1982)

# Factor function that appears in front of exponents in radioactive decay model
# inputs are time (t) and diffusion time-scale (td) both in days, progenitor radius
# (r0) in units of 10^15cm and SN ejecta velocity (vej) in units of km/s

def rad_decay_dep(t,td,r0,vej):
    return ((r0*r15/(vej*td*day*kms2cms))+t/td)*np.exp((t/td)**2+(2.*r0*r15*t/(vej*kms2cms*(td**2)*day)))

# Integrants for Ni-56 and Co-56 decay energy depositions

def rad_decay_int1(t,td,r0,vej):
    return rad_decay_dep(t,td,r0,vej)*np.exp(-t/tni)

def rad_decay_int2(t,td,r0,vej):
    return rad_decay_dep(t,td,r0,vej)*np.exp(-t/tco)

# Final radioactive decay luminosity integral function

def Lum_rad(x,Mni,td,r0,vej,A):
    res = (2.*Mni*Msun/td)*np.exp(-((x/td)**2+(2.*r0*r15*x/(vej*kms2cms*(td**2)*day))))* \
    ((eni-eco)*quad(rad_decay_int1,0,x,args=(td,r0,vej))[0] + eco*quad(rad_decay_int2,0,x,args=(td,r0,vej))[0])* \
    (1.-np.exp(-A*A0/((x+0.01)*day)**2))
    return res

# Version 2 of Radioactive decay model for small initial radius and full trapping

def Lum_rad_r0(x,Mni,td):
    r0 = 0.
    vej = 10000.
    A = 100000.
    res2 = Lum_rad(x,Mni,td,r0,vej,A)
    return res2

# OTHER VERSION YIELDING VERY SIMILAR MODELS (VALENTI ET AL. 2008)
#def lumint1(t,td):
#    return t*day*np.exp(t**2/td**2)*np.exp(-t/tni)

#def lumint2(t,td):
#    return t*day*np.exp(t**2/td**2)*np.exp(-t/tni)*np.exp(t*(tco-tni)/(tco*tni))

#def Lum_rad(x,Mni,td,r0,vej,A):
#    return (2.*Mni*Msun/td)*np.exp(-((x/td)**2))*\
#      ((eni-eco)*quad(lumint1,0,x/td,args=(td))[0] + eco*quad(lumint2,0,x/td,args=(td))[0])

#############################

#2. MAGNETAR SPIN-DOWN MODEL (Kasen & Bildsten; Woosley 2010)

# Integrand for magnetar spin-down energy release

def mag_int(t,td,tp,r0,vej):
    return (np.exp((t/td)**2+(r0*r15*t/(vej*kms2cms*(td**2)*day))))*((r0*r15/(vej*kms2cms*td*day))+ \
    (t/td))*(1./(1.+t/tp))**2

# Final magnetar spin-down luminosity integral function

def Lum_mag(x,Ep,td,tp,r0,vej):
    return (2.*Ep*E51/(tp*td*day))*(np.exp(-((x/td)**2+(r0*r15*x)/(vej*kms2cms*(td**2)*day))))* \
    quad(mag_int,0,x,args=(td,tp,r0,vej))[0]

# Version 2 of magnetar spin-down model for small initial radius

def Lum_mag_r0(x,Ep,td,tp):
    r0 = 0.
    vej = 10000.
    res2 = Lum_mag(x,Ep,td,tp,r0,vej)
    return res2

##############################

#3. FORWARD AND REVERSE SHOCK LUMINOSITY INPUT FOR SN EJECTA - CSM INTERACTION PLUS NI-56 CONTRIBUTION

# Integrands for forward and reverse shock luminosity input

def csm_int(t,Mni,Esn,Rp,Mej,kappa,n,d,Mcsm,Mdot,s,vwind):
    #Definitions of all model factors
    if int(s)==0:
        Bf, Br = 1.121, 0.974
    else:
        Bf, Br = 1.226, 0.987
    
    beta=13.8 # integration constant
    x0=0.3    # dimensionless SN ejecta density profile break radius
    A = 2.*(3.-s)**2/((n-4.)*(n-3.))
    alpha = (2.*n+6.*s-n*s-15.)/(n-s)
    rho_csm = (Mdot*Msun/year)/(4.*math.pi*vwind*kms2cms*(Rp*r15)**2)
    q = rho_csm * (Rp*r15)**s
    Rcsm = (Mcsm*Msun*(3-s)/(4*math.pi*q) + (Rp*r15)**(3-s))**(1./(3.-s))
    Rph = (-2./3.*(1.-s)/(kappa*q) + (Rcsm)**(1.-s))**(1./(1.-s))
    Mcsmth = 4.*math.pi*q/(3.-s) * (Rph**(3.-s) - (Rp*r15)**(3.-s))   
    t0 = (kappa * Mcsmth) / (beta * c * Rph) / day
    t00 = kappa * (Mej*Msun + Mcsmth) /(beta * c * Rph) /day
    
    gn = 0.5*(n-3.)*math.log((Esn*E51)*2.*(5.-d)*(n-5.)) - 0.5*(n-5.)*math.log((3.-d)*(n-3.)*Mej*Msun)
    gn = np.exp(gn)/(4*math.pi*(n-d))
   
    Kf = 2.*math.pi*((n-3.)**2)*((n-5.)/(n-s)**3)*(q**((n-5.)/(n-s)))*(Bf**(5.-s))*((A*gn)**((5.-s)/(n-s)))
    Kr = 2.*math.pi*((A*gn/q)**((5.-n)/(n-s)))*(Br**(5.-n))*gn*((3.-s)/(n-s))**3	
   
    vsn = math.sqrt(2.*(5.-d)*(n-5.)*Esn*E51/((3.-d)*(n-3.)*Mej*Msun))/x0
    
    tfs = abs((3.-s)*(q**((3.-n)/(n-s)))*((A*gn)**((s-3.)/(n-s)))/(4.*math.pi*(Bf**(3.-s))))
    tfs = (tfs**((n-s)/((n-3.)*(3.-s))))*(Mcsmth**((n-s)/((n-3.)*(3.-s))))
    tfs = tfs/day

    tfsbo = (Rph - Rp*r15)/(Bf*(A*gn/q)**(1./(n-s)))
    tfsbo = tfsbo**((n-s)/(n-3.0))
    tfsbo = tfsbo/day
   
    trs = vsn/(Br*(A*gn/q)**(1.0/(n-s))) * (1.0 - (3-n)*Mej*Msun/(4*math.pi*(vsn**(3-n))*gn))**(1.0/(3-n))
    trs = (trs**((n-s)/(s-3.)))
    trs = trs/day
    ti = Rp*r15 / vsn / day

    #Integrands for forward and reverse luminosity input and Ni-56 decay for hybrid case

    if t > tfsbo:
        L_f_int = 0.0
    else:
        L_f_int = Kf * (t*day+ti*day)**alpha
    if t > trs:
        L_r_int = 0.0
    else:
        L_r_int = Kr * (t*day+ti*day)**alpha
        
    return np.exp(t/t0)*(L_f_int+L_r_int)

# Integrand for radioactive decay contribution under assumption of "constant" photosphere in
# optically-thick CSM shell ("bloated-star")

def nicsm_int(t,Mni,Rp,Mej,kappa,Mcsm,Mdot,s,vwind):
    beta=13.8 # integration constant
    rho_csm = (Mdot*Msun/year)/(4.*math.pi*vwind*kms2cms*(Rp*r15)**2)
    q = rho_csm * (Rp*r15)**s
    Rcsm = (Mcsm*Msun*(3-s)/(4*math.pi*q) + (Rp*r15)**(3-s))**(1./(3.-s))
    Rph = (-2./3.*(1.-s)/(kappa*q) + (Rcsm)**(1.-s))**(1./(1.-s))
    Mcsmth = 4.*math.pi*q/(3.-s) * (Rph**(3.-s) - (Rp*r15)**(3.-s))   
    t00 = kappa * (Mej*Msun + Mcsmth) /(beta * c * Rph) /day
    return np.exp(t/t00)*Mni*Msun*((eni-eco)*np.exp(-t/tni)+eco*np.exp(-t/tco))

# Integrant for magnetar spin-down contribution under assumption of "constant" photosphere in
# optically-thick CSM shell ("bloated-star")

def magcsm_int(t,Ep,tp,Rp,Mej,kappa,Mcsm,Mdot,s,vwind):
    beta=13.8 # integration constant
    rho_csm = (Mdot*Msun/year)/(4.*math.pi*vwind*kms2cms*(Rp*r15)**2)
    q = rho_csm * (Rp*r15)**s
    Rcsm = (Mcsm*Msun*(3-s)/(4*math.pi*q) + (Rp*r15)**(3-s))**(1./(3.-s))
    Rph = (-2./3.*(1.-s)/(kappa*q) + (Rcsm)**(1.-s))**(1./(1.-s))
    Mcsmth = 4.*math.pi*q/(3.-s) * (Rph**(3.-s) - (Rp*r15)**(3.-s))   
    t00 = kappa * (Mej*Msun + Mcsmth) /(beta * c * Rph) /day
    return np.exp(t/t00)*(Ep*E51/(tp*day))*(1./(1+t/tp))**2

# Final SN ejecta - CSM interaction plus radioactive decay integral function
# Set Ep = 0, Mni = 0 to obtain pure CSM shock input contributions

def Lum_csm(x,Mni,Ep,tp,Esn,Rp,Mej,kappa,n,d,Mcsm,Mdot,s,vwind):
    beta=13.8 # integration constant
    n = int(round(n,0))
    d = int(round(d,0))
    s = int(round(s,0))
    rho_csm = (Mdot*Msun/year)/(4.*math.pi*vwind*kms2cms*(Rp*r15)**2)
    q = rho_csm * (Rp*r15)**s
    Rcsm = (Mcsm*Msun*(3-s)/(4*math.pi*q) + (Rp*r15)**(3-s))**(1./(3.-s))
    Rph = (-2./3.*(1.-s)/(kappa*q) + (Rcsm)**(1.-s))**(1./(1.-s))
    Mcsmth = 4.*math.pi*q/(3.-s) * (Rph**(3.-s) - (Rp*r15)**(3.-s))   
    t0 = (kappa * Mcsmth) / (beta * c * Rph) / day
    t00 = kappa * (Mej*Msun + Mcsmth) /(beta * c * Rph) / day
    
    Lum_csm = (1./t0) * np.exp(-x/t0)* \
    quad(csm_int,0,x,args=(Mni,Esn,Rp,Mej,kappa,n,d,Mcsm,Mdot,s,vwind),full_output=1)[0]+ \
    (1./t00) * np.exp(-x/t00) * \
    quad(nicsm_int,0,x,args=(Mni,Rp,Mej,kappa,Mcsm,Mdot,s,vwind),full_output=1)[0] + \
    (1./t00) * np.exp(-x/t00) * \
    quad(magcsm_int,0,x,args=(Ep,tp,Rp,Mej,kappa,Mcsm,Mdot,s,vwind),full_output=1)[0]
    return Lum_csm

# Version 2 of CSM model for only CSM+RAD contributions (wind s=2, d=0)

def Lum_csmrad_s2(x,Mni,Esn,Rp,Mej,kappa,n,Mcsm,Mdot,vwind):
    n = int(round(n,0))
    s = 2
    d = 0
    Ep = 1.0e-20
    tp = 5.
    res2 = Lum_csm(x,Mni,Ep,tp,Esn,Rp,Mej,kappa,n,d,Mcsm,Mdot,s,vwind)
    return res2

# Version 3 of CSM model for only CSM+MAG contributions (wind s=2, d=0)

def Lum_csmmag_s2(x,Ep,tp,Esn,Rp,Mej,kappa,n,Mcsm,Mdot,vwind):
    n = int(round(n,0))
    s = 2
    d = 0
    Mni = 0.
    res3 = Lum_csm(x,Mni,Ep,tp,Esn,Rp,Mej,kappa,n,d,Mcsm,Mdot,s,vwind)
    return res3

# Version 4 of CSM model for only CSM contributions (wind s=2, d=0)
def Lum_csmonly_s2(x,Esn,Rp,Mej,kappa,n,Mcsm,Mdot,vwind):
    n = int(round(n,0))
    s = 2
    d = 0
    Mni = 0.
    Ep = 0.
    tp = 5.
    res4 = Lum_csm(x,Mni,Ep,tp,Esn,Rp,Mej,kappa,n,d,Mcsm,Mdot,s,vwind)
    return res4

# Version 5 of CSM model for only CSM+RAD contributions (shell s=0, d=0)

def Lum_csmrad_s0(x,Mni,Esn,Rp,Mej,kappa,n,Mcsm,Mdot,vwind):
    n = int(round(n,0))
    s = 0
    d = 0
    Ep = 0.
    tp = 5.
    res5 = Lum_csm(x,Mni,Ep,tp,Esn,Rp,Mej,kappa,n,d,Mcsm,Mdot,s,vwind)
    return res5

# Version 6 of CSM model for only CSM+MAG contributions (shell s=0, d=0)

def Lum_csmmag_s0(x,Ep,tp,Esn,Rp,Mej,kappa,n,Mcsm,Mdot,vwind):
    n = int(round(n,0))
    s = 0
    d = 0
    Mni = 0.
    res6 = Lum_csm(x,Mni,Ep,tp,Esn,Rp,Mej,kappa,n,d,Mcsm,Mdot,s,vwind)
    return res6

# Version 7 of CSM model for only CSM contributions (shell s=0, d=0)
def Lum_csmonly_s0(x,Esn,Rp,Mej,kappa,n,Mcsm,Mdot,vwind):
    n = int(round(n,0))
    s = 0
    d = 0
    Mni = 0.
    Ep = 0.
    tp = 5.
    res7 = Lum_csm(x,Mni,Ep,tp,Esn,Rp,Mej,kappa,n,d,Mcsm,Mdot,s,vwind)
    return res7


####################################
    
#4. INPUT LUMINOSITY BY FALLBACK ACCRETION (Dexter & Kasen 2013) Equations A6-A12

# Definition of differential function to find position of ionization front (Equation A12 in DK2013)

def ion_fr(y,x,L0,R0,vsh,n,td):
    t0 = R0*r15 / (vsh*kms2cms) / day
    y0 = y[0]
    x = x * day
    Li = 4.*math.pi*((vsh*kms2cms*td*day)**2)*sb*TH**4
    y1 = -2.*y0/(5.*x)-x/(5.*y0*((td*day)**2))+(1./(5.*(y0**3)*x))*L0*L45*(x**-n)/ \
    ((t0**-n)*Li)
    return y1

def ion_func(L0,R0,vsh,n,td,nmax):
    init  = 1.0
    tt=np.linspace(1,1000,nmax)
    with stdout_redirected():
        sol = odeint(ion_fr, init, tt * day, args=(L0,R0,vsh,n,td))
    return sol

#Generate time axis grid corresponding to above-selected resolution
tlist=[]
for i in range(0,nmax):
    tlist.append(i)
    tlist[i] = 1+i*(999./nmax)

#print ion_func(1.,0.1,5000.,5./3.,30.,nmax)[10,0]

#for i in range(1,100):
#    print ion_func(1.,0.1,5000.,5./3.,30.,nmax)[i,0],ion_func(1.,0.1,5000.,5./3.,30.,nmax)[50,0]

def Lum_fb(x,L0,R0,vsh,n,td):
    t0 = R0*r15 / (vsh*kms2cms) / day

    diff = [(abs(x - xtarg),idx) for (idx,xtarg) in enumerate(tlist)]
    diff.sort()
    t_index = diff[0][1]+1
    xi = ion_func(L0,R0,vsh,n,td,nmax)[:,0][t_index]
    if xi < 1.e-10 or xi > 1.0:
        xi = 0.
        tii = x
    
    if x < t0: lum,Tph = 0.,0.

    else:   
        prefactor = L0 * L45 * ((t0/td)**n) * np.exp(-x**2/(2.*td**2)) * (-0.5**(n/2)+0j).real
        lum = prefactor * (gamma(1.-n/2.)*gammainc(1.-n/2.,t0**2/(2*td**2)) - gamma(1.-n/2.)*gammainc(1.-n/2.,x**2/(2*td**2)))
        Tph = (lum/(4.*math.pi*sb*((vsh*kms2cms*x*day)**2)))**0.25
        if Tph < TH:
            if xi < 1.e-10:
                lum = lum*(1.-np.exp(-(x**2-t0**2)/(2.*td**2)))  # or lum=0 for simplicity
            else:
                lum = 4.*math.pi*((vsh*kms2cms*x*day*xi)**2)*sb*TH**4
        
    return lum

warnings.filterwarnings("ignore")

#######################################CUSTOMIZE OTHER MODELS IN THE SPACE BELOW#############################
