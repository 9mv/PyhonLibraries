"""
Script per calcular paràmetres de EiR.
Aleix Martínez Vinent
Maig 2020
"""
import numpy as np
import scipy.constants as sci
import cmath
C = 3*(10**8)
e0=sci.epsilon_0
mu0=sci.mu_0
pi = np.pi
m = 10**(-3)
k = 10**3
M = 10**6
G = 10**9
directivity={7:[0.782,0.138],7.5:[0.824,0.146],8:[0.865,0.157],8.5:[0.892,0.165],9:[0.918,0.169],9.5:[0.935,0.174],10:[0.943,0.179],10.5:[0.957,0.182],11:[0.964,0.185]}

def log_periodic(fmin,fmax,dire):
    """
    Entrada:
    \t - Frequencia minima [Hz]
    \t - Frequencia maxima [Hz]
    \t - Directivitat [dBi]

    Sortida:
    \t - Vector de longitud dels dipols [m]
    \t - Vector de longituds entre dipols [m]


    """
    N = int(np.log10(fmin/fmax)/(np.log10(directivity[dire][0]))+1)
    L = [C/fmin]
    S = []
    for i in range(0,N):
        L.append(directivity[int(dire)][0]*L[i])
        S.append(2*directivity[int(dire)][1]*L[i])
    print(f"\n \n Nombre de branques: {len(L)} \n \n")
    return L,S

def horn(G,f,a,b,dB=True):
    """
    Entrada:
    \t - Guany [dBi]
    \t - Frequencia del guany [Hz]
    \t - a [m]
    \t - b [m]
    
    Sortida:
    \t - xi
    \t - roe [m]
    \t - roh [m]
    \t - a1 [m]
    \t - b1 [m]
    \t - pe [m]
    \t - ph [m]
    
    
    """
    if dB == True:
        G0=10**(G/10)
    xi=G0/(2*pi*np.sqrt(2*pi))
    wl=C/f
    pujar = False
    mod = 0.5
    X1=((np.sqrt(2*xi)-(b/wl))**2)*(2*xi-1)
    X2=(((G0/(2*pi))*(np.sqrt(3/(2*pi)))*(1/np.sqrt(xi))-(a/wl))**2)*(((G0**2)/(6*(pi**3)))*(1/xi)-1)
    D=np.abs(X1-X2)+1
    xant=0
    xalt=0
    xbaix=0

    """Obtencio de xi"""
    while np.abs(X1-X2)>0.0005:
        xant = xi        
        Dant = D
        D = np.abs(X1-X2)
        
        if Dant > D:                       # Si diferència entre X1 i X2 era major abans, anem per bon camí
            mod*=0.9                                # Disminuim increment per no passar-nos i anar interpolant valor
            if pujar == False:
                xi -=mod 
            else:
                xi+=mod
                    
        else:                              # Si diferencia era menor abans, ens estem allunyant del valor correcte. Si abans haviem pujat ara haurem de baixar el valor de xi.
            mod*=0.91                                # Fem un increment major perque el d'abans era incorrecte i així tornem cap enrrera.
            if pujar == False:                              # Si abans haviem baixat, hem de tornar cap enrrera i pujar
                xi+=mod
                pujar = True
            else:                                           # Si abans haviem pujat, hem de tornar cap enrrera i baixar
                xi-=mod 
                pujar = False 
        X1=((np.sqrt(2*xi)-(b/wl))**2)*(2*xi-1)
        X2=(((G0/(2*pi))*(np.sqrt(3/(2*pi)))*(1/np.sqrt(xi))-(a/wl))**2)*(((G0**2)/(6*(pi**3)))*(1/xi)-1)

    print(f"\n \n Valor de x: {xi} \n ")    
    print(str(X1) + " = " + str(X2))
    
    """Obtencio dels demes parametres"""
    roe=xi*wl
    roh=(wl/xi)*(G0**2)/(8*(pi**3))
    a1=(G0/(2*pi))*wl*np.sqrt(3/(2*pi*xi))
    b1=np.sqrt(2*xi)*wl
    pe=(b1-b)*np.sqrt(-(1/4)+(roe/b1)**2)
    ph=(a1-a)*np.sqrt(-(1/4)+(roh/a1)**2)
    return xi,roe,roh,a1,b1,pe,ph

def patch(f,er,d):
    """
    Dissenya una antena de patch a partir dels seguents parametres:
    \t - Frequencia de funcionament [Hz]
    \t - Permitivitat del substrat [F/m]
    \t - Gruixor del substrat [m]
    
    
    """
    wl=C/f
    W = (1/(2*f*np.sqrt(e0*mu0)))*np.sqrt(2/(er+1))
    ereff = (er+1)/2+(er-1)/(2*np.sqrt(1+(12*d/W)))
    dL = 0.412*d*(ereff + 0.3)*((W/d)+0.264)/((ereff-0.258)*((W/d)+0.8))
    L = (1/(2*f*np.sqrt(ereff)*np.sqrt(e0*mu0)))-(2*dL)
    Za = (90*(er**2)*((L/W)**2))/(er-1)
    df = (16*(er-1)*L*d)/(3*np.sqrt(2)*(er**2)*wl*W)
    if W > 3*wl: 
        return W,dL,L,ereff,8*(W/wl),Za,df
    
    else:
        return W,dL,L,ereff,8.2,Za,df

def max_range(f,Pt,Dt,Dr,Pl):
    """
    Calcula distància màxima que pot viatjar un senyal transmes per una antena per ser rebuda per una altra a partir dels seguents parametres:
    \t - Frequencia de funcionament [Hz]
    \t - Potencia transmesa [W]
    \t - Directivitat antena transmissora [dBi]
    \t - Directivitat antena receptora [dBi]
    \t - Sensibilitat d'antena receptora [dB]
    
    Retorna distancia en km.
    """
    L0=Dt+Dr + 10*np.log10(Pt*1000)-Pl
    r=10**((L0-32.5-20*np.log10(f/(10**6)))/20)
    return r

def min_sensibility(f,r,Pt,Dt,Dr):
    """
    Calcula sensibilitat mínima [dBi] que ha de tenir antena per rebre un senyal transmes per una antena a una certa distancia a partir dels seguents parametres:
    \t - Frequencia de funcionament [Hz]
    \t - Distància entre les dues antenes [m]
    \t - Potencia transmesa [W]
    \t - Directivitat antena transmissora [dBi]
    \t - Directivitat antena receptora [dBi]
    \t - Sensibilitat d'antena receptora [dBm]
    
    
    """
    return 10*np.log10(Pt*1000)+Dt+Dr-(32.5+20*np.log10(f/(10**6))+20*np.log10(r/1000)) 

def recieved_power(f,r1,CS,Pt,Dt,B=False,Dr=0,r2=0):
    """
    Calcula potencia rebuda per un radar. Dos modes de funcionament: per mateixa antena o per dues.
    
    1 sola antena:
    \t - Frequencia de funcionament [Hz]
    \t - Distància de l'objecte [m]
    \t - Cross Section de l'objecte [m^2]
    \t - Potencia transmesa [W]
    \t - Directivitat antena [dBi]
    \t - True (1 sola antena)
    
    2 antenes:
    \t - Frequencia de funcionament [Hz]
    \t - Distància entre radar emissor i l'objecte [m]
    \t - Cross Section de l'objecte [m^2]
    \t - Potencia transmesa [W]
    \t - Directivitat antena emissora [dBi]
    \t - False (1 sola antena)
    \t - Directivitat antena receptora [dBi]
    \t - Distància entre radar receptor i l'objecte [m]
    
    """
    wl=C/f
    Dt=10**(Dt/10)
    Dr=10**(Dr/10)
    if B==True:
        Pl=(Pt*CS*(Dt**2)*(wl**2))/(((4*pi)**3)*(r1**4))
    else:
        Pl=((Pt*CS*Dr*Dt)/(4*pi))*((wl/(4*pi*r1*r2))**2)
    return 10*np.log10(Pl)

def cross_section(f,r1,Pt,Pl,Dt,B=False, dB=False,Dr=0,r2=0):
    """
    Calcula cross section d'un objecte observat per un radar. Dos modes de funcionament: per mateixa antena o per dues.
    
    1 sola antena:
    \t - Frequencia de funcionament [Hz]
    \t - Distància de l'objecte [m]
    \t - Potencia transmesa [W]
    \t - Potencia rebuda [W]
    \t - Directivitat antena [dBi]
    \t - True (1 sola antena)
    |t - Directivitat expressat en dB [bool]
    
    2 antenes:
    \t - Frequencia de funcionament [Hz]
    \t - Distància entre radar emissor i l'objecte [m]
    \t - Potencia transmesa [W]
    \t - Potencia rebuda [W]
    \t - Directivitat antena emissora [dBi]
    \t - False (2 antenes)
    \t - Directivitat antena receptora [dBi]
    \t - Distància entre radar receptor i l'objecte [m]
    |t - Directivitat expressat en dB [bool]
    
    """
    wl=C/f
    if dB == True:
        Dt=10**(Dt/10)
        Dr=10**(Dr/10)
    
    if B==True:
        CS=(Pl/Pt)*((((4*pi)**3)*(r1**4))/((Dt**2)*(wl**2)))
    else:
        CS=(Pl/Pt)*(((4*pi)/(Dt*Dr))*(1/((wl/(4*pi*r1*r2))**2)))
    return CS
"""
def polar_to_rect_S(S11,S12,S21,S22,radians = False):
    if radians == False:
        if S11[1]>=0: 
            S11[1]=S11[1]*(pi/180)
        else:
            S11[1]=(365-S11[1])*(pi/180)
        if S12[1]>=0: 
            S12[1]=S12[1]*(pi/180)
        else:
            S12[1]=(365-S12[1])*(pi/180)
        if S21[1]>=0: 
            S21[1]=S21[1]*(pi/180)
        else:
            S21[1]=(365-S21[1])*(pi/180)
        if S22[1]>=0: 
            S22[1]=S22[1]*(pi/180)
        else:
            S22[1]=(365-S22[1])*(pi/180)

    c1 = [cmath.rect(S11[0],S11[1]), cmath.rect(S12[0],S12[1])]
    c2 = [cmath.rect(S21[0],S21[1]), cmath.rect(S22[0],S22[1])]
    x = [c1, c2]
    
    return x
"""

def polar_to_rect_S(S11,S12,S21,S22,radians = False):
    if radians == False:
            S11[1]=np.deg2rad(S11[1])
            S12[1]=np.deg2rad(S12[1])
            S21[1]=np.deg2rad(S21[1])
            S22[1]=np.deg2rad(S22[1])
    c1 = [cmath.rect(S11[0],S11[1]), cmath.rect(S12[0],S12[1])]
    c2 = [cmath.rect(S21[0],S21[1]), cmath.rect(S22[0],S22[1])]
    x = [c1, c2]
    
    return x

def polar_to_rect(P, radians = False):
    if radians == False:
        if P[1]>=0:
            P[1]=P[1]*(pi/180)
        else:
            P[1]=(365-P[1])*(pi/180)
        return cmath.rect(P[0],P[1])
    
def rect_to_polar(X,todegrees=True):
    P = cmath.polar(X)    
    if todegrees==False:
        return P
    else:
        C = (P[0], P[1]*(180/pi))
        return C 
    
def transistor_amplifier(Z0,Zs,Zl,S):
    LS = ((Zs-Z0)/(Zs+Z0))
    LL = ((Zl-Z0)/(Zl+Z0))
    Lin = S[0][0] + ((S[0][1]*S[1][0]*LL)/(1-(S[1][1]*LL)))
    Lout = S[1][1] + ((S[0][1]*S[1][0]*LS)/(1-(S[0][0]*LS)))
    
    if (LL == LS):
        GT = np.abs(S[1][0])**2
    else:
        GT = ((np.abs(S[1][0])**2)*(1-(np.abs(LS)**2))*(1-(np.abs(LL)**2)))/((np.abs(1-(LS*Lin))**2)*(np.abs(1-(S[1][1]*LL))**2))
    
    return GT, LS, LL, Lin, Lout, 10*np.log10(GT)

def two_port_TAMP(Z0,Zs,Zl,S,unilateral=False):
    LS = ((Zs-Z0)/(Zs+Z0))
    LL = ((Zl-Z0)/(Zl+Z0))
    Lin = S[0][0] + ((S[0][1]*S[1][0]*LL)/(1-(S[1][1]*LL)))
    G0 =np.abs(S[1][0])**2
    GT = (1-(np.abs(LL)**2))/(np.abs(1-(S[1][1]*LL))**2)
    if unilateral == False:
        GS = (1-(np.abs(LS)**2))/(np.abs(1-(LS*Lin))**2)
    else:
        GS = (1-(np.abs(LS)**2))/(np.abs(1-(LS*S[0][0]))**2)
    G = GS*G0*GT
    return G, 10*np.log10(G)

def rollet(S):
    diff = S[0][0]*S[1][1]-S[0][1]*S[1][0]
    K = (1-np.abs(S[0][0])**2-np.abs(S[1][1])**2+np.abs(diff)**2)/(2*np.abs(S[0][1]*S[1][0]))
    if K > 1: 
        unconditional = True
    else:
        unconditional = False
    
    return unconditional, K, diff

def stability_circles(S):
    diff = S[0][0]*S[1][1]-S[0][1]*S[1][0]
    CL = np.conj(S[1][1]-(diff*np.conj(S[0][0])))/(np.abs(S[1][1])**2-np.abs(diff)**2)
    RL = np.abs((S[0][1]*S[1][0])/((np.abs(S[1][1])**2)-(np.abs(diff)**2)))
    CS = np.conj(S[0][0]-(diff*np.conj(S[1][1])))/(np.abs(S[0][0])**2-np.abs(diff)**2)
    RS = np.abs((S[0][1]*S[1][0])/((np.abs(S[0][0])**2)-(np.abs(diff)**2)))
    return CL, RL, CS, RS

def LS_LL_LIN_LOUT(Z0,Zs,Zl,S):
    LS = ((Zs-Z0)/(Zs+Z0))
    LL = ((Zl-Z0)/(Zl+Z0))
    Lin = S[0][0] + ((S[0][1]*S[1][0]*LL)/(1-(S[1][1]*LL)))
    Lout = S[1][1] + ((S[0][1]*S[1][0]*LS)/(1-(S[0][0]*LS)))
    return LS, LL, Lin, Lout

def amp_max_gain(Z0,Zs,Zl,S):
    LS,LL,Lin,Lout = LS_LL_LIN_LOUT(Z0,Zs,Zl,S)
    if (Lin == np.conj(LS) and Lout == np.conj(LL)):
        conj = True
    if (S[0][1]<=0.01):
        uni = True
    if conj == True:
        GT_MAX = (1/(1-(np.abs(LS)**2)))*(np.abs(S[1][0])**2)*((1-(np.abs(LL)**2))/np.abs(1-(S[1][1]*LL))**2)
    elif uni == True:
        GT_MAX = (1/(1-(np.abs(S[0][0])**2)))*(np.abs(S[1][0])**2)*(1/np.abs(1-(S[1][1]*LL))**2)
    else:
        return 0
    return GT_MAX

def spec_gain_mal(Z0,Zs,Zl,S,Gs=1,Gs_MAX=0, Gl=1, Gl_MAX=0):
        LS,LL,Lin,Lout = LS_LL_LIN_LOUT(Z0,Zs,Zl,S)
        if Gs == 0 and Gs_MAX == 0:
            gs =(1-(np.abs(S[0][0])**2))*((1-(np.abs(LS)**2))/(np.abs(1-(S[0][0]*LS))**2))
        elif Gs_MAX == 0:
            gs = Gs/(1/(1-(np.abs(S[0][0])**2)))
        else:
            gs = Gs/Gs_MAX
            
        if Gl == 0 and Gl_MAX == 0:
            gl =(1-(np.abs(S[1][1])**2))*((1-(np.abs(LL)**2))/(np.abs(1-(S[1][1]*LL))**2))
        elif Gl_MAX == 0:
            gl = Gl/(1/(1-(np.abs(S[1][1])**2)))
        else:
            gl = Gl/Gl_MAX
        
        CS = (gs*np.conj(S[0][0]))/(1-(1-gs)*(np.abs(S[0][0])**2))
        RS = (np.sqrt(1-gs)*(1-(np.abs(S[0][0])**2)))/(1-(1-gs)*(np.abs(S[0][0])**2))
        CL = (gl*np.conj(S[1][1]))/(1-(1-gl)*(np.abs(S[1][1])**2))
        RL = (np.sqrt(1-gl)*(1-(np.abs(S[1][1])**2)))/(1-(1-gl)*(np.abs(S[1][1])**2))
        return CS, RS, CL, RL
    
def spec_gain(Z0,S,Gs, Gl, polar=True, dB=True, Gs_MAX=0, Gl_MAX=0):
        if dB == True:
            Gs = 10**(Gs/10)
            Gl = 10**(Gl/10)
    
        if Gs_MAX == 0:
            gs = Gs/(1/(1-(np.abs(S[0][0])**2)))
        else:
            gs = Gs/Gs_MAX
        if Gl_MAX == 0:
            gl = Gl/(1/(1-(np.abs(S[1][1])**2)))
        else:
            gl = Gl/Gl_MAX
        
        CS = (gs*np.conj(S[0][0]))/(1-(1-gs)*(np.abs(S[0][0])**2))
        RS = (np.sqrt(1-gs)*(1-(np.abs(S[0][0])**2)))/(1-(1-gs)*(np.abs(S[0][0])**2))
        CL = (gl*np.conj(S[1][1]))/(1-(1-gl)*(np.abs(S[1][1])**2))
        RL = (np.sqrt(1-gl)*(1-(np.abs(S[1][1])**2)))/(1-(1-gl)*(np.abs(S[1][1])**2))
        if polar == True:
            return rect_to_polar(CS),RS,rect_to_polar(CL),RL
        return CS, RS, CL, RL

def low_noise_amp(Z0, RN, Lopt, NF, NFmin,dB=True,polarinput=True):
    """
    Parameters
    ----------
    Z0 : float
        Characteristic impedance.
    RN : float
        Equivalent noise resistance of transistor.
    Lopt : complex (rectangular)
        OPtimum source reflection coefficient (Ls).
    NF : float
        Desired Noise Figure.
    NFmin : float
        Minimum Noise Figure of System.
    dB : boolean, optional
        True if NF and NFmin are in dB. The default is True.

    Returns
    -------
    N 
    
    CNF : complex
        Circle center.
    RNF : float
        Radius of constant NF circle.

    """
    if polarinput == True:
        Lopt = polar_to_rect(Lopt)
    if dB==True:
        NF = 10**(NF/10)
        NFmin = 10**(NFmin/10)
    N = (np.abs(1+Lopt)**2)*((NF-NFmin)/((4*RN)/(Z0)))
    CNF = Lopt/(N+1)
    RNF = np.sqrt(N*(N+1-(np.abs(Lopt)**2)))/(N+1)
    return N,CNF,RNF

def required_PN(C,S,I,BW):
    """

    Parametres
    ----------
    C : float
        Nivell desitjat de senyal (dBm).
    S : float
        Nivell desitjat de canal adjacent (dB).
    I : float
        Nivell no desitjat de senyal de soroll (dBm).
    BW : float
        Amplada de banda del filtre IF (Hz).

    Retorna
    -------
    L : float
        Maxim soroll de fase per aconseguir un aïllament de canal adjacent de S (paràmetre entrada) (dBc/Hz).

    """
    L = C -S -I -10*np.log10(BW)
    return L

def GC_LC(PIF,PRF):
    """
    Retorna Gc (Conversion Gain) i Lc (Conversion Loss)
    """
    return 10*np.log10(PIF/PRF),20*np.log10(10*np.log10(PIF/PRF))

