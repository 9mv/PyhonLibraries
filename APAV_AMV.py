"""
Llibreria amb totes les funcions i classes propies del curs.

Totes implementades per Aleix Martinez Vinent a excepció de FiltreFIR, implementada per Sisco Vallverdu.

Classes:
\t filtreFIR
\t filtreIIR
\t dft_N

Funcions:
\t lectura_Wav
\t escriptura_Wav
\t transformadaFourier
\t transformadaFourier_F1
\t zeros_pols
\t plantilla_modul
\t plantilla_guany
\t representa_FIR
\t fir_optim
\t fft
"""
import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as scs
import math
from struct import *

exps={}
N=100
pi = np.pi
dicc={8:'b', 16:'h', 32: 'i'}  
diccnp = {8:np.int8, 16:np.short, 32: np.int32}

def lectura_Wav(fitxer,norm=False):
    """
    Llegeix un fitxer WAV i la retorna juntament a la frequencia de mostratge.
    
    norm es un parametre boolea amb el que podem normalitzar els valors llegits entre 0 i 1.
    """
    f = open(fitxer, 'r+b')
    data = f.read()
    global head, bps, size                  # Creem variables globals per poder emprar posteriorment per la escriptura del fitxer modificat.
    head = data[0:44]                       # Guardem la capçalera
    f.close()                               # Tanquem el fitxer d'audio per no gastar memòria inutilment
    header = unpack('i', data [24:28])      # Obtenció de la frequencia de mostratge
    freq_mostratge = header[0]
    header = unpack('h', data[34:36])       # Obtenció dels bits per mostra
    bps = header[0]
    header = unpack('i', data[40:44])       # Obtenció de la mida del bloc de dades
    size = header[0]
    
    i = str(int(size/(bps/8))) + dicc[bps]  # Comptem el nombre de mostres que haurem de analitzar a la funcio unpack()

    senyal_audio=unpack_from(i, data, 44)         # Llegim les mostres a una variable tupla.
   
    if norm == True:
        return freq_mostratge,[(i/(2**(bps-1)))/2 + 0.5 for i in senyal_audio]    # Normalitzem per treballar el senyal    
    else:
        return freq_mostratge, np.array(senyal_audio,dtype=diccnp[bps]).T

def escriptura_Wav(fitxer, freq_mostratge, senyal_audio, norm=False):
    """
    Crea el fitxer wav amb el nom, frequencia de mostratge i senyal especificats.
    
    Si les dades del senyal estan normalitzades entre 0 i 1 hem d'especificar norm = True per desnormalitzar-los en el moment de codificar.
    """
    #Comprovem que tengui la extensió escrita. Si no l'hi afegim.
    if (fitxer[(len(fitxer)-4):len(fitxer)])!='.wav':
        fitxer = fitxer + '.wav'
    f = open(fitxer, 'wb')
    f.write(head[0:24])                     # Copiem la capçalera igual, a excepció de la frequencia de mostratge, especificada per parametre a la crida de la funcio i de la longitud del senyal, canviada per la interpolació o delmació.
    f.write(pack('i',freq_mostratge))
    f.write(head[28:40])
    f.write(pack('i',int(bps/8)*len(senyal_audio)))   #Aquí se'ns indica el número de Bytes de dades, no de mostres. Per tant, hem d'adaptar el numero de bytes segons les mostres que tinguem i els bytes que ocupin cada mostra.

    if norm==True:
        for j in senyal_audio:                        # Convertim cada mostra de senyal a tipus Bytes i la escrivim al fitxer especificat.
            dataout = pack(dicc[bps],int(((j*(2**(bps-1)))/2)-0.5))    # Desnormalitzem el senyal
            f.write(dataout)
    else:
        for j in senyal_audio:                        # Convertim cada mostra de senyal a tipus Bytes i la escrivim al fitxer especificat.
            dataout = pack(dicc[bps],j)    # Desnormalitzem el senyal
            f.write(dataout)

    f.close()
    return None

def transformadaFourier(x,Nf):
    """
    Funcio que realitza la DFT. Retorna la transformada i el nombre de punts d'aquesta.

    transformadaFourier(x,Nf)
    \t x: senyal a transformar.
    \t Nf: nombre de punts de la transformada.
    """
    N=2*(Nf-1)
    Lx = len(x)
    
    X=[]

    for k in range(Nf):
        exp = [np.exp(-2j*pi*k*n/N) for n in range(Lx)]
        X.append(x@exp)
    return X,N

def transformadaFourier_F1(x,F1):
    """
    Retorna el valor de la DFT d'un senyal x a la freqüència F1, on F1 és una frequencia discreta entre 0 i 0.5.
    """
    if F1 > 0.5:
        raise Exception("F1 ha de ser un valor real entre 0 i 0.5.")    # Retornem error si la frequencia a mirar no es admissible.
    exp = [np.exp(-2j*pi*F1*n) for n in range(len(x))]
    X = x@exp
    return X

class FiltreFIR:
    """
    Classe implementada per Sisco Vallverdú, maig 2020

    Filtre FIR d'ordre M=L-1
    nom_filtre=FiltreFIR(b,v)\n
    nom_filtre.reset() per inicialitzar l'estat intern
    y=nom_filtre(x)per filtrar x
    Implementació Directa
    """
    def __init__(self,b,v=[]):                                  # Creació del filtre
        self.b=b                                                # Coeficients
        self.L=len(b)
        self.v=v                                                # Estat intern
        if len(v)!=len(b):
            self.v=np.zeros(len(b)-1)

    def reset(self):                                            # Inicialització a 0 de l’estat intern
        self.v=np.zeros(self.L-1)

    def __call__(self,x):                                       # funció de filtratge
        if isinstance(x,(int,float,complex)):x=np.array([x])
        Lx=len(x)
        M=len(self.v)
        y=np.zeros(Lx)
        for n in range(Lx):
            y[n]=self.b[0]*x[n]+self.b[1:] @ self.v
            self.v[1:M]=self.v[0:M-1]                           #v[1:]=v[:-1]
            self.v[0]=x[n]
        return y
    def __repr__(self):
        return f"Filtre FIR({self.b}\n{self.v})"
    def __str__(self):
        return f"Filtre FIR d'ordre M={self.L-1}\nL={self.L} coeficients \n nom_filtre=FiltreFIR(b,v) \n nom_filtre.reset()per inicialitzar l'estat intern \n y=nom_filtre(x) per filtrar x"

class FiltreIIR:
    """
    Filtre IIR d'ordre M = L-1
    filtre = FiltreIIR(b,a,v) \n
    filtre.reset() per restablir l'estat intern \n
    y = filtre(x) per filtrar el senyal x    
    """
    def __init__(self,b,a,v=[]):
        self.a = a
        self.b = b
        if len(a) != len(b):
            raise Exception("A i B han de tenir la mateixa longitud.")
        self.L = len(a)
        self.v = v
        if len(v) != len(b): self.v = np.zeros(len(b)-1)
    
    def reset(self):
        self.v = np.zeros(self.L - 1)
        
    def __call__(self, x):
        if isinstance(x, (int,float,complex)): x = np.array([x])
        Lx = len(x)
        M = len(self.v)
        y = np.zeros(Lx)
        for n in range(Lx):
            y[n] = self.b[0] * x[n] + self.v[0]
            for i in range(1,M):
                self.v[i-1]=self.b[i]*x[n] - self.a[i]*y[n] + self.v[i]
            self.v[M-1] = self.b[self.L -1]*x[n] - self.a[self.L -1]*y[n]
        return y
    
    def __repr__(self):
        return f"FiltreIIR({self.b} \n {self.a} \n {self.v})"
    
    def __str__(self):
        return f"Filtre IIR d'ordre M = {self.L - 1} i L = {self.L} coeficients      nom_filtre = FiltreIIR(b,a,v)      nom_filtre.reset() per inicialitzar l'estat intern del filtre.      Per filtrar el senyal x:     y = nom_filtre(x)"

def zeros_pols(b, a=np.array([]),graf=False):
    """
    Retorna dos arrays de zeros i pols segons els vectors de coeficients del filtre.
    
    Us: \t  zeros, pols = zeros_pols(b,a,graf)
    
    graf es argument opcional: si valor es True, es representa grafic de les solucions.
    """
    zeros = np.roots(b)                                     # Resolem la equació per trobar les arrels i pols del coeficients.
    pols = np.roots(a)
    if graf==True:                                          # Si a la crida ho hem seleccionat, representem el grafic polar.
        plt.figure()
        plt.ion()
        if a!=np.array([]):
            plt.plot(np.real(pols),np.imag(pols),'xb')      # Els pols només es representen si és un filtre IIR i tenim coeficients A.
        plt.plot(np.real(zeros),np.imag(zeros),'og')
        cercle = np.exp(1j*2*np.pi*np.arange(360)/360)
        plt.plot(np.real(cercle),np.imag(cercle),':r')      # Dibuixem també el cercle imaginari unitat per comprovar si els pols/zeros hi cauen dins o no.
        plt.axis('square')
    return zeros,pols

def plantilla_modul(Fp,ap,Fa,aa,Gp=1):
    """
    Retorna una figura amb la plantilla del filtre a partir dels paràmetres desitjats.
    \t Fp: frequencia discreta limit de la banda de pas
    \t ap: arrissat en dB de la banda de pas
    \t Fa: frequencia discreta limit de la banda de rebuig
    \t aa: minima atenuacio en dB de la banda de rebuig respecte de la de pas 
    \t Gp: guany a la banda de pas [argument opcional]
    \n
    """
    fig=plt.figure()
    plt.title('Plantilla del mòdul del filtre')
    dp = (10**(ap/10)-1)/(10**(ap/20)+1)                    # Obtenció de dp i da a partir de ap i aa en dB.
    da = 10**(-aa/20)
    plt.plot([0, Fp],[Gp+dp,Gp+dp],'r')
    plt.plot([0, Fp, Fp], [Gp-dp,Gp-dp,0],'r')
    plt.plot([Fa, Fa, 0.5], [Gp, Gp*da, Gp*da],'r')
    plt.xlabel('Freqüència')
    plt.ylabel('Amplitud del mòdul del filtre')
    plt.xlim([0,0.5])
    plt.show()
    return fig

def plantilla_guany(Fp,ap,Fa,aa,Gp=0):
    """
    Retorna una figura amb la plantilla del guany del filtre a partir dels paràmetres desitjats.
    \t Fp: frequencia discreta limit de la banda de pas
    \t ap: arrissat en dB de la banda de pas
    \t Fa: frequencia discreta limit de la banda de rebuig
    \t aa: minima atenuacio en dB de la banda de rebuig respecte de la de pas
    \t Gp: guany a la banda de pas en dB [argument opcional]
    \n
    """
    fig=plt.figure()
    ap/=2
    plt.title('Plantilla del guany del filtre')
    plt.plot([0,Fp],[Gp+ap,Gp+ap],'b')
    plt.plot([0,Fp,Fp],[Gp-ap,Gp-ap,Gp-aa],'b')
    plt.plot([Fa,Fa,0.5],[0,Gp-aa,Gp-aa],'b')
    plt.xlabel('Freqüència')
    plt.ylabel('Guany del filtre [dB]')
    plt.xlim([0,0.5])
    plt.show()
    return fig

def representa_FIR(b,Fp=0,ap=0,Fa=0,aa=0):
    """
    Representa el mòdul i el guany del filtre FIR.
    Arguments:
    \t Coeficients b del filtre
    \t Arguments opcionals Fp, ap, Fa, aa. Si s'inclouen, es representa amb la plantilla.
    """
    dp = (10**(ap/10)-1)/(10**(ap/20)+1)
    da = 10**(-aa/20)
    H, Ns = transformadaFourier(b,N)
    Habs = np.abs(H)
    freq = np.linspace(0,0.5,len(Habs))
    fig,axs=plt.subplots(2,1,constrained_layout=True)
    axs[0].set_title('Mòdul del filtre')
    axs[0].plot(freq,Habs)
    if (Fp!=0 and dp!=0 and Fa!=0 and da!=0):
        axs[0].plot([0, Fp],[1+dp,1+dp],'r')
        axs[0].plot([0, Fp, Fp], [1-dp,1-dp,0],'r')
        axs[0].plot([Fa, Fa, 0.5], [1, da, da],'r')
    axs[0].set_xlabel('Freqüència')
    axs[0].set_ylabel('Amplitud del mòdul del filtre')
    axs[0].set_xlim([0,0.5])
    
    ap/=2
    axs[1].set_title('Plantilla del guany del filtre')
    axs[1].plot(freq,20*np.log10(Habs))
    if (Fp!=0 and dp!=0 and Fa!=0 and da!=0):
        axs[1].plot([0,Fp],[ap,ap],'b')
        axs[1].plot([0,Fp,Fp],[-ap,-ap,-aa],'b')
        axs[1].plot([Fa,Fa,0.5],[0,-aa,-aa],'b')
    axs[1].set_xlabel('Freqüència')
    axs[1].set_ylabel('Guany del filtre [dB]')
    axs[1].set_xlim([0,0.5])
    plt.show()
    return fig, axs

def compleix(b,Fp,dp,Fa,da):
    H, Ns = transformadaFourier(b,N)
    Habs = np.abs(H)
    # Comprovació de límits segons la resposta del filtre per cada segment de la resposta del filtre.
    if max(Habs[0:math.ceil(len(Habs)*2*Fp)+1]) > 1+dp:             # Podem trobar-nos amb un decimal. Com que els indexs han de ser enters, aproximem en aquest cas a la mostra posterior per no augmentar la diferència entre Fp i Fa.
        return False              
    if min(Habs[0:math.ceil(len(Habs)*2*Fp)+1]) < 1-dp:
        return False
    if max(Habs[math.floor(len(Habs)*2*Fa)+1:len(Habs)]) >= da:     # En aquest cas, aproximem a la mostra anterior per el mateix motiu.
        return False
    return True

def fir_optim(Fp,ap,Fa,aa,graf=False):
    """
    Donats una serie de paràmetres, retorna un array de coeficients b que formen el filtre amb les especificacions donades.
    \t Fp: frequencia discreta limit de la banda de pas
    \t ap: arrissat en dB de la banda de pas
    \t Fa: frequencia discreta limit de la banda de rebuig
    \t aa: minima atenuacio en dB de la banda de rebuig respecte de la de pas
    \t graf: representació del filtre creat. El seu valor pot ser True o False.[argument opcional]
    \n
    """
    dp = (10**(ap/10)-1)/(10**(ap/20)+1)
    da = 10**(-aa/20)
    L=2
    while True:
        b = scs.remez(L,[0,Fp,Fa,0.5],[1,0],[da,dp])        # Anem calculant conjunts de coeficients de ordre creixent fins a retornar.
        if compleix(b,Fp,dp,Fa,da):                         # Comprovem si aquells coeficients compleixen els requisits especificats per el filtre.
            if graf==True:
                representa_FIR(b,Fp,ap,Fa,aa)               # Si es compleix, retornem i el bucle acaba.
            return b
        L+=1

class dft_N:
    """
    Classe que guarda els arrays d'exponencials utilitzats per a la transformada de Fourier recursiva per una N potencia de 2.
    """
    def __init__(self, N):                                          # Creem l'objecte per crear les seves exponencals. Només es fa una vegada.
        """
        Li passem un nombre d'exponencials que tindra el vector. N és potencia de 2.
        """
        self.N = N
        self.exp = np.exp(-2j * np.pi * np.arange(self.N)/self.N)   # Calculem el vector de N exponencials.
        
    def __call__(self):                                             # Cridem l'objecte només per obtenir les seves exponencials. Es fa moltes vegades.
        """
        La crida de l'objecte simplement retorna el vector d'exponencials generat a la creacio de l'objecte.
        """
        return self.exp             

    def __repr__(self):
        return f"dft_{self.N}"
    
    def __str__(self):
        return f"Vector de {self.N} exponencials de la forma e^(-j2*pi*n) on n varia entre [0, 1)"

def fft(x):
    """
    Retorna la transformada de Fourier rapida del senyal x. El seu funcionament es basa en la crida recursiva de la funcio dins ella mateixa en "topologia d'arbre", dividint cada execució de la funció en dos fins que el senyal entrat per paràmetre a la crida de la funcio nomes estigui format per dues mostres, moment en el que es realitza la dft i es fa el recorregut invers cap a les branques superiors ponderant els resultats de les "branques filles" amb els vectors exponencials fins tenir un sol vector amb el mateix nombre de mostres que x.

    X = fft(x)
    \t x: senyal a transformar. Idealment el nombre de mostres es potencia de 2.
    \t X: senyal transformat. Te la longitud de la seguent potencia de 2 del nombre de mostres del senyal x.
    """
    Lx = len(x)
    pot = int(np.ceil(np.log2(Lx)))                                         # Com que el senyal probablement no sigui potència de 2, averiguem quin es el seguent exponent de 2.
    N = 2**pot                                                              # Calculem el nou nombre de mostres que ha de tenir el senyal.
    
    if N not in exps:
        exps[N]=dft_N(N)                                                    # Si no tenim l'objecte creat per aquella N, ho creem i guardem al diccionari format per els objectes de totes les N que tenim.
        
    if int(np.log2(Lx))!=np.log2(Lx):                                       # Comprovem que nombre de mostres de x sigui potencia de 2. Si no, emplenem amb zeros.
        x=np.concatenate([x,np.zeros(N-Lx)])
        
    if N <=2:
        X,M = transformadaFourier(x, N)                                     # Quan hem separat el senyal per parells i imparells fins haver arribat a 2 mostres, calculem la seva transformada de Fourier amb la meva funcio creada anteriorment.
        return X
    
    else:                                                                   # En cas de que no s'hagi arribat al cas anterior, tornem a separar el senyal en mostres parells i imparells i cridem la funcio, de manera que cridem a la funció des de dins la funcio reiteradament fins arribar a cridar-la amb una x de dues mostres. Despres comencen a retornar. 
        Xe = fft(x[::2])
        Xo = fft(x[1::2])
        exp = exps[2**pot]()                                                # Obtenim el vector d'exponencials corresponents al nombre de mostres amb el que estem executant la iteracio.
        X = np.concatenate([Xe + exp[:int(N/2)]*Xo, Xe+exp[int(N/2):]*Xo])  # Una vegada obtinguts els coeficients, juntem els resultats dels parells i imparells amb la ponderació dels exponencials i retornem el resultat a la funció "pare" de la branca.
    return X