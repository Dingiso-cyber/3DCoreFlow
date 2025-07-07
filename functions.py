import numpy as np
from math import factorial
from scipy.special import lpmn
from scipy.stats import linregress

import numpy as np
from math import factorial
from scipy.special import lpmn
from chaosmagpy import load_CHAOS_matfile

def recover(E, r, i, maxlm, classes): 
    
    maxlm = int(maxlm)
    
    if classes == 'es0' or classes == 'eas0':
        
        Er_r = E[:,maxlm*0:maxlm*1]
       
        Et_r  = E[:,maxlm*1:maxlm*2]
        
        Ef_r  = E[:,maxlm*2:maxlm*3]
        
        Er_i = E[:, maxlm*3:maxlm*4]
        
        Et_i = E[:,maxlm*4:maxlm*5]
        
        Ef_i = E[:,maxlm*5:maxlm*6]   
        
        ur = r*Er_r+i*Er_i

        ut = r*Et_r+i*Et_i
        
        uf = r*Ef_r+i*Ef_i

        ur = np.matmul(r,Er_r.T)+np.matmul(i,Er_i.T)
        
        ut = np.matmul(r,Et_r.T)+np.matmul(i,Et_i.T)
        
        uf = np.matmul(r,Ef_r.T)+np.matmul(i,Ef_i.T)
    
        return ur, ut, uf
    
    elif classes=='es' or classes == 'eas':
        
        Er_r   = E[:,maxlm*0:maxlm*1]
        
        Et_r   = E[:,maxlm*1:maxlm*2]
        
        Ef_r   = E[:,maxlm*2:maxlm*3]
        
        Er_i  =  E[:, maxlm*3:maxlm*4]
        
        Et_i = E[:,maxlm*4:maxlm*5]
        
        Ef_i = E[:,maxlm*5:maxlm*6] 
        
        ur = np.matmul(r,Er_r.T)+np.matmul(i,Er_i.T)
        
        ut = np.matmul(r,Et_r.T)+np.matmul(i,Et_i.T)
        
        uf = np.matmul(r,Ef_r.T)+np.matmul(i,Ef_i.T)
        
        return ur, ut, uf

    else:
        
        
        Ef   = E[:,maxlm*0:maxlm*1]
     
        uf = np.matmul(r,Ef.T)
        
        return uf
    
def lpmn_derivate2(maxl,x):

    p,dp = lpmn(maxl,maxl,x)
    
    _,dp1 = lpmn(maxl+1,maxl+1,x)
    
    dp1 = dp1[0:-1,1::]
    
    l = np.tile(np.linspace(0,maxl,maxl+1),[maxl+1,1])
    
    m = np.tile(np.linspace(0,maxl,maxl+1),[maxl+1,1]).T
    
    dp1[m>l]=0
    
    l[m>l]=0
    
    m[m>l]=0
    
    dp2 = (-(l+1)*(p+x*dp)+(l-m+1)*dp1-2*x*dp)/(x**2-1)
    
    return p, dp, dp2


def lpmn_derivate3(maxl,x):
    
    p, dp, dp2 = lpmn_derivate2(maxl,x)
    
    _, _, dp22 = lpmn_derivate2(maxl+1,x)#l+1
    
    dp22 = dp22[0:-1,1::]
    
    l = np.tile(np.linspace(0,maxl,maxl+1),[maxl+1,1])
    
    m = np.tile(np.linspace(0,maxl,maxl+1),[maxl+1,1]).T
    
    dp22[m>l]=0
    
    l[m>l]=0
    
    m[m>l]=0
    
    dp3 = (-(l+1)*(dp*2+x*dp2)+(l-m+1)*dp22-2*dp-4*x*dp2)/(x**2-1)
    
    return p, dp, dp2, dp3

def NNnm(max_n,max_m):
    N = np.zeros([max_m+1,max_n+1])
    for m in range(0,max_m+1,1):
        for n in range(m,max_n+1,1):
            if m==0:
                N[m,n] = 1
            else:
                N[m,n]= 2*factorial(n-m)/factorial(n+m)
    return N**0.5

def gen_E(t,f,max_m,truncation_m,output_list):

    N = NNnm(max_m,max_m)
    m = np.linspace(0,max_m,max_m+1)
    m = np.tile(m,[max_m+1,1]).T
    n = np.linspace(0,max_m,max_m+1)
    n = np.tile(n,[max_m+1,1])
    index = np.triu_indices(max_m+1, k=0)
    m = m[index]
    n = n[index]
    N = N[index]
    indexx = np.where(n>truncation_m)
    n = n[indexx]
    m = m[indexx]
    N = N[indexx]
    E_br_a = []
    E_br_b = []
    
    E_bt_a = []
    E_bt_b = []
    
    E_bf_a = []
    E_bf_b = []
    
    E_br_r_a = []
    E_br_r_b = []
    
    E_br_t_a = []
    E_br_t_b = []
    
    E_br_f_a = []
    E_br_f_b = []

    E_bt_t_a = []
    E_bt_t_b = []
    
    E_bt_f_a = []
    E_bt_f_b = []

    E_bf_t_a = []
    E_bf_t_b = []
    
    E_bf_f_a = []
    E_bf_f_b = []  
    
    a = 6371.2
    
    ro = a*0.5457
    
    for ti ,fi in zip(t,f):
        
        p, dp1, dp2 = lpmn_derivate2(max_m, np.cos(ti))
        
        dp2 = dp2*(np.sin(ti))**2+(-np.cos(ti))*dp1

        dp = dp1*(-np.sin(ti))
        
        p = p[index]
        
        dp = dp[index]
        
        dp2 = dp2[index]
        
        p = p[indexx]
        
        dp = dp[indexx]
        
        dp2 = dp2[indexx]
        

        
        E_br_a.append((-1)**m*N*p*np.cos(m*fi)*(a/ro)**(n+2)*(-n-1))
        E_br_b.append((-1)**m*N*p*np.sin(m*fi)*(a/ro)**(n+2)*(-n-1))
        
        E_bt_a.append((-1)**m*N*dp*np.cos(m*fi)*a**(n+2)*ro**(-n-2))
        E_bt_b.append((-1)**m*N*dp*np.sin(m*fi)*a**(n+2)*ro**(-n-2))
        
        E_bf_a.append((-1)**m*(-m)*N*p*np.sin(m*fi)*a**(n+2)*ro**(-n-2)/np.sin(ti))
        E_bf_b.append((-1)**m*(m)*N*p*np.cos(m*fi)*a**(n+2)*ro**(-n-2)/np.sin(ti))
        
        E_br_r_a.append((-1)**m*N*p*np.cos(m*fi)*a**(n+2)*ro**(-n-3)*(-n-1)*(-n-2))
        E_br_r_b.append((-1)**m*N*p*np.sin(m*fi)*a**(n+2)*ro**(-n-3)*(-n-1)*(-n-2))    
        
        E_br_t_a.append((-1)**m*N*dp*np.cos(m*fi)*(a/ro)**(n+2)*(-n-1))
        E_br_t_b.append((-1)**m*N*dp*np.sin(m*fi)*(a/ro)**(n+2)*(-n-1))
        
        E_br_f_a.append((-1)**m*-m*N*p*np.sin(m*fi)*(a/ro)**(n+2)*(-n-1))
        E_br_f_b.append((-1)**m*m*N*p*np.cos(m*fi)*(a/ro)**(n+2)*(-n-1))
        
        E_bt_t_a.append((-1)**m*N*dp2*np.cos(m*fi)*a**(n+2)*ro**(-n-2))
        E_bt_t_b.append((-1)**m*N*dp2*np.sin(m*fi)*a**(n+2)*ro**(-n-2))        
        
        E_bt_f_a.append((-1)**m*(-m)*N*dp*np.sin(m*fi)*a**(n+2)*ro**(-n-2))
        E_bt_f_b.append((-1)**m*(m)*N*dp*np.cos(m*fi)*a**(n+2)*ro**(-n-2))        
        
        E_bf_t_a.append((-1)**m*(-m)*N*dp*np.sin(m*fi)*a**(n+2)*ro**(-n-2)/np.sin(ti)-(-1)**m*(-m)*N*dp*np.sin(m*fi)*a**(n+2)*ro**(-n-2)/np.sin(ti)**2*np.cos(ti))
        E_bf_t_b.append((-1)**m*(m)*N*dp*np.cos(m*fi)*a**(n+2)*ro**(-n-2)/np.sin(ti)-(-1)**m*(-m)*N*dp*np.sin(m*fi)*a**(n+2)*ro**(-n-2)/np.sin(ti)**2*np.cos(ti))       
        
        E_bf_f_a.append((-1)**m*(-m*m)*N*p*np.cos(m*fi)*a**(n+2)*ro**(-n-2)/np.sin(ti))
        E_bf_f_b.append((-1)**m*(-m*m)*N*p*np.sin(m*fi)*a**(n+2)*ro**(-n-2)/np.sin(ti))     
        
        
    E_br_a = np.array(E_br_a)
    E_br_b = np.array(E_br_b)
    
    E_bt_a = np.array(E_bt_a)
    E_bt_b = np.array(E_bt_b)
    
    E_bf_a = np.array(E_bf_a)
    E_bf_b = np.array(E_bf_b)
    
    E_br_r_a = np.array(E_br_r_a)
    E_br_r_b = np.array(E_br_r_b)
    
    E_br_t_a = np.array(E_br_t_a)
    E_br_t_b = np.array(E_br_t_b)
    
    E_br_f_a = np.array(E_br_f_a)
    E_br_f_b = np.array(E_br_f_b)

    E_bt_t_a = np.array(E_bt_t_a)
    E_bt_t_b = np.array(E_bt_t_b)
    
    E_bt_f_a = np.array(E_bt_f_a)
    E_bt_f_b = np.array(E_bt_f_b)

    E_bf_t_a = np.array(E_bf_t_a)
    E_bf_t_b = np.array(E_bf_t_b)
    
    E_bf_f_a = np.array(E_bf_f_a)
    E_bf_f_b = np.array(E_bf_f_b)    
    
    if output_list:
        
       return np.c_[E_br_a,E_br_b,E_bt_a,E_bt_b,E_bf_a,E_bf_b,E_br_r_a,E_br_r_b,\
                    E_br_t_a,E_br_t_b,E_br_f_a,E_br_f_b,E_bt_t_a,E_bt_t_b,E_bt_f_a,E_bt_f_b,E_bf_t_a,E_bf_t_b,E_bf_f_a,E_bf_f_b]
   
    else:
        
       return E_br_a,E_br_b,E_bt_a,E_bt_b,E_bf_a,E_bf_b,E_br_r_a,E_br_r_b,\
                    E_br_t_a,E_br_t_b,E_br_f_a,E_br_f_b,E_bt_t_a,E_bt_t_b,E_bt_f_a,E_bt_f_b,E_bf_t_a,E_bf_t_b,E_bf_f_a,E_bf_f_b

def GenG(t,f,max_m,truncation_m,output_list):

    N = NNnm(max_m,max_m)
    m = np.linspace(0,max_m,max_m+1)
    m = np.tile(m,[max_m+1,1]).T
    n = np.linspace(0,max_m,max_m+1)
    n = np.tile(n,[max_m+1,1])
    index = np.triu_indices(max_m+1, k=0)
    m = m[index]
    n = n[index]
    N = N[index]
    indexx = np.where(n>truncation_m)
    n = n[indexx]
    m = m[indexx]
    N = N[indexx]
    Ea = []
    Eb = []
    Ea_t = []
    Eb_t = []
    Ea_f = []
    Eb_f = []
    
    a = 6371.2*10**3
    ro = a*0.53
    for ti ,fi in zip(t,f):
        p, dp = lpmn(max_m,max_m,np.cos(ti))
        p = p[index]
        dp = dp[index]
        p = p[indexx]
        dp = dp[indexx]        

        Ea.append((-1)**m*N*p*np.cos(m*fi)*(a/ro)**(n+2)*(-n-1))
        Eb.append((-1)**m*N*p*np.sin(m*fi)*(a/ro)**(n+2)*(-n-1))
        Ea_t.append((-1)**m*-N*dp*np.cos(m*fi)*np.sin(ti)*(a/ro)**(n+2)*(-n-1))
        Eb_t.append((-1)**m*-N*dp*np.sin(m*fi)*np.sin(ti)*(a/ro)**(n+2)*(-n-1))
        Ea_f.append((-1)**m*-m*N*p*np.sin(m*fi)*(a/ro)**(n+2)*(-n-1))
        Eb_f.append((-1)**m*m*N*p*np.cos(m*fi)*(a/ro)**(n+2)*(-n-1))
        
    Ea = np.array(Ea)
    Eb = np.array(Eb)
    Ea_t = np.array(Ea_t)
    Eb_t = np.array(Eb_t)
    Ea_f = np.array(Ea_f)
    Eb_f = np.array(Eb_f)
    if output_list:
       return np.c_[Ea,Eb,Ea_t,Eb_t,Ea_f,Eb_f]
    else:
       return Ea,Eb,Ea_t,Eb_t,Ea_f,Eb_f

def modify_brs(x,C,maxlm): 
    
    Ea = x[:,8:8+maxlm]
    Eb = x[:,8+maxlm*1:8+maxlm*2]
    Eat = x[:,8+maxlm*2:8+maxlm*3]
    Ebt = x[:,8+maxlm*3:8+maxlm*4]
    Eaf = x[:,8+maxlm*4:8+maxlm*5]
    Ebf = x[:,8+maxlm*5:8+maxlm*6]
    
    br_lm_r = C[maxlm*0:maxlm*1]
    br_lm_i = C[maxlm*1:maxlm*2]

    br = br_lm_r*Ea+br_lm_i*Eb

    dbr_t = br_lm_r*Eat+br_lm_i*Ebt

    dbr_f = br_lm_r*Eaf+br_lm_i*Ebf

    br = -np.sum(br,1)

    dbr_t = -np.sum(dbr_t,1)
  
    dbr_f = -np.sum(dbr_f,1)
   
    br = np.reshape(br,[-1,1])
   
    dbr_t = np.reshape(dbr_t,[-1,1])
 
    dbr_f = np.reshape(dbr_f,[-1,1])
    
    return br,dbr_t,dbr_f
def GenGrid(N,homo=False):
    
    if homo:
        t = np.linspace(0.01, np.pi-0.01, 40)
        f = np.linspace(-np.pi, np.pi, 80)
        t,f = np.meshgrid(t,f)
        t = np.reshape(t,[-1])
        f = np.reshape(f,[-1])  
        return t,f
        
    else:
        phi = 0.618
        x=[]
        y=[]
        z=[]
        for n in range(1,N+1,1):
            zi = (2*n-1)/N-1
            xi = np.sqrt(1-zi*zi)*np.cos(2*np.pi*n*phi)
            yi = np.sqrt(1-zi*zi)*np.sin(2*np.pi*n*phi)
            z.append(zi)
            x.append(xi)
            y.append(yi)
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        ttheta = np.arccos(z/(x**2+y**2+z**2)**0.5)
        pphi = np.arctan2(y,x)
        return ttheta,pphi


def predict_y(x,slope,intercept):
    return slope * x + intercept


def MF_Space_deriv(i, t, f, deriv=0,nmax=13):
    
    nmax=nmax; max_m=nmax; max_n=nmax
    
    if deriv==0:
        
        gh_MF = np.load('MGFMDATA/data_{}.npy'.format(i))[0:int((nmax+1)*(nmax+2)/2-1),2:4]  
        
    if deriv==1:
        
        gh_MF = np.load('MGFMDATA/data_{}.npy'.format(i))[0:int((nmax+1)*(nmax+2)/2-1),4:6] 
    
    Wm_MF = np.zeros([nmax+1,nmax+1],dtype=np.complex128)
    
    i=0
    
    for l in range(1,nmax+1,1):
        
        for m in range(0,l+1,1):
            
            Wm_MF[m,l] = gh_MF[i,0]+1j*gh_MF[i,1]
            
            i+=1
            
    N = NNnm(max_n,max_m)
    
    m = np.linspace(0,max_m,max_m+1)
    
    m = np.tile(m,[max_n+1,1]).T
    
    n = np.linspace(0,max_n,max_n+1)
    
    n = np.tile(n,[max_m+1,1])
    
    index = np.triu_indices(max_n+1, k=0)#取上三角矩阵
    
    m = m[index]
    
    n = n[index]
    
    N = N[index]

    Br=[]
    
    Br_r=[]
    Br_t=[]
    Br_f=[]
    
    Bt=[]
    Bt_t=[]
    Bt_f=[]
    
    Bf=[]
    Bf_t=[]
    Bf_f=[]
    
 

    for ti,fi in zip(t,f):

        p, dp1, dp2 = lpmn_derivate2(max_m, np.cos(ti))
        
        dp2 = dp2*(np.sin(ti))**2+(-np.cos(ti))*dp1

        dp = dp1*(-np.sin(ti))
        
        p = p[index]
        
        dp = dp[index]
        
        dp2 = dp2[index]
        
        g = Wm_MF.real[index]
        
        h = Wm_MF.imag[index]
        
        a = 6371.2
        
        ro = a*0.547

        br =    (-1)**m*N*(g*np.cos(m*fi)+h*np.sin(m*fi))*p*a**(n+2)*ro**(-n-2)*(-n-1)
        dbr_r = (-1)**m*N*(g*np.cos(m*fi)+h*np.sin(m*fi))*p*a**(n+2)*ro**(-n-3)*(-n-1)*(-n-2)
        dbr_t = (-1)**m*N*(g*np.cos(m*fi)+h*np.sin(m*fi))*dp*a**(n+2)*ro**(-n-2)*(-n-1)
        dbr_f = (-1)**m*N*(g*(-m)*np.sin(m*fi)+m*h*np.cos(m*fi))*p*a**(n+2)*ro**(-n-2)*(-n-1)
        
        dbr_f2 = (-1)**m*N*(g*(-m)*m*np.cos(m*fi)+(-m)*m*h*np.sin(m*fi))*p*a**(n+2)*ro**(-n-2)*(-n-1)
        dbr_t2 = (-1)**m*N*(g*np.cos(m*fi)+h*np.sin(m*fi))*dp2*a**(n+2)*ro**(-n-2)*(-n-1)
        dbr_r2 = (-1)**m*N*(g*np.cos(m*fi)+h*np.sin(m*fi))*p*a**(n+2)*ro**(-n-4)*(-n-1)*(-n-2)*(-n-3)

        
        bt =    (-1)**m*N*(g*np.cos(m*fi)+h*np.sin(m*fi))*dp*a**(n+2)*ro**(-n-2)
        dbt_t = (-1)**m*N*(g*np.cos(m*fi)+h*np.sin(m*fi))*dp2*a**(n+2)*ro**(-n-2)
        dbt_f = (-1)**m*N*(g*(-m)*np.sin(m*fi)+m*h*np.cos(m*fi))*dp*a**(n+2)*ro**(-n-2)
        
        bf = (-1)**m*N*(g*(-m)*np.sin(m*fi)+m*h*np.cos(m*fi))*p*a**(n+2)*ro**(-n-2)/np.sin(ti)
        dbf_t = (-1)**m*N*(g*(-m)*np.sin(m*fi)+m*h*np.cos(m*fi))*dp*a**(n+2)*ro**(-n-2)/np.sin(ti)+\
            (-1)**m*N*(g*(-m)*np.sin(m*fi)+m*h*np.cos(m*fi))*p*a**(n+2)*ro**(-n-2)/np.sin(ti)**2*-np.cos(ti)
        dbf_f = (-1)**m*N*(g*(-m*m)*np.cos(m*fi)-m*m*h*np.sin(m*fi))*p*a**(n+2)*ro**(-n-2)/np.sin(ti)

        br = np.sum(br)
        dbr_r = np.sum(dbr_r)
        dbr_t = np.sum(dbr_t)
        dbr_f = np.sum(dbr_f)
        dbr_r2 = np.sum(dbr_r2)
        dbr_t2 = np.sum(dbr_t2)
        dbr_f2 = np.sum(dbr_f2)
        
        bt = np.sum(bt)
        dbt_t = np.sum(dbt_t)
        dbt_f = np.sum(dbt_f)
        
        bf = np.sum(bf)
        dbf_t = np.sum(dbf_t)
        dbf_f = np.sum(dbf_f)        
        

        Br.append(br)
        Br_r.append(dbr_r)
        Br_t.append(dbr_t)
        Br_f.append(dbr_f)
        
        Bt.append(bt)
        Bt_t.append(dbt_t)
        Bt_f.append(dbt_f)
        
        Bf.append(bf)
        Bf_t.append(dbf_t)
        Bf_f.append(dbf_f)
   
    Br = -np.array(Br)
    Br_r = -np.array(Br_r)
    Br_t = -np.array(Br_t)
    Br_f = -np.array(Br_f)

    Bt = -np.array(Bt)
    Bt_t = -np.array(Bt_t)
    Bt_f = -np.array(Bt_f)
    
    Bf = -np.array(Bf)
    Bf_t = -np.array(Bf_t)
    Bf_f = -np.array(Bf_f)

    spectrum = np.zeros(nmax+1)
    
    for i in range(len(n)):
        
        l = int(n[i])
        
        spectrum[l] += (1)**(2*l+4)*(g[i]**2+h[i]**2)*(l+1)
    # print(spectrum)
    slope, intercept, r_value, p_value, std_err = linregress(np.linspace(1,nmax,nmax), np.log10(spectrum[1::]))
    
    new_x_values = np.linspace(nmax+1,30,30-nmax)
    
    for i in range(len(n)):
        
        l = int(n[i])
        
        spectrum[l] += (1.83)**(2*l+4)*(g[i]**2+h[i]**2)*(l+1)
        
    spectrum_small = 10**predict_y(new_x_values,slope,intercept)*1.83**(2*np.linspace(nmax+1,30,30-nmax)+4)
    
    return Br, Bt, Bf, Br_r, Br_t, Br_f, Bt_t, Bt_f, Bf_t, Bf_f, spectrum_small