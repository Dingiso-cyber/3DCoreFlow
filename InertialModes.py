#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 09:45:12 2022

@author: cyber-dingiso

see Kloss 2019

"""
import numpy as np
from scipy.special import factorial as f1
from scipy.special import factorial2 as f2
import sympy as sp
import matplotlib.pyplot as plt
import mpmath

def GenGrid(N,homo=False):
    
    if homo:
        t = np.linspace(0.01, np.pi-0.01, 40)
        f = np.linspace(-np.pi+0.01, np.pi-0.01, 80)
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
    
    
mpmath.mp.dps=80

def Cs(m,k,i,j):
    cs = (-1)**(i+j)*f2(2*(k+m+i+j)-1)\
    /2**(j+1)/f2(2*i-1)/f1(k-i-j)/f1(i)/f1(j)/f1(m+j)
    return cs

def Ca(m,k,i,j):
    ca = (-1)**(i+j)*f2(2*(k+m+i+j)+1)\
    /2**(j+1)/f2(2*i+1)/f1(k-i-j)/f1(i)/f1(j)/f1(m+j)
    return ca

def Geo_Norm(k):
    return (4*np.pi*f2(2*k+1)*f2(2*k-1)/(4*k+1)/f2(2*k)/f2(2*k-2))**0.5    

def Geostrophic_mode(max_k, theta, r=1):
    p = r*np.sin(theta)
    q = r*np.cos(theta)
    g1 = 3*p/2
    g3 = 15/16*p*(7*p**2-4)
    
    h1 = 3*q/2
    h3 = 15/16*(q*(7*p**2-4)+p*14*p*q)
    
    b1 = 3/2*np.sin(theta)
    b3 = 105/16*3*r**2*np.sin(theta)**3-15/4*np.sin(theta)
    
    if max_k==1:
        
        return np.c_[(g1/Geo_Norm(1)).T,(h1/Geo_Norm(1)).T, (b1/Geo_Norm(1)).T]
    
    if max_k==2:
        
        return np.c_[(g1/Geo_Norm(1)).T, (g3/Geo_Norm(2)).T,(h1/Geo_Norm(1)).T, (h3/Geo_Norm(2)).T, (b1/Geo_Norm(1)).T, (b3/Geo_Norm(2)).T]
    
    G = [g1/Geo_Norm(1),g3/Geo_Norm(2)]
    
    H = [h1/Geo_Norm(1),h3/Geo_Norm(2)]
    
    B = [b1/Geo_Norm(1),b3/Geo_Norm(2)]
    
    for k in range(3,max_k+1,1):
        
        g = (4*k+3)*(4*k+1)/(4*k)/(k+1)*(p**2-4*k*(2*k+1)/(4*k-1)/(4*k+3))*g3\
            -(4*k+3)*(2*k+1)*(2*k-1)/(4*k)/(k+1)/(4*k-1)*g1
            
        h = (4*k+3)*(4*k+1)/(4*k)/(k+1)*(p**2-4*k*(2*k+1)/(4*k-1)/(4*k+3))*h3+g3*2*(4*k+3)*(4*k+1)/(4*k)/(k+1)*p*q\
            -(4*k+3)*(2*k+1)*(2*k-1)/(4*k)/(k+1)/(4*k-1)*h1
            
        b = (4*k+3)*(4*k+1)/(4*k)/(k+1)*(2*r*np.sin(theta)**2*g3+b3*(p**2-4*k*(2*k+1)/(4*k-1)/(4*k+3)))-(4*k+3)*(2*k+1)*(2*k-1)/(4*k)/(k+1)/(4*k-1)*b1
        
        g1 = g3
        h1 = h3
        b1 = b3
        
        g3 = g
        h3 = h
        b3 = b
        
        G.append(g/Geo_Norm(k))
        H.append(h/Geo_Norm(k))
        B.append(b/Geo_Norm(k))

    return np.c_[np.array(G).T,np.array(H).T,np.array(B).T] # H,B is the derivative in theta and r direction, respectively

def Es0_mode_eigenvalue(k_max):
    
    eigenvaluelist = np.array([])
    
    for k in range(2, k_max+1,1):
        
        x = sp.symbols('x')
        
        for j in range(0,k):
            
            C = (-1)**j*f1(2*(2*k-j))/f1(j)/f1(2*(k-j)-1)/f1(2*k-j)
            Power = (k-j)
            if j==0:
                equation = C*x**Power
            else:
                equation += C*x**Power
                
        s = sp.solveset(sp.Eq(equation,0),x)
        s = np.array([float(element) for element in s]) 
        s = s[s>0]**0.5 
        eigenvaluelist = np.append(eigenvaluelist,s)
        
    k_list=np.array([])
    n_list = np.array([])
    for k in range(k_max):
        
        l = np.array(k*[k+1])
        p = np.arange(1,1+k,1)
        n_list = np.append(n_list,p)
        k_list = np.append(k_list,l)  
        
    return np.c_[np.zeros_like(n_list),k_list,n_list,eigenvaluelist]

def Es_mode_eigenvalue(m_max, k_max):
    
    eigenvaluelist = np.array([])
    for m in range(1, m_max+1, 1):
        
        for k in range(1, k_max+1,1):
            
            x = sp.symbols('x')
            
            for j in range(0,k):
                
                C1 = (-1)**(j+k)*f1(2*(2*k+m-j))/f1(j)/f1(2*(k-j))/f1(2*k+m-j)
                Power = 2*(k-j)-1
                C2 = m*f1(2*(k+m))/f1(k)/f1(k+m)
                if j==0:
                    equation = C1*((2*k+m-2*j)*x-2*(k-j))*x**Power+C2
                else:
                    equation += C1*((2*k+m-2*j)*x-2*(k-j))*x**Power
                    
            s = sp.solveset(sp.Eq(equation,0),x)
            s = np.array([float(element) for element in s]) 
            s = s[np.argsort(np.abs(s))]
            eigenvaluelist = np.append(eigenvaluelist,s)
        
    k_list=np.array([])
    n_list = np.array([])
    m_list = np.array([])
    for m in range(1, m_max+1,1):
        for k in range(k_max+1):
            l = np.array(2*k*[k])
            p = np.arange(1,1+k*2,1)
            n_list = np.append(n_list,p)
            k_list = np.append(k_list,l)  
            m_list = np.append(m_list,np.zeros_like(l)+m)
            
    return np.c_[m_list,k_list,n_list,eigenvaluelist]  
               
def Eas0_mode_eigenvalue(k_max):
    
    eigenvaluelist = np.array([])
    
    for k in range(1, k_max+1,1):
        
        x = sp.symbols('x')
        
        for j in range(0,k+1):
            
            C = (-1)**j*f1(2*(2*k-j+1))/f1(j)/f1(2*(k-j))/f1(2*k-j+1)
            Power = (k-j)
            if j==0:
                equation = C*x**Power
            else:
                equation += C*x**Power
                
        s = sp.solveset(sp.Eq(equation,0),x)
        s = np.array([float(element) for element in s]) 
        s = s[s>0]**0.5 
        eigenvaluelist = np.append(eigenvaluelist,s)
        
    k_list=np.array([])
    n_list = np.array([])
    for k in range(1, k_max+1, 1):
        
        l = np.array(k*[k])
        p = np.arange(1, 1+k, 1)
        n_list = np.append(n_list,p)
        k_list = np.append(k_list,l)  
        
    return np.c_[np.zeros_like(n_list),k_list,n_list,eigenvaluelist]  

def Eas_mode_eigenvalue(m_max, k_max):
    
    eigenvaluelist = np.array([])
    for m in range(1, m_max+1, 1):
        
        for k in range(0, k_max+1,1):
            
            x = sp.symbols('x')
            
            for j in range(0,k+1):
                
                C = (-1)**(j)*f1(2*(2*k+m-j+1))/f1(j)/f1(2*(k-j)+1)/f1(2*k+m-j+1)
                Power = 2*(k-j)
                if j==0:
                    equation = C*((2*k-2*j+m+1)*x-(2*k-2*j+1))*x**Power
                else:
                    equation += C*((2*k-2*j+m+1)*x-(2*k-2*j+1))*x**Power
                    
            s = sp.solveset(sp.Eq(equation,0),x)
            s = np.array([float(element) for element in s]) 
            s = s[np.argsort(np.abs(s))]
            eigenvaluelist = np.append(eigenvaluelist,s)
        
    k_list=np.array([])
    n_list = np.array([])
    m_list = np.array([])
    for m in range(1, m_max+1,1):
        for k in range(k_max+1):
            l = np.array((2*(k+1)-1)*[k])
            p = np.arange(1,2+k*2,1)
            n_list = np.append(n_list,p)
            k_list = np.append(k_list,l)  
            m_list = np.append(m_list,np.zeros_like(l)+m)

    return np.c_[m_list,k_list,n_list,eigenvaluelist]  

def col(x,l):
    
    return np.tile(np.reshape(x,[-1,1]),[1,l])
    
def row(x,l):
    
    return np.tile(np.reshape(x,[1,-1]),[l,1])    
    
def Es_Mode(listt, t, f, r=1):
    
    lt = len(t)
    ll = len(listt)
    
    mode_ur = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_ut = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_uf = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_ur_r = np.zeros(shape=[lt,ll],dtype=np.complex128)
    # mode_ur_t = np.zeros(shape=[lt,ll],dtype=np.complex128)
    # mode_ur_f = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_ut_r = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_uf_r = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_ut_t = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_ut_f = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_uf_t = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_uf_f = np.zeros(shape=[lt,ll],dtype=np.complex128)
    index = 0
    
    for m, k, n, x, norm in listt:
        
        m,k,n,x = int(m),int(k),int(n),x

        i=[];j=[]
        for xi in range(k+1):
            for xj in range(k-xi+1):
                i.append(xi);j.append(xj)
        i = np.array(i)
        j = np.array(j)
        
        li = len(i)
        tt = col(t,li)
        ff = col(f,li)
        
        cs = Cs(m,k,i,j)
        
        cc = cs*x**(2*i-1)*(1-x**2)**(j-1)*r**(m+2*(i+j)-1.0)
        cc = row(cc,lt)
        
        cc_r = (m+2*(i+j)-1.0)*cs*x**(2*i-1)*(1-x**2)**(j-1)*r**(m+2*(i+j)-2.0)
        cc_r = row(cc_r,lt)  
        
        ur = cc*np.sin(tt)**row(m+2*j,lt)*np.cos(tt)**row(2*i,lt)*np.exp(1j*m*ff)*\
            row(x*(m+m*x+2*j*x)-2*i*(1-x**2),lt)

        ut = cc*np.sin(tt)**row(m+2*j-1,lt)*np.cos(tt)**row(2*i-1,lt)*np.exp(1j*m*ff)*(\
            row(x*(m+m*x+2*j*x),lt)*np.cos(tt)**2+row(2*i*(1-x**2),lt)*np.sin(tt)**2)       
        
        uf = cc*x*np.sin(tt)**row(m+2*j-1,lt)*np.cos(tt)**row(2*i,lt)*np.exp(1j*m*ff)*\
            row(m+m*x+2*j,lt)
        
        c1 = row(m+2*j-1,lt)
        c2 = row(2*i-1,lt)
        c3 = cc*np.exp(1j*m*ff)
        c4 = row(x*(m+m*x+2*j*x),lt)
        c5 = row(2*i*(1-x**2),lt)
        
        
        # ur_t = cc*np.exp(1j*m*ff)*row(x*(m+m*x+2*j*x)-2*i*(1-x**2),lt)*\
        #     (row(m+2*j,lt)*np.sin(tt)**row(m+2*j-1.0,lt)*np.cos(tt)*np.cos(tt)**row(2*i,lt)+ np.sin(tt)**row(m+2*j,lt)*row(2*i,lt)*np.cos(tt)**row(2*i-1.0,lt)*(-np.sin(tt)))

        # ur_f = 1j*m*ur
        
        ut_t = c3*c4*(c1*np.sin(tt)**(c1-1)*np.cos(tt)**(c2+3)-(c2+2)*np.cos(tt)**(c2+1)*np.sin(tt)**(c1+1))+\
            c3*c5*((c1+2)*np.sin(tt)**(c1+1)*np.cos(tt)**(c2+1)-(c2)*np.cos(tt)**(c2-1)*np.sin(tt)**(c1+3))
                
        ut_f = 1j*m*ut

        uf_t = cc*x*np.exp(1j*m*ff)*row(m+m*x+2*j,lt)*np.cos(tt)**(row(-1+2*i,lt))*\
                np.sin(tt)**(row(-2+2*j+m,lt))*(row(-1+2*j+m,lt)*np.cos(tt)**2-row(2*i,lt)*np.sin(tt)**2)
        uf_f = 1j*m*uf
        
        ur_r = cc_r*np.sin(tt)**row(m+2*j,lt)*np.cos(tt)**row(2*i,lt)*np.exp(1j*m*ff)*\
            row(x*(m+m*x+2*j*x)-2*i*(1-x**2),lt)
            
        ut_r = cc_r*np.sin(tt)**row(m+2*j-1,lt)*np.cos(tt)**row(2*i-1,lt)*np.exp(1j*m*ff)*(\
            row(x*(m+m*x+2*j*x),lt)*np.cos(tt)**2+row(2*i*(1-x**2),lt)*np.sin(tt)**2)  
            
        uf_r = cc_r*x*np.sin(tt)**row(m+2*j-1,lt)*np.cos(tt)**row(2*i,lt)*np.exp(1j*m*ff)*\
            row(m+m*x+2*j,lt)            
       
        mode_ur[:,index] = np.sum(ur,1)/norm
        mode_ut[:,index] = np.sum(ut,1)/norm
        mode_uf[:,index] = np.sum(uf,1)/norm
        
        mode_ur_r[:,index] = np.sum(ur_r,1)/norm
        # mode_ur_t[:,index] = np.sum(ur_t,1)/norm
        # mode_ur_f[:,index] = np.sum(ur_f,1)/norm
        
        mode_ut_r[:,index] = np.sum(ut_r,1)/norm
        mode_ut_t[:,index] = np.sum(ut_t,1)/norm
        mode_ut_f[:,index] = np.sum(ut_f,1)/norm

        mode_uf_r[:,index] = np.sum(uf_r,1)/norm 
        mode_uf_t[:,index] = np.sum(uf_t,1)/norm
        mode_uf_f[:,index] = np.sum(uf_f,1)/norm
        
        
        index+=1 
                 
    return mode_ur*(-1j/2), mode_ut*(-1j/2), mode_uf*(1/2), mode_ur_r*(-1j/2), mode_ut_r*(-1j/2), mode_ut_t*(-1j/2), mode_ut_f*(-1j/2), mode_uf_r*(1/2), mode_uf_t*(1/2), mode_uf_f*(1/2)
 
def Eas_Mode(listt, t, f, r=1):
    
    lt = len(t)
    ll = len(listt)
    
    mode_ur = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_ut = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_uf = np.zeros(shape=[lt,ll],dtype=np.complex128)
    # mode_ur_t = np.zeros(shape=[lt,ll],dtype=np.complex128)
    # mode_ur_f = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_ur_r = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_ut_r = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_uf_r = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_ut_t = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_ut_f = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_uf_t = np.zeros(shape=[lt,ll],dtype=np.complex128)
    mode_uf_f = np.zeros(shape=[lt,ll],dtype=np.complex128)
    
    index = 0
    
    for m, k, n, x, norm in listt:
        
        m,k,n,x = int(m),int(k),int(n),x

        i=[];j=[]
        for xi in range(k+1):
            for xj in range(k-xi+1):
                i.append(xi);j.append(xj)
        i = np.array(i)
        j = np.array(j)
        
        li = len(i)
        tt = col(t,li)
        ff = col(f,li)
        
        ca = Ca(m,k,i,j)
        
        cc = ca*x**(2*i-1)*(1-x**2)**(j-1)*r**(m+2*(i+j))
        cc = row(cc,lt)
        
        # print()
        cc_r = (m+2*(i+j))*ca*x**(2*i-1)*(1-x**2)**(j-1)*r**(m+2*(i+j)-1.0)
        
        
        cc_r = row(cc_r,lt)
        
        ur = cc*np.sin(tt)**row(m+2*j,lt)*np.cos(tt)**row(2*i+1,lt)*np.exp(1j*m*ff)*\
            row(x*(m+m*x+2*j*x)-(2*i+1)*(1-x**2),lt)
        
        ut = cc*np.sin(tt)**row(m+2*j-1,lt)*np.cos(tt)**row(2*i,lt)*np.exp(1j*m*ff)*(\
            row(x*(m+m*x+2*j*x),lt)*np.cos(tt)**2+row((2*i+1)*(1-x**2),lt)*np.sin(tt)**2)  
        
        uf = cc*x*np.sin(tt)**row(m+2*j-1,lt)*np.cos(tt)**row(2*i+1,lt)*np.exp(1j*m*ff)*\
            row(m+m*x+2*j,lt)
        
        # ur_t = cc*np.exp(1j*m*ff)*row(x*(m+m*x+2*j*x)-(2*i+1)*(1-x**2),lt)*\
        #     (row(m+2*j,lt)*np.sin(tt)**row(m+2*j-1.0,lt)*np.cos(tt)*np.cos(tt)**row(2*i+1,lt)+ np.sin(tt)**row(m+2*j,lt)*row(2*i+1,lt)*np.cos(tt)**row(2*i,lt)*(-np.sin(tt)))

        # ur_f = 1j*m*ur
        
        ut_t = cc*np.exp(1j*m*ff)*np.cos(tt)**row(-1+2*i,lt)*np.sin(tt)**row(-2+2*j+m,lt)*(row(x*(-1+2*j+m)*\
                (m+x*(2*j+m)),lt)*np.cos(tt)**4+row(1+2*j+m-2*x*(1+i)*m+2*i*(1+m)-x**2*(1+2*i+6*j+3*m),lt)*np.cos(tt)**2*\
                np.sin(tt)**2+row(2*(-1+x**2)*i*(1+2*i),lt)*np.sin(tt)**4+row(i*(j-2*x**2*j-x**2*m),lt)*np.sin(2*tt)**2)
        
        ut_f = 1j*m*ut
        
        uf_t = cc*x*np.exp(1j*m*ff)*row(m+m*x+2*j,lt)*np.cos(tt)**row(2*i,lt)*\
                np.sin(tt)**row(-2+2*j+m,lt)*(row(-1+2*j+m,lt)*np.cos(tt)**2-row(1+2*i,lt)*np.sin(tt)**2)
        
        uf_f = 1j*m*uf
        
        ur_r = cc_r*np.sin(tt)**row(m+2*j,lt)*np.cos(tt)**row(2*i+1,lt)*np.exp(1j*m*ff)*\
            row(x*(m+m*x+2*j*x)-(2*i+1)*(1-x**2),lt)
        
        ut_r = cc_r*np.sin(tt)**row(m+2*j-1,lt)*np.cos(tt)**row(2*i,lt)*np.exp(1j*m*ff)*(\
            row(x*(m+m*x+2*j*x),lt)*np.cos(tt)**2+row((2*i+1)*(1-x**2),lt)*np.sin(tt)**2)
            
        uf_r = cc_r*x*np.sin(tt)**row(m+2*j-1,lt)*np.cos(tt)**row(2*i+1,lt)*np.exp(1j*m*ff)*\
            row(m+m*x+2*j,lt)
        
        mode_ur[:,index] = np.sum(ur,1)/norm
        mode_ut[:,index] = np.sum(ut,1)/norm
        mode_uf[:,index] = np.sum(uf,1)/norm
        
        mode_ur_r[:,index] = np.sum(ur_r,1)/norm
        
        mode_ut_r[:,index] = np.sum(ut_r,1)/norm
        mode_ut_t[:,index] = np.sum(ut_t,1)/norm
        mode_ut_f[:,index] = np.sum(ut_f,1)/norm

        mode_uf_r[:,index] = np.sum(uf_r,1)/norm 
        mode_uf_t[:,index] = np.sum(uf_t,1)/norm
        mode_uf_f[:,index] = np.sum(uf_f,1)/norm
        index+=1
       
  
                
    return mode_ur*(-1j/2), mode_ut*(-1j/2), mode_uf*(1/2), mode_ur_r*(-1j/2), mode_ut_r*(-1j/2), mode_ut_t*(-1j/2), mode_ut_f*(-1j/2), mode_uf_r*(1/2), mode_uf_t*(1/2), mode_uf_f*(1/2)


def Es_Norm(listt):
    norm=[]
    for m, k, n, sigma in listt:
        i=[];j=[];q=[];l=[]
        m = int(m);k=int(k);n=int(n)
        for xi in range(k+1):
            for xj in range(k-xi+1):
                for qi in range(k+1):
                    for li in range(k-qi+1):
                        i.append(xi);j.append(xj);l.append(li);q.append(qi)
        i = np.array(i);j = np.array(j);l = np.array(l);q = np.array(q)
        normi = Cs(m,k,i,j)*Cs(m,k,q,l)*3*2.0**(m+j+l-3)*sigma**(2*(i+q))*(1-sigma**2)**(j+l)/f2(2*(m+i+j+q+l)+1)\
            *(((m+m*sigma+2*j)*(m+m*sigma+2*l)+(m+m*sigma+2*j*sigma)*(m+m*sigma+2*l*sigma))
              *f1(m+j+l-1)*f2(2*(i+q)-1)/(1-sigma**2)**2+8*i*q*f1(m+j+l)*f2(2*(i+q)-3)/sigma**2)
        norm.append(np.sum(normi))    
    return np.c_[listt, np.abs(np.array(norm)*np.pi*4/3)**0.5] 


def Eas_Norm(listt):
    norm=[]
    for m, k, n, sigma in listt:
        i=[];j=[];q=[];l=[]
        m = int(m);k=int(k);n=int(n)
        for xi in range(k+1):
            for xj in range(k-xi+1):
                for qi in range(k+1):
                    for li in range(k-qi+1):
                        i.append(xi);j.append(xj);l.append(li);q.append(qi)
        i = np.array(i);j = np.array(j);l = np.array(l);q = np.array(q)
        normi = Ca(m,k,i,j)*Ca(m,k,q,l)*3*2.0**(m+j+l-3)*sigma**(2*(i+q))*(1-sigma**2)**(j+l)/f2(2*(m+i+j+q+l)+3)\
            *(((m+m*sigma+2*j)*(m+m*sigma+2*l)+(m+m*sigma+2*j*sigma)*(m+m*sigma+2*l*sigma))
              *f1(m+j+l-1)*f2(2*(i+q)+1)/(1-sigma**2)**2+2*(2*i+1)*(2*q+1)*f1(m+j+l)*f2(2*(i+q)-1)/sigma**2)
        norm.append(np.sum(normi))   
    return np.c_[listt, np.abs(np.array(norm)*np.pi*4/3)**0.5]  



if __name__ =='__main__' :
    
    #calculation of the eigenvalue and corresponding normalization constant
    M = 15
    
    N = 1
    
    Node = 4000
    
    K = 15
    
    r = 1
        
        # print(K)

    listt = Es_Norm(Es_mode_eigenvalue(M,K))
    n = listt[:,2]
    # p = np.where(n!=1)[0]
    p0 = np.where(n<=N)[0]
    list0 = listt[p0,:] # the simplist mode in z-direction
   
    
    listt = Eas_Norm(Eas_mode_eigenvalue(M,K))
    n = listt[:,2]
    # p = np.where(n!=1)[0]
    p0 = np.where(n<=N)[0]
    list1 = listt[p0,:]

 
    listt = Es_Norm(Es0_mode_eigenvalue(K))
    n = listt[:,2]
    # p = np.where(n!=1)[0]
    p0 = np.where(n<=N)[0]
    list2 = listt[p0,:]
    
    listt = Eas_Norm(Eas0_mode_eigenvalue(K))
    n = listt[:,2]
    # p = np.where(n!=1)[0]
    p0 = np.where(n<=N)[0]
    list3 = listt[p0,:]

    list_Es = np.r_[list2,list0]
    
    list_Eas = np.r_[list3,list1]
    
    t,f = GenGrid(N=Node)
    
    t = np.reshape(t,[-1])
    
    f = np.reshape(f,[-1])  
    
    ur, ut, uf, ur_r, ut_r, ut_t, ut_f, uf_r, uf_t, uf_f = Es_Mode(list_Es, t, f, r=1)
    
    Es = np.c_[ut, uf, ut_t, ut_f, uf_t, uf_f, ur_r, ut_r, uf_r]
    
    ur, ut, uf, ur_r, ut_r, ut_t, ut_f, uf_r, uf_t, uf_f = Eas_Mode(list_Eas, t, f, r=1)
    
    Eas = np.c_[ut, uf, ut_t, ut_f, uf_t, uf_f, ur_r, ut_r, uf_r]
    
    e = Geostrophic_mode(K, theta=t, r=1)
    
    path = 'BASE/'
    
    np.save(path+'Es_{}_{}_{}_{}.npy'.format(K,M,N,r),Es)

    np.save(path+'Eas_{}_{}_{}_{}.npy'.format(K,M,N,r),Eas)
    
    np.save(path+'Es_list_{}_{}_{}_{}.npy'.format(K,M,N,r),list_Es)

    np.save(path+'Eas_list_{}_{}_{}_{}.npy'.format(K,M,N,r),list_Eas)    
    
    np.save(path+'G_{}_{}_{}.npy'.format(K,N,r),e)  

    


