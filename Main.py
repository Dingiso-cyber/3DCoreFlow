# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:10:45 2023

@author: DELL 5820
"""


#使用惯性模反演实测数据的程序
import matplotlib.pyplot as plt
import deepxde as dde
import numpy as np
from deepxde.backend import torch
from functions import MF_Space_deriv,GenG, GenGrid, gen_E

def feature_transform(x):
    
    np.random.seed(1)
    
    a = np.random.normal(0,1,[1,40])
    
    B = torch.tensor(a,dtype=torch.float32)
    
    return torch.concat([
                      torch.cos( 2 * np.pi * torch.linalg.matmul(x[:, 0:1],B)),
                      torch.sin( 2 * np.pi * torch.linalg.matmul(x[:, 0:1],B))
                      ], axis=1)
 
def TransformationLength(x):
    
    maxlm=30
    
    lm = int((maxlm+1)*(maxlm+2)/2)
    
    coeff = torch.linalg.matmul(G30,x.T)
    
    coeff_real,coeff_imag = coeff[0:lm], coeff[lm:lm*2]
    
    coeff_real = torch.linalg.matmul(tm,coeff_real) # 30->17
    
    coeff_imag = torch.linalg.matmul(tm,coeff_imag)
    
    y = torch.linalg.matmul(G17real,coeff_real)+torch.linalg.matmul(G17imag,coeff_imag) # back to physical space
    
    return y.T

  
def modify_inertial(E, r, i, maxlm, classes): 
    
    maxlm = int(maxlm)
    
    if classes == 'es0' or classes == 'eas0':
       
        Et_r    = E[:,maxlm*0:maxlm*1]
        
        Ef_r    = E[:,maxlm*1:maxlm*2]
        
        Et_t_r  = E[:,maxlm*2:maxlm*3]
         
        Ef_t_r  = E[:,maxlm*3:maxlm*4]
        
        Er_r_r  = E[:,maxlm*4:maxlm*5]
        
        Et_r_r  = E[:,maxlm*5:maxlm*6]
        
        Ef_r_r  = E[:,maxlm*6:maxlm*7]
        
        Et_i   =  E[:,maxlm*7:maxlm*8]
        
        Ef_i   =  E[:,maxlm*8:maxlm*9]
    
        Et_t_i =  E[:,maxlm*9:maxlm*10]
        
        Ef_t_i =  E[:,maxlm*10:maxlm*11]  
        
        Er_r_i  = E[:,maxlm*11:maxlm*12]
        
        Et_r_i  = E[:,maxlm*12:maxlm*13]
        
        Ef_r_i  = E[:,maxlm*13:maxlm*14]

        ut = torch.matmul(r,Et_r.T)+torch.matmul(i,Et_i.T)
        
        uf = torch.matmul(r,Ef_r.T)+torch.matmul(i,Ef_i.T)
        
        ut_t = torch.matmul(r,Et_t_r.T)+torch.matmul(i,Et_t_i.T)
        
        uf_t = torch.matmul(r,Ef_t_r.T)+torch.matmul(i,Ef_t_i.T)
        
        ur_r = torch.matmul(r,Er_r_r.T)+torch.matmul(i,Er_r_i.T)
        
        ut_r = torch.matmul(r,Et_r_r.T)+torch.matmul(i,Et_r_i.T)
        
        uf_r = torch.matmul(r,Ef_r_r.T)+torch.matmul(i,Ef_r_i.T)
        
        return ut, uf, ut_t, uf_t, ur_r, ut_r, uf_r
    
    elif classes=='es' or classes == 'eas':
        
        Et_r   = E[:,maxlm*0:maxlm*1]
        
        Ef_r   = E[:,maxlm*1:maxlm*2]
        
        Et_t_r = E[:,maxlm*2:maxlm*3]
         
        Et_f_r = E[:,maxlm*3:maxlm*4]
        
        Ef_t_r = E[:,maxlm*4:maxlm*5]
        
        Ef_f_r = E[:,maxlm*5:maxlm*6]
        
        Er_r_r  = E[:,maxlm*6:maxlm*7]
        
        Et_r_r  = E[:,maxlm*7:maxlm*8]
        
        Ef_r_r  = E[:,maxlm*8:maxlm*9]        
        
        Et_i = E[:,maxlm*9:maxlm*10]
        
        Ef_i = E[:,maxlm*10:maxlm*11]
    
        Et_t_i = E[:,maxlm*11:maxlm*12]
        
        Et_f_i = E[:,maxlm*12:maxlm*13]   
        
        Ef_t_i = E[:,maxlm*13:maxlm*14]
        
        Ef_f_i = E[:,maxlm*14:maxlm*15] 
        
        Er_r_i  = E[:,maxlm*15:maxlm*16]
        
        Et_r_i  = E[:,maxlm*16:maxlm*17]
        
        Ef_r_i  = E[:,maxlm*17:maxlm*18]
    
        ut = torch.matmul(r,Et_r.T)+torch.matmul(i,Et_i.T)
        
        uf = torch.matmul(r,Ef_r.T)+torch.matmul(i,Ef_i.T)
        
        ut_t = torch.matmul(r,Et_t_r.T)+torch.matmul(i,Et_t_i.T)
        
        uf_t = torch.matmul(r,Ef_t_r.T)+torch.matmul(i,Ef_t_i.T)
        
        ut_f = torch.matmul(r,Et_f_r.T)+torch.matmul(i,Et_f_i.T)
        
        uf_f = torch.matmul(r,Ef_f_r.T)+torch.matmul(i,Ef_f_i.T)
        
        ur_r = torch.matmul(r,Er_r_r.T)+torch.matmul(i,Er_r_i.T)
        
        ut_r = torch.matmul(r,Et_r_r.T)+torch.matmul(i,Et_r_i.T)
        
        uf_r = torch.matmul(r,Ef_r_r.T)+torch.matmul(i,Ef_r_i.T)
    
        return ut, uf, ut_t, ut_f, uf_t, uf_f, ur_r, ut_r, uf_r
    
    else:
        
        Ef   = E[:,maxlm*0:maxlm*1]
         
        Ef_t   = E[:,maxlm*1:maxlm*2]
        
        Ef_r   = E[:,maxlm*2:maxlm*3]
     
        uf = torch.matmul(r,Ef.T)
        
        uf_t = torch.matmul(r,Ef_t.T)
        
        uf_r = torch.matmul(r,Ef_r.T)
        
        return uf, uf_t, uf_r
        
def modify_output(E,br_lm_r,br_lm_i,maxlm_br): 

    E_br_a  = E[:,maxlm_br*0:maxlm_br*1]
    
    E_br_b  = E[:,maxlm_br*1:maxlm_br*2]
    
    E_bt_a  = E[:,maxlm_br*2:maxlm_br*3]
    
    E_bt_b  = E[:,maxlm_br*3:maxlm_br*4]
    
    E_bf_a  = E[:,maxlm_br*4:maxlm_br*5]
    
    E_bf_b  = E[:,maxlm_br*5:maxlm_br*6] 
    
    E_br_r_a  = E[:,maxlm_br*6:maxlm_br*7]
    
    E_br_r_b  = E[:,maxlm_br*7:maxlm_br*8]
    
    E_br_t_a  = E[:,maxlm_br*8:maxlm_br*9]
    
    E_br_t_b  = E[:,maxlm_br*9:maxlm_br*10]
    
    E_br_f_a  = E[:,maxlm_br*10:maxlm_br*11]
    
    E_br_f_b  = E[:,maxlm_br*11:maxlm_br*12]
    
    E_bt_t_a  = E[:,maxlm_br*12:maxlm_br*13]
    
    E_bt_t_b  = E[:,maxlm_br*13:maxlm_br*14]
    
    E_bt_f_a  = E[:,maxlm_br*14:maxlm_br*15]
    
    E_bt_f_b  = E[:,maxlm_br*15:maxlm_br*16]
    
    E_bf_t_a  = E[:,maxlm_br*16:maxlm_br*17]
    
    E_bf_t_b  = E[:,maxlm_br*17:maxlm_br*18]
    
    E_bf_f_a  = E[:,maxlm_br*18:maxlm_br*19]
    
    E_bf_f_b  = E[:,maxlm_br*19:maxlm_br*20]
    
    br = torch.matmul(br_lm_r,E_br_a.T)+torch.matmul(br_lm_i,E_br_b.T)
    
    bt = torch.matmul(br_lm_r,E_bt_a.T)+torch.matmul(br_lm_i,E_bt_b.T)
    
    bf = torch.matmul(br_lm_r,E_bf_a.T)+torch.matmul(br_lm_i,E_bf_b.T)
    
    dbr_r = torch.matmul(br_lm_r,E_br_r_a.T)+torch.matmul(br_lm_i,E_br_r_b.T)

    dbr_t = torch.matmul(br_lm_r,E_br_t_a.T)+torch.matmul(br_lm_i,E_br_t_b.T)

    dbr_f = torch.matmul(br_lm_r,E_br_f_a.T)+torch.matmul(br_lm_i,E_br_f_b.T)
    
    dbt_t = torch.matmul(br_lm_r,E_bt_t_a.T)+torch.matmul(br_lm_i,E_bt_t_b.T)

    dbt_f = torch.matmul(br_lm_r,E_bt_f_a.T)+torch.matmul(br_lm_i,E_bt_f_b.T)
    
    dbf_t = torch.matmul(br_lm_r,E_bf_t_a.T)+torch.matmul(br_lm_i,E_bf_t_b.T)

    dbf_f = torch.matmul(br_lm_r,E_bf_f_a.T)+torch.matmul(br_lm_i,E_bf_f_b.T)    

    return -br,-bt,-bf,-dbr_r,-dbr_t,-dbr_f,-dbt_t,-dbt_f,-dbf_t,-dbf_f

def Induction_equation(x, y):
    """ 
    Induction system.
    dbr+1/(r*sin(t))*br*sin(t)*dudt+br*ut*cos(t)+sin(t)*ut*dbrdt+dufdf*br+dbrdf*uf-diffu br=0
    Toroidal Assumption.
    1/(r*sin(t))*[cos(t)*ut+sin(t)*dutdt+dufdf]=0
    
    """
    a = 6371.2
    
    r = a*0.547
    
    g = 0.001*y[:,0:maxlm].double(); h = 0.001*y[:,maxlm_br:2*maxlm_br].double()
    
    i1 = 2*maxlm_br
    
    uv_es_r = y[:,i1:i1+lEs] 
    
    uv_es_i = y[:,i1+lEs:i1+lEs*2]    
    
    i4 = i1+lEs*2
    
    uv_eas_r = y[:,i4:i4+lEas] 
    
    uv_eas_i = y[:,i4+lEas:i4+lEas*2]
    
    i5 = i4+lEas*2

    uv_g = y[:,i5:i5+lG]
    
    brs, bts, bfs, brs_r, brs_t, brs_f, bts_t, bts_f, bfs_t, bfs_f = modify_output(E_small_scale, g, h, maxlm_br) 
    
    br = brl+brs
    
    bt = btl+bts
    
    bf = bfl+bfs
    
    dbr_r = dbrl_r+brs_r
    
    dbr_t = dbrl_t+brs_t
    
    dbr_f = dbrl_f+brs_f
    
    dbt_t = dbtl_t+bts_t
    
    dbt_f = dbtl_f+bts_f
    
    dbf_t = dbfl_t+bfs_t
    
    dbf_f = dbfl_f+bfs_f       
    
    ut_es, uf_es, ut_t_es, ut_f_es, uf_t_es, uf_f_es, ur_r_es, ut_r_es, uf_r_es = \
        modify_inertial(E_uv_es,uv_es_r.double(),uv_es_i.double(),maxlm=lEs,classes='es')
    
    ut_eas, uf_eas, ut_t_eas, ut_f_eas, uf_t_eas, uf_f_eas,ur_r_eas, ut_r_eas, uf_r_eas  = \
        modify_inertial(E_uv_eas,uv_eas_r.double(),uv_eas_i.double(),maxlm=lEas,classes='eas') 
        
    uf_g, uf_t_g, uf_r_g = \
        modify_inertial(E_uv_g, r = uv_g.double(), i=uv_g.double(), maxlm=lG, classes='g')
    
    brs_sp = torch.matmul(g**2+h**2,d)*l

    brs_sp = torch.reshape(brs_sp,[-1,17])
    
    ut = ut_eas+ut_es

    uf = uf_eas+uf_es+uf_g
    
    dut_t = ut_t_eas+ut_t_es
    
    dut_f = ut_f_eas+ut_f_es
    
    duf_t = uf_t_eas+uf_t_es+uf_t_g
    
    duf_f = uf_f_eas+uf_f_es
    
    dut_r = (ut_r_eas+ut_r_es)/r
    
    duf_r = (uf_r_eas+uf_r_es+uf_r_g)/r
    
    dur_r = (ur_r_es+ur_r_eas)/r

    sv_r = -(1/(r*torch.sin(t)))*(br*torch.sin(t)*dut_t+br*ut*torch.cos(t)+torch.sin(t)*ut*dbr_t+duf_f*br+dbr_f*uf)
    
    sv_t = (1/r)*(ut/torch.sin(t)*dbf_f-uf/torch.sin(t)*dbt_f+bf/torch.sin(t)*dut_f+r*ut*dbr_r-bt*(duf_f/torch.sin(t)+r*dur_r)+br*(ut+r*dut_r))
    
    sv_f = (1/r)*(-ut*dbf_t+bt*duf_t+uf*(dbt_t+r*dbr_r)+br*(uf+r*duf_r)-bf*(dut_t+r*dur_r))

    sv17_r = TransformationLength(sv_r.double())
    
    sv17_t = TransformationLength(sv_t.double())
    
    sv17_f = TransformationLength(sv_f.double())
    
    l1 = (sv17_r-sv_obs_r)
    
    l2 = (sv17_f-sv_obs_f)
    
    l3 = (sv17_t-sv_obs_t)
    
    l4 = (brs_sp_true-brs_sp)

    return [l1/6000, l2/3500, l3/5000, l4/sp_ref, sv17_r, sv_obs_r, uf, ut]

if __name__=='__main__':

    M = 10; K  = 6
    
    NN = 1
    
    path = 'BASE/'
    
    Es = np.load(path+'Es_15_15_1_1.npy') #base 使用惯性模测试中的sigma生成
    
    Eas = np.load(path+'Eas_15_15_1_1.npy')
    
    G = np.load(path+'G_15_1_1.npy')
    
    lEs = int(Es.shape[1]/9)
    
    lEas = int(Eas.shape[1]/9)
    
    lG = int(G.shape[1]/3)
    
    Es_list = np.load(path+'Es_list_15_15_1_1.npy')
    
    Eas_list = np.load(path+'Eas_list_15_15_1_1.npy')
    
    index_Es = np.where((Es_list[:,1]<K+1) & (Es_list[:,0]<M+1))[0]
    
    Es = np.concatenate([Es[:,index_Es+lEs*i] for i in range(9)],axis=1)
    
    index_Eas = np.where((Eas_list[:,1]<K+1) & (Eas_list[:,0]<M+1))[0]
    
    Eas = np.concatenate([Eas[:,index_Eas+lEas*i] for i in range(9)],axis=1)
    
    index_G = np.linspace(0,K-1,K,dtype=int)
    
    G = np.concatenate([G[:,index_G+lG*i] for i in range(3)],axis=1)
    
    lEs = int(Es.shape[1]/9)
    
    lEas = int(Eas.shape[1]/9)
    
    lG = int(G.shape[1]/3)
    
    E_uv_es =  torch.tensor(np.c_[Es.real, Es.imag])
    
    E_uv_eas =  torch.tensor(np.c_[Eas.real, Eas.imag])
    
    E_uv_g = torch.tensor(G)
    
    np.random.seed(1)
    
    max_m = 30 #the max degree of small-scale field
    
    truncation_m = 13
    
    maxlm_br = int((max_m+1)*(max_m+2)/2-(truncation_m+1)*(truncation_m+2)/2)
    
    sp_ref = (2.09*10**9)
    
    N = 4000
    
    mlsv = 17
    
    mlbr = truncation_m
    
    t,f = GenGrid(N) # grid point in physics space
    
    E_small_scale = torch.tensor(gen_E(t, f, max_m, truncation_m, output_list=True))
    
    s_list=[i for i in range(21)]
    
    SV_r = np.zeros([len(s_list),N])
    
    SV_t = np.zeros([len(s_list),N])
    
    SV_f = np.zeros([len(s_list),N])
    
    Br = np.zeros([len(s_list),N])
    
    Bt = np.zeros([len(s_list),N])
    
    Bf = np.zeros([len(s_list),N])
    
    Br_r = np.zeros([len(s_list),N])
    
    Br_t = np.zeros([len(s_list),N])
    
    Br_f = np.zeros([len(s_list),N])
    
    Bt_t = np.zeros([len(s_list),N])
    
    Bt_f = np.zeros([len(s_list),N])
    
    Bf_t = np.zeros([len(s_list),N])
    
    Bf_f = np.zeros([len(s_list),N])
    
    Br_sp = np.zeros([len(s_list),17])
    
    j=0
    
    for s in s_list:
        
        Bri, Bti, Bfi, Br_ri, Br_ti, Br_fi, Bt_ti, Bt_fi, Bf_ti, Bf_fi, Br_spi = MF_Space_deriv(i=s, t=t, f=f, nmax=13)
        
        SV_ri,SV_ti,SV_fi,_,_,_,_,_,_,_,_= MF_Space_deriv(i=s, t=t, f=f, nmax=17, deriv=1)
        
        Br[j,:] = Bri
        
        Bt[j,:] = Bti
        
        Bf[j,:] = Bfi
        
        Br_r[j,:] = Br_ri
        
        Br_t[j,:] = Br_ti
        
        Br_f[j,:] = Br_fi
        
        Bt_t[j,:] = Bt_ti
        
        Bt_f[j,:] = Bt_fi
        
        Bf_t[j,:] = Bf_ti
        
        Bf_f[j,:] = Bf_fi
        
        SV_r[j,:] = SV_ri
        
        SV_t[j,:] = SV_ti
        
        SV_f[j,:] = SV_fi
        
        Br_sp[j,:] = Br_spi
        
        j+=1
    
    sv_obs_r = torch.tensor(SV_r)
    
    sv_obs_t = torch.tensor(SV_t)
    
    sv_obs_f = torch.tensor(SV_f)
    
    brl = torch.tensor(Br)
    
    btl = torch.tensor(Bt)
    
    bfl = torch.tensor(Bf)
    
    dbrl_r = torch.tensor(Br_r)
    
    dbrl_t = torch.tensor(Br_t)
    
    dbrl_f = torch.tensor(Br_f)
    
    dbtl_t = torch.tensor(Bt_t)
    
    dbtl_f = torch.tensor(Bt_f)
    
    dbfl_t = torch.tensor(Bf_t)
    
    dbfl_f = torch.tensor(Bf_f)
    
    brs_sp_true = torch.tensor(Br_sp)    
    
    
    '''
    This part is to generate the matrix that calculated energy spectrum of small-scale unsolveable part magnetic field.
    
    The ouput matrix is thd matrix d, l, and q.
    
    q is the first-order differential matrix.
    
    l and d are used to calculate the energy spectrum for the small-scale unsolveable part magnetic field.
    
    '''
    
    maxlm = int((max_m+1)*(max_m+2)/2-(truncation_m+1)*(truncation_m+2)/2)
    
    m = np.linspace(0,max_m,max_m+1)
    
    m = np.tile(m,[max_m+1,1]).T
    
    n = np.linspace(0,max_m,max_m+1)
    
    n = np.tile(n,[max_m+1,1])
    
    index0 = np.triu_indices(max_m+1, k=0)
    
    m = m[index0]
    
    n = n[index0]
    
    index1 = np.where(n>truncation_m)
    
    n = n[index1]
    
    m = m[index1]
    
    d = np.zeros(shape=[len(m),max_m-truncation_m])
    
    q = np.zeros(shape=[max_m-truncation_m,max_m-truncation_m])
    
    l = np.arange(truncation_m+1,max_m+1,1)  
    
    for i in range(max_m-truncation_m):
        
        d[np.where(n==truncation_m+i+1.0),i] = 1
        
        q[i,i] = 1
        
        if i<max_m-truncation_m-1:
            
            q[i,i+1]=-1
            
        else:
            
            q[i,i-1]=-1
            
    d = torch.tensor(d,dtype=torch.float64)
    
    l = torch.tensor((l+1)*(1.83)**(2*l+4),dtype=torch.float64)
    
    q = torch.tensor(q,dtype=torch.float64)
    
    '''
    This part is first generate the matrix that project the flow field in physcis space onto spectrum space with max order and degree ml_max.
    
    Then filter out coeff with degree and order less than or equal to ml_truncation. 
    
    Then back to the physcis space.
    
    '''
    
    ml_max = 30 # project the SV field in physcis space onto spectrum space with max order and degree ml_max
    
    ml_truncation = mlsv # project the spectrum with degree and order less than or equal to ml_truncation 
    
    G30real, G30imag, _, _, _, _ = GenG(t, f, truncation_m=-1, max_m=ml_max, output_list = False) # each matrix contains imag and real part.
    
    G17real, G17imag, _, _, _, _  = GenG(t, f, truncation_m=-1, max_m=ml_truncation, output_list = False)
    
    G17real, G17imag = torch.tensor(G17real), torch.tensor(G17imag)
    
    m = np.linspace(0, ml_max, ml_max+1)
    
    m = np.tile(m,[ml_max+1,1]).T
    
    n = np.linspace(0, ml_max, ml_max+1)
    
    n = np.tile(n,[ml_max+1,1])
    
    index = np.triu_indices(ml_max+1, k=0)
    
    m = m[index]
    
    n = n[index]
    
    maxlma = int((ml_max+1)*(ml_max+2)/2)
    
    maxlmb = int((ml_truncation+1)*(ml_truncation+2)/2)
    
    tm = np.zeros([maxlma,maxlmb])
    
    j=0
    
    for i in range(maxlma):
        
        if n[i]<=ml_truncation:
            
            tm[i,j]=1
            
            j+=1
            
        else:
            
            pass
        
    tm = torch.tensor(tm.T,dtype=torch.float64) #filter out the higher order coefficients
    
    G30 = torch.tensor(np.linalg.pinv(np.c_[G30real,G30imag]),dtype=torch.float64) # This method may unstable when the degree>30
        
    t = torch.tensor(np.reshape(t,[1,-1]))
    
    f = torch.tensor(np.reshape(f,[1,-1]))    
                
    geom = dde.geometry.PointCloud(np.reshape(np.array(s_list),[-1,1]))
        
    bcs=[]
        
    iterations=40000
    
    data = dde.data.PDE(
        geom,
        Induction_equation,
        num_domain=len(s_list),
        bcs=[],
    )
    
            
    net = dde.nn.IMNN([1]+[128]*2+[maxlm*2],[1]+[128]*2+[lEs*2+lEas*2+lG],"tanh", "Glorot uniform")
    
    # net.apply_feature_transform(feature_transform) #Using this transform, the input neural number should be changed from 1 to 80
    
    model = dde.Model(data, net) 
    
    model.compile("adam", lr=0.001, loss_weights=[1, 1, 1, 1, 0, 0, 0, 0])
    
    # model.restore("model/M.pt") 
    
    checkpointer = dde.callbacks.ModelCheckpoint("model/M", verbose=1, save_better_only=True)
    
    losshistory, train_state = model.train(iterations=iterations,callbacks=[checkpointer])  
    
#%%

    a = model.predict(np.reshape(np.array(s_list),[-1,1]),operator=Induction_equation)

    U = a[6][0]
    
    V = a[7][0]
    
    t,f = GenGrid(N)
    
    plt.figure()
    
    plt.quiver(f[0::2],np.pi/2-t[0::2],U[0::2],-V[0::2])
    


