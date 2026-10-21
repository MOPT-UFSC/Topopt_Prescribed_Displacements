import numpy as np
from scipy.sparse import csr_matrix, diags
import MESH as msh

def shapeQ4B(ssx, ttx):
    """ Q4 Não-conforme. Shape Functions and Derivatives.
    """
    #shape functions 
    phi = np.zeros(4)
    P   = np.zeros(2)
    #
    phi[0]=(1-ssx)*(1-ttx)/4
    phi[1]=(1+ssx)*(1-ttx)/4
    phi[2]=(1+ssx)*(1+ttx)/4
    phi[3]=(1-ssx)*(1+ttx)/4
    #
    P[0]=(1-ssx**2)
    P[1]=(1-ttx**2)
    #derivatives
    dphi = np.zeros((2,4))
    dP   = np.zeros((2,2))
    #
    dphi[0,0]=(-1)*(1-ttx)/4
    dphi[0,1]=(+1)*(1-ttx)/4
    dphi[0,2]=(+1)*(1+ttx)/4
    dphi[0,3]=(-1)*(1+ttx)/4
    #
    dP[0,0]= -2*ssx
    dP[0,1]= 0
    
    dphi[1,0]=(1-ssx)*(-1)/4
    dphi[1,1]=(1+ssx)*(-1)/4
    dphi[1,2]=(1+ssx)*(+1)/4
    dphi[1,3]=(1-ssx)*(+1)/4
    #
    dP[1,0]= 0
    dP[1,1]= -2*ttx
             
    return phi, P, dphi, dP

def matricesP4(ee, coord, connect, E, vv, rho, H):
    """ P4 stiffness and mass matrices.
        Simple: midplane, no integration over the thickness
    """
    E = 1
    rho = 1
    ########### Pontos de Integração ############
    # Demais matrizes                 
    nint, con, wps = 4, 1/np.sqrt(3), 1
    pint = np.array([[-con, -con],
                      [ con, -con],
                      [ con,  con],
                      [-con,  con]])
    
    ############ Matrizes Constitutivas ############
    Dp = E/(1-(vv**2))*np.array([[ 1, vv, 0        ],
                                 [ vv,  1, 0       ],
                                 [ 0,   0, (1-vv)/2]])
    
    D = H*Dp  
    ##################### INTEGRAÇÃO ########################
    Ke, Ke_, Me = 0, 0, 0
    Bm = np.zeros((3,12))  # Com aumento da bolha Bm = [B G]
    B = np.zeros((3,8))  # Sem aumento da bolha
    G = np.zeros((3,4))  # Bolha
    N = np.zeros((2,8)) #Massa sem bolha
    # integration
    for i in range(nint):
        ssx, ttx = pint[i, 0], pint[i, 1]

        phi, P, dphi, dP = shapeQ4B(ssx,ttx)
        phi0, P0, dphi0, dP0 = shapeQ4B(0.0,0.0) # funções avaliadas no centro do elemento (ssx=ttx=0.0)

        ie = connect[ee,:]-1

        dxdy = dphi[:,0:4]@coord[ie, 0:2] 
        dxdy0 = dphi0[:,0:4]@coord[ie, 0:2] #dxdy avaliado no centro do elemento (ssx=ttx=0.0)

        JAC = np.array([[dxdy[0,0], dxdy[0,1]],
                        [dxdy[1,0], dxdy[1,1]]], dtype=float) 
        JAC0 = np.array([[dxdy0[0,0], dxdy0[0,1]],
                        [dxdy0[1,0], dxdy0[1,1]]], dtype=float)  #JAC avaliado no centro do elemento (ssx=ttx=0.0)
        
        detJAC = np.linalg.det(JAC)
        detJAC0 = np.linalg.det(JAC0) #detJAC avaliado no centro do elemento (ssx=ttx=0.0)

        iJAC = np.linalg.inv(JAC) 
        iJAC0 = np.linalg.inv(JAC0) #iJAC avaliado no centro do elemento (ssx=ttx=0.0)
        #      
        dphi_t = iJAC @ dphi
        dP_t = (detJAC0/detJAC) * iJAC0 @ dP
        
        for iii in range(4):
            B[0,2*(iii)+0]=dphi_t[0,iii]
            B[0,2*(iii)+1]=0
             #
            B[1,2*(iii)+0]=0
            B[1,2*(iii)+1]=dphi_t[1,iii]
            #
            B[2,2*(iii)+0]=dphi_t[1,iii]
            B[2,2*(iii)+1]=dphi_t[0,iii]

        for iii in range(2):
            G[0,2*(iii)+0]=dP_t[0,iii]
            G[0,2*(iii)+1]=0
             #
            G[1,2*(iii)+0]=0
            G[1,2*(iii)+1]=dP_t[1,iii]
            #
            G[2,2*(iii)+0]=dP_t[1,iii]
            G[2,2*(iii)+1]=dP_t[0,iii]

        Bm[:,0:8] = B
        Bm[:,8:12] = G
            
        for iii in range(4):
            N[0,2*(iii)+0]=phi[iii]
            N[1,2*(iii)+1]=phi[iii]

        Ke_ += (Bm.T@D@Bm)*(detJAC*wps)
        Me += H*rho*N.T@N*(detJAC*wps)

    Kdd = Ke_[0:8,0:8]
    Kda = Ke_[0:8,8:12]
    Kad = Kda.T
    Kaa = Ke_[8:12,8:12]
    Ke = Kdd - Kda@np.linalg.inv(Kaa)@Kad

    # Reforço da simetria natural (evitando resíduos perto de zero assimétricos)
    Ke = 0.5 * (Ke + Ke.T)
    Me = 0.5 * (Me + Me.T)

    return Ke, Me

def generate_ind_rows_cols(connect):
    # processing the dofs indices (rows and columns) for assembly
    dofs, edofs = 2, 8
    ind_dofs =      (np.array([dofs*connect[:,0]-2, dofs*connect[:,0]-1,
                               dofs*connect[:,1]-2, dofs*connect[:,1]-1,
                               dofs*connect[:,2]-2, dofs*connect[:,2]-1,
                               dofs*connect[:,3]-2, dofs*connect[:,3]-1], dtype=int)).T
    vect_indices = ind_dofs.flatten()
    ind_rows = ((np.tile(vect_indices, (edofs,1))).T).flatten()
    ind_cols = (np.tile(ind_dofs, edofs)).flatten()
    return ind_rows, ind_cols

def stif_mass_matrices_0(coord, connect, nnode, nel, ind_rows, ind_cols, E, Emin, vv, rho, rhomin, h, p, q, ps, xval, THRSHLD):
    """ Calculates global matrices.
    """
    ngl = 2 * nnode
    data_k_ = np.zeros((nel, 8, 8), dtype=float)
    data_m_ = np.zeros((nel, 8, 8), dtype=float)
    data_k = np.zeros((nel, 64), dtype=float)
    data_ks = np.zeros((nel, 64), dtype=float)
    data_m = np.zeros((nel, 64), dtype=float)
    
    for el in range(nel):
        x = xval[el,0]
        # if THRSHLD != 0 and x <= 0.1:
        #     q = 3
        # else:
        #     q = 1
        Ke, Me = matricesP4(el, coord, connect, E, vv, rho, h)
        # if THRSHLD != 0 and x <= 0.1:
        #     # c1 = 6e5
        #     # c2= -5e6
        #     # data_m[el,:] = (rhomin + (rho-rhomin)*(c1*x**6 + c2*x**7))*Me.flatten()
        #     data_m[el,:] = (rhomin + (rho-rhomin)*(0))*Me.flatten()
        # else:
        #     data_m[el,:] = (rhomin + (rho-rhomin)*(x**1))*Me.flatten()
        #Ke, Me = matricesP4(el, coord, connect, E, vv, rho, h)
        data_k_[el,:,:] = Ke
        data_m_[el,:,:] = Me
        # data_k[el,:] = (Emin + (E-Emin)*(x**p))*Ke.flatten() 
        data_k[el,:] = (Emin + (E-Emin)*( x/(1 + p*(1 - x)) ))*Ke.flatten() 
        # data_ks[el,:] = (Emin + (E-Emin)*(x**ps))*Ke.flatten() 
        data_ks[el,:] = (Emin + (E-Emin)*( x/(1 + ps*(1 - x)) ))*Ke.flatten() 
        # r=10
        # c=0.15
        # s  = 1/(1 + np.exp(-r*(x - c)))
        # f  = x * s**3
        # data_m[el,:] = (rhomin + (rho-rhomin)*(f))*data_m_[el,:,:].flatten()
        data_m[el,:] = (rhomin + (rho-rhomin)*(x**q))*Me.flatten() 
   
    data_k = data_k.flatten()
    data_ks = data_ks.flatten()
    data_m = data_m.flatten()
    stif_matrix = csr_matrix((data_k, (ind_rows, ind_cols)), shape=(ngl, ngl))
    stif_matrix_s = csr_matrix((data_ks, (ind_rows, ind_cols)), shape=(ngl, ngl))
    mass_matrix = csr_matrix((data_m, (ind_rows, ind_cols)), shape=(ngl, ngl))

    return stif_matrix, mass_matrix, stif_matrix_s, data_k_, data_m_

def stif_mass_matrices_OPT(data_k_, data_m_, nnode, nel, ind_rows, ind_cols, p, q, ps, xval, E, Emin, rho, rhomin, THRSHLD):
    """ Calculates global matrices.
    """
    ngl = 2 * nnode
    data_k = np.zeros((nel, 64), dtype=float)
    data_ks = np.zeros((nel, 64), dtype=float)
    data_m = np.zeros((nel, 64), dtype=float)
    
    for el in range(nel):
        x = xval[el,0]
        # if THRSHLD != 0 and x <= 0.1:
        #     q = 3
        # else:
        #     q = 1
        # if THRSHLD != 0 and x <= 0.1:
        #     # c1 = 6e5
        #     # c2= -5e6
        #     # data_m[el,:] = (rhomin + (rho-rhomin)*(c1*x**6 + c2*x**7))*data_m_[el,:,:].flatten()
        #     data_m[el,:] = (rhomin + (rho-rhomin)*(0))*data_m_[el,:,:].flatten()
        # else:
        #     data_m[el,:] = (rhomin + (rho-rhomin)*(x**1))*data_m_[el,:,:].flatten()
        #data_k[el,:] = (Emin + (E-Emin)*(x**p))*data_k_[el,:,:].flatten() 
        data_k[el,:] = (Emin + (E-Emin)*( x/(1 + p*(1 - x)) ))*data_k_[el,:,:].flatten() 
        #data_ks[el,:] = (Emin + (E-Emin)*(x**ps))*data_k_[el,:,:].flatten() 
        data_ks[el,:] = (Emin + (E-Emin)*( x/(1 + ps*(1 - x)) ))*data_k_[el,:,:].flatten() 
        # r=10
        # c=0.15
        # s  = 1/(1 + np.exp(-r*(x - c)))
        # f  = x * s**3
        # data_m[el,:] = (rhomin + (rho-rhomin)*(f))*data_m_[el,:,:].flatten()
        data_m[el,:] = (rhomin + (rho-rhomin)*(x**q))*data_m_[el,:,:].flatten()
   
    data_k = data_k.flatten()
    data_ks = data_ks.flatten()
    data_m = data_m.flatten()
    stif_matrix = csr_matrix((data_k, (ind_rows, ind_cols)), shape=(ngl, ngl))
    stif_matrix_s = csr_matrix((data_ks, (ind_rows, ind_cols)), shape=(ngl, ngl))
    mass_matrix = csr_matrix((data_m, (ind_rows, ind_cols)), shape=(ngl, ngl))

    return stif_matrix, mass_matrix, stif_matrix_s

def build_selector_diag(n_dofs, one_indices):
    one_indices = np.asarray(one_indices, dtype=int).ravel()
    one_indices = np.unique(one_indices)

    if one_indices.size == 0:
        return csr_matrix((n_dofs, n_dofs))

    if one_indices.min() < 0 or one_indices.max() >= n_dofs:
        raise ValueError(
            f"indices fora do intervalo: min={one_indices.min()}, "
            f"max={one_indices.max()}, n_dofs={n_dofs}"
        )

    data = np.ones(one_indices.size, dtype=float)
    return csr_matrix((data, (one_indices, one_indices)), shape=(n_dofs, n_dofs))

def build_Lp_Lf(n_dofs, fixed_dofs):
    Lp = build_selector_diag(n_dofs, fixed_dofs)
    # Lf = I - Lp (mais seguro e sem ambiguidade)
    Lf = diags(np.ones(n_dofs), 0, format="csr") - Lp
    return Lp, Lf