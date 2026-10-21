import numpy as np

#
def objetivo(freq, alpha, xsi, data_k_, data_m_, K, M, Lp, Lf, F, F_est, U, U_adj, U_est, connect, nel, p, q, ps, xval, E, Emin, rho, rhomin, THRSHLD):


    dofs = 2
    omega = 2*np.pi*freq[0]
    # Função objetivo: potência de entrada ativa

    beta = xsi* (2*1/omega)
    C = alpha*M + beta*K
    Kd = K - (omega**2)*M + 1j*omega*C

    V = Lp @ U

    V_adj = Lf @ U_adj

    r = Lp @ Kd @ U
    F_ = F + r

    f0val_din = 0.5 * omega * np.real(1j*U.T @ np.conj(F_))

    f0val_est = F_est.T @ U_est

    # Derivada da função objetivo
    df0val_din = np.zeros((nel,1))
    df0val_est = np.zeros((nel,1))

    for el in range(nel):
        x = xval[el,0] 
        # if THRSHLD != 0 and x <= 0.1:
        #     q = 3
        # else:
        #     q = 1
        Me = data_m_[el,:,:] 
        Ke = data_k_[el,:,:]  
        # if THRSHLD != 0 and x <= 0.1:
        #     # c1 = 6e5
        #     # c2= -5e6
        #     # dMe = (rhomin + (rho-rhomin)*(6*c1*x**5 + 7*c2*x**6))*Me
        #     dMe = (rhomin + (rho-rhomin)*(0))*Me
        # else:
        #     dMe = (rhomin + (rho-rhomin)*(x**1))*Me
        # #Ke = data_k_[el,:,:]  
        # #Me = data_m_[el,:,:] 
        ind_dofs =      (np.array([dofs*connect[el,0]-2, dofs*connect[el,0]-1,
                                   dofs*connect[el,1]-2, dofs*connect[el,1]-1,
                                   dofs*connect[el,2]-2, dofs*connect[el,2]-1,
                                   dofs*connect[el,3]-2, dofs*connect[el,3]-1], dtype=int)).T
        Ue = U[ind_dofs,:]
        Uadj_e = U_adj[ind_dofs,:]
        Ve = V[ind_dofs,:]
        Vadj_e = V_adj[ind_dofs,:]
        Ue_est = np.real(U_est[ind_dofs,:])
        #dKe = (E-Emin)*(p*(x**(p-1)))*Ke
        dKe = (E-Emin)*( (1+p)/((1 + p*(1 - x))**2) )*Ke
        #dKe_s = (E-Emin)*(ps*(x**(ps-1)))*Ke
        dKe_s = (E-Emin)*( (1+ps)/((1 + ps*(1 - x))**2) )*Ke
        # r=10
        # c=0.15
        # s  = 1/(1 + np.exp(-r*(x - c)))
        # df = s**3 * (1 + 3*r*x*(1 - s))
        # dMe = (rho-rhomin)*(df)*Me
        dMe = (rho-rhomin)*(q*(x**(q-1)))*Me
        #
        dCe = alpha*dMe + beta*dKe
        dKd = dKe + 1j*omega*dCe - (omega**2)*dMe

        term_power = -0.5 * omega * np.real(1j * np.conj(Ve).T @ dKd @ Ue)
        term_adjoint = np.real(Vadj_e.T @ dKd @ Ue)
        df0val_din[el, 0] = term_power + term_adjoint
        # Termo de correção correto para potência ativa


        df0val_est[el,0] = -Ue_est.T @ dKe_s @ Ue_est  #compliance estática
    
    return f0val_din[0,0], np.real(f0val_est[0,0]), df0val_din, np.real(df0val_est)

def objetivo_freq(freq, F, U_opt, K, M, Lp, alpha, xsi):

    nfreq = len(freq)

    f_opt  = np.zeros((nfreq, 1), dtype=float)
    ux_opt  = np.zeros((nfreq, 1), dtype=float)
    uy_opt  = np.zeros((nfreq, 1), dtype=float)

    # garanta F como vetor 1D
    F = np.asarray(F).reshape(-1)

    for i in range(nfreq):

        omega = 2*np.pi*freq[i]
        beta = xsi* (2*1/omega)
        C = alpha*M + beta*K
        
        Kd = K - (omega**2)*M + 1j*omega*C

        uo = U_opt[:,  i].reshape(-1)
        u_tip_x = U_opt[2*3,  i]
        u_tip_y = U_opt[2*3 + 1,  i]

        ro  = (Lp @ (Kd @ uo)).reshape(-1)   # vetor 1D
        Fo = F + ro                          # vetor 1D

        # produto interno conjugado -> escalar complexo
        s_opt  = uo. T @ np.conjugate(Fo)

        f_opt[i, 0]  = 0.5 * omega * np.real(1j * s_opt)
        ux_opt[i, 0]  = np.abs(u_tip_x)
        uy_opt[i, 0]  = np.abs(u_tip_y)

    return f_opt, ux_opt, uy_opt