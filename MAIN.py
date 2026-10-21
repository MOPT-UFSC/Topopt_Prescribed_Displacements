#
import numpy as np
import matplotlib.pyplot as plt
import PLANE4Bolha_PT as femp4
import FILTER as filt
import SOLVE as solv
import MESH as msh
import OBJETIVO as obj
import RESTRICAO as restr
import substruct as subs
from VTK_FUNCS import vtk_write
from plots import plot_history_from_txt
# MMA
#from __future__ import division
from mma import mmasub, kktcheck
from util import setup_logger 
from typing import Tuple
import os

" OLAVO M. SILVA MOPT 2024"

if __name__ == "__main__": 

    ############### MATERIAL, AMORTECIMENTO E ESPESSURA GLOBAL (SI) ####################
    # Materiais
    E = 210e9
    Emin = E * 1e-7
    rho = 7850
    rhomin = rho * 1e-8
    v = 0.3
    # Proportinal damping
    freq = [1815]  #2000 $1010
    omega = 2 * np.pi * freq[0]
    ksi_crit = 1
    alpha_ = 0
    xsi_s = 0.05   # damping ratio during optimization (final FRFs are evaluated with 0.001, see PLOTS FRFS block)
    # Heaviside filter
    eta_ = (0.5, 0.05)  # eta0=0.5, deta=0.05
    bebeta = 0.1
    # Espessura
    h = 1
    #
    #################### MALHA QUAD4 RETANGULO ##################################
    lx = 1.5  #Comprimento do retângulo
    ly = 0.3 #Altura do retângulo
    tamanho_elemento = 0.0025   #Forçar exato
    area_do_elemento = tamanho_elemento**2
    coord, connect, nodes_faceX1, nodes_faceX2, nodes_faceY1, nodes_faceY2, nodes_excit_left, nodes_excit_right, nodes_excit_top = msh.malha2D_QUAD4(lx,ly,tamanho_elemento)

    print(coord)

    sym_map = subs.build_symmetry_map_y(coord, connect, ly)

    #elems_part1, nodes_part1, elems_part2, nodes_part2, elems_part3, nodes_part3 = subs.split_mesh_three_parts(coord, connect, ly, fractions=(0.05, 0.9, 0.05))

    nnode = len(coord)
    nel = len(connect)
    print(nel)
    
    fra_y = (0.10, 0.80, 0.10)
    #fra_x = (0.475, 0.05, 0.475)  # exemplo diferente em X
    fra_x = (0.45, 0.10, 0.45)  # exemplo diferente em X

    parts = subs.split_mesh_three_parts_xy(coord, connect, lx, ly,
                                  fractions_y=fra_y, fractions_x=fra_x)
    
    # Faixas em Y (de baixo pra cima)
    elemsY1 = parts["Y"][0]["elems"]; nodesY1 = parts["Y"][0]["nodes"]
    elemsY2 = parts["Y"][1]["elems"]; nodesY2 = parts["Y"][1]["nodes"]
    elemsY3 = parts["Y"][2]["elems"]; nodesY3 = parts["Y"][2]["nodes"]

    # Faixas em X (da esquerda pra direita)
    elemsX1 = parts["X"][0]["elems"]; nodesX1 = parts["X"][0]["nodes"]
    elemsX2 = parts["X"][1]["elems"]; nodesX2 = parts["X"][1]["nodes"]
    elemsX3 = parts["X"][2]["elems"]; nodesX3 = parts["X"][2]["nodes"]

    parts = subs.add_all_sides_to_parts(parts, coord, lx, ly,
                                      fractions_y=fra_y, fractions_x=fra_x,
                                      tol=1e-8)
    
    
    # T de cabeca pra baixo. Identifica nós nas arestas laterais do t, o os nós no topo do T invertido"
    nodesY1_left  = parts["Y"][0]["left_nodes"]
    nodesY1_right = parts["Y"][0]["right_nodes"]
    nodesY1_left  = nodesY1_left.astype(int)
    nodesY1_right = nodesY1_right.astype(int)

    # nodesX2_top = parts["X"][1]["top_nodes"]
    # nodesX2_top = nodesX2_top.astype(int)

    fixed_laterais = np.unique(np.concatenate((nodesY1_left, nodesY1_right)))

    passive = np.unique(np.concatenate((elemsY1, elemsY3)))

    active = np.setdiff1d(np.arange(nel), passive)

    n_mma = nel - len(passive)

    #passive = [elemsY1]
    #passive = [ ]

    # "faces" : linhas de contorno do retângulo. Saem com índices partindo do zero
    # "connect": sai com índices partindo do 1
    #
    ############################## FILTROS #############################################
    FILTRO = 2 #  0 -> Sem Filtro     ||   1 -> Filtro de Sensibilidades   ||    2 -> Filtro de Densidades
    diag_elem = np.sqrt(2*(tamanho_elemento)**2)
    radius = 15*diag_elem + diag_elem/10
    if FILTRO != 0:
        print("Preparação do filtro...")
        neighbors, H, pos = filt.get_neighbors_radius(nel, coord, connect, radius)
        print("Filtro armado.")
    #
    ### SEM SOMBRA ###
    # passive_mask = np.zeros(nel, dtype=bool)
    # passive_mask[passive] = True

    H_eff = H.copy()   # DEEIXAR
    neighbors_eff = neighbors.copy()  # DEEIXAR

    # 1) Remover influência dos passivos sobre os ativos
    # mask_neighbor_is_passive = (neighbors_eff != -1) & passive_mask[neighbors_eff]
    # H_eff[mask_neighbor_is_passive] = 0.0

    # 2) Isolar os passivos: só veem a si mesmos
    # for el in passive:
    #     H_eff[el, :] = 0.0
    #     H_eff[el, 0] = 1.0
    #     neighbors_eff[el, :] = -1
    #     neighbors_eff[el, 0] = el


    ################### Parâmetros da Otimização ############################
    p = 4  # Penalização da rigidez RAMP
    q = 1 # Penalização da massa
    #
    ps = 4# Penalização da rigidez RAMP - compliance estática
    #
    xinit_value = 0.25  # initial relative density of the design elements (Example 2; 0.20 in Example 1)
    xmin_value = 0
    xmax_value = 1
    xval = xinit_value*np.ones((nel, 1))
    volume_restante = 0.25
    minimo_volume = 0.05
    vtot = lx*ly*h 
    #vmax = volume_restante*vtot

    a_el = lx*ly/nel
    v_passive = len(passive) * a_el * h
    v_design = vtot - v_passive      # volume realmente projetável
    vmax = (volume_restante) * v_design 
    vmin = (minimo_volume) * v_design    

    #
    THRSHLD = 0 # 0 -> Sem Threshold   ||  1  -> Com Threshold
    PLOT_FRF = 1 # 0 -> Não plota Resposta em Frequência original vs Otimizado  || 1 -> Plota
    PARCELA_ESTATICO = 0.2 #  LC1b - 0.20 !!!!Porcentagem da função objetivo total referente a parcela estatica (0 - 1)
    ########### Condição de engaste na faceX1 ####################################
    # nnode_laterais = len(nodes_excit_top)
    nnode_base = len(nodes_faceY1) # len(nodes_faceY1)
    nnode_base_est = len(nodes_faceY1)

    fixed_dofs_x = []
    fixed_dofs_y = []
    fixed_dofs   = []
    fixed_dofs_est_x = []
    fixed_dofs_est_y = []
    fixed_dofs_est   = []

    ## CC Análise Harmônica
    for i in range(nnode_base): #no fixed dofs ... descomentar
        fixed_dofs_x_i = np.array([0]) + 2*nodes_faceY1[i]
        fixed_dofs_y_i = np.array([1]) + 2*nodes_faceY1[i]
        fixed_dofs_x = np.append(fixed_dofs_x, fixed_dofs_x_i).astype(int)
        fixed_dofs_y = np.append(fixed_dofs_y, fixed_dofs_y_i).astype(int)
    
    fixed_dofs = np.unique(np.concatenate((fixed_dofs_x, fixed_dofs_y)))
    #fixed_dofs = fixed_dofs_y.copy() #np.unique(np.concatenate((fixed_dofs_x, fixed_dofs_y)))
    all_dofs = np.arange(2*nnode)
    free_dofs = np.setdiff1d(np.arange(2*nnode), fixed_dofs)
    
    U = np.zeros((2*nnode,1),dtype=complex)
    U_adj = np.zeros((2*nnode,1),dtype=complex)
    U_est = np.zeros((2*nnode,1),dtype=complex)
    #

    AAA = 0 #1e-6 #1e-6  #* 0     # amplitude máxima
    BBB =  1e-6 #5e-7 #1e-6  #
    NNN = 5       # NNN/2 senos      
    U_edge = solv.edge_single_sine(coord, nodes_faceY1, BBB, NNN)
    

    U[fixed_dofs_x,:] = AAA #* U_edge.reshape(-1, 1)
    U[fixed_dofs_y,:] = U_edge.reshape(-1, 1)

    Up = np.asarray(U[fixed_dofs, :], dtype=complex).reshape(-1, 1)

    Lp, Lf = femp4.build_Lp_Lf(2*nnode, fixed_dofs)

    ## CC Análise Estática
    for i in range(nnode_base_est): #no fixed dofs ... descomentar
        fixed_dofs_est_x_i = np.array([0]) + 2*nodes_faceY1[i]
        fixed_dofs_est_y_i = np.array([1]) + 2*nodes_faceY1[i]
        fixed_dofs_est_x = np.append(fixed_dofs_est_x, fixed_dofs_est_x_i).astype(int)
        fixed_dofs_est_y = np.append(fixed_dofs_est_y, fixed_dofs_est_y_i).astype(int)
        #temp_est = np.append(fixed_dofs_est_x,fixed_dofs_est_y).astype(int)
        #fixed_dofs_est = np.append(fixed_dofs_est,temp_est).astype(int)
    fixed_dofs_est = np.unique(np.concatenate((fixed_dofs_est_x, fixed_dofs_est_y)))
    all_dofs_est = np.arange(2*nnode)
    free_dofs_est = np.setdiff1d(np.arange(2*nnode), fixed_dofs_est)
    #
    ########### Condição de carregamento dinamico ###############################
    nnode_excit = len(nodes_excit_top)
    load_dofs_x = []
    load_dofs_y = []
    for i in range(nnode_excit):
        load_dofs_x_i = np.array([0]) + 2*nodes_excit_top[i]
        load_dofs_y_i = np.array([1]) + 2*nodes_excit_top[i]
        load_dofs_x = np.append(load_dofs_x, load_dofs_x_i).astype(int)
        load_dofs_y = np.append(load_dofs_y, load_dofs_y_i).astype(int)
    F = np.zeros((2*nnode,1), dtype=complex)   # SEM FORÇAS EXTERNAS EM UM PRIMEIRO MOMENTO
    #F_aux = np.ones((len(F), 1), dtype=complex)
    # F[load_dofs_x[0],:] = 1000
    # F[load_dofs_x[-1],:] = -1000
    F[load_dofs_y,:] = -1000 #---:> ZEREI PARA PLOTAR COMO SAO OS DESLOCAMENTOS PRESCRITOS NO VTU!!!! LOAD CASE 2 tem força
    #F[2*127+1,:] = -1
    #F[:,:] = -1
    ########### Condição de carregamento estatico ###############################
    nnode_excit_est = len(nodes_faceY2)
    load_dofs_est_x = []
    load_dofs_est_y = []
    for i in range(nnode_excit_est):
        load_dofs_est_x_i = np.array([0]) + 2*nodes_faceY2[i]
        load_dofs_est_y_i = np.array([1]) + 2*nodes_faceY2[i]
        load_dofs_est_x = np.append(load_dofs_est_x, load_dofs_est_x_i).astype(int)
        load_dofs_est_y = np.append(load_dofs_est_y, load_dofs_est_y_i).astype(int)
    F_est = np.zeros((2*nnode,1), dtype=complex)
    #F_aux = np.ones((len(F), 1), dtype=complex)
    #F[load_dofs_x,:] = 1
    #F[load_dofs_r,:] = -1
    #F[load_dofs_x,:] = -1000
    F_est[load_dofs_est_y,:] = -1000
    #F[2*127+1,:] = -1
    #F[:,:] = -1
    ################### SE FILTRO DE DENSIDADES ##################################
    xval[passive] = 1
    xval_unf = xval.copy()
    #xval_unf = 0.5 * (xval_unf + xval_unf[sym_map])

    xval_unf_R = np.zeros((n_mma, 1))
    xval_unf_R = xval_unf[active]
    
    if FILTRO == 2:
        if bebeta is None:
            xval = filt.calc_xnew(nel, H_eff, neighbors_eff, xval_unf, bebeta=None, eta=eta_)  # densidade pura
            xD = xval.copy()  # só pra sua restrição funcionar do mesmo jeito
        else:
            xE,dE,xI,dI,xD,dD,_ = filt.calc_xnew_robust(nel, H_eff, neighbors_eff, xval_unf, bebeta, eta_)
            xE[passive] = 1.0
            xD[passive] = 1.0
            dE[passive] = 0.0
            dD[passive] = 0.0
            xval = xE.copy()  # (se quiser, pode usar xI aqui)
    #    
    xval_initial = xval.copy()
    ###################### ASSEMBLY ################################################
    print('Assembly...')
    ind_rows, ind_cols = femp4.generate_ind_rows_cols(connect)
    stif_matrix, mass_matrix, stif_matrix_s, data_k_, data_m_ = femp4.stif_mass_matrices_0(coord, connect, nnode, nel, ind_rows, ind_cols, E, Emin, v, rho, rhomin, h, p, q, ps, xval, THRSHLD)
    print('Assembly finalizado!')
    #
    # ################# ANÁLISE HARMÔNICA ##########################################
    ###### Ainda não usar para otimização acima da primeira ressonância ############
    #
    #damp_matrix = alpha_*mass_matrix.copy() + beta_*stif_matrix.copy()
    #KD = stif_matrix.copy() - (omega**2)*mass_matrix.copy() + 1j*omega*damp_matrix.copy()

    #U = np.zeros((2*nnode,1),dtype=complex)
    #U[free_dofs,:] = solv.harmonic_analysys(freq, stif_matrix[free_dofs, :][:, free_dofs], mass_matrix[free_dofs, :][:, free_dofs], alpha_, xsi_s, F[free_dofs,:])

    ####### Harmonic problem ##########
    U[free_dofs,:] = solv.harmonic_analysys(freq, stif_matrix, mass_matrix, alpha_, xsi_s, F, Up, free_dofs, fixed_dofs)
    # U[fixed_dofs_x,:] = 0
    # U[fixed_dofs_y,:] = AAA #* U_edge.reshape(-1, 1)
    #
    ####### Adjoint problem ###############
    #Fadj = -0.5*1j*omega*( np.conjugate(F) - (KD@Lp - Lp@np.conjugate(KD))@np.conjugate(U) )
    #U_adj= solv.adjoint_analysys(freq, stif_matrix, mass_matrix, alpha_, xsi_s, Fadj)
    U_adj[free_dofs,:] = solv.harmonic_adjoint_analysys(freq, stif_matrix, mass_matrix, alpha_, xsi_s, F, Up, free_dofs, fixed_dofs)
    # U_adj[fixed_dofs_x,:] = 0
    # U_adj[fixed_dofs_y,:] = 0 #* U_edge.reshape(-1, 1)
    #

    if PARCELA_ESTATICO != 0:
        U_est[free_dofs_est,:] = solv.static_analysys(stif_matrix_s[free_dofs_est, :][:, free_dofs_est], F_est[free_dofs_est,:])
        U_est[fixed_dofs_est,:] = 0
    #
    
    ####################################################################################################################################################
    ####################################################################################################################################################
    #########################################################             MMA            ###############################################################
    ####################################################################################################################################################
    ####################################################################################################################################################
    #
    # Logger
    path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(path, "EPT_OPT.log")
    logger = setup_logger(file)
    logger.info("Started\n")
    
    # Set numpy print options
    np.set_printoptions(precision=8, formatter={'float': '{: 0.4e}'.format})
    # Initial settings
    m, n = 2, n_mma
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    zerom = np.zeros((m, 1))
    df0dx_R = np.zeros((nel, 1))
    df0dx = np.zeros((n, 1))
    dfdx_R = np.zeros((m, nel))
    fval, dfdx = np.zeros((m, 1)), np.zeros((m, n))
    #xval = xval.copy() # Variáveis serão as espessuras elementares
    xold1 = xval_unf_R.copy()
    xold2 = xval_unf_R.copy()
    xmin = xmin_value * eeen
    ##xmin[passive] = 0.999
    xmax = xmax_value * eeen
    ##xmax[passive] = 1.001
    low = xmin.copy()
    ##low[passive] = 0.999
    upp = xmax.copy()
    ##upp[passive] = 1.001
    move = 0.1 #0.1
    c = 1000 * eeem
    d = eeem.copy()
    a0 = 1
    a = zerom.copy()
    innerit = 0
    outeriter = 0
    maxoutit = 2000
    kkttol = 0
    convergencia = np.zeros((maxoutit+1,3))
    
    modes = 20
    nat_freqs = np.zeros((maxoutit+1, modes))

    hist_outer = []
    hist_f0val_dyn = []
    hist_f0val_est = []
    hist_fval = []
    hist_bebeta = []
    hist_kktnorm = []
    hist_change = []

    # Calculate function values and gradients of the objective and constraints functions
    if outeriter == 0:

        natural_frequencies, _ = solv.modal_analysis(stif_matrix[free_dofs, :][:, free_dofs], mass_matrix[free_dofs, :][:, free_dofs], modes, which='LM', sigma=0.01)
        nat_freqs[0,:] =natural_frequencies 
        print(natural_frequencies)

        f0val_din_, f0val_est, df0dx_din_, df0dx_est = obj.objetivo(freq, alpha_, xsi_s, data_k_, data_m_, stif_matrix, mass_matrix, Lp, Lf, F, F_est, U, U_adj, U_est, connect, nel, p, q, ps, xval, E, Emin, rho, rhomin, THRSHLD)
        fval_, dfdx_ = restr.restricao(m, nel, active, a_el, h, xD, v_passive)

        print('f0val_dyn = ',f0val_din_)

        fval[0,0] = 100*((fval_-vmax)/v_design)
        dfdx_R[0,:] = 100*dfdx_/v_design

        print(fval[0,0])
        
        fval[1,0] = -100*((fval_-vmin)/v_design)
        dfdx_R[1,:] = -100*dfdx_/v_design

        f0val_din = 100 + 10*np.log10(f0val_din_)
        df0dx_din = 10*(1/(np.log(10)*f0val_din_))*df0dx_din_

        # f0val_din = f0val_din_
        # df0dx_din = df0dx_din_

        f_ini = fval[0,0].copy()
        f0_din_ini = f0val_din.copy()
        f0_est_ini = f0val_est.copy()
        if PARCELA_ESTATICO == 0:
            f0_est_ini = 1        
        
        if FILTRO == 2:
            if bebeta is None:
                df0dx_din, df0dx_est, dfdx_R = filt.density_filter(
                    nel, H_eff, neighbors_eff, df0dx_din, df0dx_est, dfdx_R,
                    xval=xval_unf, bebeta=None, eta=eta_
                )
            else:
                # Objetivo: (nel,1) -> (1,nel) -> filtra -> (nel,1)
                df0dx_din = filt.density_filter_with_dHdx(
                    nel, H_eff, neighbors_eff, np.asarray(df0dx_din).reshape(1, -1), dE
                ).reshape(-1, 1)

                df0dx_est = filt.density_filter_with_dHdx(
                    nel, H_eff, neighbors_eff, np.asarray(df0dx_est).reshape(1, -1), dE
                ).reshape(-1, 1)

                # Restrições: cada linha vira (1,nel) e volta para (nel,)
                # Em robust típico, restrição de volume usa dD (design), não dE (eroded).
                dfdx_R[0, :] = filt.density_filter_with_dHdx(
                    nel, H_eff, neighbors_eff, np.asarray(dfdx_R[0, :]).reshape(1, -1), dD
                ).reshape(-1)

                dfdx_R[1, :] = filt.density_filter_with_dHdx(
                    nel, H_eff, neighbors_eff, np.asarray(dfdx_R[1, :]).reshape(1, -1), dD
                ).reshape(-1)
                
        f0val = PARCELA_ESTATICO*(100*f0val_est/f0_est_ini) + (1-PARCELA_ESTATICO)*(100*f0val_din/f0_din_ini)
        df0dx_R = PARCELA_ESTATICO*(100*df0dx_est/f0_est_ini) + (1-PARCELA_ESTATICO)*(100*df0dx_din/f0_din_ini)

        outvector1 = np.concatenate((np.array([outeriter, innerit, f0val, fval[0,0] + 100*vmax/v_design, fval[1,0] - 100*vmin/v_design]), []))
        outvector2 = xval.copy().flatten()

        print('volume=',fval[0,0] + 100*vmax/v_design)
        
        convergencia[0,0] = 0
        convergencia[0,1] = f0val
        convergencia[0,2] = fval[0,0]

        kktnorm = 10
        change = 1
        hist_outer.append(outeriter)
        hist_f0val_dyn.append(100*f0val_din/f0_din_ini)
        hist_f0val_est.append(100*f0val_est/f0_est_ini)
        hist_fval.append(fval[0,0] + 100*vmax/v_design)
        hist_bebeta.append(bebeta if bebeta is not None else 0.0)
        hist_kktnorm.append(kktnorm)
        hist_change.append(change)

        # Log
        logger.info("outvector1 = {}".format(outvector1))
        #logger.info("outvector2 = {}\n".format(outvector2))
    
    # The iterations start
    kktnorm = kkttol + 10
    outit = 0
    change = 1
    mean_change = 1

    kk = 0
    am = 150
    ak = 20
    al = 30
    flags = 1

    flags = 1

    df0dx = df0dx_R[active,:]
    dfdx = dfdx_R[:,active]

    xval_old = xval.copy()

    while kktnorm > kkttol and outit < maxoutit and (bebeta is None or bebeta < 64):

        if kktnorm < 1 and change < 0.01 and kk > am:
            if kktnorm < 0.005 and (abs(fval[0,0]) < 0.1 or abs(fval[1,0]) < 0.1) and bebeta < 1:
                am = 10
                bebeta = min(bebeta * 1.3, bebeta + 3)
            else:
                am = 15
                bebeta = min(bebeta * 1.2, bebeta + 2)
            kk = 0

        kk = kk + 1

        if outit > 200 and kk > ak and change < 0.05 and (abs(fval[0,0]) < 0.1 or abs(fval[1,0]) < 0.1):
            bebeta = min(bebeta * 1.3, bebeta + 3)
            kk = 0

        if outit > 300 and kk > al and change < 0.1:
            bebeta = min(bebeta * 1.2, bebeta + 2)
            kk = 0

        if outit > 400 and kk > al and change < 0.3:
            bebeta = min(bebeta * 1.1, bebeta + 1)
            kk = 0

        if bebeta > 64:
            break

        print('bebeta = ', bebeta)
        print('kk = ', kk)

        outit += 1
        outeriter += 1        

        # The MMA subproblem is solved at the point xval:
        print("MMA ...")
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
            m, n, outeriter, xval_unf_R, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move)
        print("MMA finalizado!")

        #xmma[passive] = 1.0
        
        # Some vectors are updated:
        xold2 = xold1.copy()
        xold1 = xval_unf_R.copy()
        xval_unf_R = xmma.copy()

        xval_unf[active] = xmma.copy()
        xval_unf[passive] = 1.0
        #xval_unf = 0.5 * (xval_unf + xval_unf[sym_map])
        if FILTRO == 2:
            if bebeta is None:
                xval = filt.calc_xnew(nel, H_eff, neighbors_eff, xval_unf, bebeta=None, eta=eta_)
                xD = xval.copy()
            else:
                xE,dE,xI,dI,xD,dD,_ = filt.calc_xnew_robust(nel, H_eff, neighbors_eff, xval_unf, bebeta, eta_)
                xE[passive] = 1.0
                xD[passive] = 1.0
                dE[passive] = 0.0
                dD[passive] = 0.0
                xval = xE.copy()     
               
        ##natural_frequencies, _ = solv.modal_analysis(stif_matrix[free_dofs, :][:, free_dofs], mass_matrix[free_dofs, :][:, free_dofs], modes, which='LM', sigma=0.01)
        nat_freqs[outit,:] = natural_frequencies.copy() 

        # Re-calculate function values and gradients of the objective and constraints functions
        stif_matrix, mass_matrix, stif_matrix_s = femp4.stif_mass_matrices_OPT(data_k_, data_m_, nnode, nel, ind_rows, ind_cols, p, q, ps, xval, E, Emin, rho, rhomin, THRSHLD)
        #damp_matrix = alpha_*mass_matrix.copy() + beta_*stif_matrix.copy()
        
        #KD = stif_matrix.copy() - (omega**2)*mass_matrix.copy() + 1j*omega*damp_matrix.copy()

        U[free_dofs,:] = solv.harmonic_analysys(freq, stif_matrix, mass_matrix, alpha_, xsi_s, F, Up, free_dofs, fixed_dofs)
        # U[fixed_dofs_x,:] = 0
        # U[fixed_dofs_y,:] = AAA #* U_edge.reshape(-1, 1)

        ####### Adjoint problem ###############
        #Fadj = -0.5*1j*omega*( np.conjugate(F) - (KD@Lp - Lp@np.conjugate(KD))@np.conjugate(U) )
        #U_adj= solv.adjoint_analysys(freq, stif_matrix, mass_matrix, alpha_, xsi_s, Fadj)
        U_adj[free_dofs,:] = solv.harmonic_adjoint_analysys(freq, stif_matrix, mass_matrix, alpha_, xsi_s, F, Up, free_dofs, fixed_dofs)
        # U_adj[fixed_dofs_x,:] = 0
        # U_adj[fixed_dofs_y,:] = 0 #* U_edge.reshape(-1, 1)
       
        U_est = np.zeros((2*nnode,1),dtype=complex)
        if PARCELA_ESTATICO != 0:
            U_est[free_dofs_est,:] = solv.static_analysys(stif_matrix_s[free_dofs_est, :][:, free_dofs_est], F_est[free_dofs_est,:])
            U_est[fixed_dofs_est,:] = 0
    
        f0val_din_, f0val_est, df0dx_din_, df0dx_est = obj.objetivo(freq, alpha_, xsi_s, data_k_, data_m_, stif_matrix, mass_matrix, Lp, Lf, F, F_est, U, U_adj, U_est, connect, nel, p, q, ps, xval, E, Emin, rho, rhomin, THRSHLD)
        fval_, dfdx_ = restr.restricao(m, nel, active, a_el, h, xD, v_passive)

        #print('f0val_dyn = ',f0val_din_)

        f0val_din = 100 + 10*np.log10(f0val_din_)
        print('f0val_din = ',f0val_din)
        df0dx_din = 10*(1/(np.log(10)*f0val_din_))*df0dx_din_

        # f0val_din = f0val_din_
        # df0dx_din = df0dx_din_

        fval[0,0] = 100*((fval_-vmax)/v_design)
        dfdx_R[0,:] = 100*dfdx_/v_design

        fval[1,0] = -100*((fval_-vmin)/v_design)
        dfdx_R[1,:] = -100*dfdx_/v_design


        if FILTRO == 2:
            if bebeta is None:
                df0dx_din, df0dx_est, dfdx_R = filt.density_filter(
                    nel, H_eff, neighbors_eff, df0dx_din, df0dx_est, dfdx_R,
                    xval=xval_unf, bebeta=None, eta=eta_
                )
            else:
                df0dx_din = filt.density_filter_with_dHdx(
                    nel, H_eff, neighbors_eff, np.asarray(df0dx_din).reshape(1, -1), dE
                ).reshape(-1, 1)

                df0dx_est = filt.density_filter_with_dHdx(
                    nel, H_eff, neighbors_eff, np.asarray(df0dx_est).reshape(1, -1), dE
                ).reshape(-1, 1)

                dfdx_R[0, :] = filt.density_filter_with_dHdx(
                    nel, H_eff, neighbors_eff, np.asarray(dfdx_R[0, :]).reshape(1, -1), dD
                ).reshape(-1)

                dfdx_R[1, :] = filt.density_filter_with_dHdx(
                    nel, H_eff, neighbors_eff, np.asarray(dfdx_R[1, :]).reshape(1, -1), dD
                ).reshape(-1)
        
        f0val = PARCELA_ESTATICO*(100*f0val_est/f0_est_ini) + (1-PARCELA_ESTATICO)*(100*f0val_din/f0_din_ini)
        df0dx_R = PARCELA_ESTATICO*(100*df0dx_est/f0_est_ini) + (1-PARCELA_ESTATICO)*(100*df0dx_din/f0_din_ini)

        df0dx = df0dx_R[active,:]
        dfdx = dfdx_R[:,active]
        
        # The residual vector of the KKT conditions is calculated
        residu, kktnorm, residumax = kktcheck(
            m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)
        
        outvector1 = np.concatenate((np.array([outeriter, innerit, f0val, fval[0,0] + 100*vmax/v_design, fval[1,0] - 100*vmin/v_design]), []))
        outvector2 = xval.copy().flatten()

        convergencia[outit,0] = outit
        convergencia[outit,1] = f0val
        convergencia[outit,2] = fval[0,0]

        ###################### Critério de convergência baseado nas densidades relativas (norma) #######################
        change = np.max(np.abs(xval.flatten() - xval_old.flatten()))
        xval_old = xval.copy()

        hist_outer.append(outeriter)
        hist_f0val_dyn.append(100*f0val_din/f0_din_ini)
        hist_f0val_est.append(100*f0val_est/f0_est_ini)
        hist_fval.append(fval[0,0] + 100*vmax/v_design)
        hist_bebeta.append(bebeta if bebeta is not None else 0.0)
        hist_kktnorm.append(kktnorm)
        hist_change.append(change)
        
        # Log
        logger.info("outvector1 = {}".format(outvector1))
        logger.info("Critério de convergência = {}".format(change))
        #logger.info("outvector2 = {}".format(outvector2))
        logger.info("kktnorm    = {}\n".format(kktnorm))
    
    # Final log
    logger.info("Finished")

    iterx = np.arange(len(nat_freqs))
    for i in range(modes):
        plt.plot(iterx, nat_freqs[:, i], label=f"Linha {i+1}")
    plt.show()

    #### SAVE ####
    np.savetxt("f0val_dyn.txt",
           np.column_stack([hist_outer, hist_f0val_dyn]),
           header="outeriter  f0val_dyn")

    np.savetxt("f0val_est.txt",
           np.column_stack([hist_outer, hist_f0val_est]),
           header="outeriter  f0val_est")

    np.savetxt("fval.txt",
           np.column_stack([hist_outer, hist_fval]),
           header="outeriter  fval")

    np.savetxt("bebeta.txt",
           np.column_stack([hist_outer, hist_bebeta]),
           header="outeriter  bebeta")

    np.savetxt("kktnorm.txt",
           np.column_stack([hist_outer, hist_kktnorm]),
           header="outeriter  kktnorm")

    np.savetxt("change.txt",
           np.column_stack([hist_outer, hist_change]),
           header="outeriter  change")
    
    pos = 0 
    vtk_write(U[:,pos], xval, coord, connect, nnode, nel)

    ################### PLOTS #######################################################
    # Plotar curva de evolução das funções
    # fig, ax1 = plt.subplots()
    # fig, ax2 = plt.subplots()
    # ax1.plot(convergencia[:,0], convergencia[:,1], 'k', label='Função Objetivo: Compliance')
    # ax2.plot(convergencia[:,0], convergencia[:,2] + 100*volume_restante, 'r', label='Restrição: Volume')
    # ax2.axhline(y=100*volume_restante, color='k', linestyle='--')
    # plt.show()
    # 


    ################### PLOTS  FRFS #######################################################
    # Plotar FRF f0val
    xsi_s = 0.001 ####* (2*ksi_crit/omega)  ---> Calculado dentro do harmonic freq e do objetivo freq
    alpha_ = 0 
    if PLOT_FRF != 0:
        freq = np.linspace(1,3000,501)
        
        U_init = np.zeros((2*nnode,len(freq)),dtype=complex)
        U_opt = np.zeros((2*nnode,len(freq)),dtype=complex)

        U_init[fixed_dofs_x,:] = AAA
        U_init[fixed_dofs_y,:] = U_edge.reshape(-1, 1)

        U_opt[fixed_dofs_x,:] = AAA
        U_opt[fixed_dofs_y,:] = U_edge.reshape(-1, 1)

        stif_matrix, mass_matrix, stif_matrix_s = femp4.stif_mass_matrices_OPT(data_k_, data_m_, nnode, nel, ind_rows, ind_cols, p, q, ps, xval_initial, E, Emin, rho, rhomin, THRSHLD)
        U_init[free_dofs,:] = solv.harmonic_analysys(freq, stif_matrix, mass_matrix, alpha_, xsi_s, F, Up, free_dofs, fixed_dofs)
        
        f_init, ux_init, uy_init = obj.objetivo_freq(freq, F, U_init, stif_matrix, mass_matrix, Lp, alpha_, xsi_s)

        stif_matrix, mass_matrix, stif_matrix_s = femp4.stif_mass_matrices_OPT(data_k_, data_m_, nnode, nel, ind_rows, ind_cols, p, q, ps, xval, E, Emin, rho, rhomin, THRSHLD)
        U_opt[free_dofs,:] = solv.harmonic_analysys(freq, stif_matrix, mass_matrix, alpha_, xsi_s, F, Up, free_dofs, fixed_dofs)
   
        f_opt, ux_opt, uy_opt = obj.objetivo_freq(freq, F, U_opt, stif_matrix, mass_matrix, Lp, alpha_, xsi_s)

        np.savetxt("FRF_init.txt",
           np.column_stack([freq, f_init]),
           header="freq  f_init")
        
        np.savetxt("Ux_init.txt",
           np.column_stack([freq, ux_init]),
           header="freq  Ux_init")
        
        np.savetxt("Uy_init.txt",
           np.column_stack([freq, uy_init]),
           header="freq  Uy_init")

        np.savetxt("FRF_opt.txt",
           np.column_stack([freq, f_opt]),
           header="freq  f_opt")
        
        np.savetxt("Ux_opt.txt",
           np.column_stack([freq, ux_opt]),
           header="freq  Ux_opt")
        
        np.savetxt("Uy_opt.txt",
           np.column_stack([freq, uy_opt]),
           header="freq  Uy_opt")
        
        plot_history_from_txt()

        # fig, ax3 = plt.subplots()
        # ax3.semilogy(freq, f_init, 'k', label='Active Input Power')
        # ax3.semilogy(freq, f_opt, 'r', label='Potencia de Entrada Otimizada')
        # plt.grid()
        # plt.title("Active Input Power")
        # plt.xlabel("Frequency [Hz]")
        # plt.ylabel("Active Input Power [W]")
        # plt.show()

        # fig1, ax4 = plt.subplots()
        # ax4.plot(freq, W_reac_init, 'b--', label='Reactive Input Power')
        # ax4.semilogy(freq, Ep_init, 'k', label='Time-Averaged Potential Energy')
        # ax4.semilogy(freq, Ek_init, 'r', label='Time-Averaged Kinetic Energy')
        # ax4.legend()
        # plt.grid()
        # plt.title("Reactive Input Power, Time-Averaged Potential and Kinetic Energies ")
        # plt.xlabel("Frequency [Hz]")
        # plt.ylabel("Reactive Input Power [W] and Energies [J]") 
        # plt.show()

        # fig2, ax4 = plt.subplots()
        # ax4.plot(freq, W_reac_init, 'b--', label='Reactive Input Power')
        # ax4.semilogy(freq, U_drive, 'k', label='Abs. Vertical Disp. Driving Point')
        # ax4.legend()
        # plt.grid()
        # plt.title("Reactive Input Power and Abs. Vertical Disp. Driving Point ")
        # plt.xlabel("Frequency [Hz]")
        # plt.ylabel("Reactive Input Power[W] and Displacement [mm]")
        # plt.show()

        # fig3, ax5 = plt.subplots()
        # ax6 = ax5.twinx()
        # ax5.semilogy(freq, W_reac_init, 'b--', label='Reactive Input Power')
        # plt.xlabel("Frequency [Hz]")
        # ax5.legend()
        # ax6.plot(freq, R_init, 'k', label='Reactive Input Power')
        # ax6.legend()
        # plt.grid()

        # plt.ylabel("W and mm")
        # plt.show()
    
    #

    
    

    