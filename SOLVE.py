import numpy as np
from scipy.sparse.linalg import eigs, spsolve
from pypardiso.pardiso_wrapper import PyPardisoSolver
from scipy.sparse import triu, csr_matrix
import time

" Olavo M Silva 2024 MOPT"

def modal_analysis(K=[], M=[], modes=20, which='LM', sigma=0.01, normalize=True):
    """
    """
    KT = K
    MT = M
       
    eigen_values, eigen_vectors = eigs(KT, M=MT, k=modes, which=which, sigma=sigma)

    positive_real = np.absolute(np.real(eigen_values))
    natural_frequencies = np.sqrt(positive_real)/(2*np.pi)
    modal_shape = np.real(eigen_vectors)

    index_order = np.argsort(natural_frequencies)
    natural_frequencies = natural_frequencies[index_order]
    modal_shape = modal_shape[:, index_order]
    
    if normalize:
        modal_shape /= np.max(np.abs(modal_shape), axis=0)

    return natural_frequencies, modal_shape

# def harmonic_analysys(freq, K, M, alpha, xsi, F, Up, free_dofs, fixed_dofs):

#     # Up = np.asarray(Up, dtype=complex).reshape(-1, 1)
#     # F  = np.asarray(F, dtype=complex).reshape(-1, 1)

#     # free_dofs  = np.asarray(free_dofs, dtype=int).ravel()
#     # fixed_dofs = np.asarray(fixed_dofs, dtype=int).ravel()
    
#     ps = PyPardisoSolver(mtype=6)
#     #ps.set_iparm(8, 2)
#     print('Rodando Análise Harmônica...')
#     U = np.empty([len(free_dofs),len(freq)]).astype(complex)

#     Kff = K[free_dofs, :][:, free_dofs]
#     Mff = M[free_dofs, :][:, free_dofs]
#     Ff =  F[free_dofs,:]

#     Kfp = K[free_dofs, :][:, fixed_dofs]
#     Mfp = M[free_dofs, :][:, fixed_dofs]

#     counter_1 = 0
#     for a in freq:
#         omega = 2 * (np.pi) * a
#         beta = xsi * (2*1/omega)
#         Cff = alpha*Mff + beta*Kff
#         Cfp = alpha*Mfp + beta*Kfp
#         Aff = Kff - (omega**2)*Mff +1j*omega*Cff
#         Aff = triu(Aff, format="csr")
#         Afp = Kfp - (omega**2)*Mfp +1j*omega*Cfp
#         F_ = Ff - Afp @ Up
#         #t = time.time()
#         #print('Solving freq: ',a)
#         U_aux = ps.solve(Aff, F_)
#         ps.free_memory(everything=True)
#         #elapsed = time.time() - t
#         #print('=====> Elapsed time: ',elapsed)
#         U[:,counter_1] = U_aux[:,0]
#         counter_1 +=  1
#     print("Análise Harmônica Finalizada com Sucesso!")
#     return U

def harmonic_analysys(freq, K, M, alpha, xsi, F, Up, free_dofs, fixed_dofs):
    from scipy.sparse import issparse
    from scipy.sparse import triu
    import numpy as np

    print('Rodando Análise Harmônica...')

    # --- Sub-matrizes (constantes) ---
    Kff = K[free_dofs, :][:, free_dofs]
    Mff = M[free_dofs, :][:, free_dofs]
    Ff  = F[free_dofs, :]

    Kfp = K[free_dofs, :][:, fixed_dofs]
    Mfp = M[free_dofs, :][:, fixed_dofs]

    # Pardiso mtype=6 (hermitiano complexo) espera upper triangle em CSR.
    # K e M são simétricas reais => triu(aK+bM) = a*triu(K) + b*triu(M)
    Kff_u = triu(Kff, format="csr")
    Mff_u = triu(Mff, format="csr")

    # Garante que Ff e Up sejam arrays densos 2D (coluna),
    # porque PyPardiso espera ndarray como RHS.
    Ff_dense = Ff.toarray() if issparse(Ff) else np.asarray(Ff)
    Up_dense = Up.toarray() if issparse(Up) else np.asarray(Up)

    has_prescribed = np.any(Up_dense != 0)

    # Solver criado UMA vez — reutiliza análise simbólica
    ps = PyPardisoSolver(mtype=6)

    n_freq = len(freq)
    n_dof  = len(free_dofs)
    U = np.empty((n_dof, n_freq), dtype=complex)

    omegas = 2.0 * np.pi * np.asarray(freq, dtype=float)

    for i, omega in enumerate(omegas):
        beta = xsi * (2.0 / omega)

        # A = (1 + j*omega*beta)*K + (-omega² + j*omega*alpha)*M
        c_K = 1.0 + 1j * omega * beta
        c_M = -(omega**2) + 1j * omega * alpha

        Aff = (Kff_u * c_K + Mff_u * c_M).tocsr()

        if has_prescribed:
            Afp = Kfp * c_K + Mfp * c_M
            rhs = Ff_dense - (Afp @ Up_dense)
        else:
            rhs = Ff_dense

        # Garante RHS como array contíguo Fortran-order (Pardiso prefere)
        rhs = np.ascontiguousarray(rhs, dtype=complex)

        U[:, i] = ps.solve(Aff, rhs).ravel()

    ps.free_memory(everything=True)

    print("Análise Harmônica Finalizada com Sucesso!")
    return U

# def harmonic_adjoint_analysys(freq, K, M, alpha, xsi, F, Up, free_dofs, fixed_dofs):

#     # Up = np.asarray(Up, dtype=complex).reshape(-1, 1)
#     # F  = np.asarray(F, dtype=complex).reshape(-1, 1)

#     # free_dofs  = np.asarray(free_dofs, dtype=int).ravel()
#     # fixed_dofs = np.asarray(fixed_dofs, dtype=int).ravel()

#     ps = PyPardisoSolver(mtype=6)
#     #ps.set_iparm(8, 2)
#     print('Rodando Análise Harmônica - Prob. Adjunto...')
#     Uadj = np.empty([len(free_dofs),len(freq)]).astype(complex)

#     Kff = K[free_dofs, :][:, free_dofs]
#     Mff = M[free_dofs, :][:, free_dofs]
#     Ff =  F[free_dofs,:]

#     Kfp = K[free_dofs, :][:, fixed_dofs]
#     Mfp = M[free_dofs, :][:, fixed_dofs]

#     counter_1 = 0
#     for a in freq:
#         omega = 2 * (np.pi) * a
#         beta = xsi * (2*1/omega)
#         Cff = alpha*Mff + beta*Kff
#         Cfp = alpha*Mfp + beta*Kfp
#         Aff = Kff - (omega**2)*Mff +1j*omega*Cff
#         Aff = triu(Aff, format="csr")
#         Afp = Kfp - (omega**2)*Mfp +1j*omega*Cfp
#         #
#         Fadj = 0.5*omega*1j*( Afp @ np.conjugate(Up) - np.conjugate(Ff) )
#         #
#         #t = time.time()
#         #print('Solving freq: ',a)
#         U_aux = ps.solve(Aff, Fadj)
#         ps.free_memory(everything=True)
#         #elapsed = time.time() - t
#         #print('=====> Elapsed time: ',elapsed)
#         Uadj[:,counter_1] = U_aux[:,0]
#         counter_1 +=  1
#     print("Análise Harmônica - Prob. Adjunto - Finalizada com Sucesso!")
#     return Uadj

def harmonic_adjoint_analysys(freq, K, M, alpha, xsi, F, Up, free_dofs, fixed_dofs):
    from scipy.sparse import issparse, triu
    import numpy as np

    print('Rodando Análise Harmônica - Prob. Adjunto...')

    # --- Sub-matrizes (constantes) ---
    Kff = K[free_dofs, :][:, free_dofs]
    Mff = M[free_dofs, :][:, free_dofs]
    Ff  = F[free_dofs, :]

    Kfp = K[free_dofs, :][:, fixed_dofs]
    Mfp = M[free_dofs, :][:, fixed_dofs]

    # Upper triangle para Pardiso mtype=6
    Kff_u = triu(Kff, format="csr")
    Mff_u = triu(Mff, format="csr")

    # Denso para RHS — PyPardiso espera ndarray
    Ff_dense  = Ff.toarray() if issparse(Ff) else np.asarray(Ff)
    Up_dense  = Up.toarray() if issparse(Up) else np.asarray(Up)
    Up_conj   = np.conjugate(Up_dense)  # constante, calcula uma vez
    Ff_conj   = np.conjugate(Ff_dense)

    ps = PyPardisoSolver(mtype=6)

    n_freq = len(freq)
    n_dof  = len(free_dofs)
    Uadj = np.empty((n_dof, n_freq), dtype=complex)

    omegas = 2.0 * np.pi * np.asarray(freq, dtype=float)

    for i, omega in enumerate(omegas):
        beta = xsi * (2.0 / omega)

        # A = (1 + j*omega*beta)*K + (-omega² + j*omega*alpha)*M
        c_K = 1.0 + 1j * omega * beta
        c_M = -(omega**2) + 1j * omega * alpha

        Aff = (Kff_u * c_K + Mff_u * c_M).tocsr()

        Afp = Kfp * c_K + Mfp * c_M

        # Fadj = 0.5 * omega * j * (Afp @ conj(Up) - conj(Ff))
        Fadj = (0.5 * omega * 1j) * (Afp @ Up_conj - Ff_conj)
        Fadj = np.ascontiguousarray(Fadj, dtype=complex)

        Uadj[:, i] = ps.solve(Aff, Fadj).ravel()

    ps.free_memory(everything=True)

    print("Análise Harmônica - Prob. Adjunto - Finalizada com Sucesso!")
    return Uadj

# def adjoint_analysys(freq, K, M, alpha, beta, F):
#     ps = PyPardisoSolver(mtype=6)
#     print('Rodando Análise Harmônica...')
#     U = np.empty([len(F),len(freq)]).astype(complex)

#     counter_1 = 0
#     for a in freq:
#         omega = 2 * (np.pi) * a
#         C = alpha*M + beta*K
#         A = K - (omega**2)*M +1j*omega*C
#         A = triu(A, format="csr")
#         #t = time.time()
#         #print('Solving freq: ',a)
#         U_aux = ps.solve(A, F)
#         ps.free_memory(everything=True)
#         #elapsed = time.time() - t
#         #print('=====> Elapsed time: ',elapsed)
#         U[:,counter_1] = U_aux[:,0]
#         counter_1 +=  1
#     print("Análise Harmônica Finalizada com Sucesso!")
#     return U

def static_analysys(K, F):

    F = np.asarray(F).reshape((-1,1))

    # garante matriz real e triangular superior (como antes)
    K = np.real(K)
    F = np.real(F)
    K = csr_matrix(triu(K, format="csr"))

    # solver correto para estática
    ps = PyPardisoSolver(mtype=2)

    U_aux = ps.solve(K, F)

    ps.free_memory(everything=True)

    # ---- mantém EXATAMENTE o formato antigo ----
    U = np.empty((len(F), 1), dtype=complex)
    U[:, 0] = U_aux[:, 0]

    print("Análise Estática Finalizada com Sucesso!")
    return U


def msup_analysis(K=[], M=[], modes=20, which='LM', sigma=0.01, normalize=True):
    """
    """
    KT = K
    MT = M
       
    eigen_values, eigen_vectors = eigs(KT, M=MT, k=modes, which=which, sigma=sigma)

    positive_real = np.absolute(np.real(eigen_values))
    natural_frequencies = np.sqrt(positive_real)/(2*np.pi)
    modal_shape = np.real(eigen_vectors)

    index_order = np.argsort(natural_frequencies)
    natural_frequencies = natural_frequencies[index_order]
    modal_shape = modal_shape[:, index_order]
    
    if normalize:
        modal_shape /= np.max(np.abs(modal_shape), axis=0)

    return natural_frequencies, modal_shape

def edge_single_sine(node_coords, nodes_edge, A, n=1):
    """
    nodes_edge : índices dos nós da aresta (ordenados em x!)
    A          : amplitude máxima
    n          : número do modo (1=meio seno, 2=seno inteiro, ...)
    """
    x = node_coords[nodes_edge, 0]

    # coordenada normalizada s ∈ [0,1]
    s = (x - x[0]) / (x[-1] - x[0])

    # seno puro
    U_edge = A * np.sin(n * np.pi * s)

    return U_edge