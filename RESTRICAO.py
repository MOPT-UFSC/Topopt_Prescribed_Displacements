import numpy as np

# def restricao(m,n,lx,ly,nel,h,xval):
#     #
#     a_el = lx*ly/nel
#     fval = np.zeros((m, 1))
#     dfval = np.zeros((m,n))
#     for el in range(n):
#         x = xval[el,0]
#         fval[0,0] += x*a_el*h
#         dfval[0,el] = a_el*h

#     return fval, dfval

def restricao(m, nel, active, a_el, h, xval, v_passive):
    #fval = np.zeros((m,1))
    fval = 0
    dfdx = np.zeros((1,nel))

    for el in active:
        #fval[0,0] += xval[el,0] * a_el * h
        fval += xval[el,0] * a_el * h
        dfdx[0,el] = a_el * h

    return fval, dfdx
