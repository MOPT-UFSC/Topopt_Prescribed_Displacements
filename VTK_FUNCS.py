import numpy as np
from vtk import vtkUnstructuredGrid, vtkPoints, vtkDoubleArray, vtkXMLUnstructuredGridWriter

def vtk_write(U, data, coord, connect, nnode, nel):

# Gera VTU da Malha ###################################################
    my_vtk_dataset = vtkUnstructuredGrid()
    points = vtkPoints()
    VTK_QUAD = 9
    for id in range(nnode):
        points.InsertPoint(id, [coord[id,0], coord[id,1], coord[id,2]])
        my_vtk_dataset.SetPoints(points)
    my_vtk_dataset.Allocate(nel)
    for id in range(nel):
        point_ids = [connect[id,0]-1, connect[id,1]-1, connect[id,2]-1, connect[id,3]-1]
        my_vtk_dataset.InsertNextCell(VTK_QUAD, 4, point_ids)

    unod1 = np.zeros((nnode, 2), dtype=complex)
    eigm1 = np.zeros((nnode, 2), dtype=float)
    
    for i in range(nnode):
        #eigm1[i,0] = modal_shape[2*i]
        #eigm1[i,1] = modal_shape[2*i+1]
        #
        unod1[i,0] = U[2*i]
        unod1[i,1] = U[2*i+1]
                        
    # array1 = vtkDoubleArray()
    # array1.SetNumberOfComponents(3)
    # array1.SetNumberOfTuples(nnode)
    # array1.SetName('Eigenvector')

    array1 = vtkDoubleArray()
    array1.SetNumberOfComponents(3)
    array1.SetNumberOfTuples(nnode)
    array1.SetName('Displacement - Real')
    #

    for id in range(nnode):
        # values1 = [(eigm1[id,0]), (eigm1[id,1]), 0.]
        # array1.SetTuple(id, values1)
        # my_vtk_dataset.GetPointData().AddArray(array1)
        values1 = [np.real(unod1[id, 0]), np.real(unod1[id, 1]), 0]
        array1.SetTuple(id, values1)
        my_vtk_dataset.GetPointData().AddArray(array1)
        #
 
    array3 = vtkDoubleArray()
    array3.SetNumberOfComponents(1)
    array3.SetNumberOfTuples(nel)
    array3.SetName('Artificial Density')

    for i in range(nel):
        array3.SetTuple(i, [np.real(data[i])])
        my_vtk_dataset.GetCellData().AddArray(array3)

    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName("result.vtu")
    writer.SetInputData(my_vtk_dataset)
    writer.Write()