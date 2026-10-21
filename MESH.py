import gmsh
import numpy as np
import vtk
import math
from vtk.util import numpy_support

def malha2D_QUAD4(lx,ly,size):

    # Inicializar o Gmsh
    gmsh.initialize()

    # Criar ou carregar a geometria
    gmsh.model.add("Quadrangular Mesh Example")
    # Aqui você pode criar uma geometria, por exemplo, um retângulo
    # Para um arquivo IGES, use: gmsh.open("sua_geometria.iges")

    lc = size  # Tamanho do elemento da malha
    # Criar um retângulo como exemplo
    p1 = gmsh.model.geo.addPoint(0,  0,  0, lc)
    p2 = gmsh.model.geo.addPoint(lx, 0,  0, lc)
    p3 = gmsh.model.geo.addPoint(lx, ly, 0, lc)
    p4 = gmsh.model.geo.addPoint(0,  ly, 0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Criar uma curva fechada e uma superfície plana
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    # Aplicar malha transfinita à superfície
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface)  # Recombinar para obter quadriláteros

   # Sincronizar a geometria
    gmsh.model.geo.synchronize()

    # Gerar a malha bidimensional
    gmsh.model.mesh.generate(2)

    # Recompor para gerar elementos quadrangulares
    gmsh.model.mesh.recombine()

    # Sincronizar novamente para garantir que todas as operações foram aplicadas
    gmsh.model.geo.synchronize()

    # Obter os nós da malha
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords).reshape(-1, 3)

    # === REINDEXAÇÃO: tags do Gmsh podem não ser sequenciais ===
    node_tags = node_tags.astype(int)
    sort_order = np.argsort(node_tags)
    node_coords = node_coords[sort_order]       # reordena coords por tag crescente
    tag_to_idx = np.empty(node_tags.max() + 1, dtype=int)
    tag_to_idx[node_tags[sort_order]] = np.arange(len(node_tags))

    # Obter conectividade dos elementos quadrangulares
    element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements(dim=2)

    # Procurar elementos quadrangulares (element_type 3)
    quad_elements = []
    for i, element_type in enumerate(element_types):
        if element_type == 3:
            raw = np.array(node_tags_per_element[i], dtype=int).reshape(-1, 4)
            quad_elements = tag_to_idx[raw] + 1   # remapeia para índices sequenciais, 1-based

    # Finalizar a API do Gmsh
    gmsh.finalize()

    # Exibir informações sobre a malha quadrangular gerada
    # print("Nós da malha:\n", node_coords)
    # print("Conectividades dos elementos quadrangulares:\n", quad_elements)

    # Filtrar nós que estão dentro dos limites especificados
    nodes_in_faceX1 = np.where(
        (node_coords[:, 0] >= 0-(size/8)) & (node_coords[:, 0] <= 0+(size/8))  # Dentro do intervalo x
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

    nodes_in_faceX2 = np.where(
        (node_coords[:, 0] >= lx-(size/8)) & (node_coords[:, 0] <= lx+(size/8))  # Dentro do intervalo x
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

      # Filtrar nós que estão dentro dos limites especificados
    nodes_in_faceY1 = np.where(
        (node_coords[:, 1] >= 0-(size/8)) & (node_coords[:, 1] <= 0+(size/8))  # Dentro do intervalo y
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

    nodes_in_faceY1 = nodes_in_faceY1[np.argsort(node_coords[nodes_in_faceY1, 0])]

    nodes_in_faceY2 = np.where(
        (node_coords[:, 1] >= ly-(size/8)) & (node_coords[:, 1] <= ly+(size/8))  # Dentro do intervalo y
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

    nodes_in_faceY2 = nodes_in_faceY2[np.argsort(node_coords[nodes_in_faceY2, 0])]

    nodes_excit_right = np.where(
        (node_coords[:, 1] >= (ly/2)-(3*size + 1e-5)) & (node_coords[:, 1] <= (ly/2)+(3*size + 1e-5)) & (node_coords[:, 0] >= lx-(size/8)) & (node_coords[:, 0] <= lx+(size/8))  # Dentro do intervalo y metade da aresta
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

    nodes_excit_left = np.where(
        (node_coords[:, 1] >= (ly/2)-(3*size + 1e-5)) & (node_coords[:, 1] <= (ly/2)+(3*size + 1e-5)) & (node_coords[:, 0] >= 0-(size/8)) & (node_coords[:, 0] <= 0+(size/8))  # Dentro do intervalo y metade da aresta
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

    nodes_excit_top = np.where(
        (node_coords[:, 0] >= (lx/2)-(3*size + 1e-5)) & (node_coords[:, 0] <= (lx/2)+(3*size + 1e-5)) & (node_coords[:, 1] >= ly-(size/8)) & (node_coords[:, 1] <= ly+(size/8))  # Dentro do intervalo y metade da aresta
    )[0]  # np.where retorna uma tupla, então usamos [0] para obter o array de índices

    # Criação do objeto vtkPoints para armazenar as coordenadas dos nós
    points = vtk.vtkPoints()
    for coord in node_coords:
        points.InsertNextPoint(coord[0], coord[1], coord[2])

    # Criar o objeto vtkUnstructuredGrid para armazenar a malha
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)

    # Criar e adicionar elementos quadrangulares ao vtkUnstructuredGrid
    for quad in quad_elements:
        quad_vtk = vtk.vtkQuad()
        for i in range(4):
            quad_vtk.GetPointIds().SetId(i, quad[i] - 1)  # Subtrair 1 devido à indexação baseada em 0 no VTK
        ugrid.InsertNextCell(quad_vtk.GetCellType(), quad_vtk.GetPointIds())

    # Escrever a malha em um arquivo VTU
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName("quad_mesh.vtu")
    writer.SetInputData(ugrid)
    writer.Write()

    print("Arquivo VTU gerado com sucesso!")

    return node_coords, quad_elements, nodes_in_faceX1, nodes_in_faceX2, nodes_in_faceY1, nodes_in_faceY2, nodes_excit_left, nodes_excit_right, nodes_excit_top