# .obj parser
import torch
class Mesh:
    def __init__(self, vertices_pos, face_vertices_indices):
        # vertices_pos [vertices, 3 xyz]
        self.vertices_pos = vertices_pos
        # face_vertices_indices [faces, 3 vertice index]
        self.face_vertices_indices = face_vertices_indices

        # preprocess field
        # vertices_degree [vertices]
        self.vertices_degree = None
        # vertices_degree [face, neighbor_faces]
        self.face_neighbor_faces = None
        self.build_auxiliary_data()
        pass

    def build_auxiliary_data(self):
        # edge_dict {(edge vertex 1, edge vertex 2): numbering}
        edge_dict = {}
        counter = 0
        for face in self.face_vertices_indices:
            for i in range(3):
                key = sorted((face[i], face[(i+1) % 3]))
                if edge_dict.get(key) != None:
                    pass
                else:
                    edge_dict[key] = counter
        pass

    def extract_features(self):
        pass

