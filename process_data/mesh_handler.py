from threading import local
from process_data.custom_types import *
from process_data.constants import CACHE_ROOT, DATA_ROOT
from process_data import mesh_utils
import torch
import os
from process_data.options import Options
from process_data import files_utils


class MeshHandler:
    # This data structure only store raw mesh (vertices, faces) and processed data structure (MeshDS)
    # FIXME upsampler should only be used as a function

    # vs [batch, vertex size, 3]
    # faces [face size, 3]

    def __init__(self, path_or_mesh: Union[str, T_Mesh], opt: Options):
        self.opt = opt
        if type(path_or_mesh) is str:
            self.raw_mesh = mesh_utils.load_mesh(path_or_mesh)
        else:
            self.raw_mesh: T_Mesh = path_or_mesh

        if self.raw_mesh[0].ndim == 3:
            # [batch, vertices size, 3 xyz]
            self.mesh_ds = mesh_utils.MeshDS((self.raw_mesh[0][0], self.raw_mesh[1]))
        else:
            self.mesh_ds = mesh_utils.MeshDS(self.raw_mesh)
        # Placeholder for local_axes, its origin and local axes relative to other vertices according to face topology
        self.valid_axes = False

    @property
    def vs(self) -> T:
        # [batch, vertices, 3]
        return self.raw_mesh[0]

    @property
    def faces(self) -> T:
        # [faces, 3]
        return self.raw_mesh[1]

    @vs.setter
    def vs(self, vs_new):
        self.valid_axes = False
        self.raw_mesh = vs_new, self.faces

    @faces.setter
    def faces(self, faces_new):
        self.raw_mesh = self.vs, faces_new

    @property
    def gfmm(self) -> T:
        return self.mesh_ds.gfmm

    @property
    def device(self) -> D:
        return self.vs.device

    @property
    def mesh_copy(self) -> T_Mesh:
        return self.vs.detach().cpu().clone(), self.faces.detach().cpu().clone()

    def __call__(self, z: Union[T, float] = 0, noise_before: bool = False) -> T:
        return self.extract_features(z, noise_before)

    def __len__(self) -> int:
        return self.faces.shape[0]

    def repeat(self, batch):
        self.raw_mesh = (self.raw_mesh[0].repeat(batch, 1, 1), self.raw_mesh[1])

    def ensure_local_axes(self):
        if self.valid_axes == False:
            self.local_axes = self.extract_local_axes()
            self.valid_axes = True

    def copy(self, batch=None):
        mesh = self.vs.clone().detach(), self.faces.clone().detach()
        new_mesh_handler = MeshHandler(mesh, self.opt)
        if batch != None:
            new_mesh_handler.repeat(batch)
        new_mesh_handler.ensure_local_axes()
        return new_mesh_handler

    # in place
    def to(self, device: D):
        self.raw_mesh = (self.vs.to(device), self.faces.to(device))
        self.mesh_ds.to(device)
        if self.valid_axes == True:
            self.local_axes = self.local_axes[0].to(device), self.local_axes[1].to(device)
        return self

    def detach(self):
        self.vs = self.vs.detach()
        self.local_axes = self.local_axes[0].detach(), self.local_axes[1].detach()
        return self

    # in place
    # def upsample(self, in_place: bool = True):
    #     if not in_place:
    #         return self.copy().upsample()
    #     self.raw_mesh = self.upsampler(self.raw_mesh)
    #     self.vs = self.opt.scale_vs_factor * self.vs
    #     return self

    def __add__(self, deltas: T):
        mesh = self.copy().to(self.device)
        mesh += deltas
        return mesh

    def projet_displacemnets(self, deltas: T) -> T:
        # deltas [1, 3, 960]
        _, local_axes = self.extract_local_axes()
        # deltas [face size, xyz]
        batch = deltas.shape[0]
        deltas = deltas.permute(0, 2, 1).reshape(batch, -1, 3)
        # # linear projection according to local axis, this is displacement vector
        # # global_vecs [face * 3 vertex(each face vertex), 3 xyz]??
        global_vecs = torch.einsum('bfsad,bfa->bfsd', [local_axes, deltas]).view(batch, -1, 3)
        # deltas = deltas.squeeze(0).t().reshape(-1, 3)
        # deltas [960, 3]
        # linear projection according to local axis, this is displacement vector
        # global_vecs [face * 3 vertex(each face vertex), 3 xyz]??
        # global_vecs = torch.einsum('fsad,fa->fsd', [local_axes, deltas])
        # global_vecs [960, 3, 3]
        # global_vecs = global_vecs.view(-1, 3)
        # global_vecs [2880, 3]
        # vs_deltas [vertex, face, 3xyz]
        # global_vecs[:, self.mesh_ds.vertex2faces] [batch, vertex, vertex's each face, 3 xyz]
        vs_deltas = global_vecs[:, self.mesh_ds.vertex2faces] * self.mesh_ds.vertex2faces_ma[None, :, :, None]
        vs_deltas = vs_deltas.sum(2) / self.mesh_ds.vs_degree[None, :, None]
        return vs_deltas

    def __iadd__(self, deltas: T):
        self.vs = self.vs + self.projet_displacemnets(deltas)
        return self

    @staticmethod
    def get_local_axes(mesh: T_Mesh) -> Tuple[T, T]:
        vs, faces = mesh
        batch = vs.shape[0]
        _, normals = mesh_utils.compute_face_areas(mesh)
        # vs_faces [batch, face size, 3 vertex, 3 xyz]
        vs_faces = vs[:, faces]
        # origins 3 vertex * [batch, face size, 1, 3xyz]
        origins = [((vs_faces[:, :, i] + vs_faces[:, :, (i + 1) % 3]) / 2).unsqueeze(2) for i in range(3)]
        # x 3 vertex * [batch, face size, 3xyz]
        x = [(vs_faces[:, :, (i + 1) % 3] - vs_faces[:, :, i]) for i in range(3)]
        y = [torch.cross(x[i], normals, dim=2) for i in range(3)]  # 3 f 3
        # axes 3 vertex * [batch, face size, 3 xyznormal, 3 xyz]
        axes = [torch.cat(list(map(lambda v: v.unsqueeze(2), [x[i], y[i], normals])), 2) for i in range(3)]
        # axes [batch, face size, 3 vertex, 3 xyznormal, 3 xyz]
        axes = torch.cat(axes, dim=2).view(batch, -1, 3, 3, 3)
        # origins [batch, face size, 3 vertex, 3xyz]
        origins = torch.cat(origins, dim=2).view(batch, -1, 3, 3)
        # normalize xyz
        axes = axes / torch.norm(axes, p=2, dim=4)[:, :, :, :, None]
        return origins, axes

    def extract_local_axes(self) -> Tuple[T, T]:
        if self.valid_axes:
            return self.local_axes
        self.local_axes = self.get_local_axes(self.raw_mesh)
        self.valid_axes = True
        return self.local_axes

    def extract_features(self, z: Union[T, float], noise_before: bool) -> TS:

        def extract_local_cords() -> T:
            vs_ = self.vs
            if self.opt.noise_before and type(z) is T:
                if self.opt.fix_vs_noise:
                    vs_ = vs_ + z
                else:
                    vs_ += z
                if self.opt.update_axes:
                    self.vs = vs_
                    origins, local_axes = self.extract_local_axes()
                else:
                    origins, local_axes = self.get_local_axes((vs_, self.faces))
            else:
                origins, local_axes = self.extract_local_axes()

            # vs_[self.mesh_ds.face2points] [batch, face size, 3 neighbor face size, 3 xyz]
            global_cords = vs_[:, self.mesh_ds.face2points] - origins
            local_cords = torch.einsum('bfsd,bfsad->bfsa', [global_cords, local_axes])
            return local_cords, vs_

        def get_edge_lengths() -> T:
            nonlocal vs
            # vs_faces [batch, face size, 3 vertex, 3 xyz]
            vs_faces = vs[:, self.faces]
            # lengths 3 * [batch, face size, 1]
            lengths = list(map(lambda i: (vs_faces[:, :, (i + 1) % 3] - vs_faces[:, :, i]).norm(p=2, dim=2).unsqueeze(2), range(3)))
            # lengths [batch, face size, 3]
            lengths = torch.cat(lengths, 2)
            # lengths [batch, face size, 3, 1]
            return lengths.unsqueeze(3)

        opposite_vs, vs = extract_local_cords()
        batch = vs.shape[0]
        edge_lengths = get_edge_lengths()
        fe = torch.cat((opposite_vs, edge_lengths), 3).view(batch, len(self), -1).permute(0, 2, 1)
        if not noise_before:
            fe = fe + z
        # return torch.rand(1, 12, len(self), device=self.device)
        # [batch, features, face size]
        return fe

    def export(self, path):
        if not self.opt.debug:
            mesh_utils.export_mesh((self.vs / (self.opt.scale_vs_factor ** self.level), self.faces), path)

class MeshHandlerOrigin:

    _mesh_dss: List[Union[mesh_utils.MeshDS, N]] = []
    _upsamplers: List[Union[mesh_utils.Upsampler, N]] = []

    def __init__(self, path_or_mesh: Union[str, T_Mesh], opt: Options, level: int, local_axes: Union[N, TS] = None):
        self.level = level
        self.opt = opt
        if type(path_or_mesh) is str:
            self.raw_mesh = mesh_utils.load_mesh(path_or_mesh)
        else:
            self.raw_mesh: T_Mesh = path_or_mesh
        self.update_ds(self.raw_mesh, level)
        self.valid_axes = local_axes is not None
        if local_axes is None:
            self.local_axes = self.extract_local_axes()
        else:
            self.local_axes = local_axes

    @staticmethod
    def reset():
        MeshHandlerOrigin._mesh_dss = []
        MeshHandlerOrigin._upsamplers = []

    # in place
    def to(self, device: D):
        self.raw_mesh = (self.vs.to(device), self.faces.to(device))
        self.ds.to(device)
        self.upsampler.to(device)
        self.local_axes = self.local_axes[0].to(device), self.local_axes[1].to(device)
        return self

    def detach(self):
        self.vs = self.vs.detach()
        self.local_axes = self.local_axes[0].detach(), self.local_axes[1].detach()
        return self

    @staticmethod
    def pad_ds(level: int):
        for i in range(len(MeshHandlerOrigin._mesh_dss), level + 1):
            MeshHandlerOrigin._mesh_dss.append(None)
            MeshHandlerOrigin._upsamplers.append(None)

    def fill_ds(self, mesh: T_Mesh, level: int):
        MeshHandlerOrigin._mesh_dss[level] = mesh_utils.MeshDS(mesh).to(mesh[0].device)
        MeshHandlerOrigin._upsamplers[level] = mesh_utils.Upsampler(mesh).to(mesh[0].device)

    def update_ds(self, mesh: T_Mesh, level: int):
        self.pad_ds(level)
        if MeshHandlerOrigin._mesh_dss[level] is None:
            self.fill_ds(mesh, level)

    # in place
    def upsample(self, in_place: bool = True):
        if not in_place:
            return self.copy().upsample()
        self.raw_mesh = self.upsampler(self.raw_mesh)
        self.vs = self.opt.scale_vs_factor * self.vs
        self.level += 1
        self.update_ds(self.raw_mesh, self.level)
        return self

    def __add__(self, deltas: T):
        mesh = self.copy()
        mesh += deltas
        return mesh

    def projet_displacemnets(self, deltas: T) -> T:
        _, local_axes = self.extract_local_axes()
        # deltas [face size, xyz]
        local_axes = local_axes
        deltas = deltas.squeeze(0).t().reshape(-1, 3)
        # linear projection according to local axis, this is displacement vector
        # global_vecs [face * 3 vertex(each face vertex), 3 xyz]??
        global_vecs = torch.einsum('fsad,fa->fsd', [local_axes, deltas]).view(-1, 3)
        # vs_deltas [vertex, face, 3xyz]
        vs_deltas = global_vecs[self.ds.vertex2faces] * self.ds.vertex2faces_ma[:, :, None]
        vs_deltas = vs_deltas.sum(1) / self.ds.vs_degree[:, None]
        return vs_deltas

    def __iadd__(self, deltas: T):
        self.vs = self.vs + self.projet_displacemnets(deltas)
        return self

    @staticmethod
    def get_local_axes(mesh: T_Mesh) -> Tuple[T, T]:
        vs, faces = mesh
        _, normals = mesh_utils.compute_face_areas(mesh)
        # vs_faces [face size, 3 vertex, 3 xyz]
        vs_faces = vs[faces]
        origins = [((vs_faces[:, i] + vs_faces[:, (i + 1) % 3]) / 2).unsqueeze(1) for i in range(3)]
        # x 3 vertex * [1, face size, 3xyz]
        x = [(vs_faces[:, (i + 1) % 3] - vs_faces[:, i]) for i in range(3)]
        y = [torch.cross(x[i], normals) for i in range(3)]  # 3 f 3
        # axes 3 vertex * [1, 3 xyznormal, face size, 3 xyz]
        axes = [torch.cat(list(map(lambda v: v.unsqueeze(1), [x[i], y[i], normals])), 1) for i in range(3)]
        # axes [face size, 3 vertex, 3 xyznormal, 3 xyz]
        axes = torch.cat(axes, dim=1).view(-1, 3, 3, 3)
        origins = torch.cat(origins, dim=1).view(-1, 3, 3)
        axes = axes / torch.norm(axes, p=2, dim=3)[:, :, :, None]
        return origins, axes

    def extract_local_axes(self) -> Tuple[T, T]:
        if self.valid_axes:
            return self.local_axes
        self.local_axes = self.get_local_axes(self.raw_mesh)
        self.valid_axes = True
        return self.local_axes

    def extract_features(self, z: Union[T, float], noise_before: bool) -> TS:

        def extract_local_cords() -> T:
            vs_ = self.vs
            if self.opt.noise_before and type(z) is T:
                if self.opt.fix_vs_noise:
                    vs_ = vs_ + z
                else:
                    vs_ += z
                if self.opt.update_axes:
                    self.vs = vs_
                    origins, local_axes = self.extract_local_axes()
                else:
                    origins, local_axes = self.get_local_axes((vs_, self.faces))
            else:
                origins, local_axes = self.extract_local_axes()

            global_cords = vs_[self.ds.face2points] - origins
            local_cords = torch.einsum('fsd,fsad->fsa', [global_cords, local_axes])
            return local_cords, vs_

        def get_edge_lengths() -> T:
            nonlocal vs
            vs_faces = vs[self.faces]
            lengths = list(map(lambda i: (vs_faces[:, (i + 1) % 3] - vs_faces[:, i]).norm(2, 1).unsqueeze(1), range(3)))
            lengths = torch.cat(lengths, 1)
            return lengths.unsqueeze(2)

        opposite_vs, vs = extract_local_cords()
        edge_lengths = get_edge_lengths()
        fe = torch.cat((opposite_vs, edge_lengths), 2).view(len(self), -1).t()
        if not noise_before:
            fe = fe + z
        # return torch.rand(1, 12, len(self), device=self.device)
        return fe.unsqueeze(0)

    def export(self, path):
        if not self.opt.debug:
            mesh_utils.export_mesh((self.vs / (self.opt.scale_vs_factor ** self.level), self.faces), path)

    @property
    def vs(self) -> T:
        return self.raw_mesh[0]

    @vs.setter
    def vs(self, vs_new):
        self.valid_axes = False
        self.raw_mesh = vs_new, self.faces

    @property
    def faces(self) -> T:
        return self.raw_mesh[1]

    @faces.setter
    def faces(self, faces_new):
        self.raw_mesh = self.vs, faces_new

    @property
    def device(self) -> D:
        return self.vs.device

    # IMPORTANT
    @property
    def ds(self) -> mesh_utils.MeshDS:
        return MeshHandlerOrigin._mesh_dss[self.level]

    @property
    def upsampler(self) -> mesh_utils.Upsampler:
        return MeshHandlerOrigin._upsamplers[self.level]

    @property
    def gfmm(self) -> T:
        return self.ds.gfmm

    @property
    def mesh_copy(self) -> T_Mesh:
        return self.vs.detach().cpu().clone(), self.faces.detach().cpu().clone()

    def __call__(self, z: Union[T, float] = 0, noise_before: bool = False) -> T:
        return self.extract_features(z, noise_before)

    def __len__(self) -> int:
        return self.faces.shape[0]

    def copy(self):
        mesh = self.vs.clone(), self.faces.clone()
        if self.valid_axes:
            local_axes = self.local_axes[0].clone(), self.local_axes[1].clone()
        else:
            local_axes = None
        return MeshHandlerOrigin(mesh, self.opt, self.level, local_axes=local_axes)

class MeshInference(MeshHandlerOrigin):

    def __init__(self, mesh_name: str, path_or_mesh: Union[str, T_Mesh], opt: Options, level: int,
                 local_axes: Union[N, TS] = None):
        self.mesh_name = mesh_name
        super(MeshInference, self).__init__(path_or_mesh, opt, level, local_axes)

    def grow_add(self, deltas: T):
        displacemnets = self.projet_displacemnets(deltas)
        self.vs = self.vs + self.projet_displacemnets(deltas)
        return self, displacemnets

    def fill_ds(self, mesh: T_Mesh, level: int):
        if not self.load(level):
            super(MeshInference, self).fill_ds(mesh, level)
            self.save(level)

    def load(self, level: int) -> bool:
        cache = []
        if files_utils.load_pickle(self.cache_path(level), cache):
            cache = cache[0]
            MeshHandler._mesh_dss[level] = cache['mesh_dss'].to(self.device)
            MeshHandler._upsamplers[level] = cache['upsamplers'].to(self.device)
            return True
        return False

    def save(self, level: int):
        files_utils.save_pickle({'mesh_dss': self._mesh_dss[level].to(CPU),
                                 'upsamplers': self._upsamplers[level].to(CPU)}, self.cache_path(level))
        self._mesh_dss[level].to(self.device)
        self._upsamplers[level].to(self.device)

    def cache_path(self, level: int):
        return f'{CACHE_ROOT}/{self.mesh_name}/{self.mesh_name}_{level:02d}.pkl'

    def copy(self):
        mesh = self.vs.clone(), self.faces.clone()
        if self.valid_axes:
            local_axes = self.local_axes[0].clone(), self.local_axes[1].clone()
        else:
            local_axes = None
        return MeshInference(self.mesh_name, mesh, self.opt, self.level, local_axes=local_axes)

def load_template_mesh(opt: Options, level) -> Tuple[str, T_Mesh]:
    mesh_path = f'{DATA_ROOT}/{opt.mesh_name}/{opt.mesh_name}_template.obj'
    if not os.path.isfile(mesh_path):
        return opt.template_name, mesh_utils.load_real_mesh(opt.template_name, level)
    else:
        mesh = mesh_utils.scale_mesh(mesh_path, False, opt.mesh_name, 0)
        if level > 0:
            mesh_handler = MeshHandler(mesh, opt, 0)
            for i in range(level):
                mesh_handler.upsample()
            mesh = mesh_handler.raw_mesh
        return f'{opt.mesh_name}_template', mesh
