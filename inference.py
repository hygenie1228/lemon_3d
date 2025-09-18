import torch
import numpy as np
import os
import open3d as o3d
import argparse
import pdb
import yaml
from tools.models.model_LEMON_d import LEMON
from tools.models.LEMON_noCur import LEMON_wocur
from PIL import Image
from dataset_utils.dataset_3DIR import _3DIR
from tools.utils.build_layer import create_mesh, build_smplh_mesh, Pelvis_norm
from tools.utils.evaluation import visual_pred, generate_proxy_sphere
from dataset_utils.dataset_3DIR import img_normalize, pc_normalize
from tools.utils.mesh_sampler import get_sample
from torch.utils.data import DataLoader

import trimesh


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def inference_batch(opt, dict, val_loader, model, device):

    checkpoint = torch.load(dict['best_checkpoint'], map_location=device)

    model.load_state_dict(checkpoint)
    model = model.to(device)
    model = model.eval()
    contact_color = np.array([255.0, 191.0, 0.])

    def save_path(path):
        file_name = path.split('/')[-1]
        obj, aff = file_name.split('_')[0], file_name.split('_')[1]
        hm_save_folder = dict['contact_result_folder'] + obj + '/' + aff
        spatial_folder = dict['spatial_result_folder'] + obj
        if not os.path.exists(hm_save_folder):
            os.makedirs(hm_save_folder)
        if not os.path.exists(spatial_folder):
            os.makedirs(spatial_folder)
        file_name = file_name.split('.')[0] + '.ply'
        hm_save_file = os.path.join(hm_save_folder, file_name)
        spatial_save_file = os.path.join(spatial_folder, file_name)
        return hm_save_file, spatial_save_file
    
    with torch.no_grad():
        for i, data_info in enumerate(val_loader):
            img = data_info['img'].to(device)
            B = img.size(0)
            img_paths = data_info['img_path']
            pts_paths = data_info['Pts_path']
            H, face = build_smplh_mesh(data_info['human'])
            H = H.to(device)
            H, pelvis = Pelvis_norm(H, device)
            O = data_info['Pts'].float().to(device)
            C_h = data_info['hm_curvature'].to(device)
            C_o = data_info['obj_curvature'].to(device)
            pre_contact, pre_affordance, pre_spatial, _, _ = model(img, O, H, C_h, C_o)
            pre_affordance = pre_affordance.cpu().detach().numpy()
            contact_fine = pre_contact[-1]

            for j in range(B):
                vertices = H[j].detach().cpu().numpy()
                spatial_center = pre_spatial[j].detach().cpu().numpy()
                spatial_sphere = generate_proxy_sphere(spatial_center, pts_paths[j])
                colors = np.array([255.0, 255.0, 255.0])[None, :].repeat(6890, axis=0)
                contact_id = torch.where(contact_fine[j] > 0.5)[0].cpu()
                contact_id = np.asarray(contact_id)
                colors[contact_id] = contact_color
                colors = colors / 255.0

                contact_mesh = create_mesh(vertices=vertices, faces=face, colors=colors)
                mesh_save_path, spatial_save_path = save_path(data_info['img_path'][j])
                o3d.io.write_triangle_mesh(mesh_save_path, contact_mesh)
                o3d.io.write_triangle_mesh(spatial_save_path, spatial_sphere)                
                visual_pred(img_paths[j], pre_affordance[j], pts_paths[j], dict['affordance_result_folder'])

def mask_img(img_path, mask_path):
    img = Image.open(img_path).convert('RGB')

    mask_path1 = img_path.replace("image.jpg", "person_mask.png")
    mask_path2 = img_path.replace("image.jpg", "obj_mask.png")

    mask1 = Image.open(mask_path1).convert('RGB')
    mask2 = Image.open(mask_path2).convert('RGB')

    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    mask = np.maximum(mask1, mask2)

    img, mask = np.asarray(img), np.asarray(mask)
    back_ground = np.array([0, 0, 0])
    mask_bi = np.all(mask == back_ground, axis=2)
    mask_img = np.ones_like(mask)
    mask_img[mask_bi] = back_ground
    masked_img = img * mask_img
    masked_img = Image.fromarray(masked_img)
    return masked_img

def extract_point_file(path):
    with open(path,'r') as f:
        coordinates = []
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.strip(' ')
        data = line.split(' ')
        coordinate = [float(x) for x in data]
        coordinates.append(coordinate)
    data_array = np.array(coordinates)
    points_coordinates = data_array[:, 0:3]
    affordance_label = data_array[: , 3:]

    return points_coordinates, affordance_label

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

# def get_human_param(path):
#     smplh_param = {}
#     param_data = np.load(path, allow_pickle=True)
#     smplh_param['shape'] = torch.tensor(param_data['shape']).unsqueeze(0)
#     smplh_param['transl'] = torch.tensor(param_data['transl']).unsqueeze(0)
#     smplh_param['body_pose'] = torch.tensor(param_data['body_pose']).reshape(1, 21, 3, 3)
#     smplh_param['left_hand_pose'] = torch.tensor(param_data['left_hand_pose']).reshape(1, 15, 3, 3)
#     smplh_param['right_hand_pose'] = torch.tensor(param_data['right_hand_pose']).reshape(1, 15, 3, 3)
#     smplh_param['global_orient'] = torch.tensor(param_data['global_orient']).reshape(1, 3, 3)

#     return smplh_param

def get_human_param(path):
    import json

    smplh_param = {}
    # param_data = np.load(path, allow_pickle=True)
    # smplh_param['shape'] = torch.tensor(param_data['shape']).unsqueeze(0)
    # smplh_param['transl'] = torch.tensor(param_data['transl']).unsqueeze(0)
    # smplh_param['body_pose'] = torch.tensor(param_data['body_pose']).reshape(1, 21, 3, 3)
    # smplh_param['left_hand_pose'] = torch.tensor(param_data['left_hand_pose']).reshape(1, 15, 3, 3)
    # smplh_param['right_hand_pose'] = torch.tensor(param_data['right_hand_pose']).reshape(1, 15, 3, 3)
    # smplh_param['global_orient'] = torch.tensor(param_data['global_orient']).reshape(1, 3, 3)

    # smplh_param['shape'] = smplh_param['shape'][:, :10]

    param_data = json.load(open(path))
    smplh_param['shape'] = torch.tensor(param_data['shape'])
    smplh_param['transl'] = torch.tensor(param_data['cam_trans'])   
    smplh_param['body_pose'] = batch_rodrigues(torch.tensor(param_data['body_pose']).reshape(-1, 3)).reshape(1, 21, 3, 3)
    smplh_param['left_hand_pose'] = batch_rodrigues(torch.tensor(param_data['lhand_pose']).reshape(-1, 3)).reshape(1, 15, 3, 3)
    smplh_param['right_hand_pose'] = batch_rodrigues(torch.tensor(param_data['rhand_pose']).reshape(-1, 3)).reshape(1, 15, 3, 3)
    smplh_param['global_orient'] = batch_rodrigues(torch.tensor(param_data['root_pose']).reshape(-1, 3)).reshape(1, 3, 3)
    
    smplh_param['shape'] = torch.cat((smplh_param['shape'], torch.zeros((1,6))), -1)
    return smplh_param

def inference_single(model, opt, dict, device, outdir):

    checkpoint = torch.load(dict['best_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model = model.eval()

    #load image
    img_size = (224, 224)
    img = mask_img(opt.img_path, opt.mask)
    Img = img.resize(img_size)
    I = img_normalize(Img).unsqueeze(0).to(device)

    #load human
    human_param = get_human_param(opt.human_param_path)
    import pdb; pdb.set_trace()
    vertices, face = build_smplh_mesh(human_param)
    mesh_sampler = get_sample(device=None)
    hm_curvature = np.load(opt.C_h, allow_pickle=True)
    hm_curvature = torch.from_numpy(hm_curvature).to(torch.float32)
    C_h = mesh_sampler.downsample(hm_curvature).unsqueeze(0).to(device)
    vertices = vertices.to(device)
    H, pelvis = Pelvis_norm(vertices, device)
    H = H.to(device)

    #load object
    Pts, affordance_label = extract_point_file(opt.object)
    Pts = pc_normalize(Pts)
    Pts = Pts.transpose()
    O = torch.from_numpy(Pts).float().unsqueeze(0).to(device)
    C_o = np.load(opt.C_o, allow_pickle=True)
    C_o = torch.from_numpy(C_o).to(torch.float32).unsqueeze(dim=-1).unsqueeze(dim=0).to(device)
    pre_contact, pre_affordance, pre_spatial, _, _ = model(I, O, H, C_h, C_o)
    contact_fine = pre_contact[-1]
    pre_affordance = pre_affordance[0].cpu().detach().numpy()

    #save
    contact_color = np.array([255.0, 191.0, 0.])
    vert = H.detach().cpu().numpy()
    spatial_center = pre_spatial[0].detach().cpu().numpy()
    spatial_sphere = generate_proxy_sphere(spatial_center, opt.object)
    colors = np.array([255.0, 255.0, 255.0])[None, :].repeat(6890, axis=0)
    contact_id = torch.where(contact_fine[0] > 0.5)[0].cpu()
    contact_id = np.asarray(contact_id)
    colors[contact_id] = contact_color
    colors = colors / 255.0

    contact_mesh = create_mesh(vertices=vert[0], faces=face, colors=colors)
    mesh_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_contact.ply')
    spatial_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_spatial.ply')
    o3d.io.write_triangle_mesh(mesh_save_path, contact_mesh)
    o3d.io.write_triangle_mesh(spatial_save_path, spatial_sphere)

    reference_color = np.array([255, 0, 0])
    back_color = np.array([190, 190, 190])
    pred_point = o3d.geometry.PointCloud()
    pred_point.points = o3d.utility.Vector3dVector(O[0].detach().cpu().numpy().transpose())
    pred_color = np.zeros((O.shape[2],3))
    for i, pred in enumerate(pre_affordance):
        scale_i = pred
        pred_color[i] = (reference_color-back_color) * scale_i + back_color
    pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)
    object_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_object.ply')
    o3d.io.write_point_cloud(object_save_path, pred_point)

def inference_single_wo_curvature(model, opt, dict, device, outdir):

    checkpoint = torch.load(dict['best_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model = model.eval()

    from glob import glob
    from tqdm import tqdm
    sample_list = sorted(glob(f"Data/open3dhoi_p1/*"))
    sample_list = [sample.split("/")[-1] for sample in sample_list]

    for sample in tqdm(sample_list):
        opt.img_path = f"/home/namhj/Lemon3D/Data/open3dhoi_p1/{sample}/image.jpg"
        opt.human_param_path = f"/home/namhj/Lemon3D/Data/open3dhoi_p1/{sample}/smplh_parameters.json"
        opt.object = f"/home/namhj/Lemon3D/Data/open3dhoi_p1/{sample}/obj_pcd_h_align.obj"
        opt.mask = None

        #load image
        img_size = (224, 224)
        img = mask_img(opt.img_path, opt.mask)
        Img = img.resize(img_size)
        I = img_normalize(Img).unsqueeze(0).to(device)

        #load human
        human_param = get_human_param(opt.human_param_path) ## !!!!
        # human_param
        # ['shape', 'transl', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'global_orient']
        # 'shape': (1, 16)
        # 'transl': (1, 3)
        # 'body_pose': (1, 21, 3, 3)
        # 'left_hand_pose': (1, 15, 3, 3)
        # 'global_orient': (1, 3, 3)


        vertices, face = build_smplh_mesh(human_param)
        vertices = vertices.to(device)
        H, pelvis = Pelvis_norm(vertices, device)
        H = H.to(device)

        #load object
        # Pts, affordance_label = extract_point_file(opt.object)
        mesh = trimesh.load(opt.object)
        Pts = mesh.vertices
        affordance_label = None
        Pts = pc_normalize(Pts)
        Pts = Pts.transpose()
        O = torch.from_numpy(Pts).float().unsqueeze(0).to(device)
        pre_contact, pre_affordance, pre_spatial, _, _ = model(I, O, H)
        contact_fine = pre_contact[-1]
        pre_affordance = pre_affordance[0].cpu().detach().numpy()

        import pdb; pdb.set_trace()

        #save
        contact_color = np.array([255.0, 191.0, 0.])
        vert = H.detach().cpu().numpy()
        spatial_center = pre_spatial[0].detach().cpu().numpy()
        # spatial_sphere = generate_proxy_sphere(spatial_center, opt.object)
        colors = np.array([255.0, 255.0, 255.0])[None, :].repeat(6890, axis=0)
        contact_id = torch.where(contact_fine[0] > 0.5)[0].cpu()
        contact_id = np.asarray(contact_id)
        colors[contact_id] = contact_color
        colors = colors / 255.0

        contact_mesh = create_mesh(vertices=vert[0], faces=face, colors=colors)
        mesh_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_contact.ply')
        spatial_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_spatial.ply')
        o3d.io.write_triangle_mesh(mesh_save_path, contact_mesh)
        # o3d.io.write_triangle_mesh(spatial_save_path, spatial_sphere)

        reference_color = np.array([255, 0, 0])
        back_color = np.array([190, 190, 190])
        pred_point = o3d.geometry.PointCloud()
        pred_point.points = o3d.utility.Vector3dVector(O[0].detach().cpu().numpy().transpose())
        pred_color = np.zeros((O.shape[2],3))
        for i, pred in enumerate(pre_affordance):
            scale_i = pred
            pred_color[i] = (reference_color-back_color) * scale_i + back_color
        pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)
        object_save_path = os.path.join(outdir, opt.img_path.split('/')[-1].split('.')[0]+'_object.ply')
        o3d.io.write_point_cloud(object_save_path, pred_point)
        import pdb; pdb.set_trace()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu to run')
    parser.add_argument('--yaml', type=str, default='config/infer.yaml', help='infer setting')
    #single
    parser.add_argument('--img_path', type=str, default='Demo/Backpack_carry_demo.jpg', help='single test image')
    parser.add_argument('--mask', type=str, default='Demo/Backpack_carry_mask.png', help='single test mask')
    parser.add_argument('--human_param_path', type=str, default='Demo/Backpack_human_demo.npz', help='single test human')
    parser.add_argument('--object', type=str, default='Demo/Backpack_object_demo.txt', help='single test object')
    parser.add_argument('--C_o', type=str, default='Demo/Backpack_curvature.pkl', help='single test object curvature')
    parser.add_argument('--C_h', type=str, default='Demo/Human_curvature.pkl', help='single test object curvature')
    parser.add_argument('--outdir', type=str, default='Demo/output', help='single test ouput dir')

    opt = parser.parse_args()
    dict = read_yaml(opt.yaml)
    if opt.use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    curvature = dict['curvature']
    print('whether to use curvature:', curvature)
    if curvature:
        model = LEMON(dict['emb_dim'], run_type='infer', device=device)
    else:
        model = LEMON_wocur(dict['emb_dim'], run_type='infer', device=device)
    #batch

    infer_type = dict['infer_type']
    if infer_type == 'batch':
        val_dataset = _3DIR(dict['val_image'], dict['val_pts'], dict['human_3DIR'], dict['behave'], mode='val')
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
        inference_batch(opt, dict, val_loader, model, device)
    elif infer_type == 'single':
        if curvature:
            inference_single(model, opt, dict, device, opt.outdir)
        else:
            inference_single_wo_curvature(model, opt, dict, device, opt.outdir)