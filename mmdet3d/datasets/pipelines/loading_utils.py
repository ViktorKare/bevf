import os
import numpy as np
import torch
import math
import random
__all__ = ["load_augmented_point_cloud", "reduce_LiDAR_beams", "rotation_matrix"]

# LiDAR reduction of beams: Credit BEVFusion: https://github.com/mit-han-lab/bevfusion

def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    # NOTE: following Tianwei's implementation, it is hard coded for nuScenes
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    # NOTE: path definition different from Tianwei's implementation.
    tokens = path.split("/")
    vp_dir = "_VIRTUAL" if reduce_beams == 32 else f"_VIRTUAL_{reduce_beams}BEAMS"
    seg_path = os.path.join(
        *tokens[:-3],
        "virtual_points",
        tokens[-3],
        tokens[-2] + vp_dir,
        tokens[-1] + ".pkl.npy",
    )
    assert os.path.exists(seg_path)
    data_dict = np.load(seg_path, allow_pickle=True).item()

    virtual_points1 = data_dict["real_points"]
    # NOTE: add zero reflectance to virtual points instead of removing them from real points
    virtual_points2 = np.concatenate(
        [
            data_dict["virtual_points"][:, :3],
            np.zeros([data_dict["virtual_points"].shape[0], 1]),
            data_dict["virtual_points"][:, 3:],
        ],
        axis=-1,
    )

    points = np.concatenate(
        [
            points,
            np.ones([points.shape[0], virtual_points1.shape[1] - points.shape[1] + 1]),
        ],
        axis=1,
    )
    virtual_points1 = np.concatenate(
        [virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1
    )
    # note: this part is different from Tianwei's implementation, we don't have duplicate foreground real points.
    if len(data_dict["real_points_indice"]) > 0:
        points[data_dict["real_points_indice"]] = virtual_points1
    if virtual:
        virtual_points2 = np.concatenate(
            [virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1
        )
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)
    return points
def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    # print(pts.size())
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta)
    phi = torch.atan2(pts[:, 1], pts[:, 0])

    top_ang = 0.1862
    down_ang = -0.5353

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang

    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275
    # beam_range = [1, 0.18, 0.15, 0.13, 0.11, 0.085, 0.065, 0.03, 0.01, -0.01, -0.03, -0.055, -0.08, -0.105, -0.13, -0.155, -0.18, -0.205, -0.228, -0.251, -0.275,
    #                -0.295, -0.32, -0.34, -0.36, -0.38, -0.40, -0.425, -0.45, -0.47, -0.49, -0.52, -0.54]

    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)
    if reduce_beams_to == 16:
        for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        for id in [7, 9, 11, 13]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    # [?] pick the 14th beam
    elif reduce_beams_to == 1:
        chosen_beam_id = 9
        mask = (theta < (beam_range[chosen_beam_id - 1] - 0.012)) * (
            theta > (beam_range[chosen_beam_id] - 0.012)
        )
    elif reduce_beams_to == 0:
        mask = mask.bool()
    else:
        raise NotImplementedError
    # points = copy.copy(pts)
    points = pts[mask]
    # print(points.size())
    return points.numpy()

def simulate_close_lidar_occlusions(pts, scene_token='x' ,occlusion_deg=10):
    occlusion_deg = occlusion_deg/2 #fix left and right of occlusion angle. Caused by abs.
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    occlusion_dir = (get_seed_from_token(scene_token) % 360) - 180
    theta = torch.atan2(pts[:, 0], pts[:, 1])
    theta = ((occlusion_dir * np.pi / 180) - theta)
    theta = np.absolute(theta%(2*math.pi) - math.pi)
    mask = theta >= np.absolute(occlusion_deg * np.pi / 180)
    points=pts[mask]

    return points.numpy()

def simulate_missing_lidar_points(pts, percentage_of_lidar_points_to_remove, seed='x'):
    torch.manual_seed(seed)
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    size = pts.size(0)
    nr_of_samps = round(size*((100-percentage_of_lidar_points_to_remove)/100))
    permutations = torch.randperm(size)
    ind = permutations[:nr_of_samps]
    points = pts[ind]
    return points.numpy()

def rotation_matrix(theta1, theta2, theta3):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    matrix=np.array([[c2*c3, -c2*s3, s2],
    [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
    [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])

    return matrix



def get_seed_from_front_cam_name(s):
    return int(s.split("__CAM_FRONT__")[1].split(".jpg")[0])%(2**18)
def get_seed_from_token(s):
    return int(str(int.from_bytes(s.encode(), 'little'))[-6:])