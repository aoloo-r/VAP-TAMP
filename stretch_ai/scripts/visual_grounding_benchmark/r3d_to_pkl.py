import torch
import pickle as pkl
import argparse
# You will need to ensure the r3d_loader is available in your environment.
# It can often be found within the Record3D SDK or related projects.
from a_star.dataset_class import R3DDataset 

# This function remains unchanged from your original script.
def get_xyz_coordinates(depth, pose, intrinsic):
    """
    Calculates the 3D world coordinates from a depth map, camera pose, and intrinsics.
    """
    _, height, width = depth.shape

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(width, device=depth.device),
        torch.arange(height, device=depth.device),
        indexing="xy",
    )

    x = (xs - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (ys - intrinsic[1, 2]) / intrinsic[1, 1]

    z = depth[0]

    camera_coords = torch.stack((x * z, y * z, z, torch.ones_like(z)), axis=-1)
    
    # Transform to world coordinates using the pose matrix
    world_coords = camera_coords @ pose.T

    return world_coords[..., :3], camera_coords[..., :3]

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Converts an .r3d file to a .pkl file, including frame timestamps and all required placeholder keys."
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input .r3d file."
    )
    args = parser.parse_args()

    # Initialize the data dictionary with all expected keys
    data = {
        'base_poses': [],
        'feats': [],
        'obs': [],
        'xyz': [],
        'world_xyz': [],
        'rgb': [],
        'depth': [],
        'camera_K': [],
        'camera_poses': [],
        'timestamps': [],
        'instance': [] # <-- FIXED: Added the missing 'instance' key
    }

    print(f"Loading dataset from {args.input_file}...")
    # The R3DDataset class must yield the timestamp as the third element.
    # The format should be (image, depth, timestamp, intrinsics, camera_pose)
    dataset = R3DDataset(path=args.input_file, subsample_freq=10)
    
    for item in dataset:
        # Unpack the data from the dataset loader
        image, depth, timestamp, intrinsics, camera_pose = item

        # Your original processing
        image_uint8 = (image * 255).to(torch.uint8).permute(1, 2, 0)
        
        # Appending real data to the dictionary
        data['rgb'].append(image_uint8)
        data['depth'].append(depth[0])
        data['camera_K'].append(intrinsics)
        data['camera_poses'].append(camera_pose)
        data['timestamps'].append(timestamp)

        # Appending placeholders to match the required format
        data['base_poses'].append(None)
        data['world_xyz'].append(None)
        data['feats'].append(None)
        data['obs'].append(None)
        data['xyz'].append(None) # Added to be thorough, matching original script
        data['instance'].append(None) # <-- FIXED: Added the placeholder value

    output_filename = args.input_file.replace('.r3d', '_unaligned.pkl')
    print(f"Saving unaligned data to {output_filename}...")
    with open(output_filename, 'wb') as file:
        pkl.dump(data, file)
    print("Done.")

if __name__ == "__main__":
    main()

