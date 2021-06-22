#creates 3D point clouds from dataset rgb images and their ground truth/predicted depth map 

import open3d
import matplotlib.pyplot as plt

color_raw = open3d.io.read_image("C:/Users/alexk/OneDrive/Desktop/Thesis/EvalPics/rgb9.png")
depth_gt = open3d.io.read_image("C:/Users/alexk/OneDrive/Desktop/Thesis/EvalPics/depth9.png")
pred_depth = open3d.io.read_image("C:/Users/alexk/OneDrive/Desktop/Thesis/EvalPics/pred9.png")

#create rgbd images from ground truth and predicted depth maps
rgbd_gt = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_gt, convert_rgb_to_intensity = False)
rgbd_pred = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, pred_depth, convert_rgb_to_intensity = False)

#create point clouds from rgbd image
pcd_gt = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_gt, open3d.camera.PinholeCameraIntrinsic(
        open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
pcd_pred = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_pred, open3d.camera.PinholeCameraIntrinsic(
        open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))


# flip the orientation, so it looks upright, not upside-down
pcd_gt.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
plt.subplot(1, 2, 1)
plt.title('RGB image')
plt.imshow(rgbd_gt.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd_gt.depth)
plt.show()
open3d.visualization.draw_geometries([pcd_gt])    # visualize the point cloud from ground truth rgbd

pcd_pred.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
plt.subplot(1, 2, 1)
plt.title('RGB image')
plt.imshow(rgbd_pred.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd_pred.depth)
plt.show()
open3d.visualization.draw_geometries([pcd_pred]) # visualize the point cloud from predicted rgbd
