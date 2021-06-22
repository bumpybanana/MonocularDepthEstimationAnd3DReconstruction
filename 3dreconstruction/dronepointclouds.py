import open3d
import matplotlib.pyplot as plt

rgb_file = "C:/Users/alexk/OneDrive/Desktop/Crazyflie/RGB/rgb11.png"
depth_file = "C:/Users/alexk/OneDrive/Desktop/Crazyflie/RGB/dronepredict11.png"

depth = open3d.io.read_image(depth_file)
rgb = open3d.io.read_image(rgb_file)
rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb,depth, convert_rgb_to_intensity = False)

#point cloud from drone image and corresponding depth map with found out camera intrinsics
pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, open3d.camera.PinholeCameraIntrinsic(width=480,height=320,fx=217.67,fy=216.49,cx=149.64,cy=164.05))

# flip the orientation, so it looks upright, not upside-down
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
plt.subplot(1, 2, 1)
plt.title('RGB image')
plt.imshow(rgbd.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd.depth)
plt.show()
open3d.visualization.draw_geometries([pcd])    # visualize the point cloud
