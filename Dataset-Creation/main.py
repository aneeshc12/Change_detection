# use python 3.7.16

import os
import cv2
import math
import random
from ai2thor.controller import Controller
import numpy as np
import open3d as o3d

# local_executable_path for a local build using the Unity app, else use scene for a in-built scene
controller = Controller(scene='FloorPlan201', renderDepthImage=True, renderInstanceSegmentation=True, width=600, height=600)

# Disable all pickupable objects
for object in controller.last_event.metadata["objects"]:
    if object["pickupable"]:
        controller.step(
        action="DisableObject",
        objectId=object["objectId"]
)
        
# Disable/Remove all moveable objects which are not required in the scene
list = ["Television|-02.36|+01.21|+06.24",
"HousePlant|-02.93|+00.60|-00.09",
"GarbageCan|-04.86|00.00|+00.27",
"FloorLamp|-00.57|+00.00|+00.02",
"DeskLamp|-00.27|+00.70|+03.61",
"SideTable|-00.25|+00.00|+03.37",
"ArmChair|-04.38|00.00|+06.02",
"DiningTable|-02.27|-00.02|+01.42",
"CoffeeTable|-02.33|+00.00|+04.92",
"Sofa|-02.40|00.00|+03.42",
"SideTable|-02.94|+00.00|-00.10",
"Chair|-01.86|+00.02|+01.04",
"Chair|-03.12|+00.02|+01.41",
"Chair|-01.86|+00.02|+01.84",
"Chair|-02.51|+00.02|+00.99",
"Chair|-02.52|+00.02|+01.88",
]
for object in list:
    controller.step(
    action="DisableObject",
    objectId=object
    )
    # controller.step(
    # action="RemoveFromScene",
    # objectId=object
    # )

# store the required Object IDs in a list
objects = []
for obj in controller.last_event.metadata["objects"]:
    if (obj['moveable'] == True):
        # print("\""+obj['objectId']+"\""+",")
        objects.append(obj['objectId'])
objects2 = []
print("Object IDs: ")
for obj in objects:
    for object in controller.last_event.metadata['objects']:
        if object['objectId'] == obj:
            print(object['objectId'])
            objects2.append(object)
            break

# precomputed poses for the agent on 4 corners and 4 sides of the room 
views = [{'name': 'agent', 'position': {'x': -2.5, 'y': 0.9026566743850708, 'z': 6.25}, 'rotation': {'x': -0.0, 'y': 180.00001525878906, 'z': 0.0}, 'cameraHorizon': 0.0, 'isStanding': False, 'inHighFrictionArea': False}, 
        {'name': 'agent', 'position': {'x': -4.5, 'y': 0.9026566743850708, 'z': 6.25}, 'rotation': {'x': -0.0, 'y': 135.00001525878906, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False}, 
        {'name': 'agent', 'position': {'x': -4.5, 'y': 0.9026566743850708, 'z': 3.25}, 'rotation': {'x': -0.0, 'y': 90.00000762939453, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False}, 
        {'name': 'agent', 'position': {'x': -4.5, 'y': 0.9026566743850708, 'z': 1.4901161193847656e-08}, 'rotation': {'x': -0.0, 'y': 45.000003814697266, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False}, 
        {'name': 'agent', 'position': {'x': -2.75, 'y': 0.9026566743850708, 'z': 1.4901161193847656e-08}, 'rotation': {'x': -0.0, 'y': 0.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False}, 
        {'name': 'agent', 'position': {'x': -0.5, 'y': 0.9026566743850708, 'z': 1.4901161193847656e-08}, 'rotation': {'x': 0.0, 'y': 315.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False}, 
        {'name': 'agent', 'position': {'x': -0.5, 'y': 0.9026566743850708, 'z': 3.25}, 'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False}, 
        {'name': 'agent', 'position': {'x': -0.5, 'y': 0.9026566743850708, 'z': 6.25}, 'rotation': {'x': 0.0, 'y': 224.99998474121094, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False}]

# views = [{'name': 'agent', 'position': {'x': -2.5177674293518066, 'y': 0.9011250734329224, 'z': 0.4822334945201874}, 'rotation': {'x': 0.0, 'y': -5.768936262029456e-06, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': -0.7677674293518066, 'y': 0.9011250734329224, 'z': 0.4822337031364441}, 'rotation': {'x': 0.0, 'y': 315.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': -0.26776745915412903, 'y': 0.9011250734329224, 'z': 3.2322335243225098}, 'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': -0.7677674293518066, 'y': 0.9011250734329224, 'z': 6.73223352432251}, 'rotation': {'x': 0.0, 'y': 224.99998474121094, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': -2.5177674293518066, 'y': 0.9011250734329224, 'z': 6.73223352432251}, 'rotation': {'x': -0.0, 'y': 179.99998474121094, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': -4.767767429351807, 'y': 0.9011250734329224, 'z': 6.73223352432251}, 'rotation': {'x': -0.0, 'y': 134.99996948242188, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': -4.767767429351807, 'y': 0.9011250734329224, 'z': 3.7322335243225098}, 'rotation': {'x': -0.0, 'y': 89.9999771118164, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': -4.767767429351807, 'y': 0.9011250734329224, 'z': 0.23223352432250977}, 'rotation': {'x': -0.0, 'y': 44.999977111816406, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': False, 'inHighFrictionArea': False},
# ]

# views = [{'name': 'agent', 'position': {'x': 8.0, 'y': 0.9028512835502625, 'z': 1.249999761581421}, 'rotation': {'x': -0.0, 'y': 90.00000762939453, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': 8.0, 'y': 0.9028512835502625, 'z': -1.0000001192092896}, 'rotation': {'x': -0.0, 'y': 45.000003814697266, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': 11.25, 'y': 0.9028512835502625, 'z': -1.0000001192092896}, 'rotation': {'x': -0.0, 'y': 0.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': 14.5, 'y': 0.9028512835502625, 'z': -1.0000001192092896}, 'rotation': {'x': 0.0, 'y': 315.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': 14.5, 'y': 0.9028512835502625, 'z': 0.9999998211860657}, 'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': 14.5, 'y': 0.9028512835502625, 'z': 3.499999761581421}, 'rotation': {'x': 0.0, 'y': 224.99998474121094, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': 11.0, 'y': 0.9028512835502625, 'z': 3.249999761581421}, 'rotation': {'x': -0.0, 'y': 179.99998474121094, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
# {'name': 'agent', 'position': {'x': 7.823223114013672, 'y': 0.9028512835502625, 'z': 3.42677640914917}, 'rotation': {'x': -0.0, 'y': 134.99996948242188, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
# ]

# views = [{'name': 'agent', 'position': {'x': 0.0, 'y': 0.9009992480278015, 'z': -1.0}, 'rotation': {'x': -0.0, 'y': 5.768936262029456e-06, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': 1.75, 'y': 0.9009992480278015, 'z': -1.0}, 'rotation': {'x': 0.0, 'y': 315.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': 1.75, 'y': 0.9009992480278015, 'z': 1.0000001192092896}, 'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': 1.75, 'y': 0.9009992480278015, 'z': 3.750000238418579}, 'rotation': {'x': 0.0, 'y': 225.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': 0.0, 'y': 0.9009992480278015, 'z': 3.750000238418579}, 'rotation': {'x': -0.0, 'y': 180.0, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': -1.75, 'y': 0.9009992480278015, 'z': 3.750000238418579}, 'rotation': {'x': -0.0, 'y': 135.0, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': -1.75, 'y': 0.9009992480278015, 'z': 1.250000238418579}, 'rotation': {'x': -0.0, 'y': 89.99999237060547, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': -1.5, 'y': 0.9009992480278015, 'z': -1.4999996423721313}, 'rotation': {'x': -0.0, 'y': 44.9999885559082, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#         ]

# views = [{'name': 'agent', 'position': {'x': -2.5, 'y': 0.9011250734329224, 'z': 0.2499997615814209}, 'rotation': {'x': -0.0, 'y': 1.8389482647762634e-05, 'z': -0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': -0.75, 'y': 0.9011250734329224, 'z': 0.2499992400407791}, 'rotation': {'x': -0.0, 'y': 315.0, 'z': 0.0}, 'cameraHorizon': 0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': -0.4999992847442627, 'y': 0.9011250734329224, 'z': 3.2499990463256836}, 'rotation': {'x': -0.0, 'y': 270.0, 'z': 0.0}, 'cameraHorizon': 0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': -0.24999846518039703, 'y': 0.9011250734329224, 'z': 6.749999046325684}, 'rotation': {'x': -0.0, 'y': 225.0, 'z': 0.0}, 'cameraHorizon': 0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': -2.4999985694885254, 'y': 0.9011250734329224, 'z': 6.749999046325684}, 'rotation': {'x': -0.0, 'y': 180.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': -4.749998569488525, 'y': 0.9011250734329224, 'z': 6.749999046325684}, 'rotation': {'x': -0.0, 'y': 135.0, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': -4.749998569488525, 'y': 0.9011250734329224, 'z': 3.4999990463256836}, 'rotation': {'x': -0.0, 'y': 89.99999237060547, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False},
#          {'name': 'agent', 'position': {'x': -4.749998569488525, 'y': 0.9011250734329224, 'z': 0.2499990612268448}, 'rotation': {'x': -0.0, 'y': 44.9999885559082, 'z': 0.0}, 'cameraHorizon': -0.0, 'isStanding': True, 'inHighFrictionArea': False}]

viewnames = ['view'+str(i) for i in range(1, len(views)+1)]

def save_images(folder):
    # initialise positions
    for i in objects2:
        c = None
        while c is None:
            c = controller.step(
                action="PlaceObjectAtPoint",
                objectId=i['objectId'],
                position={
                    "x": random.random() * 2 + (views[2]['position']['x'] + views[6]['position']['x'])/2 - 1,
                    "y": i['position']['y'],
                    "z": random.random() * 2 + (views[0]['position']['z'] + views[4]['position']['z'])/2 - 1
                },
                rotation={
                    "x": 0,
                    "y": random.random() * 360,
                    "z": 0
                }
            ).metadata["actionReturn"]
        print("Placed object: ", i['objectId'])
    
    try:
        os.mkdir(folder)
    except:
        pass
    
    for orientation in range(1, 9):
        os.mkdir("{}/orientation{}".format(folder, orientation))
        for i in range(len(views)):
            os.mkdir("{}/orientation{}/{}".format(folder, orientation, viewnames[i]))
            controller.step(
                        action="Teleport",
                        position=views[i]['position'],
                        rotation=views[i]['rotation'],
                        horizon=0,
                        standing=False
                    )
            width = 600
            height = 600
            fov = 90

            focal_length = 0.5 * width / math.tan(math.radians(fov/2))

            # camera intrinsics
            fx, fy, cx, cy = (focal_length, focal_length, width/2, height/2)

            # Obtain point cloud
            event = controller.last_event
            color = o3d.geometry.Image(event.frame.astype(np.uint8))
            d = event.depth_frame
            d1 = d/np.max(d)
            depth = o3d.geometry.Image(d1)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                                                                            depth_scale=1.0,
                                                                            depth_trunc=0.7,
                                                                            convert_rgb_to_intensity=False)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            pcd.transform([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
            
            o3d.io.write_point_cloud("{}/orientation{}/{}/{}.pcd".format(folder, orientation, viewnames[i], viewnames[i]), pcd)

            rgb = controller.last_event.cv2img
            cv2.imwrite("{}/orientation{}/{}/{}.png".format(folder,orientation, viewnames[i], viewnames[i]), rgb)
            depth = controller.last_event.depth_frame
            np.save("{}/orientation{}/{}/{}.npy".format(folder, orientation, viewnames[i], viewnames[i]), depth)
            sem = controller.last_event.instance_segmentation_frame
            cv2.imwrite("{}/orientation{}/{}/{}_sem.png".format(folder, orientation, viewnames[i], viewnames[i]), sem)
            
            # poses
            f = open("{}/orientation{}/{}/{}_poses.txt".format(folder, orientation, viewnames[i], viewnames[i]), "w")
            f.write(str([controller.last_event.metadata['agent']['position'], controller.last_event.metadata['agent']['rotation']]))
            f.write("\n")
            for object in controller.last_event.metadata['objects']:
                f.write(str([object['objectId'], object['position']['x'], object['position']['y'], object['position']['z'], object['rotation']['x'], object['rotation']['y'], object['rotation']['z']]))
                f.write("\n")

        for i in objects2:
            c = None
            while c is None:
                c = controller.step(
                    action="PlaceObjectAtPoint",
                    objectId=i['objectId'],
                    position={
                        "x": random.random() * 2 + (views[2]['position']['x'] + views[6]['position']['x'])/2 - 1,
                        "y": i['position']['y'],
                        "z": random.random() * 2 + (views[0]['position']['z'] + views[4]['position']['z'])/2 - 1
                    },
                    rotation={
                        "x": 0,
                        "y": random.random() * 360,
                        "z": 0
                    }
                ).metadata["actionReturn"]
            print("Placed object: ", i['objectId'])
            
            
save_images("dataset")
print("Done")