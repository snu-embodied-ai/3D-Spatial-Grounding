import numpy as np

from dataloader.objects import pcObject


def is_suppported(supported: pcObject,
                  surface: pcObject,
                  floating_margin: float = 0.01,
                  threshold_of_z_rate: float = 0.8,
                  projection_bound: float = 0.3):
    
    # 1. Check: target must have larger support area
    surface_area = np.ptp(surface.points[:,:2], axis=0).prod()
    object_area = np.ptp(supported.points[:,:2], axis=0).prod()

    if not surface_area > object_area:
        return False
    
    supported_z_min = supported.points[:,2].min()   # z_min
    supported_z_max = supported.points[:,2].max()   # z_max
    supported_x_min, supported_y_min = supported.points[:,:2].min(axis=0)
    supported_x_max, supported_y_max = supported.points[:,:2].max(axis=0)

    x_min_bound = supported_x_min - projection_bound <= surface.points[:,0]
    x_max_bound = supported_x_max + projection_bound >= surface.points[:,0]
    y_min_bound = supported_y_min - projection_bound <= surface.points[:,1]
    y_max_bound = supported_y_max + projection_bound >= surface.points[:,1]
    x_range = np.logical_and(x_min_bound, x_max_bound)
    y_range = np.logical_and(y_min_bound, y_max_bound)
    proj_surface = surface.points[np.logical_and(x_range, y_range)]

    if proj_surface.shape[0] == 0:
        surface_z_min = surface.points[:,2].min()
        surface_z_max = surface.points[:,2].max()
    else:
        surface_z_min = proj_surface[:,2].min()              # tz_min
        surface_z_max = proj_surface[:,2].max()              # tz_max

    obj_height = supported_z_max - supported_z_min
    vertical_gap = supported_z_min - surface_z_max  # how much obj is above target
    z_rate = abs(vertical_gap) / obj_height
    
    # Case 1: Supporting object is floor
    if surface.obj_type == 'floor':
        if supported_z_min > surface_z_max:
            return False        # object not actually sitting on floor
        
    # Case 2: Supporting object is NOT floor
    else:
        # Floating check: if bottom of obj is too far above top of target
        if supported_z_min > surface_z_max + floating_margin:
            return False        # too high to be supported

        # If bottom of obj is below bottom of target, likely invalid
        if supported_z_min < surface_z_min:
            return False
        
        # If vertical gap is too big compared to object's height, reject
        if vertical_gap > obj_height * 0.2:
            return False
    
    # Check if obj center is inside target's top surface (must be well aligned)
    center = (supported.points.min(axis=0) + supported.points.max(axis=0)) / 2
    in_surface_x = surface.points[:,0].min() < center[0] < surface.points[:,0].max()
    in_surface_y = surface.points[:,1].min() < center[1] < surface.points[:,1].max()

    if not (in_surface_x and in_surface_y):
        return False
    
    # Support types based on z overlap
    if surface.obj_type == 'floor':
        return 'support_express'
    elif z_rate < threshold_of_z_rate:
        return 'support_express'
    elif z_rate < 0.95:
        return 'embed_express'
    else:
        return 'inside_express'
    