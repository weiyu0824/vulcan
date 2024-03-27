person_id, vehicle_id = 0, 1
def get_dict_nuimage_category_to_id():
    NuimageCategoryToId = {}
    NuimageCategoryToId['human.pedestrian.adult'] = person_id
    NuimageCategoryToId['human.pedestrian.child'] = person_id
    NuimageCategoryToId['human.pedestrian.construction_worker'] = person_id
    NuimageCategoryToId['human.pedestrian.personal_mobility'] = person_id
    NuimageCategoryToId['human.pedestrian.police_officer'] = person_id
    NuimageCategoryToId['human.pedestrian.stroller'] = person_id
    NuimageCategoryToId['human.pedestrian.wheelchair'] = person_id
    NuimageCategoryToId['vehicle.bicycle'] = vehicle_id
    NuimageCategoryToId['vehicle.car'] = vehicle_id
    NuimageCategoryToId['vehicle.motorcycle'] = vehicle_id 
    NuimageCategoryToId['vehicle.truck'] = vehicle_id 
    NuimageCategoryToId['vehicle.bus.rigid'] = vehicle_id
    NuimageCategoryToId['vehicle.bus.bendy'] = vehicle_id
    NuimageCategoryToId['vehicle.trailer'] = vehicle_id
    return NuimageCategoryToId 

def get_dict_coco_category_to_id():
    CocoCategoryToId = {}
    CocoCategoryToId[0] = person_id
    CocoCategoryToId[1] = vehicle_id
    CocoCategoryToId[2] = vehicle_id
    CocoCategoryToId[3] = vehicle_id
    CocoCategoryToId[5] = vehicle_id
    CocoCategoryToId[7] = vehicle_id
    return CocoCategoryToId
