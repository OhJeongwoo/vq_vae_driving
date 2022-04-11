import os
import json

Cv = 0.1

def load_json(file_name):
    with open(file_name, 'r') as jf:
        return json.load(jf)


def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def count_files(dir):
    return len(os.listdir(dir))


def cubic_function(coeff, x):
    return coeff['c0'] + coeff['c1'] * x + coeff['c2'] * x * x + coeff['c3'] * x * x * x


def cubic_derivative(coeff, x):
    return coeff['c1'] + 2.0 * coeff['c2'] * x + 3.0 * coeff['c3'] * x * x


def get_obs(data):
    """
    output (dictionary format)
    ego
        lin_vel
        ang_vel
        lin_acc
        deviation
        heading
        curvature
    vehicles : list of surrounding vehicles, its length is 6.
          0  |  1  |  2
        -----|-ego-|-----
          3  |  4  |  5
        distance
        deviation
        heading
        lin_vel
        ang_vel
    """
    rt = {}

    ego = {}
    # lane coefficient
    l_lane = data['lanes'][0]
    r_lane = data['lanes'][1]
    ego_lane = {'c0': -(l_lane['c0'] + r_lane['c0']) / 2,
                   'c1': -(l_lane['c1'] + r_lane['c1']) / 2,
                   'c2': -(l_lane['c2'] + r_lane['c2']) / 2,
                   'c3': -(l_lane['c3'] + r_lane['c3']) / 2}
    
    # lane deviation
    ego['deviation'] = (l_lane['c0'] + r_lane['c0']) / 2

    # heading
    ego['heading'] = (l_lane['c1'] + r_lane['c1']) / 2

    # curvature
    ego['curvature'] = (l_lane['c2'] + r_lane['c2'])

    # ego state
    ego['lin_vel'] = data['v']
    ego['ang_vel'] = data['omega']
    ego['lin_acc'] = data['ax']

    rt['ego'] = ego

    v_idx = [[-1, -1] for i in range(6)]
    for idx, obj in enumerate(data['objects']):
        x = obj['x']
        y = obj['y']
        d = (x ** 2 + y ** 2) ** 0.5
        f = cubic_function(l_lane, x)
        g = cubic_function(r_lane, x)
        type = 0
        if x < 0:
            type = 3
        if f + y < 0:
            type += 1
        if g + y < 0:
            type += 1
        if v_idx[type][0] == -1:
            v_idx[type] = [idx, d]
        elif v_idx[type][1] > d:
            v_idx[type] = [idx, d]
        
    vehicles = []
    for i, cand in enumerate(v_idx):
        vehicle = {}
        idx = cand[0]
        d = cand[1]
        if idx == -1:
            # default
            if i == 0:
                vehicle['distance'] = 40.0
                vehicle['deviation'] = abs(l_lane['c0'] - r_lane['c0'])
                vehicle['heading'] = 0.0
                vehicle['lin_vel'] = ego['lin_vel'] * (1.0 + Cv)
                vehicle['ang_vel'] = 0.0
            elif i == 1:
                vehicle['distance'] = 40.0
                vehicle['deviation'] = 0.0
                vehicle['heading'] = 0.0
                vehicle['lin_vel'] = ego['lin_vel'] * (1.0 + Cv)
                vehicle['ang_vel'] = 0.0
            elif i == 2:
                vehicle['distance'] = 40.0
                vehicle['deviation'] = -abs(l_lane['c0'] - r_lane['c0'])
                vehicle['heading'] = 0.0
                vehicle['lin_vel'] = ego['lin_vel'] * (1.0 + Cv)
                vehicle['ang_vel'] = 0.0
            elif i == 3:
                vehicle['distance'] = -20.0
                vehicle['deviation'] = abs(l_lane['c0'] - r_lane['c0'])
                vehicle['heading'] = 0.0
                vehicle['lin_vel'] = ego['lin_vel'] * (1.0 - Cv)
                vehicle['ang_vel'] = 0.0
            elif i == 4:
                vehicle['distance'] = -20.0
                vehicle['deviation'] = 0.0
                vehicle['heading'] = 0.0
                vehicle['lin_vel'] = ego['lin_vel'] * (1.0 - Cv)
                vehicle['ang_vel'] = 0.0
            elif i == 5:
                vehicle['distance'] = -20.0
                vehicle['deviation'] = -abs(l_lane['c0'] - r_lane['c0'])
                vehicle['heading'] = 0.0
                vehicle['lin_vel'] = ego['lin_vel'] * (1.0 - Cv)
                vehicle['ang_vel'] = 0.0
            
        else:
            obj = data['objects'][idx]
            vehicle['distance'] = obj['x']
            vehicle['deviation'] = obj['y'] - cubic_function(ego_lane, x)
            vehicle['heading'] = obj['theta'] - cubic_derivative(ego_lane, x)
            vehicle['lin_vel'] = obj['v']
            vehicle['ang_vel'] = obj['omega']
        vehicles.append(vehicle)
    rt['vehicles'] = vehicles

    return rt


def get_act(data):
    dt = 0.1
    rt = []
    s = 0
    l = data[0]['ego']['deviation']
    for i in range(len(data)-1):
        s += data[i]['ego']['lin_vel'] * dt
        d_dev = data[i+1]['ego']['deviation'] - data[i]['ego']['deviation']
        if abs(d_dev) < 0.1:
            l += d_dev
        rt.append({'s': s, 'l': l})
    return rt


