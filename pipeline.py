from typing import Tuple, Dict, List
import yaml

# define a DAG
class Node:
    # knobs: List[Tuple[str, List[str]]]
    def __init__(self, id, func_name):
        self.id: int = id
        self.func_name: str = func_name
class Edge:
    def __init__(self, from_id, to_id):
        from_id: int = from_id
        to_id: int = to_id

class Pipeline:

    def __init__(self, task: str):
        # read config.yaml
        with open(f'{task}/config.yaml', 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        # create pipeline 
        self.node_list = []
        self.knob_list: List[Dict]  = []
        for node in data['nodes']:
            id = node['id'] 
            func_name = node['name']
            if node['knobs'] != None:
                for knob in node['knobs']:
                    knob['id'] = id
                    self.knob_list.append(knob) 
            self.node_list.append(Node(id, func_name))

        self.adj_list = {} 
        for edge in data['edges']:
            source, to = edge['source'], edge['to'] 
            if source not in self.adj_list:
                self.adj_list[source] = []
                
            self.adj_list[source].append(to)


    def get_node_orders(self):
        # right not we only have pipeline instead of DAG, so just hack!!
        return self.node_list 

    def get_knobs(self):
        return self.knob_list

pipeline = Pipeline('video_analytics') 
print(pipeline.get_knobs())

