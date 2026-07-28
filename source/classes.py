class Node:
    def __init__(self,neighbours=[],value=None):
        self.neighbours = neighbours
        self.value = value

class Group:
    def __init__(self,keywords,memeber_ids,graph,group_id):
        self.keywords = keywords
        self.memeber_ids = memeber_ids
        self.graph = graph
        self.group_id = group_id