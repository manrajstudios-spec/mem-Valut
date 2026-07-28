class Node:
    def __init__(self,neighbours=[],value=None):
        self.neighbours = neighbours
        self.value = value

class Groups:
    def __init__(self,topics,keywords,memeber_ids,graph,group_id):
        self.topics = topics
        self.keywords = keywords
        self.memeber_ids = memeber_ids
        self.graph = graph
        self.group_id = group_id