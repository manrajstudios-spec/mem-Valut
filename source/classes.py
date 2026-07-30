class Node:
    def __init__(self,neighbours=[],value=None):
        self.neighbours = neighbours
        self.value = value

class Group:
    __slots__ = ("group_id","memebers")
    
    def __init__(self,group_id,members):
        self.group_id = group_id
        self.memebers = members