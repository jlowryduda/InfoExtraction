from nltk.tree import Tree

class AttributeTree(Tree):
    def __init__(self, node, children=None):
        super(self.__class__, self).__init__(node, children)
        self.token = None
        self.pos = None
        self.index = None
        self.entity_type = None

    def set_token(self, token):
        self.token = token

    def set_pos(self, pos):
        self.pos = pos

    def set_index(self, index):
        self.index = index

    def set_entity_type(self, entity_type):
        self.entity_type = entity_type

    
