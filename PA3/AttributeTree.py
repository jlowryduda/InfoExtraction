from nltk.tree import ParentedTree

class AttributeTree(ParentedTree):
    def __init__(self, node, children=None):
        super(self.__class__, self).__init__(node, children)
        self._pos = None
        self._index = None
        self._entity_type = None
        self._hypernym = None

    def set_pos(self, pos):
        self._pos = pos

    def set_index(self, index):
        self._index = index

    def set_entity_type(self, entity_type):
        self._entity_type = entity_type

    def set_hypernym(self, hypernym):
        self._hypernym = hypernym
