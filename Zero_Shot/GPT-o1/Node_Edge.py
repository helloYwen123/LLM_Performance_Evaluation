class Node:
    def __init__(self, state: NodeState = None):
        self.state = state
        logger.info("Initialized node")


class Edge:
    def __init__(self, source: Node, target_node: Node):
        self.source: Node = source
        self.target: Node = target_node
        logger.info("Initialized edge")

# help me finish two classes: one is named Node and another is Edge
# the Node class takes state: NodeState = None as its input and has member variable. In the __init__ function it will initialize it with input state
# the Edge class takes source: Node, target_node: Node as inputs and in the __init__ it will initialize its member variables  self.source: Node and self.target: Node. 
# after successful initialization it will do logger.info 
####################################################################################################################
####################################################################################################################
####################################################################################################################
class Node:
    def __init__(self, state: Optional[NodeState] = None):
        """
        Initialize a Node with an optional state.

        :param state: An instance of NodeState or None.
        """
        self.state = state
        logger.debug(f"Initialized Node with state: {self.state}")

    def __repr__(self):
        return f"Node(state={self.state})"

class Edge:
    def __init__(self, source: Node, target: Node):
        """
        Initialize an Edge connecting two nodes.

        :param source: The source Node.
        :param target: The target Node.
        """
        self.source = source
        self.target = target
        logger.info(f"Edge created from {self.source} to {self.target}")

    def __repr__(self):
        return f"Edge(source={self.source}, target={self.target})"
################################################################################################################################
####################################################################################################################
####################################################################################################################
class Node:
    def __init__(self, state: Optional[NodeState] = None):
        """
        Initialize a Node with an optional state.

        :param state: An instance of NodeState or None.
        """
        self.state = state
        logger.debug(f"Initialized Node with state: {self.state}")

    def __repr__(self):
        return f"Node(state={self.state})"

class Edge:
    def __init__(self, source: Node, target_node: Node):
        """
        Initialize an Edge connecting two nodes.

        :param source: The source Node.
        :param target_node: The target Node.
        """
        self.source = source
        self.target = target_node
        logger.info(f"Edge created from {self.source} to {self.target}")

    def __repr__(self):
        return f"Edge(source={self.source}, target={self.target})"
#############################################################################3###############################3
class Node:
    def __init__(self, state: Optional[NodeState] = None):
        """
        Initialize a Node with an optional state.

        :param state: An instance of NodeState or None.
        """
        self.state = state
        logger.info("Node initialized with state: %s", self.state)

    def __repr__(self):
        return f"Node(state={self.state})"

class Edge:
    def __init__(self, source: Node, target_node: Node):
        """
        Initialize an Edge connecting two nodes.

        :param source: The source Node.
        :param target_node: The target Node.
        """
        self.source: Node = source
        self.target: Node = target_node
        logger.info("Edge initialized from %s to %s", self.source, self.target)

    def __repr__(self):
        return f"Edge(source={self.source}, target={self.target})"