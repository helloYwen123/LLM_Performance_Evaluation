##################################################3


#####################################################
class Node:
    def __init__(self, state: 'NodeState' = None):
        """
        Initialize a Node with a given state.

        Args:
            state (NodeState, optional): The initial state of the node. Defaults to None.
        """
        self.state = state
        logger.info(f"Node is genreated from {state}")
class Edge:
    def __init__(self, source: Node, target_node: Node):
        """
        Initialize an Edge with a source and target node.

        Args:
            source (Node): The source node of the edge.
            target_node (Node): The target node of the edge.
        """
        self.source = source
        self.target = target_node
        logger.info(f"Edge created from {source} to {target_node}")
#######################################################################################

class Node:
    def __init__(self, state: 'NodeState' = None):
        self.state = state
        logger.info("Node created")

class Edge:
    def __init__(self, source: Node, target_node: Node):
        self.source: Node = source
        self.target: Node = target_node
        logger.info("Initialized Edge")