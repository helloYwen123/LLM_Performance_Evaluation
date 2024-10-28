#Ground Truth 
###################

###################
class Graph:
    def __init__(self, root_node: Node):
        self.nodes = list[Node]()
        self.edges = list[Edge]()

        self.root: Node = root_node
        self.add_node(self.root)

        logger.info("Initialized video reasoning graph")

    def get_edges_from_node(self, node) -> list[Edge]:
        return [edge for edge in list(self.edges) if edge.source == node]

    def get_successors_of_node(self, node) -> list[Node]:
        return [edge.target for edge in self.get_edges_from_node(node)]

    def get_predecessors_of_node(self, node) -> list[Node]:
        return [edge.source for edge in self.get_edges_from_node(node)]

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def add_edges(self, edges: list[Edge]) -> None:
        self.edges.extend(edges)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_nodes(self, nodes: list[Node]) -> None:
        self.nodes.extend(nodes)

    def get_derivable_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not node.state.derived and node != self.root]

    def get_unranked_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not node.state.ranking and node != self.root]

    def get_leaf_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not self.get_successors_of_node(node)]

    def get_derived_root_successor_nodes(self) -> list[Node]:
        root_successors = self.get_successors_of_node(self.root)
        return [node for node in root_successors if node.state.derived]

    def get_decidable_leaf_nodes(self) -> list[Node]:
        return [node for node in self.nodes if
                # consider leaf nodes
                (not self.get_successors_of_node(node)
                 # consider leaf nodes that are not waiting
                 and not node.state.waiting)]

    def get_concludable_nodes(self) -> list[Node]:
        return [node for node in self.nodes if
                # consider leaf nodes
                (not self.get_successors_of_node(node)
                 # consider leaf nodes that have states that are derived
                 and node.state.derived)]
    
#########################################################################################################
#########################################################################################################
# here, you are an very experienced Programmer, who is very good at programming under others' instruction
# and can also complete code with many blank space or gaps very well 
# you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.
# Next please you complete the code, including classes and functions for my algorithm framework under my instruction and decription

# please help me to writin a class named Graph , it takes root_node: Node as inputs and has following class variables:self.nodes = list[Node]();self.edges = list[Edge]();self.root: Node = root_node;self.add_node(self.root)
# the first function is named get_edges_from_node, it takes node as input and returns list[Edge] ; it can derive the edge from list[Edge] if the input node is the edge.source.
# the second function is named get_successors_of_node taking node as input and then returns list[Node]. it will use for-loop calling get_edges_from_node(node) function to get edges list sourcing from this node
# return all successor of this node via edge.target
# the third function is named get_predecessors_of_node, which takes node as inputs and returns list of node
# also is via self.get_edges_from_node and edge.source to get predecessors of node
# the next several functions are add_edge, add_edges, add_node, add_nodes , they take edges and edges list or node and node list . they will use append or extend to do operation
# next the function are get_derivable_nodes, which is used to get node.state that is not derived.
# the another function is named get_unranked_nodes , it returns list of node. it will return node if node.state.ranking is false and node is not root
# the next function is get_leaf_nodes, it returns list[node] . it will return node whose get_successors_of_node is false
# the next function is named get_derived_root_successor_nodes , it will return rootnode's successors and the successors have been derived
# the next functionis named get_decidable_leaf_nodes which returns list[node] , the nodes in list have no successor_node and state is waiting then return these node
# the next function is named get_concludable_nodes, which returns list[node] , the nodes in list have no successor_node and state is derived then return these node.
class Graph:
    def __init__(self, root_node):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.root: Node = root_node
        self.add_node(self.root)
        
    def add_node(self, node: Node):
        """Add a single node to the graph."""
        if node not in self.nodes:
            self.nodes.append(node)

    def add_nodes(self, nodes: List[Node]):
        """Add multiple nodes to the graph."""
        for node in nodes:
            self.add_node(node)
    
    def add_edge(self, edge: Edge):
        """Add a single edge to the graph."""
        if edge not in self.edges:
            self.edges.append(edge)
            self.add_node(edge.source)
            self.add_node(edge.target)
    
    def add_edges(self, edges: List[Edge]):
        """Add multiple edges to the graph."""
        for edge in edges:
            self.add_edge(edge)
    
    def get_edges_from_node(self, node: Node) -> List[Edge]:
        """Get all edges originating from a given node."""
        return [edge for edge in self.edges if edge.source == node]
    
    def get_successors_of_node(self, node: Node) -> List[Node]:
        """Get all successor nodes of a given node."""
        return [edge.target for edge in self.get_edges_from_node(node)]
    
    def get_predecessors_of_node(self, node: Node) -> List[Node]:
        """Get all predecessor nodes of a given node."""
        return [edge.source for edge in self.edges if edge.target == node]
    
    def get_derivable_nodes(self) -> List[Node]:
        """Get nodes whose state has not been derived."""
        return [node for node in self.nodes if not node.state.derived]
    
    def get_unranked_nodes(self) -> List[Node]:
        """Get nodes that are unranked and not the root."""
        return [node for node in self.nodes if not node.state.ranking and node != self.root]
    
    def get_leaf_nodes(self) -> List[Node]:
        """Get all leaf nodes in the graph."""
        return [node for node in self.nodes if not self.get_successors_of_node(node)]
    
    def get_derived_root_successor_nodes(self) -> List[Node]:
        """Get all derived successor nodes of the root node."""
        return [node for node in self.get_successors_of_node(self.root) if node.state.derived]
    
    def get_decidable_leaf_nodes(self) -> List[Node]:
        """Get all leaf nodes with state 'waiting'."""
        return [node for node in self.get_leaf_nodes() if node.state.waiting]
    
    def get_concludable_nodes(self) -> List[Node]:
        """Get all leaf nodes with state 'derived'."""
        return [node for node in self.get_leaf_nodes() if node.state.derived]
    
###########################################################################################################
##############################################################################################################
# Now i will give you the partly context of one class which i have finished, help me to continue to write and complete it based on the following requirment
# and description. 
# Do not start writing when i have not given you the context of code

class Graph:
    def __init__(self, root_node: Node):
        self.nodes = list[Node]()
        self.edges = list[Edge]()

        self.root: Node = root_node
        self.add_node(self.root)

        logger.info("Initialized video reasoning graph")

    def get_edges_from_node(self, node) -> list[Edge]:
        return [edge for edge in list(self.edges) if edge.source == node]

    def get_successors_of_node(self, node) -> list[Node]:
        return [edge.target for edge in self.get_edges_from_node(node)]

    def get_predecessors_of_node(self, node) -> list[Node]:
        return [edge.source for edge in self.get_edges_from_node(node)]

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def add_edges(self, edges: list[Edge]) -> None:
        self.edges.extend(edges)
      def add_node(self, node: Node) -> None:
        """Adds a single node to the graph if not already present."""
        if node not in self.nodes:
            self.nodes.append(node)

    def add_nodes(self, nodes: list[Node]) -> None:
        """Adds multiple nodes to the graph."""
        for node in nodes:
            self.add_node(node)

    def get_derivable_nodes(self) -> list[Node]:
        """Returns a list of nodes that have not been derived."""
        return [node for node in self.nodes if not node.state.derived]

    def get_unranked_nodes(self) -> list[Node]:
        """Returns nodes that are unranked and not the root."""
        return [node for node in self.nodes if not node.state.ranking and node != self.root]

    def get_leaf_nodes(self) -> list[Node]:
        """Returns all leaf nodes, i.e., nodes without successors."""
        return [node for node in self.nodes if not self.get_successors_of_node(node)]

    def get_derived_root_successor_nodes(self) -> list[Node]:
        """Returns derived successor nodes of the root node."""
        return [node for node in self.get_successors_of_node(self.root) if node.state.derived]

    def get_decidable_leaf_nodes(self) -> list[Node]:
        """Returns leaf nodes with a 'waiting' state."""
        return [node for node in self.get_leaf_nodes() if node.state.waiting]

    def get_concludable_nodes(self) -> list[Node]:
        """Returns leaf nodes with a 'derived' state."""
        return [node for node in self.get_leaf_nodes() if node.state.derived]
#################################################################################################33
#################################################################################################33
# Now i will give you the one class which i have finished and but with many blank spaces and gaps 
# that I hope you can help me fill in and complete it in order to make it fuctional and run smoothly.
# continue to write and complete it based on the following requirment and description. 
# Please do not start writing when i still given you my code.
#################################################################################################33
#################################################################################################33
class Graph:
    def __init__(self, root_node: 'Node'):
        self.nodes: List['Node'] = []
        self.edges: List['Edge'] = []

        self.root: 'Node' = root_node
        self.add_node(self.root)

        logger.info("Initialized video reasoning graph")

    def get_edges_from_node(self, node: 'Node') -> List['Edge']:
        return [edge for edge in self.edges if edge.source == node]

    def get_successors_of_node(self, node: 'Node') -> List['Node']:
        return [edge.target for edge in self.get_edges_from_node(node)]

    def get_predecessors_of_node(self, node: 'Node') -> List['Node']:
        return [edge.source for edge in self.edges if edge.target == node]

    def add_edge(self, edge: 'Edge') -> None:
        if edge not in self.edges:
            self.edges.append(edge)
            self.add_node(edge.source)
            self.add_node(edge.target)
            logger.info(f"Added edge from {edge.source} to {edge.target}")

    def add_edges(self, edges: List['Edge']) -> None:
        for edge in edges:
            self.add_edge(edge)

    def add_node(self, node: 'Node') -> None:
        if node not in self.nodes:
            self.nodes.append(node)
            logger.info(f"Added node {node}")

    def add_nodes(self, nodes: List['Node']) -> None:
        for node in nodes:
            self.add_node(node)

    def get_derivable_nodes(self) -> List['Node']:
        """Returns nodes whose state has not been derived and are not the root."""
        return [node for node in self.nodes if not node.state.derived and node != self.root]

    def get_unranked_nodes(self) -> List['Node']:
        """Returns nodes that are unranked and not the root."""
        return [node for node in self.nodes if not node.state.ranking and node != self.root]

    def get_leaf_nodes(self) -> List['Node']:
        """Returns all leaf nodes, i.e., nodes without successors."""
        return [node for node in self.nodes if not self.get_successors_of_node(node)]

    def get_derived_root_successor_nodes(self) -> List['Node']:
        """Returns derived successor nodes of the root node."""
        root_successors = self.get_successors_of_node(self.root)
        derived_successors = [node for node in root_successors if node.state.derived]
        logger.info(f"Derived successors of root: {derived_successors}")
        return derived_successors

    def get_decidable_leaf_nodes(self) -> List['Node']:
        """
        Returns leaf nodes with state 'waiting'.
        These nodes have no successors and are in a waiting state.
        """
        decidable_leaves = [
            node for node in self.get_leaf_nodes()
            if node.state.waiting
        ]
        logger.info(f"Decidable leaf nodes: {decidable_leaves}")
        return decidable_leaves

    def get_concludable_nodes(self) -> List['Node']:
        """
        Returns leaf nodes with state 'derived'.
        These nodes have no successors and their state is derived.
        """
        concludable_nodes = [
            node for node in self.get_leaf_nodes()
            if node.state.derived
        ]
        logger.info(f"Concludable nodes: {concludable_nodes}")
        return concludable_nodes