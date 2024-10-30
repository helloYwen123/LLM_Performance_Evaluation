###############################################################3
class Graph:
    def __init__(self, root_node: Node):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.root: Node = root_node
        self.add_node(self.root)
        logger.info("Graph initialized with root node")

    def get_edges_from_node(self, node: Node) -> List[Edge]:
        edges = [edge for edge in self.edges if edge.source == node]
        logger.info(f"Retrieved {len(edges)} edges from node")
        return edges

    def get_successors_of_node(self, node: Node) -> List[Node]:
        edges = self.get_edges_from_node(node)
        successors = [edge.target for edge in edges]
        logger.info(f"Node has {len(successors)} successors")
        return successors

    def get_predecessors_of_node(self, node: Node) -> List[Node]:
        predecessors = [edge.source for edge in self.edges if edge.target == node]
        logger.info(f"Node has {len(predecessors)} predecessors")
        return predecessors

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        logger.info("Added edge to graph")

    def add_edges(self, edges: List[Edge]):
        self.edges.extend(edges)
        logger.info(f"Added {len(edges)} edges to graph")

    def add_node(self, node: Node):
        self.nodes.append(node)
        logger.info("Added node to graph")

    def add_nodes(self, nodes: List[Node]):
        self.nodes.extend(nodes)
        logger.info(f"Added {len(nodes)} nodes to graph")

    def get_derivable_nodes(self) -> List[Node]:
        derivable_nodes = [node for node in self.nodes if not node.state.derived]
        logger.info(f"Found {len(derivable_nodes)} derivable nodes")
        return derivable_nodes

    def get_unranked_nodes(self) -> List[Node]:
        unranked_nodes = [node for node in self.nodes if not node.state.ranking and node != self.root]
        logger.info(f"Found {len(unranked_nodes)} unranked nodes")
        return unranked_nodes

    def get_leaf_nodes(self) -> List[Node]:
        leaf_nodes = [node for node in self.nodes if not self.get_successors_of_node(node)]
        logger.info(f"Found {len(leaf_nodes)} leaf nodes")
        return leaf_nodes

    def get_derived_root_successor_nodes(self) -> List[Node]:
        successors = self.get_successors_of_node(self.root)
        derived_successors = [node for node in successors if node.state.derived]
        logger.info(f"Found {len(derived_successors)} derived root successor nodes")
        return derived_successors

    def get_decidable_leaf_nodes(self) -> List[Node]:
        decidable_leaf_nodes = [
            node for node in self.get_leaf_nodes()
            if node.state.waiting
        ]
        logger.info(f"Found {len(decidable_leaf_nodes)} decidable leaf nodes")
        return decidable_leaf_nodes

    def get_concludable_nodes(self) -> List[Node]:
        concludable_nodes = [
            node for node in self.get_leaf_nodes()
            if node.state.derived
        ]
        logger.info(f"Found {len(concludable_nodes)} concludable nodes")
        return concludable_nodes
    ##################################################################################################
    class Graph:
    def __init__(self, root_node: Node):
        self.nodes: List[Node] = list()
        self.edges: List[Edge] = list()

        self.root: Node = root_node
        self.add_node(self.root)

        logger.info("Initialized video reasoning graph")

    def get_edges_from_node(self, node: Node) -> List[Edge]:
        edges = [edge for edge in self.edges if edge.source == node]
        logger.info(f"Retrieved {len(edges)} edges from node")
        return edges

    def get_successors_of_node(self, node: Node) -> List[Node]:
        successors = [edge.target for edge in self.get_edges_from_node(node)]
        logger.info(f"Node has {len(successors)} successors")
        return successors

    def get_predecessors_of_node(self, node: Node) -> List[Node]:
        predecessors = [edge.source for edge in self.edges if edge.target == node]
        logger.info(f"Node has {len(predecessors)} predecessors")
        return predecessors

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        logger.info("Added edge to graph")

    def add_edges(self, edges: List[Edge]) -> None:
        self.edges.extend(edges)
        logger.info(f"Added {len(edges)} edges to graph")

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        logger.info("Added node to graph")

    def add_nodes(self, nodes: List[Node]) -> None:
        self.nodes.extend(nodes)
        logger.info(f"Added {len(nodes)} nodes to graph")

    def get_derivable_nodes(self) -> List[Node]:
        derivable_nodes = [node for node in self.nodes if not node.state.derived]
        logger.info(f"Found {len(derivable_nodes)} derivable nodes")
        return derivable_nodes

    def get_unranked_nodes(self) -> List[Node]:
        unranked_nodes = [node for node in self.nodes if not node.state.ranking and node != self.root]
        logger.info(f"Found {len(unranked_nodes)} unranked nodes")
        return unranked_nodes

    def get_leaf_nodes(self) -> List[Node]:
        leaf_nodes = [node for node in self.nodes if not self.get_successors_of_node(node)]
        logger.info(f"Found {len(leaf_nodes)} leaf nodes")
        return leaf_nodes

    def get_derived_root_successor_nodes(self) -> List[Node]:
        successors = self.get_successors_of_node(self.root)
        derived_successors = [node for node in successors if node.state.derived]
        logger.info(f"Found {len(derived_successors)} derived root successor nodes")
        return derived_successors

    def get_decidable_leaf_nodes(self) -> List[Node]:
        decidable_leaf_nodes = [
            node for node in self.get_leaf_nodes()
            if node.state.waiting
        ]
        logger.info(f"Found {len(decidable_leaf_nodes)} decidable leaf nodes")
        return decidable_leaf_nodes

    def get_concludable_nodes(self) -> List[Node]:
        concludable_nodes = [
            node for node in self.get_leaf_nodes()
            if node.state.derived
        ]
        logger.info(f"Found {len(concludable_nodes)} concludable nodes")
        return concludable_nodes
    
    ###############################################################################################################
    
class Graph:
    def __init__(self, root_node: Node):
        self.nodes: List[Node] = list()
        self.edges: List[Edge] = list()

        self.root: Node = root_node
        self.add_node(self.root)

        logger.info("Initialized video reasoning graph")

    def get_edges_from_node(self, node: Node) -> List[Edge]:
        edges = [edge for edge in self.edges if edge.source == node]
        logger.info(f"Retrieved {len(edges)} edges from node")
        return edges

    def get_successors_of_node(self, node: Node) -> List[Node]:
        edges = self.get_edges_from_node(node)
        successors = [edge.target for edge in edges]
        logger.info(f"Node has {len(successors)} successors")
        return successors

    def get_predecessors_of_node(self, node: Node) -> List[Node]:
        predecessors = [edge.source for edge in self.edges if edge.target == node]
        logger.info(f"Node has {len(predecessors)} predecessors")
        return predecessors

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        logger.info("Added edge to graph")

    def add_edges(self, edges: List[Edge]) -> None:
        self.edges.extend(edges)
        logger.info(f"Added {len(edges)} edges to graph")

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        logger.info("Added node to graph")

    def add_nodes(self, nodes: List[Node]) -> None:
        self.nodes.extend(nodes)
        logger.info(f"Added {len(nodes)} nodes to graph")

    def get_derivable_nodes(self) -> List[Node]:
        derivable_nodes = [node for node in self.nodes if not node.state.derived]
        logger.info(f"Found {len(derivable_nodes)} derivable nodes")
        return derivable_nodes

    def get_unranked_nodes(self) -> List[Node]:
        unranked_nodes = [node for node in self.nodes if not node.state.ranking and node != self.root]
        logger.info(f"Found {len(unranked_nodes)} unranked nodes")
        return unranked_nodes

    def get_leaf_nodes(self) -> List[Node]:
        leaf_nodes = [node for node in self.nodes if not self.get_successors_of_node(node)]
        logger.info(f"Found {len(leaf_nodes)} leaf nodes")
        return leaf_nodes

    def get_derived_root_successor_nodes(self) -> List[Node]:
        root_successors = self.get_successors_of_node(self.root)
        derived_successors = [node for node in root_successors if node.state.derived]
        logger.info(f"Found {len(derived_successors)} derived root successor nodes")
        return derived_successors

    def get_decidable_leaf_nodes(self) -> List[Node]:
        decidable_leaf_nodes = [
            node for node in self.get_leaf_nodes()
            if node.state.waiting
        ]
        logger.info(f"Found {len(decidable_leaf_nodes)} decidable leaf nodes")
        return decidable_leaf_nodes

    def get_concludable_nodes(self) -> List[Node]:
        concludable_nodes = [
            node for node in self.get_leaf_nodes()
            if node.state.derived
        ]
        logger.info(f"Found {len(concludable_nodes)} concludable nodes")
        return concludable_nodes