class Graph:
    def __init__(self, root_node: Node):
        self.nodes = []
        self.edges = []
        self.root: Node = root_node
        self.add_node(self.root)

    def get_edges_from_node(self, node: Node) -> list[Edge]:
        return [edge for edge in self.edges if edge.source == node]

    def get_successors_of_node(self, node: Node) -> list[Node]:
        edges = self.get_edges_from_node(node)
        return [edge.target for edge in edges]

    def get_predecessors_of_node(self, node: Node) -> list[Node]:
        edges = self.get_edges_from_node(node)
        return [edge.source for edge in edges]

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def add_edges(self, edges: list[Edge]):
        self.edges.extend(edges)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_nodes(self, nodes: list[Node]):
        self.nodes.extend(nodes)

    def get_derivable_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not node.state.derived]

    def get_unranked_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not node.state.ranking and node!= self.root]

    def get_leaf_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not self.get_successors_of_node(node)]

    def get_derived_root_successor_nodes(self) -> list[Node]:
        successors = self.get_successors_of_node(self.root)
        return [node for node in successors if node.state.derived]

    def get_decidable_leaf_nodes(self) -> list[Node]:
        leaf_nodes = self.get_leaf_nodes()
        return [node for node in leaf_nodes if node.state == NodeState.WAITING]

    def get_concludable_nodes(self) -> list[Node]:
        leaf_nodes = self.get_leaf_nodes()
        return [node for node in leaf_nodes if node.state.derived]
    #######################################################################################333
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
            return [node for node in self.nodes if not node.state.derived]

        def get_unranked_nodes(self) -> list[Node]:
            return [node for node in self.nodes if not node.state.ranking and node!= self.root]

        def get_leaf_nodes(self) -> list[Node]:
            return [node for node in self.nodes if not self.get_successors_of_node(node)]

        def get_derived_root_successor_nodes(self) -> list[Node]:
            successors = self.get_successors_of_node(self.root)
            return [node for node in successors if node.state.derived]

        def get_decidable_leaf_nodes(self) -> list[Node]:
            leaf_nodes = self.get_leaf_nodes()
            return [node for node in leaf_nodes if node.state == NodeState.WAITING]

        def get_concludable_nodes(self) -> list[Node]:
            leaf_nodes = self.get_leaf_nodes()
            return [node for node in leaf_nodes if node.state.derived]
    ###################################################################################
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
        return [node for node in self.nodes if not node.state.derived and node!= self.root]

    def get_unranked_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not node.state.ranking and node!= self.root]

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
                 and node.state.waiting)]

    def get_concludable_nodes(self) -> list[Node]:
        return [node for node in self.nodes if
                # consider leaf nodes
                (not self.get_successors_of_node(node)
                 # consider leaf nodes that have states that are derived
                 and node.state.derived)]
    