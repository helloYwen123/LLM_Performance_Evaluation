# #######################################################
# #######################################################
# Based on Description to generate code
# #######################################################
# #######################################################
class Graph:
    def __init__(self, root_node: Node):
        self.nodes = list[Node]()
        self.edges = list[Edge]()
        self.root = root_node
        self.add_node(root_node)

    def get_edges_from_node(self, node: Node) -> List[Edge]:
        """Derives Edges from given node."""
        return [
            edge
            for edge in self.edges
            if edge.source == node
        ]

    def get_successors_of_node(self, node: Node) -> List[Node]:
        """Returns Successors (nodes connected by outgoing edges)."""
        return [
            edge.target
            for edge in self.get_edges_from_node(node)
        ]

    def get_predecessors_of_node(self, node: Node) -> List[Node]:
        """Returns Predecessors (nodes connected by incoming edges)."""
        return [
            edge.source
            for edge in self.get_edges_from_node(node)
        ]

    def add_edge(self, *edges: Edge):
        self.edges.extend(edges)

    def add_edges(self, *edges_lists: List[Edge]):
        for edges in edges_lists:
            self.edges.extend(edges)

    def add_node(self, *nodes: Node):
        self.nodes.extend(nodes)

    def add_nodes(self, *nodes_lists: List[Node]):
        for nodes in nodes_lists:
            self.nodes.extend(nodes)

    def get_derivable_nodes(self) -> List[Node]:
        """Get Nodes where their states haven't yet been derived."""
        return filter(lambda n: isinstance(n.state, Derivable), self.nodes)

    def get_unranked_nodes(self) -> List[Node]:
        """Return unranked nodes - those who aren't ranked but also don't belong at top level."""
        return filter(
            lambda n: (
                hasattr(n.state, 'ranking') and \
                getattr(n.state, 'ranking', False)) and\
                n != self.root,
            self.nodes
        )

    def get_leaf_nodes(self) -> List[Node]:
        """Return leaf nodes - those having zero successors."""
        return filter(
            lambda n: len(self.get_successors_of_node(n)) == 0,
            self.nodes
        )

    def get_derived_root_successor_nodes(self) -> List[Node]:
        """
        Returns Root's successors' nodes that were already derived.
        """
        return set([
            succ
            for pred in map(self.get_predecessors_of_node, self.get_successors_of_node(self.root))
            for succ in self.get_successors_of_node(pred)
            if isinstance(succ.state, Derived)
        ])

    def get_decidable_leaf_nodes(self) -> List[Node]:
        """
        Returns decidable leaves - those having no successors AND waiting for derivation.
        """
        return filter(
            lambda n: (
                len(self.get_successors_of_node(n)) == 0 and
                n.state.wait_for_derivation),
            self.get_leaf_nodes()
        )

    def get_concludable_nodes(self) -> List[Node]:
        """
        Returns concludable nodes - those having no successors AND deriving.
        """
        return filter(
            lambda n: (
                len(self.get_successors_of_node(n)) == 0 and
                type(n.state).__name__.endswith('Derived')),
            self.get_leaf_nodes()
        )
# #######################################################
# #######################################################
# Completion given Half of the code
# \n \n Now i will give you the part of one class which i have finished, help me to complete it based on the above requirment and description and give me the completed code finally.
# #######################################################
# #######################################################
class Graph:
    def __init__(self, root_node: Node):
        self.nodes = list[Node]()
        self.edges = list[Edge]()
        self.root: Node = root_node
        self.add_node(root_node)

        logger.info("Initialized video reasoning graph")

    def get_edges_from_node(self, node) -> list[Edge]:
        return [edge for edge in self.edges if edge.source == node]

    def get_predecessors_of_node(self, node) -> list[Node]:
        return [edge.source for edge in self.get_edges_from_node(node)]

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def add_edges(self, edges: list[Edge]) -> None:
        self.edges.extend(edges)

    def add_nodes(self, nodes: list[Node]) -> None:
        self.nodes.extend(nodes)

    def get_derivable_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not node.state.derived and node != self.root]

    def get_leaf_nodes(self) -> list[Node]:
        return [node for node in self.nodes if len(self.get_successors_of_node(node)) == 0]

    def get_derived_root_successor_nodes(self) -> list[Node]:
        root_successors = self.get_successors_of_node(self.root)
        return [
            succ for succ in root_successors
            if isinstance(succ.state, DerivedStateType)  
        ] 

    def get_decidable_leaf_nodes(self) -> list[Node]:
        return [node for node in self.nodes if (
            not self.get_successors_of_node(node) and
            not node.state.waiting
        )]

    def get_concludable_nodes(self) -> list[Node]:
        return [node for node in self.nodes if (
            not self.get_successors_of_node(node) and
            node.state.derived
        )]
# #######################################################
# #######################################################
# Filling blanks in the code
# \n\n Now i will give you the part of needed class with many blanks which i have finished, help me to fill and complete it based on the following requirment and description. Give me the completed code finally.
# #######################################################
# #######################################################
class Graph:
    def __init__(self, root_node: Node):
        self.nodes = list[Node]()
        self.edges = list[Edge]()  # Initialize empty lists for nodes and edges
        
        self.root: Node = root_node
        self.add_node(self.root)

        logger.info("Initialized video reasoning graph")

    def get_edges_from_node(self, node) -> list[Edge]:
        return [edge for edge in self.edges if edge.source == node]

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
                (not self.get_successors_of_node(node)
                 and not node.state.waiting)]  # Consider only those who aren't waiting

    def get_concludable_nodes(self) -> list[Node]:
        return [node for node in self.nodes if
                (not self.get_successors_of_node(node)
                 and node.state.derived)]  # Only concludable when their state is derived
