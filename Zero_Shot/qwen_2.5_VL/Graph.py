"""Here, you are an very experienced Programmer, who is very good at programming under others' instruction
and can also complete code with many blank space or gaps very well 
you can also perfectly finish the code given the part of a frameworks to make sure it functional fully.especially using Python completely satisfying the requirement.
Next please you complete the code, including classes and functions for my algorithm framework under my instruction and decription

please help me to writin a class named Graph , it takes root_node: Node as inputs and has following class variables:self.nodes = list[Node]();self.edges = list[Edge]();self.root: Node = root_node;self.add_node(self.root)
the first function is named get_edges_from_node, it takes node as input and returns list[Edge] ; it can derive the edge from list[Edge] if the input node is the edge.source.
the second function is named get_successors_of_node taking node as input and then returns list[Node]. it will use for-loop calling get_edges_from_node(node) function to get edges list sourcing from this node
return all successor of this node via edge.target
the third function is named get_predecessors_of_node, which takes node as inputs and returns list of node
also is via self.get_edges_from_node and edge.source to get predecessors of node
the next several functions are add_edge, add_edges, add_node, add_nodes , they take edges and edges list or node and node list . they will use append or extend to do operation
next the function are get_derivable_nodes, which is used to get node.state that is not derived.
the another function is named get_unranked_nodes , it returns list of node. it will return node if node.state.ranking is false and node is not root
the next function is get_leaf_nodes, it returns list[node] . it will return node whose get_successors_of_node is false
the next function is named get_derived_root_successor_nodes , it will return rootnode's successors and the successors have been derived
the next functionis named get_decidable_leaf_nodes which returns list[node] , the nodes in list have no successor_node and state is waiting then return these node
the next function is named get_concludable_nodes, which returns list[node] , the nodes in list have no successor_node and state is derived then return these node."""