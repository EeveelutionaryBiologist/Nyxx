
import json
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph


class KnowledgeRelationshipGraph:
    def __init__(self, filepath):
        self.filepath = filepath
        # Load the previously stored graph as a Json
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.G = json_graph.node_link_graph(data)
        else:
            self.G = nx.DiGraph() 

    def retrieve_relationships(self, node_name: str, depth: int = 1) -> list[str]:
        """
        Yields information pieces formatted for the LLM context.
        E.g., "User [OWNS] Max", "Max [CHEWS_ON] Shoes"
        """
        # If the graph doesn't know this entity, return nothing
        if not self.G.has_node(node_name):
            return []

        # ego_graph extracts the target node and all neighbors within the 'depth' radius
        subgraph = nx.ego_graph(self.G, node_name, radius=depth, undirected=False)
        
        extracted_facts = []
        for source, target, data in subgraph.edges(data=True):
            predicate = data.get('relation', 'RELATES_TO')
            extracted_facts.append(f"{source} [{predicate}] {target}")
            
        return extracted_facts

    def add_relationship(self, subject: str, predicate: str, object_: str):
        # NetworkX handles duplicates automatically; adding an existing node does nothing
        self.G.add_node(subject)
        self.G.add_node(object_)
        
        # Add or update the edge with the relationship predicate
        self.G.add_edge(subject, object_, relation=predicate)

    def remove_relationship(self, subject: str, object_: str):
        """Removes an edge, and cleans up orphaned nodes if they are left floating."""
        if self.G.has_edge(subject, object_):
            self.G.remove_edge(subject, object_)
            
            # If nodes have 0 connections left after edge removal, delete them
            if self.G.degree(subject) == 0:
                self.G.remove_node(subject)
            if self.G.degree(object_) == 0:
                self.G.remove_node(object_)

    def write_graph(self):
        with open(self.filepath, 'w') as outfile:
            data = json_graph.node_link_data(self.G)
            json.dump(data, outfile, indent=4)
            
    def dump_all_facts(self) -> list[str]:
        """Utility function to view the entire knowledge base."""
        facts = []
        for source, target, data in self.G.edges(data=True):
            predicate = data.get('relation', 'RELATES_TO')
            facts.append(f"{source} [{predicate}] {target}")
        return facts
    