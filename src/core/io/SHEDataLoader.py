# Enhanced data loader with weighted simplices
class SHEDataLoader:
    """Enhanced data loader with weight support"""
    
    @staticmethod
    def from_weighted_networkx(G, weight_attr: str = 'weight', 
                             include_cliques: bool = True, 
                             max_clique_size: int = 4) -> SHESimplicialComplex:
        """Load from NetworkX graph with weights"""
        complex = SHESimplicialComplex("from_weighted_networkx")
        
        # Add nodes with weights
        for node, data in G.nodes(data=True):
            weight = data.get(weight_attr, 1.0)
            complex.add_node(node, weight=weight, **data)
        
        # Add edges with weights
        for u, v, data in G.edges(data=True):
            weight = data.get(weight_attr, 1.0)
            complex.add_edge((u, v), weight=weight, **data)
        
        # Add higher-order simplices from cliques
        if include_cliques:
            try:
                import networkx as nx
                cliques = list(nx.find_cliques(G))
                for clique in cliques:
                    if 3 <= len(clique) <= max_clique_size:
                        # Weight of clique = average of edge weights
                        edge_weights = []
                        for i in range(len(clique)):
                            for j in range(i + 1, len(clique)):
                                if G.has_edge(clique[i], clique[j]):
                                    edge_weights.append(G[clique[i]][clique[j]].get(weight_attr, 1.0))
                        
                        clique_weight = np.mean(edge_weights) if edge_weights else 1.0
                        complex.add_simplex(list(clique), weight=clique_weight)
            except Exception as e:
                logger.warning(f"Could not compute weighted cliques: {e}")
        
        return complex
