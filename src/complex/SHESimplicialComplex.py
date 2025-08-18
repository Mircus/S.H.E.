class SHESimplicialComplex:
    """
    Enhanced wrapper around TopoX SimplicialComplex with diffusion capabilities
    """
    
    def __init__(self, name: str = "SHE_Complex", config: Optional[SHEConfig] = None):
        if not TOPOX_AVAILABLE:
            raise ImportError("TopoX is required for SHE. Install with: pip install toponetx")
            
        self.name = name
        self.config = config or SHEConfig()
        self.complex = SimplicialComplex()
        self.node_features = {}
        self.edge_features = {}
        self.face_features = {}
        self.metadata = {}
        self._cached_matrices = {}
        
    def add_node(self, node_id: Any, features: Optional[Dict[str, Any]] = None, **attr):
        """Add a node (0-simplex) to the complex"""
        self.complex.add_node(node_id, **attr)
        if features:
            self.node_features[node_id] = features
    
    def add_edge(self, edge: Tuple, features: Optional[Dict[str, Any]] = None, **attr):
        """Add an edge (1-simplex) to the complex"""
        self.complex.add_simplex(edge, rank=1, **attr)
        if features:
            self.edge_features[edge] = features
    
    def add_simplex(self, simplex: Union[List, Tuple], rank: Optional[int] = None, 
                   features: Optional[Dict[str, Any]] = None, **attr):
        """Add a general k-simplex to the complex"""
        if rank is None:
            rank = len(simplex) - 1
        
        self.complex.add_simplex(simplex, rank=rank, **attr)
        
        if features:
            if rank == 0:
                self.node_features[simplex[0]] = features
            elif rank == 1:
                self.edge_features[tuple(sorted(simplex))] = features
            elif rank == 2:
                self.face_features[tuple(sorted(simplex))] = features
    
    def get_hodge_laplacians(self, use_cache: bool = True) -> Dict[int, csr_matrix]:
        """Get Hodge Laplacian matrices for all dimensions using TopoX"""
        if use_cache and 'hodge_laplacians' in self._cached_matrices:
            return self._cached_matrices['hodge_laplacians']
        
        laplacians = {}
        max_dim = min(self.complex.dim, self.config.max_dimension)
        
        for k in range(max_dim + 1):
            try:
                # Use TopoX's built-in Hodge Laplacian computation
                L_k = self.complex.hodge_laplacian_matrix(rank=k, signed=True)
                if L_k is not None and L_k.shape[0] > 0:
                    laplacians[k] = L_k.tocsr()
                    logger.info(f"Computed Hodge Laplacian L_{k} with shape {L_k.shape}")
                else:
                    logger.warning(f"Empty or None Hodge Laplacian for dimension {k}")
            except Exception as e:
                logger.warning(f"Could not compute Hodge Laplacian L_{k}: {e}")
        
        if use_cache:
            self._cached_matrices['hodge_laplacians'] = laplacians
        
        return laplacians
    
    def get_incidence_matrices(self, use_cache: bool = True) -> Dict[str, csr_matrix]:
        """Get boundary/incidence matrices using TopoX"""
        if use_cache and 'incidence_matrices' in self._cached_matrices:
            return self._cached_matrices['incidence_matrices']
        
        matrices = {}
        max_dim = min(self.complex.dim, self.config.max_dimension)
        
        for k in range(max_dim):
            try:
                # Boundary matrix from k+1 to k simplices
                B_k = self.complex.incidence_matrix(rank=k, to_rank=k+1, signed=True)
                if B_k is not None and B_k.shape[0] > 0:
                    matrices[f"B_{k}"] = B_k.tocsr()
                    logger.info(f"Computed incidence matrix B_{k} with shape {B_k.shape}")
            except Exception as e:
                logger.warning(f"Could not compute incidence matrix B_{k}: {e}")
        
        if use_cache:
            self._cached_matrices['incidence_matrices'] = matrices
        
        return matrices
    
    def get_simplex_weights(self, dimension: int) -> Dict[Any, float]:
        """Extract weights for simplices of given dimension"""
        weights = {}
        
        try:
            for simplex in self.complex.skeleton(dimension):
                attrs = self.complex.get_simplex_attributes(simplex, dimension)
                weight = attrs.get('weight', 1.0)
                weights[simplex] = weight
        except Exception as e:
            logger.warning(f"Could not extract weights for dimension {dimension}: {e}")
        
        return weights
    
    def get_simplex_list(self, dimension: int) -> List[Any]:
        """Get ordered list of simplices for a given dimension"""
        try:
            return list(self.complex.skeleton(dimension))
        except:
            return []
