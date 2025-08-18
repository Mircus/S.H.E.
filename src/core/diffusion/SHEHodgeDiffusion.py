class SHEHodgeDiffusion:
    """
    Advanced diffusion analysis using Hodge Laplacians
    """
    
    def __init__(self, config: Optional[SHEConfig] = None):
        self.config = config or SHEConfig()
    
    def compute_spectral_properties(self, laplacian: csr_matrix, 
                                  k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of Hodge Laplacian"""
        k = k or min(self.config.spectral_k, laplacian.shape[0] - 1)
        
        if laplacian.shape[0] <= 1:
            return np.array([0.0]), np.array([[1.0]])
        
        try:
            # For symmetric matrices, use eigsh for efficiency
            if k >= laplacian.shape[0] - 1:
                # Use dense solver for small matrices
                eigenvals, eigenvecs = eigh(laplacian.toarray())
            else:
                # Use sparse solver for large matrices
                eigenvals, eigenvecs = eigsh(laplacian, k=k, which='SM', sigma=0.0)
            
            # Sort by eigenvalue
            idx = np.argsort(eigenvals)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            return eigenvals, eigenvecs
            
        except Exception as e:
            logger.warning(f"Spectral computation failed: {e}")
            return np.array([0.0]), np.array([[1.0]])
    
    def compute_diffusion_centrality(self, laplacian: csr_matrix, 
                                   weights: Dict[Any, float],
                                   simplex_list: List[Any]) -> Dict[Any, float]:
        """Compute diffusion centrality for simplices"""
        
        if laplacian.shape[0] == 0:
            return {}
        
        # Create weight vector
        weight_vector = np.array([weights.get(simplex, 1.0) for simplex in simplex_list])
        
        try:
            # Solve diffusion equation: (I + λL)x = w
            # where λ controls diffusion strength
            lambda_diff = 0.1
            A = diags([1.0], shape=laplacian.shape) + lambda_diff * laplacian
            
            if A.shape[0] > 0:
                diffusion_result = spsolve(A, weight_vector)
                
                # Normalize to get centrality scores
                if np.max(np.abs(diffusion_result)) > 0:
                    centrality_scores = np.abs(diffusion_result) / np.max(np.abs(diffusion_result))
                else:
                    centrality_scores = np.ones_like(diffusion_result)
                
                return {simplex: score for simplex, score in zip(simplex_list, centrality_scores)}
            
        except Exception as e:
            logger.warning(f"Diffusion centrality computation failed: {e}")
        
        # Fallback: uniform centrality
        return {simplex: 1.0 for simplex in simplex_list}
    
    def compute_heat_kernel(self, laplacian: csr_matrix, t: float = 1.0) -> np.ndarray:
        """Compute heat kernel exp(-tL)"""
        if laplacian.shape[0] <= 1:
            return np.eye(laplacian.shape[0])
        
        try:
            eigenvals, eigenvecs = self.compute_spectral_properties(laplacian)
            
            # Heat kernel: exp(-t * eigenval)
            heat_eigenvals = np.exp(-t * eigenvals)
            
            # Reconstruct: U * diag(exp(-t*λ)) * U^T
            heat_kernel = eigenvecs @ np.diag(heat_eigenvals) @ eigenvecs.T
            
            return heat_kernel
            
        except Exception as e:
            logger.warning(f"Heat kernel computation failed: {e}")
            return np.eye(laplacian.shape[0])
    
    def hodge_decomposition(self, complex: SHESimplicialComplex, 
                          dimension: int) -> Dict[str, np.ndarray]:
        """Compute Hodge decomposition for k-forms"""
        
        try:
            # Get relevant matrices
            hodge_laplacians = complex.get_hodge_laplacians()
            incidence_matrices = complex.get_incidence_matrices()
            
            if dimension not in hodge_laplacians:
                return {"harmonic": np.array([]), "exact": np.array([]), "coexact": np.array([])}
            
            L_k = hodge_laplacians[dimension]
            
            # Compute kernel (harmonic forms)
            eigenvals, eigenvecs = self.compute_spectral_properties(L_k)
            
            # Harmonic forms: kernel of Laplacian (eigenvalue ≈ 0)
            tol = 1e-6
            harmonic_idx = np.where(np.abs(eigenvals) < tol)[0]
            harmonic_forms = eigenvecs[:, harmonic_idx] if len(harmonic_idx) > 0 else np.array([]).reshape(L_k.shape[0], 0)
            
            # For exact and coexact forms, we'd need boundary operators
            # This is a simplified version
            exact_forms = np.array([]).reshape(L_k.shape[0], 0)
            coexact_forms = np.array([]).reshape(L_k.shape[0], 0)
            
            return {
                "harmonic": harmonic_forms,
                "exact": exact_forms,
                "coexact": coexact_forms
            }
            
        except Exception as e:
            logger.warning(f"Hodge decomposition failed for dimension {dimension}: {e}")
            return {"harmonic": np.array([]), "exact": np.array([]), "coexact": np.array([])}
    
    def analyze_diffusion(self, complex: SHESimplicialComplex) -> DiffusionResult:
        """Comprehensive diffusion analysis"""
        
        hodge_laplacians = complex.get_hodge_laplacians()
        
        all_eigenvals = {}
        all_eigenvecs = {}
        diffusion_maps = {}
        key_diffusers = {}
        hodge_decompositions = {}
        
        for dim, laplacian in hodge_laplacians.items():
            logger.info(f"Analyzing diffusion for dimension {dim}")
            
            # Spectral analysis
            eigenvals, eigenvecs = self.compute_spectral_properties(laplacian)
            all_eigenvals[dim] = eigenvals
            all_eigenvecs[dim] = eigenvecs
            
            # Diffusion map (using first few non-trivial eigenvectors)
            if len(eigenvals) > 1:
                # Skip first eigenvalue (should be 0 or very small)
                start_idx = 1 if eigenvals[0] < 1e-6 else 0
                end_idx = min(start_idx + 3, len(eigenvals))
                diffusion_maps[f"dim_{dim}"] = eigenvecs[:, start_idx:end_idx]
            
            # Key diffusers analysis
            weights = complex.get_simplex_weights(dim)
            simplex_list = complex.get_simplex_list(dim)
            
            if weights and simplex_list:
                centrality = self.compute_diffusion_centrality(laplacian, weights, simplex_list)
                
                # Sort by centrality score
                sorted_diffusers = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                key_diffusers[dim] = sorted_diffusers[:min(10, len(sorted_diffusers))]
            
            # Hodge decomposition
            hodge_decompositions[f"dim_{dim}"] = self.hodge_decomposition(complex, dim)
        
        return DiffusionResult(
            eigenvalues=all_eigenvals,
            eigenvectors=all_eigenvecs,
            diffusion_maps=diffusion_maps,
            key_diffusers=key_diffusers,
            hodge_decomposition=hodge_decompositions
        )
