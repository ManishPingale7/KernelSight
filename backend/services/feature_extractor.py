from typing import Dict, Any

class FeatureExtractor:
    """
    Takes raw instruction counts from PTXParser and computes normalized features
    for machine learning input.
    """
    
    def extract_features(self, ptx_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Computes derived features:
        - mem_ratio
        - arithmetic_intensity
        - branch_density
        - register_pressure (approximate, assuming warp size 32)
        - total_instr
        """
        features = {}
        
        total_ops = ptx_counts.get('total_instructions', 0)
        compute_ops = ptx_counts.get('compute', 0)
        mem_loads = ptx_counts.get('memory_load', 0)
        mem_stores = ptx_counts.get('memory_store', 0)
        mem_ops = mem_loads + mem_stores
        branch_ops = ptx_counts.get('branch', 0)
        reg_count = ptx_counts.get('register_count', 0)
        
        # mem_ratio & branch_density
        if total_ops > 0:
            features['mem_ratio'] = mem_ops / total_ops
            features['branch_density'] = branch_ops / total_ops
        else:
            features['mem_ratio'] = 0.0
            features['branch_density'] = 0.0
            
        # arithmetic_intensity
        if mem_ops > 0:
            features['arithmetic_intensity'] = compute_ops / mem_ops
        else:
            # If no memory ops, arithmetic intensity is effectively infinite. 
            features['arithmetic_intensity'] = float(compute_ops)
            
        # Warp size is typically 32 in NVIDIA GPUs
        features['register_pressure'] = reg_count / 32.0 
        
        features['total_instr'] = float(total_ops)
        
        return features
