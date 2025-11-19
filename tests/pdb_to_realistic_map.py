#!/usr/bin/env python3
"""
pdb_to_realistic_map.py
----------------------
Convert PDB files to realistic cryo-EM density maps with:
- Multiple separated virions (3 copies with ≥150Å spacing)
- Random orientations
- Realistic CTF effects
- Gaussian/Poisson noise
- Resolution filtering (3-12 Å)
- EMDB-like dimensions and voxel sizes

Requires: EMAN2 (or uses fallback method with Biopython + scipy)
"""

import os
import subprocess
import numpy as np
import mrcfile
from Bio.PDB import PDBParser, PDBIO
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional
import warnings
import hashlib


class PDBToRealisticMap:
    """Convert PDB files to realistic cryo-EM density maps."""
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize converter.
        
        Args:
            cache_dir: Directory to cache converted maps
        """
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), 
            "../data_directories/.cache/pdb_maps"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if EMAN2 is available
        self.has_eman2 = self._check_eman2()
        
        # Default parameters
        self.default_params = {
            'apix': 1.2,              # Voxel size (Å/pixel)
            'resolution': 6.0,        # Target resolution (Å)
            'box_size': 256,          # Box dimensions
            'n_virions': 3,           # Number of virion copies
            'min_separation': 150.0,  # Minimum separation (Å)
            'noise_level': 2.0,       # Noise strength
            'defocus_min': 15000,     # Min defocus (Å)
            'defocus_max': 25000,     # Max defocus (Å)
        }
    
    def _check_eman2(self) -> bool:
        """Check if EMAN2 is installed."""
        try:
            result = subprocess.run(
                ['e2version.py'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _get_cache_key(self, pdb_file: str, params: dict) -> str:
        """Generate cache key from PDB file and parameters."""
        # Create hash from filename and parameters
        key_str = f"{os.path.basename(pdb_file)}_{str(sorted(params.items()))}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_map(self, cache_key: str) -> Optional[str]:
        """Check if cached map exists."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.mrc")
        if os.path.exists(cache_path):
            return cache_path
        return None
    
    def convert(self, pdb_file: str, output_file: str = None, 
                **kwargs) -> str:
        """
        Convert PDB to realistic cryo-EM map.
        
        Args:
            pdb_file: Input PDB file path
            output_file: Output MRC file path (optional)
            **kwargs: Override default parameters
            
        Returns:
            str: Path to output MRC file
        """
        # Merge parameters
        params = {**self.default_params, **kwargs}
        
        # Check cache
        cache_key = self._get_cache_key(pdb_file, params)
        cached_map = self._get_cached_map(cache_key)
        
        if cached_map:
            print(f"[CACHE] Using cached map: {os.path.basename(cached_map)}")
            if output_file and output_file != cached_map:
                import shutil
                shutil.copy(cached_map, output_file)
                return output_file
            return cached_map
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = os.path.join(
                self.cache_dir,
                f"{cache_key}.mrc"
            )
        
        print(f"[PDB→MAP] Converting {os.path.basename(pdb_file)}...")
        
        # Choose conversion method
        if self.has_eman2:
            print(f"[PDB→MAP] Using EMAN2 pipeline (realistic)")
            self._convert_with_eman2(pdb_file, output_file, params)
        else:
            print(f"[PDB→MAP] EMAN2 not found, using fallback method")
            print(f"[PDB→MAP] Warning: Fallback method is less realistic")
            print(f"[PDB→MAP] Install EMAN2 for best results")
            self._convert_fallback(pdb_file, output_file, params)
        
        print(f"[PDB→MAP] ✓ Saved: {os.path.basename(output_file)}")
        return output_file
    
    def _convert_with_eman2(self, pdb_file: str, output_file: str, 
                            params: dict):
        """Convert using EMAN2 (most realistic method)."""
        import tempfile
        
        temp_dir = tempfile.mkdtemp(prefix='pdb2map_')
        
        try:
            # Step 1: Create separated and rotated virions
            multi_pdb = os.path.join(temp_dir, 'multi_virion.pdb')
            self._create_multi_virion_pdb(pdb_file, multi_pdb, params)
            
            # Step 2: PDB → Ideal density map
            ideal_map = os.path.join(temp_dir, 'ideal.mrc')
            cmd = [
                'e2pdb2mrc.py', multi_pdb,
                f'--apix={params["apix"]}',
                f'--res={params["resolution"]}',
                f'--box={params["box_size"]}',
                f'--output={ideal_map}'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Step 3: Generate 2D projections
            particles = os.path.join(temp_dir, 'particles.hdf')
            cmd = [
                'e2project3d.py', ideal_map,
                f'--outfile={particles}',
                '--n=1000',  # Fewer projections for speed
                f'--apix={params["apix"]}'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Step 4: Apply CTF
            cmd = [
                'e2ctf.py', particles,
                f'--defocusmin={params["defocus_min"]}',
                f'--defocusmax={params["defocus_max"]}',
                '--astigmatism=500',
                '--voltage=300',
                '--cs=2.7'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Step 5: Add realistic noise
            noisy = os.path.join(temp_dir, 'noisy.hdf')
            cmd = [
                'e2proc2d.py', particles, noisy,
                f'--process=math.addnoise:noise={params["noise_level"]}'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Step 6: Reconstruct 3D map
            cmd = [
                'e2make3dpar.py',
                f'--input={noisy}',
                f'--output={output_file}',
                '--parallel=thread:4',
                '--keep=0.9',
                '--sym=c1'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
        finally:
            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _convert_fallback(self, pdb_file: str, output_file: str, 
                         params: dict):
        """
        Fallback conversion without EMAN2.
        Less realistic but functional.
        """
        # Step 1: Create multi-virion PDB
        import tempfile
        temp_pdb = tempfile.NamedTemporaryFile(
            suffix='.pdb', delete=False
        ).name
        
        self._create_multi_virion_pdb(pdb_file, temp_pdb, params)
        
        # Step 2: Convert to density map
        density = self._pdb_to_density_grid(
            temp_pdb,
            params['apix'],
            params['box_size']
        )
        
        # Step 3: Apply resolution filtering
        sigma = params['resolution'] / (2.355 * params['apix'])
        density = gaussian_filter(density, sigma=sigma)
        
        # Step 4: Add realistic noise
        noise = np.random.normal(0, params['noise_level'], density.shape)
        density = density + noise
        
        # Step 5: Apply CTF-like effects (simplified)
        density = self._apply_simplified_ctf(
            density, 
            params['defocus_min'],
            params['apix']
        )
        
        # Step 6: Save as MRC
        with mrcfile.new(output_file, overwrite=True) as mrc:
            mrc.set_data(density.astype(np.float32))
            mrc.voxel_size = params['apix']
            mrc.header.origin.x = -params['box_size'] * params['apix'] / 2
            mrc.header.origin.y = -params['box_size'] * params['apix'] / 2
            mrc.header.origin.z = -params['box_size'] * params['apix'] / 2
        
        # Cleanup
        os.unlink(temp_pdb)
    
    def _create_multi_virion_pdb(self, input_pdb: str, output_pdb: str,
                                 params: dict):
        """
        Create PDB with multiple separated and rotated virions.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', input_pdb)
        
        # Get original structure bounds
        coords = []
        for atom in structure.get_atoms():
            coords.append(atom.get_coord())
        coords = np.array(coords)
        
        center = coords.mean(axis=0)
        size = coords.max(axis=0) - coords.min(axis=0)
        
        print(f"[PDB→MAP] Original structure size: {size}")
        print(f"[PDB→MAP] Creating {params['n_virions']} virions...")
        
        # Generate separation vectors
        separations = self._generate_separations(
            params['n_virions'],
            params['min_separation']
        )
        
        # Generate random rotations
        rotations = [self._random_rotation_matrix() 
                    for _ in range(params['n_virions'])]
        
        # Create output structure
        io = PDBIO()
        
        with open(output_pdb, 'w') as f:
            for i in range(params['n_virions']):
                print(f"[PDB→MAP]   Virion {i+1}: "
                      f"offset={separations[i]}, "
                      f"rotation=random")
                
                # Clone and transform each chain
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                # Get original coordinates
                                coord = atom.get_coord() - center
                                
                                # Apply rotation
                                coord = rotations[i] @ coord
                                
                                # Apply translation
                                coord = coord + separations[i]
                                
                                # Write PDB line
                                f.write(self._format_pdb_line(
                                    atom, coord, i
                                ))
    
    def _generate_separations(self, n_virions: int, 
                             min_sep: float) -> np.ndarray:
        """Generate separation vectors for virions."""
        separations = np.zeros((n_virions, 3))
        
        # First virion at origin
        separations[0] = [0, 0, 0]
        
        # Space others along X and Y axes
        if n_virions >= 2:
            separations[1] = [min_sep, 0, 0]
        
        if n_virions >= 3:
            separations[2] = [0, min_sep, 0]
        
        # Additional virions in grid pattern
        for i in range(3, n_virions):
            x = (i % 3) * min_sep
            y = (i // 3) * min_sep
            separations[i] = [x, y, 0]
        
        return separations
    
    def _random_rotation_matrix(self) -> np.ndarray:
        """Generate random 3D rotation matrix."""
        # Random Euler angles
        alpha = np.random.uniform(0, 2*np.pi)  # Z rotation
        beta = np.random.uniform(0, 2*np.pi)   # Y rotation
        gamma = np.random.uniform(0, 2*np.pi)  # X rotation
        
        # Rotation matrices
        Rz = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])
        
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)]
        ])
        
        return Rz @ Ry @ Rx
    
    def _format_pdb_line(self, atom, coord: np.ndarray, 
                        virion_id: int) -> str:
        """Format PDB ATOM line."""
        return (
            f"ATOM  {atom.get_serial_number():5d} "
            f"{atom.get_name():>4s} "
            f"{atom.get_parent().get_resname():3s} "
            f"{chr(65 + virion_id)}"  # Chain ID: A, B, C, ...
            f"{atom.get_parent().get_id()[1]:4d}    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
            f"{atom.get_occupancy():6.2f}"
            f"{atom.get_bfactor():6.2f}          "
            f"{atom.element:>2s}\n"
        )
    
    def _pdb_to_density_grid(self, pdb_file: str, apix: float, 
                            box_size: int) -> np.ndarray:
        """Convert PDB atoms to density grid."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        # Initialize grid
        grid = np.zeros((box_size, box_size, box_size), dtype=np.float32)
        
        # Get atom coordinates
        coords = []
        for atom in structure.get_atoms():
            coords.append(atom.get_coord())
        coords = np.array(coords)
        
        # Center coordinates
        center = coords.mean(axis=0)
        coords -= center
        
        # Convert to grid indices
        origin = -box_size * apix / 2
        indices = ((coords - origin) / apix).astype(int)
        
        # Filter valid indices
        valid = (
            (indices >= 0) & 
            (indices < box_size)
        ).all(axis=1)
        indices = indices[valid]
        
        # Fill grid with Gaussian atoms
        for idx in indices:
            x, y, z = idx
            # 3x3x3 Gaussian kernel
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if (0 <= nx < box_size and 
                            0 <= ny < box_size and 
                            0 <= nz < box_size):
                            r = np.sqrt(dx**2 + dy**2 + dz**2)
                            grid[nx, ny, nz] += np.exp(-r**2 / 2)
        
        return grid
    
    def _apply_simplified_ctf(self, density: np.ndarray, 
                             defocus: float, apix: float) -> np.ndarray:
        """Apply simplified CTF effects in Fourier space."""
        # FFT
        fft = np.fft.fftn(density)
        fft_shifted = np.fft.fftshift(fft)
        
        # Create frequency grid
        shape = density.shape
        fx = np.fft.fftfreq(shape[0], d=apix)
        fy = np.fft.fftfreq(shape[1], d=apix)
        fz = np.fft.fftfreq(shape[2], d=apix)
        FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
        freq = np.sqrt(FX**2 + FY**2 + FZ**2)
        
        # Simple CTF (oscillating contrast)
        wavelength = 0.0197  # 300kV electrons
        ctf = -np.sin(
            np.pi * wavelength * freq**2 * defocus
        )
        
        # Apply CTF
        fft_shifted *= ctf
        
        # Inverse FFT
        result = np.fft.ifftn(np.fft.ifftshift(fft_shifted))
        return np.real(result)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def convert_pdb_to_map(pdb_file: str, output_file: str = None,
                       cache_dir: str = None, **kwargs) -> str:
    """
    Convenience function to convert PDB to realistic cryo-EM map.
    
    Args:
        pdb_file: Input PDB file
        output_file: Output MRC file (optional)
        cache_dir: Cache directory (optional)
        **kwargs: Additional parameters
        
    Returns:
        str: Path to output MRC file
    """
    converter = PDBToRealisticMap(cache_dir=cache_dir)
    return converter.convert(pdb_file, output_file, **kwargs)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdb_to_realistic_map.py <input.pdb> [output.mrc]")
        sys.exit(1)
    
    input_pdb = sys.argv[1]
    output_mrc = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_pdb):
        print(f"Error: File not found: {input_pdb}")
        sys.exit(1)
    
    print("="*70)
    print("PDB TO REALISTIC CRYO-EM MAP CONVERTER")
    print("="*70)
    
    result = convert_pdb_to_map(input_pdb, output_mrc)
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"Output: {result}")