#!/usr/bin/env python3
"""
Inspect TensorRT Engine File
Extracts metadata about how the engine was built
"""

import sys
import os
try:
    import tensorrt as trt
except ImportError:
    print("ERROR: TensorRT Python bindings not installed")
    print("Install with: pip install tensorrt")
    sys.exit(1)

def inspect_engine(engine_path):
    if not os.path.exists(engine_path):
        print(f"ERROR: Engine file not found: {engine_path}")
        return
    
    print("=" * 60)
    print("TensorRT Engine Inspector")
    print("=" * 60)
    print(f"\nEngine file: {engine_path}")
    print(f"File size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB")
    print()
    
    # Create TensorRT logger and runtime
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    try:
        # Load the engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("ERROR: Failed to deserialize engine")
            print("This usually means:")
            print("  1. TensorRT version mismatch")
            print("  2. CUDA version incompatibility")
            print("  3. Corrupted engine file")
            print("\nRECOMMENDATION: Delete and regenerate the engine")
            return
        
        print("✓ Engine loaded successfully!")
        print()
        
        # Get engine info
        print("Engine Information:")
        print("-" * 60)
        print(f"Number of bindings: {engine.num_bindings}")
        print(f"Number of layers: {engine.num_layers}")
        print(f"Device memory size: {engine.device_memory_size / (1024*1024):.2f} MB")
        print(f"Max batch size: {engine.max_batch_size}")
        print(f"Has implicit batch dimension: {engine.has_implicit_batch_dimension}")
        print()
        
        # Binding information
        print("Bindings (Inputs/Outputs):")
        print("-" * 60)
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            is_input = engine.binding_is_input(i)
            
            print(f"  [{i}] {name}")
            print(f"      Type: {'INPUT' if is_input else 'OUTPUT'}")
            print(f"      Data type: {dtype}")
            print(f"      Shape: {shape}")
            print()
        
        # Check for optimization profiles (for dynamic shapes)
        if hasattr(engine, 'num_optimization_profiles'):
            print(f"Optimization profiles: {engine.num_optimization_profiles}")
            print()
        
        print("TensorRT Build Info:")
        print("-" * 60)
        
        # Try to extract version info from the engine
        import subprocess
        result = subprocess.run(
            ['strings', engine_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            
            # Look for version strings
            print("Version strings found in engine:")
            for line in lines:
                if any(keyword in line.lower() for keyword in 
                       ['tensorrt', 'cuda', 'cudnn', 'version']):
                    if len(line) < 100 and len(line) > 5:
                        print(f"  {line}")
        
        print()
        print("=" * 60)
        print("Current System Info:")
        print("=" * 60)
        print(f"TensorRT version: {trt.__version__}")
        
        # Get CUDA version if available
        try:
            import pycuda.driver as cuda
            cuda.init()
            print(f"CUDA devices available: {cuda.Device.count()}")
            if cuda.Device.count() > 0:
                device = cuda.Device(0)
                print(f"Primary GPU: {device.name()}")
                print(f"Compute capability: {device.compute_capability()}")
        except:
            print("pycuda not available - install with: pip install pycuda")
        
        print()
        print("✓ Engine is compatible with current system")
        print()
        
    except Exception as e:
        print(f"\n✗ ERROR loading engine: {e}")
        print()
        print("Common causes:")
        print("  1. TensorRT version mismatch")
        print("     - Engine built with different TensorRT version")
        print("  2. CUDA version incompatibility")
        print("     - Engine built with different CUDA version")
        print("  3. GPU architecture mismatch")
        print("     - Engine built for different GPU compute capability")
        print()
        print("SOLUTION: Delete the engine and regenerate:")
        print(f"  rm {engine_path}")
        print("  Then run your application to auto-regenerate")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_tensorrt_engine.py <engine_file>")
        print("Example: python3 inspect_tensorrt_engine.py model.engine")
        sys.exit(1)
    
    inspect_engine(sys.argv[1])
