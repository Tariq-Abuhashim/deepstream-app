import tensorrt as trt
import numpy as np
import subprocess
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# tested for TensorRT 10
def build_and_serialize_engine(onnx_model_path, engine_path, max_batch_size=1, half=True, int8=False, strip_weights=False):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    cache = config.create_timing_cache(b"")
    config.set_timing_cache(cache, ignore_mismatch=False)

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, "rb") as f:
        if not parser.parse(f.read()):
            print(f"ERROR: Failed to parse the ONNX file {onnx_model_path}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for input in inputs:
        print(f"Model {input.name} shape: {input.shape} {input.dtype}")
    for output in outputs:
        print(f"Model {output.name} shape: {output.shape} {output.dtype}")

    # Create optimization profile
    profile = builder.create_optimization_profile()
    for input in inputs:
        min_shape = [1 if dim == -1 else dim for dim in input.shape]
        opt_shape = [max(1, int(max_batch_size / 2)) if dim == -1 else dim for dim in input.shape]
        max_shape = [max(1, max_batch_size) if dim == -1 else dim for dim in input.shape]

        # Ensure min <= opt <= max
        for i in range(len(min_shape)):
            if min_shape[i] > opt_shape[i]:
                opt_shape[i] = min_shape[i]
            if opt_shape[i] > max_shape[i]:
                max_shape[i] = opt_shape[i]
    
        print(f"min_shape is set to {min_shape}")
        print(f"opt_shape is set to {opt_shape}")
        print(f"max_shape is set to {max_shape}")

        profile.set_shape(input.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    elif int8:
        config.set_flag(trt.BuilderFlag.INT8)

    if strip_weights:
        config.set_flag(trt.BuilderFlag.STRIP_PLAN)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("ERROR: Failed to build engine!")
        return

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

# tested for TensorRT 8
def run_trtexec(onnx_file, engine_file):
    # Command to run trtexec
    command = f"trtexec --onnx={onnx_file} --saveEngine={engine_file}"
    
    # Run the command
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Print the output and error messages (if any)
    print("Output:", result.stdout.decode('utf-8'))
    print("Error:", result.stderr.decode('utf-8'))

# cuda-memcheck python3 detect_trt.py
# cuda-gdb python3 detect_trt.py
# Example usage
def main(args):
    # Example usage
    tensorrt_version = trt.__version__
    if tensorrt_version.startswith('10'):
        build_and_serialize_engine("model.onnx", "model.engine",max_batch_size=args.max_batch_size) # Set appropriate batch size
    elif tensorrt_version.startswith('8'):
        run_trtexec("model.onnx", "model.engine")
    print("--- finished ---")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict and recreate images with missing red channel.')
    parser.add_argument('--max_batch_size', type=int, default=1, help='Maximum input dimensions size')
    args = parser.parse_args()
    main(args)
