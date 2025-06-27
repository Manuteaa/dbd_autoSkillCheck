import tensorrt as trt

onnx_model_path = "model.onnx"
engine_file = "model_fp32.engine"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags=network_flags)

print("🔄 Parsing ONNX model...")
parser = trt.OnnxParser(network, logger)
with open(onnx_model_path, 'rb') as model_file:
    if not parser.parse(model_file.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("❌ Failed to parse ONNX model")

# Create builder config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB

# Set optimization profile
profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
config.add_optimization_profile(profile)
print("⚙️  Creating TensorRT engine...")

# Build serialized engine
serialized_engine = builder.build_serialized_network(network, config)


if serialized_engine is None:
    raise RuntimeError("❌ Engine build failed")

print("💾 Exporting engine file...")
with open(engine_file, "wb") as f:
    f.write(serialized_engine)

print(f"✅ Engine saved as '{engine_file}'")
