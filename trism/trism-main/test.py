from trism import TritonModel
import numpy as np
# Create triton model.
model = TritonModel(
  model="viencoder",     # Model name.
  version=1,            # Model version.
  url="localhost:8001", # Triton Server URL.
  grpc=True             # Use gRPC or Http.
)

# View metadata.
for inp in model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
  print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")

# Inference.
outputs = model.run(data = [np.array(["maidz"])])
print(outputs)