# Python TRISM
**TR**iton **I**nference **S**erver **M**odel is a simple Python package that helps us infer with Triton Inference Server easily.
```bash
pip install trism
# Or
pip install https://github.com/hieupth/trism
```
## How to use
```python
from trism import TritonModel

# Create triton model.
model = TritonModel(
  model="my_model",     # Model name.
  version=0,            # Model version.
  url="localhost:8001", # Triton Server URL.
  grpc=True             # Use gRPC or Http.
)

# View metadata.
for inp in model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
  print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")

# Inference.
outputs = model.run(data = [np.array(...)])
```
## License
[GNU AGPL v3.0](LICENSE).<br>
Copyright &copy; 2024 [Hieu Pham](https://github.com/hieupth). All rights reserved.