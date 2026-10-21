import onnx

m = onnx.load(r"C:\Users\admin\Desktop\rust_yolo\yolo26n.onnx")
print("ir_version:", m.ir_version, "opsets:", [(o.domain, o.version) for o in m.opset_import])
print("producer:", m.producer_name, m.producer_version)
g = m.graph
print("=== inputs ===")
for i in g.input:
    shape = [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
    print(i.name, shape, i.type.tensor_type.elem_type)
print("=== outputs ===")
for o in g.output:
    shape = [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]
    print(o.name, shape, o.type.tensor_type.elem_type)
print("=== metadata ===")
for f in m.metadata_props:
    v = f.value
    print(f.key, "=", (v[:200] + "..." if len(v) > 200 else v))
print("=== initializer count:", len(g.initializer))
