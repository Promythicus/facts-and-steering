#!/usr/bin/env python3
"""Write an edited copy of a GGUF model file. The original is never modified.
    python gguf_patch.py <base.gguf> <out.gguf> <patch.pt> [Q8_0|F16]
    python gguf_patch.py <base.gguf> <out.gguf> reencode:<tensor>[,<tensor>...] [Q8_0|F16]     (no edit: re-encoding only, the control)
A patch is  {tensor name: {"columns": LongTensor, "added": Tensor[rows, len(columns)]}}  (torch.save), in the
orientation torch uses ([out, in]; for blk.N.ffn_down.weight that is [hidden, neurons]).
Every other tensor is copied byte for byte. An edited tensor is decoded, changed and stored as Q8_0 by default
(8 bits a weight: Python cannot write the file's own 4-bit encodings, and 8 bits loses nothing that was there).
The copy records what was done in the metadata key `patch.description`."""
import os, sys, json, hashlib
sys.path.insert(0, os.path.expanduser("~/models/llama.cpp/src/gguf-py"))
import numpy as np, gguf
from gguf import quants


def main(base, out, what, enc="Q8_0"):
    assert not os.path.exists(out), "refusing to overwrite " + out
    enc = gguf.GGMLQuantizationType[enc]; r = gguf.GGUFReader(base)
    if what.startswith("reencode:"):
        patch = {n: None for n in what[9:].split(",")}; desc = {"reencode_only": sorted(patch)}
    else:
        import torch
        patch = torch.load(what); desc = {"patch_file": os.path.basename(what), "sha256": hashlib.sha256(open(what, "rb").read()).hexdigest(), "tensors": {n: len(p["columns"]) for n, p in patch.items()}}
    names = {t.name for t in r.tensors}; assert set(patch) <= names, set(patch) - names
    w = gguf.GGUFWriter(out, arch=r.get_field(gguf.Keys.General.ARCHITECTURE).contents(), endianess=r.endianess)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]; w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    w.add_string("patch.description", json.dumps({"base": os.path.basename(base), "stored_as": enc.name, **desc}))
    new = {}
    for t in r.tensors:
        if t.name in patch:
            x = quants.dequantize(t.data, t.tensor_type).astype(np.float32)
            if patch[t.name] is not None:
                p = patch[t.name]; x[:, p["columns"].numpy()] += p["added"].float().numpy()
            new[t.name] = quants.quantize(x, enc)
            w.add_tensor_info(t.name, new[t.name].shape, new[t.name].dtype, new[t.name].nbytes, enc)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(new.pop(t.name) if t.name in new else t.data, tensor_endianess=r.endianess)
    w.close(); print("wrote", out, round(os.path.getsize(out) / 2**30, 3), "GiB (base", round(os.path.getsize(base) / 2**30, 3), "GiB)", json.dumps(desc))


if __name__ == "__main__":
    main(*sys.argv[1:])
