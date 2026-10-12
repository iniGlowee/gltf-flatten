#!/usr/bin/env python3
"""
gltf-flatten - bake a glTF/GLB's NODE transforms into the vertex data, in pure
Python, no Blender.

For loaders that grab a mesh by name and ignore the node hierarchy (so a model that
relies on node translation/rotation/scale lands in the wrong place or on its side),
this transforms each static mesh's POSITION/NORMAL/TANGENT by its node's WORLD matrix
and then zeroes every node transform - so the geometry is world-space and renders the
same whether or not the loader honours nodes. The Blender-free companion to
gltf-axis-bake.

    python gltf_flatten.py model.glb [out.glb]
    python gltf_flatten.py scene.gltf out.gltf

Dependency-free (Python standard library only). MIT.

Scope / refusals (honest limits): operates on UNCOMPRESSED glTF with FLOAT
POSITION/NORMAL/TANGENT. It refuses (with a clear message) on skins (skinned geometry
ignores its node transform by spec - flattening would be wrong), EXT_meshopt /
KHR_draco compression, sparse accessors, non-float positions (quantized), and a mesh
instanced at two different world matrices (de-instance first). Morph-target deltas are
transformed as directions.
"""
from __future__ import annotations
import argparse
import base64
import json
import math
import os
import struct
import sys

FLOAT = 5126
TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


# ------------------------------------------------------------------ mat4 (column-major, glTF order)

def mat_identity():
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def quat_to_mat(x, y, z, w):
    xx, yy, zz = x + x, y + y, z + z
    return [
        1 - y * yy - z * zz, x * yy + w * zz, x * zz - w * yy, 0,
        x * yy - w * zz, 1 - x * xx - z * zz, y * zz + w * xx, 0,
        x * zz + w * yy, y * zz - w * xx, 1 - x * xx - y * yy, 0,
        0, 0, 0, 1,
    ]


def mat_mul(a, b):  # column-major: c[c*4+r] = sum_k a[k*4+r]*b[c*4+k]
    c = [0.0] * 16
    for col in range(4):
        for row in range(4):
            c[col * 4 + row] = sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
    return c


def node_local(node):
    if "matrix" in node:
        return list(node["matrix"])
    t = node.get("translation", [0, 0, 0])
    r = node.get("rotation", [0, 0, 0, 1])
    s = node.get("scale", [1, 1, 1])
    # M = T * R * S (column-major): scale the rotation's columns 0,1,2 by s, set T.
    sm = quat_to_mat(*r)
    for col in range(3):
        for row in range(3):
            sm[col * 4 + row] *= s[col]
    sm[12], sm[13], sm[14] = t[0], t[1], t[2]
    return sm


def mul_point(m, x, y, z):
    return (m[0] * x + m[4] * y + m[8] * z + m[12],
            m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14])


def mul_dir(m3, x, y, z):  # m3 = [m0..m8] upper-left 3x3 in the same column-major slots
    return (m3[0] * x + m3[3] * y + m3[6] * z,
            m3[1] * x + m3[4] * y + m3[7] * z,
            m3[2] * x + m3[5] * y + m3[8] * z)


def upper3(m):
    return [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]


def normal_matrix(m):
    a = upper3(m)  # column-major 3x3
    # determinant
    det = (a[0] * (a[4] * a[8] - a[5] * a[7])
           - a[3] * (a[1] * a[8] - a[2] * a[7])
           + a[6] * (a[1] * a[5] - a[2] * a[4]))
    if abs(det) < 1e-20:
        return [1, 0, 0, 0, 1, 0, 0, 0, 1]
    inv = 1.0 / det
    # inverse (column-major) then transpose = inverse-transpose for normals
    c = [
        (a[4] * a[8] - a[5] * a[7]) * inv, (a[2] * a[7] - a[1] * a[8]) * inv, (a[1] * a[5] - a[2] * a[4]) * inv,
        (a[5] * a[6] - a[3] * a[8]) * inv, (a[0] * a[8] - a[2] * a[6]) * inv, (a[2] * a[3] - a[0] * a[5]) * inv,
        (a[3] * a[7] - a[4] * a[6]) * inv, (a[1] * a[6] - a[0] * a[7]) * inv, (a[0] * a[4] - a[1] * a[3]) * inv,
    ]
    return c  # already inverse-transpose in column-major direction form


# ------------------------------------------------------------------ IO

def load(path):
    data = open(path, "rb").read()
    if data[:4] == b"glTF":
        jlen = struct.unpack("<I", data[12:16])[0]
        gltf = json.loads(data[20:20 + jlen])
        rest = data[20 + jlen:]
        bin_buf = bytearray()
        if len(rest) >= 8 and rest[4:8] == b"BIN\x00":
            blen = struct.unpack("<I", rest[0:4])[0]
            bin_buf = bytearray(rest[8:8 + blen])
        return gltf, {0: bin_buf}, True, None
    gltf = json.loads(data.decode("utf-8"))
    base = os.path.dirname(os.path.abspath(path))
    bufs = {}
    for i, b in enumerate(gltf.get("buffers", [])):
        uri = b.get("uri", "")
        if uri.startswith("data:"):
            bufs[i] = bytearray(base64.b64decode(uri.split(",", 1)[1]))
        elif uri:
            bufs[i] = bytearray(open(os.path.join(base, uri), "rb").read())
        else:
            bufs[i] = bytearray()
    return gltf, bufs, False, base


def save(path, gltf, bufs, is_glb, base):
    if is_glb:
        b = bufs[0]
        gltf.setdefault("buffers", [{}])[0] = {"byteLength": len(b)}
        j = json.dumps(gltf, separators=(",", ":")).encode()
        j += b" " * ((4 - len(j) % 4) % 4)
        pad = (4 - len(b) % 4) % 4
        bc = bytes(b) + b"\x00" * pad
        total = 12 + 8 + len(j) + 8 + len(bc)
        out = bytearray(b"glTF") + struct.pack("<II", 2, total)
        out += struct.pack("<I", len(j)) + b"JSON" + j
        out += struct.pack("<I", len(bc)) + b"BIN\x00" + bc
        open(path, "wb").write(out)
    else:
        for i, b in enumerate(bufs):
            bb = bufs[i]
            uri = gltf["buffers"][i].get("uri", "")
            if uri.startswith("data:") or not uri:
                gltf["buffers"][i]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(bytes(bb)).decode()
            else:
                open(os.path.join(os.path.dirname(os.path.abspath(path)), uri), "wb").write(bytes(bb))
            gltf["buffers"][i]["byteLength"] = len(bb)
        open(path, "w").write(json.dumps(gltf, separators=(",", ":")))


# ------------------------------------------------------------------ flatten

class Refuse(Exception):
    pass


def world_matrices(gltf):
    """node index -> world matrix (walk the scene hierarchy)."""
    nodes = gltf.get("nodes", [])
    parent = {}
    for ni, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = ni
    cache = {}

    def world(ni):
        if ni in cache:
            return cache[ni]
        local = node_local(nodes[ni])
        w = local if ni not in parent else mat_mul(world(parent[ni]), local)
        cache[ni] = w
        return w

    return {ni: world(ni) for ni in range(len(nodes))}


def flatten(gltf, bufs):
    ext = set(gltf.get("extensionsUsed", []))
    if "EXT_meshopt_compression" in ext or "KHR_draco_mesh_compression" in ext:
        raise Refuse("compressed geometry (meshopt/draco) - flatten the UNCOMPRESSED model first")
    nodes = gltf.get("nodes", [])
    meshes = gltf.get("meshes", [])
    accessors = gltf.get("accessors", [])
    views = gltf.get("bufferViews", [])
    worlds = world_matrices(gltf)

    # map accessor -> the one world matrix that uses it (refuse on conflict)
    used_with = {}
    jobs = []  # (accessor_idx, kind, world_matrix)
    for ni, n in enumerate(nodes):
        mi = n.get("mesh")
        if mi is None:
            continue
        if n.get("skin") is not None:
            raise Refuse("skinned node present - skinned geometry ignores its node transform; flatten is for static meshes")
        w = worlds[ni]
        for prim in meshes[mi].get("primitives", []):
            attrs = prim.get("attributes", {})
            if "JOINTS_0" in attrs:
                raise Refuse("skinned primitive present - not flattenable")
            targets = prim.get("targets", [])
            for key, kind in (("POSITION", "point"), ("NORMAL", "normal"), ("TANGENT", "tangent")):
                a = attrs.get(key)
                if a is None:
                    continue
                prev = used_with.get(a)
                if prev is not None and prev != tuple(round(x, 6) for x in w):
                    raise Refuse(f"accessor {a} is instanced at two different world transforms - de-instance first")
                used_with[a] = tuple(round(x, 6) for x in w)
                jobs.append((a, kind, w))
            for t in targets:  # morph deltas = directions (no translation)
                for key, kind in (("POSITION", "delta"), ("NORMAL", "delta"), ("TANGENT", "delta")):
                    a = t.get(key)
                    if a is not None:
                        jobs.append((a, kind, w))

    def check(a):
        acc = accessors[a]
        if acc.get("componentType") != FLOAT:
            raise Refuse(f"accessor {a} is not FLOAT (quantized?) - unsupported")
        if "sparse" in acc:
            raise Refuse(f"accessor {a} is sparse - unsupported")
        bv = views[acc["bufferView"]]
        return acc, bv

    # transform each accessor's floats IN PLACE
    done = set()
    for a, kind, w in jobs:
        if a in done:
            continue
        done.add(a)
        acc, bv = check(a)
        n = TYPE_N[acc["type"]]
        count = acc["count"]
        buf = bufs[bv.get("buffer", 0)]
        stride = bv.get("byteStride") or (n * 4)
        base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        m3 = upper3(w)
        nm = normal_matrix(w)
        for i in range(count):
            off = base + i * stride
            comps = struct.unpack_from(f"<{n}f", buf, off)
            if kind == "point":
                x, y, z = mul_point(w, comps[0], comps[1], comps[2])
                struct.pack_into("<3f", buf, off, x, y, z)
            elif kind == "normal":
                x, y, z = mul_dir(nm, comps[0], comps[1], comps[2])
                ln = math.sqrt(x * x + y * y + z * z) or 1.0
                struct.pack_into("<3f", buf, off, x / ln, y / ln, z / ln)
            elif kind == "tangent":  # vec4: xyz dir (3x3), keep w
                x, y, z = mul_dir(m3, comps[0], comps[1], comps[2])
                ln = math.sqrt(x * x + y * y + z * z) or 1.0
                struct.pack_into("<3f", buf, off, x / ln, y / ln, z / ln)
            elif kind == "delta":  # morph delta: direction, no translation/renorm
                x, y, z = mul_dir(m3, comps[0], comps[1], comps[2])
                struct.pack_into("<3f", buf, off, x, y, z)
        # the accessor's own min/max are now stale - recompute for POSITION
        if kind == "point":
            lo = [math.inf] * 3
            hi = [-math.inf] * 3
            for i in range(count):
                off = base + i * stride
                p = struct.unpack_from("<3f", buf, off)
                for c in range(3):
                    lo[c] = min(lo[c], p[c]); hi[c] = max(hi[c], p[c])
            acc["min"] = lo
            acc["max"] = hi

    # zero every node transform (geometry is world-space now)
    for n in nodes:
        for k in ("matrix", "translation", "rotation", "scale"):
            n.pop(k, None)

    return len(done)


def main():
    ap = argparse.ArgumentParser(description="Bake glTF node transforms into the vertices (no Blender).")
    ap.add_argument("input", help=".glb or .gltf")
    ap.add_argument("output", nargs="?", help="output (default: overwrite input)")
    args = ap.parse_args()
    try:
        gltf, bufs, is_glb, base = load(args.input)
        n = flatten(gltf, bufs)
    except Refuse as e:
        print(f"gltf-flatten: refused - {e}", file=sys.stderr)
        return 2
    out = args.output or args.input
    save(out, gltf, bufs, is_glb, base)
    print(f"{args.input}: baked {n} accessor(s) to world space, zeroed node transforms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
