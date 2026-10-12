# gltf-flatten

**Bake a glTF/GLB's node transforms into the vertex data — in pure Python, no
Blender.**

For loaders that grab a mesh by name and **ignore the node hierarchy** (so a model
that relies on node translation/rotation/scale lands in the wrong place or on its
side), this transforms each static mesh's `POSITION` / `NORMAL` / `TANGENT` by its
node's **world** matrix and then zeroes every node transform. The geometry ends up in
world space, so it renders the same whether or not the loader honours nodes.

It's the headless, Blender-free sibling of
[gltf-axis-bake](https://github.com/iniGlowee/gltf-axis-bake): same goal (orientation
lives in the vertices, not a node), no DCC round-trip.

## Usage

```bash
python gltf_flatten.py model.glb                 # overwrite in place
python gltf_flatten.py model.glb flat.glb        # write a copy
python gltf_flatten.py scene.gltf flat.gltf      # .gltf (embedded or external .bin)
```

```
model.glb: baked 410 accessor(s) to world space, zeroed node transforms -> flat.glb
```

Dependency-free (Python standard library only). Handles `.glb` and `.gltf` (embedded
data-URI or external `.bin`), interleaved or tight buffer views, and morph-target
deltas (transformed as directions).

## What it does, precisely

- Computes each node's **world** matrix by walking the scene hierarchy.
- Transforms `POSITION` by the full 4×4, `NORMAL` by the inverse-transpose 3×3 (and
  renormalises), `TANGENT.xyz` by the 3×3 (renormalises, keeps `w`), and morph-target
  deltas by the 3×3 (no translation).
- Recomputes each `POSITION` accessor's `min`/`max`.
- **Zeroes every node transform**, so a compliant loader *and* a node-ignoring loader
  both render world-space geometry correctly.

Verify the result's orientation/scale with the bounds inspector shipped in
`gltf-axis-bake` (`gltf_bounds.py`).

## Scope & refusals (honest limits)

It operates on **uncompressed glTF with FLOAT** `POSITION`/`NORMAL`/`TANGENT`, and
**refuses with a clear message** (exit 2) rather than corrupt your file when it hits:

- **skins / skinned primitives** — skinned geometry ignores its node transform by the
  glTF spec, so flattening it would be wrong (and unnecessary);
- **`EXT_meshopt_compression` / `KHR_draco_mesh_compression`** — flatten the
  *uncompressed* model first, then compress;
- **sparse accessors**, or **non-float (quantized) positions**;
- a mesh **instanced at two different world transforms** — de-instance first (baking
  one world matrix into a shared accessor would collapse the instances).

This is a tool for **static** meshes/scenes — exactly the case where a node-ignoring
loader bites you.

## Provenance

From the AE1 ("Angel's Egg") engine's asset pipeline, whose loader grabs named meshes
and drops node transforms. `gltf-axis-bake` solves this in Blender; `gltf-flatten`
solves it headless, straight on the binary.

## License

MIT — see [LICENSE](LICENSE). Code and prose only; no models included.
