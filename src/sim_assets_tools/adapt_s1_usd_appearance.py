#!/usr/bin/env python3
"""Build a non-destructive USD appearance layer from the S1 MuJoCo assets.

Run with Python containing numpy, scipy, Pillow, trimesh and pxr (usd-core), or from an
initialized Isaac Sim Python session. The original USD is never rewritten.
"""

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


ROOT = Path(__file__).resolve().parents[2]
DESCRIPTIONS = ROOT / "astribot_descriptions"
MJCF = DESCRIPTIONS / "mjcf/astribot_s1_mjcf"
USD_DIR = DESCRIPTIONS / "usd/astribot_s1_usd"
ROBOT = "/World/astribot_whole_body"


def read_obj(path):
    """Keep OBJ's independent corner UV/normal indices, including UV seams."""
    points, uvs, normals, faces, uv_indices, normal_indices = [], [], [], [], [], []
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                points.append([float(x) for x in fields[1:4]])
            elif fields[0] == "vt":
                uvs.append([float(x) for x in fields[1:3]])
            elif fields[0] == "vn":
                normals.append([float(x) for x in fields[1:4]])
            elif fields[0] == "f":
                corners = [x.split("/") for x in fields[1:]]
                if len(corners) != 3 or any(len(x) != 3 or not all(x) for x in corners):
                    raise ValueError(f"Expected UV-mapped triangles with normals: {path}")
                # The repository's generated OBJ files use positive indices.
                if any(int(i) <= 0 for corner in corners for i in corner):
                    raise ValueError(f"Expected positive OBJ indices: {path}")
                faces.append([int(x[0]) - 1 for x in corners])
                uv_indices.extend(int(x[1]) - 1 for x in corners)
                normal_indices.extend(int(x[2]) - 1 for x in corners)
    return (np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int32),
            np.asarray(uvs, dtype=np.float32)[uv_indices],
            np.asarray(normals, dtype=np.float32)[normal_indices])


def source_assets():
    asset_file = MJCF / "assets/astribot_s1_asset_for_gripper.xml"
    asset = ET.parse(asset_file).getroot().find("asset")
    textures = {x.get("name"): (asset_file.parent / x.get("file")).resolve()
                for x in asset.findall("texture")}
    meshes = {x.get("name"): (asset_file.parent / x.get("file")).resolve()
              for x in asset.findall("mesh")}
    materials = {x.get("name"): dict(x.attrib) for x in asset.findall("material")}
    assignments = {}
    files = [MJCF / "model_with_effector/astribot_s1_model_with_gripper.xml"]
    files += sorted(MJCF.glob("astribot_gripper/*/*_model.xml"))
    for path in files:
        for geom in ET.parse(path).getroot().iter("geom"):
            if geom.get("type") != "mesh" or geom.get("contype") != "0":
                continue
            name = geom.get("mesh")
            if geom.get("material"):
                assignment = geom.get("material")
            else:
                rgba = geom.get("rgba", "1 1 1 1")
                assignment = "M_gripper_shell" if name == "gripper_base_link" else "M_gripper_finger"
                materials[assignment] = {"rgba": rgba, "specular": "0.2", "shininess": "0.15"}
            if name in assignments and assignments[name] != assignment:
                raise ValueError(f"Conflicting materials for {name}")
            assignments[name] = assignment
    return textures, meshes, materials, assignments


def create_material(stage, name, data, textures, output_dir):
    material = UsdShade.Material.Define(stage, f"{ROBOT}/Looks/Mujoco/{name}")
    rgba = tuple(float(x) for x in data.get("rgba", "1 1 1 1").split())
    shininess = float(data.get("shininess", "0.3"))
    # MuJoCo's normalized Phong shininess is only an approximation of PBR roughness.
    roughness = float(np.clip(np.sqrt(2 / (2 + 128 * shininess)), 0.2, 0.9))
    mdl = UsdShade.Shader.Define(stage, f"{material.GetPath()}/OmniPBR")
    mdl.SetSourceAsset("OmniPBR.mdl", "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgba[:3]))
    mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    mdl.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0)
    mdl.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(float(data.get("specular", "0.4")))
    mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")

    preview = UsdShade.Shader.Define(stage, f"{material.GetPath()}/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgba[:3]))
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0)
    preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(rgba[3])
    preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    if "texture" in data:
        texture = textures[data["texture"]]
        if not texture.is_file():
            raise FileNotFoundError(texture)
        relative = os.path.relpath(texture, output_dir)
        mdl.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(relative))
        mdl.CreateInput("diffuse_texture_color_space", Sdf.ValueTypeNames.Token).Set("sRGB")
        reader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/UVReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        sampler = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Texture")
        sampler.CreateIdAttr("UsdUVTexture")
        sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(relative))
        sampler.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        sampler.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        sampler.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        preview.GetInput("diffuseColor").ConnectToSource(sampler.ConnectableAPI(), "rgb")
    return material


def write_mesh(mesh, points, faces, normals, uvs=None):
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.flatten()))
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateExtentAttr().Set([Gf.Vec3f(*points.min(0).tolist()),
                                Gf.Vec3f(*points.max(0).tolist())])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    if uvs is not None:
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray.FromNumpy(uvs))


def pad_texture(source, corners, destination):
    """Extend UV island interiors into gutters without averaging the background.

    The source atlases have pale, blurred gutters. RTX's bilinear/mipmap sampling
    exposes them as white lines on dark panels. Keep interior pixels (including
    the chest logo), and fill the island edges/gutters from nearest interior.
    """
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_erosion, distance_transform_edt, label

    image = Image.open(source).convert("RGB")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    size = np.asarray(image.size) - 1
    pixels = corners.copy()
    pixels[:, 1] = 1 - pixels[:, 1]
    pixels = (pixels * size).astype(int).reshape(-1, 3, 2)
    for triangle in pixels:
        draw.polygon([tuple(p) for p in triangle], fill=255)
    occupied = np.asarray(mask) > 0
    interior = binary_erosion(occupied, iterations=2)
    # Do not lose tiny islands which have no two-pixel interior.
    components, _ = label(occupied)
    surviving = np.unique(components[interior])
    interior |= occupied & ~np.isin(components, surviving)
    nearest = distance_transform_edt(~interior, return_distances=False, return_indices=True)
    padded = np.asarray(image)[tuple(nearest)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(padded).save(destination)


def build(base, output):
    textures, meshes, definitions, assignments = source_assets()
    output.parent.mkdir(parents=True, exist_ok=True)
    appearance = output.with_name("astribot_s1_appearance.usdc")
    # Build in memory so regenerating never opens/truncates an existing layer in use.
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths = [str(base.resolve())]
    textured_geometry = {}
    for name, data in definitions.items():
        if "texture" not in data:
            continue
        source_name = next(mesh for mesh, material in assignments.items() if material == name)
        geometry = read_obj(meshes[source_name])
        textured_geometry[source_name] = geometry
        texture_name = data["texture"]
        padded_path = output.parent / "textures" / textures[texture_name].name
        pad_texture(textures[texture_name], geometry[2], padded_path)
        textures[texture_name] = padded_path
    materials = {name: create_material(stage, name, data, textures, output.parent)
                 for name, data in definitions.items()}
    robot = stage.GetPrimAtPath(ROBOT)
    if not robot:
        raise ValueError(f"Missing {ROBOT} in {base}")
    visuals = [p for p in Usd.PrimRange(robot) if p.GetName() == "visuals"]
    for prim in visuals:
        prim.SetInstanceable(False)
    count, textured = 0, 0
    for prim in Usd.PrimRange(robot):
        if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in str(prim.GetPath()):
            continue
        parent = prim.GetParent()
        name = parent.GetName()
        source_name = name.removeprefix("astribot_") if name.startswith("astribot_gripper_") else name
        if source_name not in assignments:
            raise ValueError(f"No MuJoCo appearance for {prim.GetPath()} ({source_name})")
        material_name = assignments[source_name]
        # Bind on the visual Xform to replace the imported strongerThanDescendants binding.
        UsdShade.MaterialBindingAPI.Apply(parent).Bind(materials[material_name],
                                                       UsdShade.Tokens.strongerThanDescendants)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[material_name])
        if "texture" in definitions[material_name]:
            mesh = UsdGeom.Mesh(prim)
            points, faces, uvs, normals = textured_geometry[source_name]
            old_points = np.asarray(mesh.GetPointsAttr().Get())
            if not np.allclose([old_points.min(0), old_points.max(0)],
                               [points.min(0), points.max(0)], atol=2e-5, rtol=0):
                raise ValueError(f"OBJ and USD coordinate bounds differ: {name}")
            write_mesh(mesh, points, faces, normals, uvs)
            textured += 1
        count += 1

    # The fixed-base USD omitted the four wheel visuals. Add their MJCF geometry
    # under the chassis only, without introducing joints, mass or collision shapes.
    import trimesh
    model = ET.parse(MJCF / "model_with_effector/astribot_s1_model_with_gripper.xml").getroot()
    wheels = [b for b in model.iter("body") if b.get("name", "").startswith("wheel_")]
    for body in wheels:
        name = body.get("name")
        xform = UsdGeom.Xform.Define(stage, f"{ROBOT}/astribot_torso_base/appearance_wheels/{name}")
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, body.get("pos").split())))
        q = np.asarray(list(map(float, body.get("quat").split())))
        q /= np.linalg.norm(q)
        xform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:].tolist())))
        geometry = trimesh.load_mesh(meshes[name], process=False)
        points = np.asarray(geometry.vertices, dtype=np.float32)
        faces = np.asarray(geometry.faces, dtype=np.int32)
        normals = np.repeat(np.asarray(geometry.face_normals, dtype=np.float32), 3, axis=0)
        mesh = UsdGeom.Mesh.Define(stage, f"{xform.GetPath()}/mesh")
        write_mesh(mesh, points, faces, normals)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[assignments[name]])

    # The appearance layer contains only opinions; the wrapper composes it over
    # the original so physics and cameras continue to follow future base changes.
    stage.GetRootLayer().subLayerPaths = []
    stage.GetRootLayer().Export(str(appearance))
    wrapper = Sdf.Layer.CreateAnonymous("wrapper.usda")
    wrapper.subLayerPaths = [appearance.name, os.path.relpath(base, output.parent)]
    wrapper.defaultPrim = "World"
    wrapper.pseudoRoot.SetInfo("upAxis", "Z")
    wrapper.pseudoRoot.SetInfo("metersPerUnit", 1.0)
    wrapper.Export(str(output))
    print(f"Adapted {count} visual meshes: {textured} OBJ/PNG pairs, "
          f"{count - textured} solid materials, {len(wheels)} wheel visuals.")
    print(f"Model: {output}\nAppearance: {appearance}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=USD_DIR / "astribot_s1.usda")
    parser.add_argument("--output", type=Path, default=USD_DIR / "astribot_s1_textured.usda")
    args = parser.parse_args()
    if args.output.resolve() == args.base.resolve():
        parser.error("--output must differ from --base")
    build(args.base.resolve(), args.output.resolve())
