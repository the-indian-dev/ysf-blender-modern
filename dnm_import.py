# -*- coding: utf-8 -*-
"""
dnm_import.py - Import YSFlight SRF and DNM files
Original Copyright 2009 Vincent A (Vincentweb)
Modified by Ritabrata Das (https://theindiandev.in)
Updated for Blender 4.0
"""

import bpy
import os
import math
import bmesh
from mathutils import Vector, Matrix, Euler
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator
from datetime import datetime
from . import ysfsConfig
from .libysfs import *

bl_info = {
    "name": "Import YSFlight SRF/DNM",
    "author": "VincentWeb, Ritabrata Das",
    "version": (2, 0, 0),
    "blender": (4, 0, 0),
    "location": "File > Import > YSFlight SRF/DNM",
    "description": "Import YSFlight 3D format (.srf/.dnm)",
    "warning": "",
    "wiki_url": "",
    "category": "Import-Export"
}

# Initialize error and warning handlers
e = Error()
w = Warning()
log = Log("logYSFSimport.txt", False)

# Initialize textures
zatex = Ztex("za")
if not zatex.success:
    e.outl(0, "Failed to load some ZA textures, see the console.")

# Global configuration from ysfsConfig
use_triangulation = getattr(ysfsConfig, 'triangulate', False)
animation_sta_length = getattr(ysfsConfig, 'animationSTALength', 10)
animation_fps = getattr(ysfsConfig, 'animationFPS', 24)
animation_end = getattr(ysfsConfig, 'animationEND', 100)
cla_start_anim = getattr(ysfsConfig, 'CLAstartAnim', {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0})

class DNMPart:
    """Data structure for DNM parts"""
    def __init__(self, name="", fileName="", CNT=(0, 0, 0), POS=(0, 0, 0, 0, 0, 0, 0), CLA=0):
        self.name = name
        self.fileName = fileName
        self.CNT = list(CNT)
        self.POS = list(POS)
        self.CLA = CLA
        self.STAlist = []
        self.CLDlist = []
        self.sons = []
        self.father = ""
        self.ZA = {}
        self.ZZ = {}
        self.ZL = {}

def process_srf_data(data_lines, offset=(0, 0, 0)):
    """Process SRF data from lines"""
    vertices = []
    faces = []
    face_colors = []
    face_transparency = []
    face_materials = []

    current_face = None
    face_verts = []
    face_color = (1, 1, 1)
    face_bright = False
    face_za = False
    face_za_val = 0
    face_zz = False
    face_zl = False

    # For ZA, ZZ, ZL processing
    face_idx = 0
    za_dict = {}
    zz_list = []
    zl_list = []

    line_idx = 0
    while line_idx < len(data_lines):
        line = data_lines[line_idx].strip()
        line_idx += 1

        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        cmd = parts[0]

        # New face
        if cmd == 'F' or cmd == 'FAC':
            if current_face is not None and face_verts:
                # Add previous face
                faces.append(face_verts)
                face_colors.append(face_color)
                face_materials.append({
                    'bright': face_bright,
                    'za': face_za,
                    'za_val': face_za_val,
                    'zz': face_zz,
                    'zl': face_zl
                })

            current_face = face_idx
            face_idx += 1
            face_verts = []
            face_bright = False
            face_za = False
            face_za_val = 0
            face_zz = False
            face_zl = False

        # Vertex outside of face context
        elif (cmd == 'V' or cmd == 'VER') and current_face is None:
            # Parse vertex coordinates
            if parts[-1] == 'R':  # Rounded vertex
                rounded = True
                coords = parts[1:-1]
            else:
                rounded = False
                coords = parts[1:]

            try:
                y, z, x = map(float, coords)
                # Apply coordinate conversion and offset
                # Original: vertices.append((x - offset[2], -y + offset[0], z - offset[1]))

                # YSFlight CNT is in (Y, Z, X) format
                try:
                    y_cnt, z_cnt, x_cnt = offset
                except ValueError:
                    print(f"Warning: Malformed CNT value: {offset}")
                    y_cnt, z_cnt, x_cnt = 0, 0, 0

                # First adjust in YSFlight coordinate system
                y_adj = y - y_cnt
                z_adj = z - z_cnt
                x_adj = x - x_cnt

                # Then convert to Blender coordinates
                x_bl = y_adj  # YS Y → Blender X
                y_bl = -x_adj  # YS X → Blender -Y
                z_bl = z_adj  # YS Z → Blender Z

                vertices.append((x_bl, y_bl, z_bl))
            except Exception as ex:
                e.out(f"Cannot read vertex coordinates: {str(ex)}")

        # Vertex inside face context
        elif (cmd == 'V' or cmd == 'VER') and current_face is not None:
            # Parse vertex indices for face
            try:
                face_verts = list(map(int, parts[1:]))
            except Exception as ex:
                e.out(f"Invalid face vertex indices: {str(ex)}")

        # Color
        elif cmd == 'C' or cmd == 'COL':
            if len(parts) == 2:  # 24-bit color
                try:
                    col_value = int(parts[1])
                    rgb_col = Color()
                    rgb_col.from24b(col_value)
                    face_color = (rgb_col.r / 255.0, rgb_col.g / 255.0, rgb_col.b / 255.0)
                except:
                    e.out("Invalid 24-bit color")
            else:  # RGB color
                try:
                    r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
                    face_color = (r / 255.0, g / 255.0, b / 255.0)
                except:
                    e.out("Invalid RGB color")

        # Brightness
        elif cmd == 'B':
            face_bright = True

        # End of section
        elif cmd == 'E' or cmd == 'END':
            if current_face is not None and face_verts:
                # Add the face
                faces.append(face_verts)
                face_colors.append(face_color)
                face_materials.append({
                    'bright': face_bright,
                    'za': face_za,
                    'za_val': face_za_val,
                    'zz': face_zz,
                    'zl': face_zl
                })
                current_face = None

        # ZA transparency info
        elif cmd == 'ZA':
            za_parts = parts[1:]
            for i in range(0, len(za_parts), 2):
                if i+1 < len(za_parts):
                    face_idx = int(za_parts[i])
                    za_val = int(za_parts[i+1])
                    za_dict[face_idx] = za_val

        # ZZ billboard info
        elif cmd == 'ZZ':
            zz_parts = parts[1:]
            zz_list.extend(map(int, zz_parts))

        # ZL halo info
        elif cmd == 'ZL':
            zl_parts = parts[1:]
            zl_list.extend(map(int, zl_parts))

    # Apply ZA, ZZ, ZL data to faces
    for i, props in enumerate(face_materials):
        if i in za_dict:
            props['za'] = True
            props['za_val'] = za_dict[i]
        if i in zz_list:
            props['zz'] = True
        if i in zl_list:
            props['zl'] = True

    return vertices, faces, face_colors, face_materials

def create_mesh_from_srf_data(srf_data, name, offset=(0, 0, 0)):
    """Create a mesh from SRF data string"""
    # Split the data into lines
    data_lines = srf_data.split('\n')

    # Process the SRF data
    vertices, faces, face_colors, face_materials = process_srf_data(data_lines, offset)

    # Create mesh
    mesh = bpy.data.meshes.new(name)

    # Create mesh from data
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Create materials dictionary
    material_dict = {}

    # Add materials and set face properties
    for i, (face, color, mat_props) in enumerate(zip(mesh.polygons, face_colors, face_materials)):
        # Create or get material
        mat_key = (color, mat_props['bright'], mat_props['za'], mat_props['zz'], mat_props['zl'])

        if mat_key not in material_dict:
            mat = bpy.data.materials.new(f"{name}_Material_{i}")
            mat.use_nodes = True
            principled = mat.node_tree.nodes.get('Principled BSDF')

            if principled:
                # Set base color
                principled.inputs['Base Color'].default_value = (*color, 1.0)

                # Handle transparency
                if mat_props['za']:
                    mat.blend_method = 'BLEND'
                    alpha = 1.0 - (mat_props['za_val'] / 255.0)
                    principled.inputs['Alpha'].default_value = alpha

                # Handle emission
                if mat_props['bright']:
                    principled.inputs['Emission Strength'].default_value = 1.0
                    principled.inputs['Emission Color'].default_value = (*color, 1.0)

                # Handle billboard/double-sided
                if mat_props['zz']:
                    mat.use_backface_culling = False

                # Handle ZL (halo)
                if mat_props['zl']:
                    mat.blend_method = 'HASHED'

            material_dict[mat_key] = mat
            mesh.materials.append(mat)

        # Assign material to face
        mat_index = list(material_dict.keys()).index(mat_key)
        face.material_index = mat_index

    return mesh

def read_surf_file(file_path, offset=(0, 0, 0)):
    """Read a SRF file and create a mesh"""
    # Load file content
    try:
        with open(file_path, 'r') as f:
            data_lines = f.readlines()
    except Exception as ex:
        e.outl(0, f"Error opening SRF file: {str(ex)}")
        return None

    # Process the data and create mesh
    name = os.path.basename(file_path).rsplit('.', 1)[0]
    vertices, faces, face_colors, face_materials = process_srf_data(data_lines, offset)

    # Create mesh
    mesh = bpy.data.meshes.new(name)

    # Create mesh from data
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Create materials dictionary
    material_dict = {}

    # Add materials and set face properties
    for i, (face, color, mat_props) in enumerate(zip(mesh.polygons, face_colors, face_materials)):
        # Create or get material
        mat_key = (color, mat_props['bright'], mat_props['za'], mat_props['zz'], mat_props['zl'])

        if mat_key not in material_dict:
            mat = bpy.data.materials.new(f"{name}_Material_{i}")
            mat.use_nodes = True
            principled = mat.node_tree.nodes.get('Principled BSDF')

            if principled:
                # Set base color
                principled.inputs['Base Color'].default_value = (*color, 1.0)

                # Handle transparency
                if mat_props['za']:
                    mat.blend_method = 'BLEND'
                    alpha = 1.0 - (mat_props['za_val'] / 255.0)
                    principled.inputs['Alpha'].default_value = alpha

                # Handle emission
                if mat_props['bright']:
                    principled.inputs['Emission Strength'].default_value = 1.0
                    principled.inputs['Emission Color'].default_value = (*color, 1.0)

                # Handle billboard/double-sided
                if mat_props['zz']:
                    mat.use_backface_culling = False

                # Handle ZL (halo)
                if mat_props['zl']:
                    mat.blend_method = 'HASHED'

            material_dict[mat_key] = mat
            mesh.materials.append(mat)

        # Assign material to face
        mat_index = list(material_dict.keys()).index(mat_key)
        face.material_index = mat_index

    return mesh

def read_dnm_file(file_path):
    """Read a DNM file and create objects"""
    parts = []
    file_directory = os.path.dirname(file_path)

    # Dictionary to store packed meshes by name
    packed_meshes = {}
    # Dictionary to map part names to part data for parent-child relationships
    part_data_dict = {}

    # First pass: extract PCK data and read structure
    with open(file_path, 'r') as f:
        content = f.read()

    lines = content.splitlines()
    i = 0
    line_count = len(lines)

    current_part = None
    in_pck = False
    pck_name = None
    pck_lines = 0
    pck_data = []

    # Process all lines
    while i < line_count:
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        # Handle PCK sections
        if line.startswith('PCK '):
            parts_line = line.split()
            try:
                pck_name = parts_line[1]
                pck_lines = int(parts_line[2])

                # Extract the PCK data
                pck_data = lines[i:i+pck_lines]
                i += pck_lines

                # Create a mesh from the PCK data
                srf_data = "\n".join(pck_data)
                try:
                    mesh = create_mesh_from_srf_data(srf_data, pck_name)
                    packed_meshes[pck_name] = mesh
                    print(f"Successfully loaded packed mesh: {pck_name}")
                except Exception as ex:
                    e.outl(0, f"Error processing packed SRF data '{pck_name}': {str(ex)}")
            except Exception as ex:
                e.outl(0, f"Invalid PCK line format: {line}, error: {str(ex)}")
            continue

        # Process DNM structure
        parts_line = line.split()
        if not parts_line:
            continue

        cmd = parts_line[0]

        if cmd == 'SRF':
            # Start a new part
            if current_part:
                parts.append(current_part)

            current_part = DNMPart()
            current_part.name = line[3:].strip().replace('"', '')

        elif cmd == 'FIL' and current_part:
            current_part.fileName = line[3:].strip().replace('"', '')

        elif cmd == 'CLA' and current_part:
            try:
                current_part.CLA = int(parts_line[1])
            except:
                e.outl(i, "Invalid CLA")

        elif cmd == 'CNT' and current_part:
            try:
                current_part.CNT = list(map(float, parts_line[1:4]))
            except:
                e.outl(i, "Invalid CNT")

        elif cmd == 'POS' and current_part:
            try:
                current_part.POS = list(map(float, parts_line[1:]))
                if len(current_part.POS) == 6:
                    current_part.POS.append(0)  # Add visibility if not present
            except:
                e.outl(i, "Invalid POS")

        elif cmd == 'STA' and current_part:
            try:
                current_part.STAlist.append(list(map(float, parts_line[1:])))
            except:
                e.outl(i, "Invalid STA")

        elif cmd == 'CLD' and current_part:
            cld_name = line[3:].strip().replace('"', '')
            current_part.CLDlist.append(cld_name)

        elif cmd == 'END':
            if current_part:
                parts.append(current_part)
                current_part = None

    # Store parts by name for easy lookup
    for part in parts:
        part_data_dict[part.name] = part

    # Set up parent-child relationships
    for part in parts:
        for child_name in part.CLDlist:
            if child_name in part_data_dict:
                part_data_dict[child_name].father = part

    # Create objects dictionary
    objects = {}

    # Create all basic objects first
    for part in parts:
        if part.fileName:
            mesh = None

            # First check if we have this mesh in our packed data
            if part.fileName in packed_meshes:
                mesh = packed_meshes[part.fileName]
            else:
                # If not in packed data, try to load external SRF file
                srf_path = os.path.join(file_directory, part.fileName)
                if os.path.exists(srf_path):
                    # Pass the CNT offset to correctly position the vertices
                    mesh = read_surf_file(srf_path, part.CNT)
                else:
                    e.outl(0, f"SRF file not found: {srf_path}")

            # Create object if we have a mesh
            if mesh:
                obj = bpy.data.objects.new(part.name, mesh)
                bpy.context.collection.objects.link(obj)
                objects[part.name] = obj

                # Set custom property for CLA
                obj['CLA'] = part.CLA
            else:
                # Create empty if no mesh
                obj = bpy.data.objects.new(part.name, None)
                bpy.context.collection.objects.link(obj)
                objects[part.name] = obj
                obj['CLA'] = part.CLA
        else:
            # Create empty
            obj = bpy.data.objects.new(part.name, None)
            bpy.context.collection.objects.link(obj)
            objects[part.name] = obj
            obj['CLA'] = part.CLA

    # Create dictionary to map part names to empty objects for parenting
    empty_objects = {}

    # Create empty objects as intermediaries for proper parenting
    for part in parts:
        # Create an empty object for each part
        empty = bpy.data.objects.new(part.name + "_Empty", None)
        empty.empty_display_size = 0.1
        empty.empty_display_type = 'PLAIN_AXES'
        bpy.context.collection.objects.link(empty)
        empty_objects[part.name] = empty

        # Set position and rotation for empty object
        if part.name in objects:
            obj = objects[part.name]
            # Parent the mesh object to its empty
            obj.parent = empty

            # Apply transformations to empty
            try:
                y_ys, z_ys, x_ys, heading, pitch, roll, visib = part.POS

                # Convert to Blender coordinates
                x_bl = y_ys  # YS Y → Blender X
                y_bl = -x_ys  # YS X → Blender -Y
                z_bl = z_ys  # YS Z → Blender Z

                # Convert angles
                angles = Angles(-roll, -pitch, heading)
                angles.YS2Radian()

                # Set position and rotation to empty
                empty.location = (x_bl, y_bl, z_bl)
                empty.rotation_euler = (angles.ax, angles.ay, angles.az)

                # Set visibility based on visib parameter
                if visib == 0:
                    empty.hide_viewport = True
                    obj.hide_viewport = True
            except Exception as ex:
                e.outl(0, f"Error setting position for {part.name} empty: {str(ex)}")

    # Set up parent-child relationships properly using the empties
    for part in parts:
        for child_name in part.CLDlist:
            if part.name in empty_objects and child_name in empty_objects:
                try:
                    # Get the empty objects
                    child_empty = empty_objects[child_name]
                    parent_empty = empty_objects[part.name]

                    # When setting parent relationship, we need to adjust for CNT offset
                    if part.CNT != [0, 0, 0]:
                        # Convert CNT to Blender coordinates
                        y_cnt, z_cnt, x_cnt = part.CNT
                        x_bl_cnt = y_cnt  # YS Y → Blender X
                        y_bl_cnt = -x_cnt  # YS X → Blender -Y
                        z_bl_cnt = z_cnt  # YS Z → Blender Z

                        # Store current location
                        loc = child_empty.location

                        # Adjust location
                        child_empty.location = (loc[0] - x_bl_cnt,
                                               loc[1] - y_bl_cnt,
                                               loc[2] - z_bl_cnt)

                    # Set parent relationship
                    child_empty.parent = parent_empty
                except Exception as ex:
                    e.outl(0, f"Error setting parent relationship between {part.name} and {child_name}: {str(ex)}")

    # Create animation if there are STA entries
    for part in parts:
        if part.name in empty_objects and part.STAlist:
            empty = empty_objects[part.name]

            # Set keyframes for each STA
            for i, sta in enumerate(part.STAlist):
                frame = i * animation_sta_length + 1 + cla_start_anim.get(part.CLA, 0)

                # Set frame
                bpy.context.scene.frame_set(frame)

                try:
                    # STA format is also: Y, Z, X, heading, pitch, roll, visib
                    y_ys, z_ys, x_ys, heading, pitch, roll, visib = sta

                    # Convert to Blender coordinates
                    x_bl = y_ys  # YS Y → Blender X
                    y_bl = -x_ys  # YS X → Blender -Y
                    z_bl = z_ys  # YS Z → Blender Z

                    # Convert angles
                    angles = Angles(-roll, -pitch, heading)
                    angles.YS2Radian()

                    # Set properties
                    empty.location = (x_bl, y_bl, z_bl)
                    empty.rotation_euler = (angles.ax, angles.ay, angles.az)
                    empty.hide_viewport = (visib == 0)

                    if part.name in objects:
                        objects[part.name].hide_viewport = (visib == 0)

                    # Insert keyframes
                    empty.keyframe_insert(data_path="location", frame=frame)
                    empty.keyframe_insert(data_path="rotation_euler", frame=frame)
                    empty.keyframe_insert(data_path="hide_viewport", frame=frame)

                    if part.name in objects:
                        objects[part.name].keyframe_insert(data_path="hide_viewport", frame=frame)

                except Exception as ex:
                    e.outl(0, f"Error setting animation frame {frame} for {part.name}: {str(ex)}")

    # Set scene animation settings
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = animation_end
    bpy.context.scene.render.fps = animation_fps

    # Ensure all objects are properly visible unless explicitly hidden
    for obj in bpy.data.objects:
        if not obj.hide_viewport:  # Only reset those that aren't deliberately hidden
            obj.hide_viewport = False
            obj.hide_render = False

    return objects

class IMPORT_OT_ysfs_srf(Operator, ImportHelper):
    """Import YSFlight SRF File"""
    bl_idname = "import_mesh.ysf_srf"
    bl_label = "Import YSFlight SRF"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".srf"
    filter_glob: StringProperty(
        default="*.srf",
        options={'HIDDEN'},
    )

    def execute(self, context):
        # Import SRF file
        mesh = read_surf_file(self.filepath)
        if mesh:
            obj = bpy.data.objects.new(os.path.basename(self.filepath)[:-4], mesh)
            context.collection.objects.link(obj)

            # Select and make active
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj

            self.report({'INFO'}, f"Imported SRF: {self.filepath}")
            return {'FINISHED'}
            for obj in bpy.data.objects:
                obj.hide_viewport = False # Reset the monitor-icon visibility (viewport)
                obj.hide_render = False # Reset the camera-icon visibility (render)
        else:
            self.report({'ERROR'}, f"Failed to import SRF: {self.filepath}")
            return {'CANCELLED'}

class IMPORT_OT_ysfs_dnm(Operator, ImportHelper):
    """Import YSFlight DNM File"""
    bl_idname = "import_mesh.ysf_dnm"
    bl_label = "Import YSFlight DNM"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".dnm"
    filter_glob: StringProperty(
        default="*.dnm",
        options={'HIDDEN'},
    )

    def execute(self, context):
        # Import DNM file
        objects = read_dnm_file(self.filepath)
        if objects:
            # Select the objects
            bpy.ops.object.select_all(action='DESELECT')
            for obj in objects.values():
                obj.select_set(True)

            # Set active object to the first one
            if objects:
                context.view_layer.objects.active = next(iter(objects.values()))

            for obj in bpy.data.objects:
                obj.hide_viewport = False # Reset the monitor-icon visibility (viewport)
                obj.hide_render = False # Reset the camera-icon visibility (render)

            self.report({'INFO'}, f"Imported DNM: {self.filepath}")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, f"Failed to import DNM: {self.filepath}")
            return {'CANCELLED'}

def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_ysfs_srf.bl_idname, text="YSFlight SRF (.srf)")
    self.layout.operator(IMPORT_OT_ysfs_dnm.bl_idname, text="YSFlight DNM (.dnm)")

def register():
    bpy.utils.register_class(IMPORT_OT_ysfs_srf)
    bpy.utils.register_class(IMPORT_OT_ysfs_dnm)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(IMPORT_OT_ysfs_srf)
    bpy.utils.unregister_class(IMPORT_OT_ysfs_dnm)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()
