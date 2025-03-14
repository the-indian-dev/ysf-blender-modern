bl_info = {
    "name": "YSF Blender tools updated for Blender 4",
    "author": "Cobra, Vincentweb, Ritabrata Das",
    "version": (4, 0, 0),
    "blender": (4, 0, 0),
    "location": "File > Import/Export > YSFlight",
    "description": "This allows you to import/export YSFlight SRF/DNM files into latest blender 4",
    "warning": "",
    "doc_url": "https://theindiandev.in",
    "category": "Import-Export",
}

import bpy
from bpy.props import StringProperty, BoolProperty, FloatProperty
from bpy.types import Operator, Panel, AddonPreferences
import os

# Import all modules we've ported
from . import libysfs
from . import ysfsConfig
#from . import DNMExport
#from . import SRFExporter
from . import dnm_import
#from . import libysfsExport

# Registration functions
def register():
    # Register modules
    #DNMExport.register()
    #SRFExporter.register()
    dnm_import.register()

def unregister():
    # Unregister modules
    #DNMExport.unregister()
    #SRFExporter.unregister()
    dnm_import.unregister()

if __name__ == "__main__":
    register()
