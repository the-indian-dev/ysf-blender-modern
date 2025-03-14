## ------------------------ Ground Preferences ---------------------

ysfs_path = "/home/vincentweb/Win/ysflighttest20070405diff"
# Path of the YSFlight installation
# DON'T USE THE CHARACTER '\' but '\\' or '/' !!!!!!!!!!!!!!!!!!!!!!!!!

## ------------------------ Import Preferences ---------------------

triangulate = False
# (True/False)
#True -> all the faces are converted to triangles
#False -> try to make decent quadrilaterals, because Blender doesn't accept faces with more than 4 vertices


CLAstartAnim = {}    
CLAstartAnim[0]   = 30       # Body or gear wheels
CLAstartAnim[1]   = 70     # Variable geometry wing
CLAstartAnim[2]   = 50       # Afterburner
CLAstartAnim[3]   = 0         # Propeller, doesn't have STA
CLAstartAnim[4]   = 0       # Spoiler
CLAstartAnim[5]   = 60       # Flap
CLAstartAnim[6]   = 30       # Elevator
CLAstartAnim[7]   = 0         # Aileron
CLAstartAnim[8]   = 50     # Rudder
CLAstartAnim[9]   = 70     # Bomb door
CLAstartAnim[10] = 70     # Thrust vector (VTOL, nozzles)
CLAstartAnim[11] = 40       # Thrust reverser
CLAstartAnim[12] = 60       # TV interlock
CLAstartAnim[13] = 30       # High speed TV interlock
CLAstartAnim[14] = 0       # Gear door
CLAstartAnim[15] = 20       # gear casing
CLAstartAnim[16] = 0         # Arresting hook
CLAstartAnim[17] = 20       # gear door
CLAstartAnim[18] = 30       # Low speed Propeller
CLAstartAnim[19] = 0         # unknown
CLAstartAnim[20] = 60       # High speed Propeller
CLAstartAnim[21] = 0         # Auto gun?
CLAstartAnim[22] = 0         # unknown
CLAstartAnim[23] = 0         # unknown
CLAstartAnim[24] = 0         # unknown
CLAstartAnim[25] = 0         # unknown
CLAstartAnim[26] = 0         # unknown
CLAstartAnim[27] = 0         # unknown
CLAstartAnim[28] = 0         # unknown
CLAstartAnim[29] = 0         # unknown
CLAstartAnim[30] = 0         # Nav lights
CLAstartAnim[31] = 0         # Strobe lights
CLAstartAnim[32] = 0         # Anti collision lights
CLAstartAnim[33] = 0         # Landing lights
CLAstartAnim[34] = 0         # Landing gear lights



# Here we define at what frame the animation of the CLA x will start
# For example, here, then animation of the CLA 16 is played first, then is played CLA 14, 0, 15...
# We go 30 by 30 because most of the animations have 3 STA = 30 frames = 3 seconds


animationSTALength = 10
# Defined the animation duration of one STA

animationFPS = 20
# We define here the number of frame per seconds
# Here 10 frames = 1 second, so one STA lests 1 second
# If animationSTALength was equal to 20, one STA would last 20 frames = 2 seconds

animationEND = 100
# We define here the last frame which will be played
# As the animation of CLA 1 will start at frame 150 and so will end at frame 180, we should use a number lower than 180


## ------------------------ Export Preferences ---------------------

use24bitColor = False
# (True/False)
#True -> we will export using the old 24 bits color format

exportMode = 2
# (0, 1, 2)
# 0 -> only the last mesh will be exported
# 1 -> all the objects of the scene will be exported into different files called "objectName.srf"
# 2 -> only the first selected object will be exported
