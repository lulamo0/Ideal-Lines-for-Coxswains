# Ideal-Lines-for-Coxswains
Using uploaded images, users can map out the shortest line down a rowing course. 

============================================================================================================

Process: 
1. download the spokane river map and Comp_Project.py into the same folder
2. open in VSCode (or program of your choosing)
3. be sure required libraries are installed (numpy, matplotlib, Pillow, scikit-image, scipy, networkx)
4. run code 

=============================================================================================================

What the Code Does:
1. Automatic River Extraction
    - Loads image of the river for use
    - Isolates water regions using RGB method
    - Cleans the mask and extracts the centerline of the river
2. Pathfinding (A*) and Isolation to starboard side (traffic pattern)
    - Enforces rowing traffic pattern (stick to starboard)
    - Builds a grid graph of appropriate water pixels for use in analysis
    - Gives shortest path along the starboard half of the river
3. Path Smoothing
    - Uses scipy.interpolate.splprep and splev to smooth out the A* path
    - Produces a cleaner “ideal” racing line
4. Boat Animation
    - Uses matplotlib.animation.FuncAnimation
    - Renders a small boat icon moving along the path
