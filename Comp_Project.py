# importa correct libraries to complete code 
import numpy as np
from PIL import Image # image module for opening map image 
from collections import deque # for pathfinding 
import matplotlib.pyplot as plt 
from skimage.morphology import (
    binary_closing, binary_dilation, remove_small_objects, disk, skeletonize
) # for processing the map image and getting centerline  
from skimage.measure import label # for labeling compoments in images 
import networkx as nx # for graph and A* pathfinding (shortest line)
from scipy.interpolate import splprep, splev, interp1d # for smoothing paths 


# ============================================================
# 1. LOAD IMAGE & BUILD RIVER MASK
# ============================================================

img_path = "SpokaneRiverMapShaded.jpeg"   # inputs image path (in the same folder as the code)
img = Image.open(img_path).convert("RGB") # opens image and converts to RGB
arr = np.array(img) # converts image to numpy array 
H, W, _ = arr.shape # gathers height and width of image 
print("image size", W, "x", H) # prints image size 

# Builds river mask based on RGB 
R = arr[:, :, 0]
G = arr[:, :, 1]
B = arr[:, :, 2]

# Threshold for shaded river (red)
river_mask_raw = (R > 150) & (G < 140) & (B < 140)
print("river mask pixels", river_mask_raw.sum())

# cleans river mask and thickens pre skeletonizing 
mask = remove_small_objects(river_mask_raw, min_size=200)
mask = binary_closing(mask, footprint=disk(5))
mask = binary_dilation(mask, footprint=disk(3))

# prints number of pixels in cleaned mask 
print("river pixels post clean", mask.sum())

# keeps largest connected component in map only 
labels = label(mask)
n_components = labels.max()

# if more than one compoment only keeps largest mask 
if n_components > 1:
    counts = np.bincount(labels.flatten())
    counts[0] = 0
    largest_label = counts.argmax()
    mask = labels == largest_label
    print("largest component for use", largest_label)

# print final number of pixels in mask to check 
print("Final mask pixels", mask.sum())

# plots clean river mask to see 
plt.figure(figsize=(6, 10))
plt.imshow(mask, cmap="gray", origin="upper")
plt.title("River Mask from Image")
plt.axis("off")
plt.show()


# ============================================================
# 2. SKELETON & CENTERLINE 
# ============================================================

# gets skeleton of river mask for finding centerline 
skeleton = skeletonize(mask)
print("Skeleton pixels:", skeleton.sum())

# function to get endpoints of skeleton for help with centerline pathfinding 
def get_neighbors(r, c, arr_bool):
    H, W = arr_bool.shape
    nbrs = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and arr_bool[rr, cc]:
                nbrs.append((rr, cc))
    return nbrs

# finds enpoints of skeleton using function 
endpoints = []
for r in range(H):
    for c in range(W):
        if skeleton[r, c]:
            if len(get_neighbors(r, c, skeleton)) == 1:
                endpoints.append((r, c))

# prints the endpoints found, checks if enough were found for pathfinding 
print("Endpoints", len(endpoints))
if len(endpoints) < 2:
    raise RuntimeError("Not enough endpoints boooo")

# selects farthest endpoint based on x coordinates for mapping
start_pix = max(endpoints, key=lambda p: p[1])
end_pix   = min(endpoints, key=lambda p: p[1])

# prints start and end pixels for centerline pathfinding 
print("Skeleton start values", start_pix)
print("Skeleton end values", end_pix)

# function for initial search to find path along skeleton between endpoints 
def bfs_path(start, end, arr_bool): # breadth first search 
    q = deque([start])
    visited = {start}
    parent = {start: None}

    while q:
        r, c = q.popleft() 
        if (r, c) == end:
            break

        for nr, nc in get_neighbors(r, c, arr_bool): 
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

    if end not in parent:
        return None

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path

# finds skeleton path with BFS function (breadth first search)
skel_path = bfs_path(start_pix, end_pix, skeleton)
if skel_path is None: # error check for when skeleton wasn't working 
    raise RuntimeError("No path along skeleton (bruh)")


print("Skeleton path length =", len(skel_path))

# centerline in (x,y) pixel coords
centerline = np.array([[c, r] for (r, c) in skel_path], dtype=float)

# plots centerline over map 
plt.figure(figsize=(6, 10))
plt.imshow(arr, origin="upper")
plt.plot(centerline[:, 0], centerline[:, 1], "cyan", linewidth=1)
plt.title("Extracted Centerline")
plt.axis("off")
plt.show()


# ============================================================
# 3. BUILD GRID: 0 = WATER, 1 = SHORE
# ============================================================

# mask==true symbolizes river thus water = 0 and shore = 1
grid = (~mask).astype(int)   
rows, cols = grid.shape
print("grid shape", grid.shape)

# plots river grid for viewing
plt.figure(figsize=(6, 10))
plt.imshow(grid, cmap="gray", origin="upper")
plt.title("River Grid")
plt.axis("off")
plt.show()


# ============================================================
# 4. START & END FROM CENTERLINE
# ============================================================

# rightmost point = start | leftmost point = end
start = (int(round(centerline[0, 1])),  int(round(centerline[0, 0])))   # (row, col)
end  = (int(round(centerline[-1, 1])), int(round(centerline[-1, 0])))

# print for check while completing code 
print("start Coords", start)
print("end Coords", end)

# plots start and end points on river grid (not full map)
plt.figure(figsize=(6, 10))
plt.imshow(grid, cmap="gray", origin="upper")
plt.scatter(start[1], start[0], c="green", s=60, label="Start")
plt.scatter(end[1],  end[0],  c="red",   s=60, label="End")
plt.legend()
plt.title("Start & End on River Grid")
plt.axis("off")
plt.show()


# ============================================================
# 5. ISOLATE STARBOARD SIDE (Racing line needs to be in traffic pattern)
# ============================================================

# variables for centerline coordinates 
xs = centerline[:, 0]
ys = centerline[:, 1]

# sorts centerline points by x values 
idx = np.argsort(xs)
xs_sorted = xs[idx]
ys_sorted = ys[idx]

# makes values unique to avoid issues with interpolation 
xs_unique, unique_idx = np.unique(xs_sorted, return_index=True)
ys_unique = ys_sorted[unique_idx]

# interpolates centerline y values based on the given x values 
centerline_y = interp1d(xs_unique, ys_unique, bounds_error=False, fill_value="extrapolate")

# creates new upper half grid (starboard side going down the course)
upper_grid = grid.copy()

# blocks pixels below the centerline (full starboard isolation)
for c in range(cols):
    cy = int(centerline_y(c))
# leaves centerline row as water, blocks everything below
    if cy + 1 < rows:
        upper_grid[cy + 1 :, c] = 1

# keeps start and end points as water 
upper_grid[start[0], start[1]] = 0
upper_grid[end[0],  end[1]]  = 0

# plots starboard (upper) half of river grid for viewers to see 
plt.figure(figsize=(6, 10))
plt.imshow(upper_grid, cmap="gray", origin="upper")
plt.title("Starboard Half of River Grid")
plt.axis("off")
plt.show()


# ============================================================
# 6. FINDS SHORTEST PATH USING A* FUNCTION ONLY ON UPPER HALF OF WATER 
# ============================================================

# function that converts the grid to a graph to use for A* pathfinding function 
def grid_to_graph(grid_arr):
    rmax, cmax = grid_arr.shape
    G = nx.Graph()
    dirs4  = [(-1,0), (1,0), (0,-1), (0,1)]
    diag   = [(-1,-1),(-1,1),(1,-1),(1,1)]

# adds nodes and edges to graph based on river pixels
    for r in range(rmax):
        for c in range(cmax):
            if grid_arr[r, c] == 0:  # water
                G.add_node((r, c))
                for dr, dc in dirs4 + diag:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rmax and 0 <= nc < cmax and grid_arr[nr, nc] == 0:
                        w = np.hypot(dr, dc)
                        G.add_edge((r, c), (nr, nc), weight=w)
    return G

# heuristis function (prioritizes more promising nodes) for A* pathfinding 
def heuristic(a, b):
    ar, ac = a
    br, bc = b
    return np.hypot(ar - br, ac - bc)

# uses grid to graph function 
G = grid_to_graph(upper_grid)

# Shortest path in the correct traffic pattern with A*
astar_path = nx.astar_path(G, start, end, heuristic=heuristic, weight="weight")
astar_path = np.array(astar_path)
print("shortest path length:", len(astar_path))

# plots the shortest path on the river grid 
plt.figure(figsize=(6, 10))
plt.imshow(upper_grid, cmap="gray", origin="upper")
plt.plot(astar_path[:, 1], astar_path[:, 0], "yellow", linewidth=1)
plt.title("Shortest Line Down the Course")
plt.axis("off")
plt.show()


# ============================================================
# 7. MAKES IDEAL LINE SMOOTHER (was reccomended but don't think this did much, may omit from final code)
# ============================================================

# creates path smoothing function using spline interpolation 
def smooth_path(path_arr, smoothing=200.0, samples=400):
    x = path_arr[:, 1]
    y = path_arr[:, 0]
    t = np.linspace(0, 1, len(path_arr))
    tck, _ = splprep([x, y], u=t, s=smoothing)
    t_f = np.linspace(0, 1, samples)
    xs_s, ys_s = splev(t_f, tck)
    return np.array(xs_s), np.array(ys_s)

# uses smoothing function on A* path
sx, sy = smooth_path(astar_path)

# plots centerline v raw line v smoothed line to compare 
plt.figure(figsize=(6, 10))
plt.imshow(arr, origin="upper")
plt.plot(centerline[:, 0], centerline[:, 1], "cyan", linewidth=1, label="Centerline")
plt.plot(astar_path[:, 1], astar_path[:, 0], "blue", linewidth=1, label="A* raw")
plt.plot(sx, sy, "yellow", linewidth=1, label="Smoothed Line")
plt.legend()
plt.title("Centerline vs Raw Shortest Line vs Smoothed Ideal Line")
plt.axis("off")
plt.show()


# ============================================================
# 8. PLOT ONLY THE IDEAL LINE OVER THE MAP (for best visualization for coxswains)
# ============================================================

# plotting process 
plt.figure(figsize=(6, 10))
plt.imshow(arr, origin="upper")
plt.plot(sx, sy, "yellow", linewidth=1)
plt.title("Ideal Racing Line (Down River)")
plt.axis("off")
plt.show()

# ============================================================
# 9. ANIMATION OF A BOAT TRAVELING DOWN THE RIVER
# ============================================================

from matplotlib.animation import FuncAnimation

# prep figure 
fig, ax = plt.subplots(figsize=(6, 10))
ax.imshow(arr, origin="upper")
ax.plot(sx, sy, color="yellow", linewidth=1.5, label="Ideal Line")

# boat marker (triangle like a little boat)
boat_marker, = ax.plot([], [], marker="^", markersize=10, color="blue")

ax.set_title("Boat Animation Down the River")
ax.axis("off")

Nframes = len(sx)

# initializing function for blitting
def init():
    boat_marker.set_data([], [])
    return boat_marker,

# update function called every frame
def update(frame):
    # IMPORTANT: wrap in lists so x, y are sequences
    boat_marker.set_data([sx[frame]], [sy[frame]])
    return boat_marker,

# animation call 
anim = FuncAnimation(
    fig,
    update,
    frames=Nframes,   # or range(Nframes)
    init_func=init,
    interval=20,      # ms between frames (speed control)
    blit=True
)

plt.show()
