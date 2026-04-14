import math

import numpy as np
from scipy.spatial import ConvexHull
from shapely import Point, Polygon


def find_major_axis(polygon):
    """
    Find the major axis of a polygon.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon
        Input polygon

    Returns:
    --------
    dict
        Dictionary containing major axis length and endpoints
    """
    # Get vertices of the polygon
    vertices = np.array(polygon.exterior.coords)

    # Find all pairs of vertices and calculate distances
    max_dist = 0
    max_pair = None

    for i in range(len(vertices) - 1):  # -1 because the last point is the same as the first
        for j in range(i + 1, len(vertices) - 1):
            dist = np.sqrt(np.sum((vertices[i] - vertices[j]) ** 2))
            if dist > max_dist:
                max_dist = dist
                max_pair = (i, j)

    # Return the major axis details
    return {
        "major_axis_length": max_dist,
        "major_axis_endpoints": [vertices[max_pair[0]], vertices[max_pair[1]]],
    }


def find_minor_axis(polygon, major_axis_endpoints):
    """
    Find the minor axis of a polygon perpendicular to the major axis.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon
        Input polygon
    major_axis_endpoints : list
        List of two points defining the major axis

    Returns:
    --------
    dict
        Dictionary containing minor axis length and endpoints
    """

    # Get direction vector of major axis
    dx = major_axis_endpoints[1][0] - major_axis_endpoints[0][0]
    dy = major_axis_endpoints[1][1] - major_axis_endpoints[0][1]

    # Get perpendicular direction (rotate 90 degrees)
    perp_dx, perp_dy = -dy, dx

    # Normalize
    mag = math.sqrt(perp_dx**2 + perp_dy**2)
    if mag > 0:
        perp_dx, perp_dy = perp_dx / mag, perp_dy / mag

    # Find midpoint of major axis
    midpoint = [
        (major_axis_endpoints[0][0] + major_axis_endpoints[1][0]) / 2,
        (major_axis_endpoints[0][1] + major_axis_endpoints[1][1]) / 2,
    ]

    # Get vertices of the polygon
    vertices = np.array(polygon.exterior.coords)

    # Project vertices onto perpendicular line and find the extreme points
    max_proj_dist = 0
    minor_axis_endpoints = None

    for i in range(len(vertices) - 1):
        # Vector from midpoint to vertex
        vec_x = vertices[i][0] - midpoint[0]
        vec_y = vertices[i][1] - midpoint[1]

        # Project this vector onto the perpendicular direction
        proj = abs(vec_x * perp_dx + vec_y * perp_dy)

        if proj > max_proj_dist:
            max_proj_dist = proj
            point1 = [midpoint[0] + perp_dx * proj, midpoint[1] + perp_dy * proj]
            point2 = [midpoint[0] - perp_dx * proj, midpoint[1] - perp_dy * proj]
            minor_axis_endpoints = [point1, point2]

    # Calculate the minor axis length
    minor_axis_length = max_proj_dist * 2

    return {
        "minor_axis_length": minor_axis_length,
        "minor_axis_endpoints": minor_axis_endpoints,
    }


def convex_hull_polygon(points):
    """
    Create a convex hull polygon from points.

    Parameters:
    -----------
    points : numpy.ndarray
        Array of points (x, y)

    Returns:
    --------
    shapely.geometry.Polygon
        Convex hull polygon
    """
    if isinstance(points, Polygon):
        points = np.array(points.exterior.coords)

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    return Polygon(hull_points)


def eccentricity(polygon):
    """
    Calculate the eccentricity of a polygon.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon or numpy.ndarray
        Input polygon or points

    Returns:
    --------
    float
        Eccentricity value (minor axis length / major axis length)
    """

    major_axis = find_major_axis(polygon)
    minor_axis = find_minor_axis(polygon, major_axis["major_axis_endpoints"])

    return minor_axis["minor_axis_length"] / major_axis["major_axis_length"]


def sphericity(polygon):
    """
    Calculate the sphericity of a polygon.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon or numpy.ndarray
        Input polygon or points

    Returns:
    --------
    float
        Sphericity value (inner circle radius / outer circle radius)
    """

    # Get bounding box
    minx, miny, maxx, maxy = polygon.bounds

    # Calculate outer circle (bounding circle)
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    center = Point(center_x, center_y)

    # Find distance to farthest vertex (outer radius)
    vertices = np.array(polygon.exterior.coords)
    outer_radius = max(np.sqrt(np.sum((vertices - [center_x, center_y]) ** 2, axis=1)))

    # Find the largest circle that can be inscribed in the polygon (inner circle)
    # This is a simplification - finding the true inscribed circle is complex
    # Here we use the distance from center to the nearest point on the boundary
    inner_radius = center.distance(polygon.boundary)

    return inner_radius / outer_radius


def solidity(polygon, chull_polygon=None):
    """
    Calculate the solidity of a polygon.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon or numpy.ndarray
        Input polygon or points
    chull_polygon : shapely.geometry.Polygon, optional
        Convex hull polygon

    Returns:
    --------
    float
        Solidity value (polygon area / convex hull area)
    """

    if chull_polygon is None:
        points = np.array(polygon.exterior.coords)
        chull_polygon = convex_hull_polygon(points)

    area_p = polygon.area
    ch_area = chull_polygon.area

    return area_p / ch_area


def convexity(polygon, chull_polygon=None):
    """
    Calculate the convexity of a polygon.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon or numpy.ndarray
        Input polygon or points
    chull_polygon : shapely.geometry.Polygon, optional
        Convex hull polygon

    Returns:
    --------
    float
        Convexity value (convex hull perimeter / polygon perimeter)
    """
    if chull_polygon is None:
        points = np.array(polygon.exterior.coords)
        chull_polygon = convex_hull_polygon(points)

    perimeter = polygon.length
    ch_perimeter = chull_polygon.length

    return ch_perimeter / perimeter


def circularity(polygon, chull_polygon=None):
    """
    Calculate the circularity of a polygon.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon or numpy.ndarray
        Input polygon or points
    chull_polygon : shapely.geometry.Polygon, optional
        Convex hull polygon

    Returns:
    --------
    float
        Circularity value (4π × area / (convex hull perimeter)²)
    """

    if chull_polygon is None:
        points = np.array(polygon.exterior.coords)
        chull_polygon = convex_hull_polygon(points)

    area = polygon.area
    ch_perimeter = chull_polygon.length

    return (4 * math.pi * area) / (ch_perimeter**2)


def compactness(polygon):
    """
    Calculate the compactness of a polygon.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon or numpy.ndarray
        Input polygon or points

    Returns:
    --------
    float
        Compactness value (4π × area / perimeter²)
    """
    area = polygon.area
    perimeter = polygon.length

    return (4 * math.pi * area) / (perimeter**2)
