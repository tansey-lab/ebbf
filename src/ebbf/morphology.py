import math

import numpy as np
import pandas
from scipy.spatial import ConvexHull
from shapely import Point, Polygon
from tqdm.auto import tqdm as auto_tqdm


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


def calculate_morphological_statistics(adata, cell_boundaries, nucleus_boundaries=None):
    """
    Calculate morphological statistics for cells.

    Parameters
    ----------
    adata : AnnData
        AnnData object whose obs index contains the cell IDs to process.
    cell_boundaries : geopandas.GeoDataFrame
        GeoDataFrame of cell polygon boundaries indexed by cell_id.
    nucleus_boundaries : geopandas.GeoDataFrame, optional
        GeoDataFrame of nucleus polygon boundaries indexed by cell_id.
        When provided, nucleus metrics and cn_ratio are also computed.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by string cell_id with one column per metric.
    """
    auto_tqdm.pandas(desc="processing cell shapes")

    def _make_geoseries(obs_df, boundaries):
        """Inner-join adata obs with a boundary GeoDataFrame; return plain Series.

        spatialdata_io sets integer obs_names (0, 1, 2, ...) while storing the
        actual Xenium cell IDs in obs['cell_id'].  Prefer the cell_id column so
        that we join on the right key for both adata_default and adata_ebbf.
        """
        bnd = boundaries.copy()
        bnd.index = bnd.index.astype(str)
        if "cell_id" in obs_df.columns:
            lookup = obs_df["cell_id"].astype(str)
        else:
            lookup = obs_df.index.astype(str)
        common_mask = lookup.isin(bnd.index)
        common_ids = lookup[common_mask]
        # Use a plain Series so progress_apply returns float Series, not a
        # GeoSeries (which preserves geometry dtype even for scalar results).
        return pandas.Series(list(bnd.loc[common_ids, "geometry"]), index=common_ids.values)

    geoseries = _make_geoseries(adata.obs, cell_boundaries)
    stats_df = pandas.DataFrame(index=geoseries.index)

    stats_df["cell_area"] = geoseries.progress_apply(lambda x: x.area)
    stats_df["cell_perimeter"] = geoseries.progress_apply(lambda x: x.length)
    if "cell_area" in adata.obs.columns and "nucleus_area" in adata.obs.columns:
        # Re-index obs by cell_id so we can look up by the same key used in geoseries
        obs_aligned = adata.obs.copy()
        if "cell_id" in obs_aligned.columns:
            obs_aligned.index = obs_aligned["cell_id"].astype(str)
        else:
            obs_aligned.index = obs_aligned.index.astype(str)
        stats_df["cn_ratio"] = (
            obs_aligned.loc[geoseries.index, "cell_area"]
            / obs_aligned.loc[geoseries.index, "nucleus_area"]
        )
    stats_df["cell_eccentricity"] = geoseries.progress_apply(lambda x: eccentricity(x))
    stats_df["cell_sphericity"] = geoseries.progress_apply(lambda x: sphericity(x))
    stats_df["cell_solidity"] = geoseries.progress_apply(lambda x: solidity(x))
    stats_df["cell_convexity"] = geoseries.progress_apply(lambda x: convexity(x))
    stats_df["cell_circularity"] = geoseries.progress_apply(lambda x: circularity(x))
    stats_df["cell_compactness"] = geoseries.progress_apply(lambda x: compactness(x))

    if nucleus_boundaries is not None:
        nuc_geoseries = _make_geoseries(adata.obs, nucleus_boundaries)
        stats_df["nucleus_area"] = nuc_geoseries.progress_apply(lambda x: x.area)
        stats_df["nucleus_eccentricity"] = nuc_geoseries.progress_apply(lambda x: eccentricity(x))
        stats_df["nucleus_sphericity"] = nuc_geoseries.progress_apply(lambda x: sphericity(x))
        stats_df["nucleus_solidity"] = nuc_geoseries.progress_apply(lambda x: solidity(x))
        stats_df["nucleus_convexity"] = nuc_geoseries.progress_apply(lambda x: convexity(x))
        stats_df["nucleus_circularity"] = nuc_geoseries.progress_apply(lambda x: circularity(x))
        stats_df["nucleus_compactness"] = nuc_geoseries.progress_apply(lambda x: compactness(x))
        if "cn_ratio" not in stats_df.columns:
            stats_df["cn_ratio"] = stats_df["cell_area"] / stats_df["nucleus_area"]

    return stats_df
