"""
Terrain Analysis Tools for Semantic Kernel Agent

These are the tools the terrain agent can call to perform geospatial analysis.
Each tool is a kernel function that can be invoked by the agent during reasoning.
"""

import logging
import os
from typing import Annotated, Dict, Any, Optional, List
import numpy as np
from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)


class TerrainAnalysisTools:
    """
    Terrain analysis tools that can be invoked by Semantic Kernel agents.
    Each method decorated with @kernel_function becomes a callable tool.
    """
    
    def __init__(self):
        """Initialize terrain tools with STAC client."""
        self.stac_endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self._catalog = None
        logger.info("✅ TerrainAnalysisTools initialized")
    
    @property
    def catalog(self):
        """Lazy-load STAC catalog."""
        if self._catalog is None:
            from pystac_client import Client
            self._catalog = Client.open(self.stac_endpoint)
        return self._catalog
    
    def _calculate_bbox(self, latitude: float, longitude: float, radius_km: float) -> List[float]:
        """Calculate bounding box from center and radius."""
        lat_deg_per_km = 1 / 111.0
        lon_deg_per_km = 1 / (111.0 * np.cos(np.radians(latitude)))
        
        lat_delta = radius_km * lat_deg_per_km
        lon_delta = radius_km * lon_deg_per_km
        
        return [
            longitude - lon_delta,
            latitude - lat_delta,
            longitude + lon_delta,
            latitude + lat_delta
        ]
    
    @kernel_function(
        name="get_elevation_analysis",
        description="Analyze elevation data for a location. Returns min, max, mean elevation in meters, elevation range, and terrain classification (flat, hilly, mountainous). Use this when the user asks about elevation, altitude, height, or topography."
    )
    async def get_elevation_analysis(
        self,
        latitude: Annotated[float, "Center latitude of the area to analyze"],
        longitude: Annotated[float, "Center longitude of the area to analyze"],
        radius_km: Annotated[float, "Radius in kilometers for analysis area"] = 5.0
    ) -> Dict[str, Any]:
        """Fetch DEM data and calculate elevation statistics."""
        try:
            import rasterio
            from rasterio.windows import from_bounds
            import planetary_computer
            
            logger.info(f"🏔️ [TOOL] get_elevation_analysis at ({latitude:.4f}, {longitude:.4f}), radius={radius_km}km")
            
            bbox = self._calculate_bbox(latitude, longitude, radius_km)
            
            # Search for DEM
            search = self.catalog.search(
                collections=["cop-dem-glo-30"],
                bbox=bbox,
                limit=1
            )
            
            items = list(search.items())
            if not items:
                return {"error": "No DEM data available for this location", "elevation_stats": {}}
            
            item = items[0]
            signed_item = planetary_computer.sign(item)
            dem_url = signed_item.assets["data"].href
            
            with rasterio.open(dem_url) as src:
                window = from_bounds(*bbox, src.transform)
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                if window.width < 1 or window.height < 1:
                    return {"error": "Area too small for DEM analysis", "elevation_stats": {}}
                
                elevation = src.read(1, window=window)
                elevation = np.ma.masked_equal(elevation, src.nodata or -9999)
                
                if elevation.count() == 0:
                    return {"error": "No valid elevation data", "elevation_stats": {}}
                
                elev_min = float(elevation.min())
                elev_max = float(elevation.max())
                elev_mean = float(elevation.mean())
                elev_range = elev_max - elev_min
                
                # Classify terrain
                if elev_range < 50:
                    terrain_type = "flat plains"
                elif elev_range < 200:
                    terrain_type = "gently rolling hills"
                elif elev_range < 500:
                    terrain_type = "hilly terrain"
                elif elev_range < 1000:
                    terrain_type = "rugged hills"
                else:
                    terrain_type = "mountainous terrain"
                
                result = {
                    "elevation_min_meters": round(elev_min, 1),
                    "elevation_max_meters": round(elev_max, 1),
                    "elevation_mean_meters": round(elev_mean, 1),
                    "elevation_range_meters": round(elev_range, 1),
                    "terrain_type": terrain_type,
                    "data_source": "Copernicus DEM GLO-30 (30m resolution)"
                }
                
                logger.info(f"✅ [TOOL] Elevation: {elev_min:.0f}m - {elev_max:.0f}m, type: {terrain_type}")
                return result
                
        except Exception as e:
            logger.error(f"❌ [TOOL] Elevation analysis failed: {e}")
            return {"error": str(e), "elevation_stats": {}}
    
    @kernel_function(
        name="get_slope_analysis", 
        description="Analyze terrain slope (steepness) for a location. Returns min, max, mean slope in degrees, and identifies steep areas. Use this when user asks about slope, steepness, gradient, or terrain difficulty."
    )
    async def get_slope_analysis(
        self,
        latitude: Annotated[float, "Center latitude"],
        longitude: Annotated[float, "Center longitude"],
        radius_km: Annotated[float, "Radius in kilometers"] = 5.0
    ) -> Dict[str, Any]:
        """Calculate slope from DEM data."""
        try:
            import rasterio
            from rasterio.windows import from_bounds
            import planetary_computer
            
            logger.info(f"📐 [TOOL] get_slope_analysis at ({latitude:.4f}, {longitude:.4f})")
            
            bbox = self._calculate_bbox(latitude, longitude, radius_km)
            
            search = self.catalog.search(collections=["cop-dem-glo-30"], bbox=bbox, limit=1)
            items = list(search.items())
            
            if not items:
                return {"error": "No DEM data available", "slope_stats": {}}
            
            item = planetary_computer.sign(items[0])
            
            with rasterio.open(item.assets["data"].href) as src:
                window = from_bounds(*bbox, src.transform)
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                elevation = src.read(1, window=window).astype(float)
                
                # CRITICAL: Convert cell size from degrees to meters
                # The DEM resolution is in degrees, but elevation is in meters
                # At the equator, 1 degree ≈ 111,320 meters
                # At latitude φ: 1 degree longitude ≈ 111,320 * cos(φ) meters
                cell_size_deg = src.res[0]  # Resolution in degrees
                cell_size_meters = cell_size_deg * 111320 * np.cos(np.radians(latitude))
                
                # Calculate slope using proper cell size in meters
                dy, dx = np.gradient(elevation, cell_size_meters)
                slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
                
                slope_min = float(np.min(slope))
                slope_max = float(np.max(slope))
                slope_mean = float(np.mean(slope))
                
                # Classify areas
                flat_pct = float(np.sum(slope < 5) / slope.size * 100)
                moderate_pct = float(np.sum((slope >= 5) & (slope < 15)) / slope.size * 100)
                steep_pct = float(np.sum(slope >= 15) / slope.size * 100)
                
                result = {
                    "slope_min_degrees": round(slope_min, 1),
                    "slope_max_degrees": round(slope_max, 1),
                    "slope_mean_degrees": round(slope_mean, 1),
                    "flat_area_percent": round(flat_pct, 1),
                    "moderate_slope_percent": round(moderate_pct, 1),
                    "steep_area_percent": round(steep_pct, 1),
                    "traversability": "easy" if slope_mean < 5 else "moderate" if slope_mean < 15 else "difficult"
                }
                
                logger.info(f"✅ [TOOL] Slope: mean {slope_mean:.1f}°, {flat_pct:.0f}% flat")
                return result
                
        except Exception as e:
            logger.error(f"❌ [TOOL] Slope analysis failed: {e}")
            return {"error": str(e)}
    
    @kernel_function(
        name="get_aspect_analysis",
        description="Analyze terrain aspect (slope direction/facing). Returns dominant direction (N, NE, E, etc) and distribution. Use this when user asks about which way slopes face, sun exposure, or orientation."
    )
    async def get_aspect_analysis(
        self,
        latitude: Annotated[float, "Center latitude"],
        longitude: Annotated[float, "Center longitude"],
        radius_km: Annotated[float, "Radius in kilometers"] = 5.0
    ) -> Dict[str, Any]:
        """Calculate aspect (slope direction) from DEM."""
        try:
            import rasterio
            from rasterio.windows import from_bounds
            import planetary_computer
            
            logger.info(f"🧭 [TOOL] get_aspect_analysis at ({latitude:.4f}, {longitude:.4f})")
            
            bbox = self._calculate_bbox(latitude, longitude, radius_km)
            search = self.catalog.search(collections=["cop-dem-glo-30"], bbox=bbox, limit=1)
            items = list(search.items())
            
            if not items:
                return {"error": "No DEM data available"}
            
            item = planetary_computer.sign(items[0])
            
            with rasterio.open(item.assets["data"].href) as src:
                window = from_bounds(*bbox, src.transform)
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                elevation = src.read(1, window=window).astype(float)
                dy, dx = np.gradient(elevation, src.res[0])
                
                # Aspect in degrees (0=N, 90=E, 180=S, 270=W)
                aspect = np.degrees(np.arctan2(-dx, dy))
                aspect = np.where(aspect < 0, aspect + 360, aspect)
                
                # Categorize
                directions = {
                    "N": ((aspect >= 337.5) | (aspect < 22.5)).sum(),
                    "NE": ((aspect >= 22.5) & (aspect < 67.5)).sum(),
                    "E": ((aspect >= 67.5) & (aspect < 112.5)).sum(),
                    "SE": ((aspect >= 112.5) & (aspect < 157.5)).sum(),
                    "S": ((aspect >= 157.5) & (aspect < 202.5)).sum(),
                    "SW": ((aspect >= 202.5) & (aspect < 247.5)).sum(),
                    "W": ((aspect >= 247.5) & (aspect < 292.5)).sum(),
                    "NW": ((aspect >= 292.5) & (aspect < 337.5)).sum()
                }
                
                total = sum(directions.values())
                distribution = {k: round(v / total * 100, 1) for k, v in directions.items()}
                dominant = max(directions, key=directions.get)
                
                result = {
                    "dominant_direction": dominant,
                    "direction_distribution_percent": distribution,
                    "sun_exposure": "good" if dominant in ["S", "SE", "SW"] else "moderate" if dominant in ["E", "W"] else "limited"
                }
                
                logger.info(f"✅ [TOOL] Aspect: dominant {dominant}, sun exposure {result['sun_exposure']}")
                return result
                
        except Exception as e:
            logger.error(f"❌ [TOOL] Aspect analysis failed: {e}")
            return {"error": str(e)}
    
    # NOTE: NDVI (get_vegetation_index) and NDWI (identify_water_bodies) tools have been removed
    # due to unreliable Sentinel-2 tile coverage and intersection errors.
    # Visual analysis via GPT-4 Vision provides better vegetation/water detection.
    
    @kernel_function(
        name="find_flat_areas",
        description="Find flat areas suitable for landing zones, construction, or camps. Returns percentage of flat land and locations. Use when user asks about landing zones, flat ground, or buildable areas."
    )
    async def find_flat_areas(
        self,
        latitude: Annotated[float, "Center latitude"],
        longitude: Annotated[float, "Center longitude"],
        radius_km: Annotated[float, "Radius in kilometers"] = 5.0,
        max_slope_degrees: Annotated[float, "Maximum slope to consider 'flat'"] = 5.0
    ) -> Dict[str, Any]:
        """Find flat areas based on slope threshold."""
        try:
            import rasterio
            from rasterio.windows import from_bounds
            import planetary_computer
            
            logger.info(f"🛬 [TOOL] find_flat_areas at ({latitude:.4f}, {longitude:.4f}), max_slope={max_slope_degrees}°")
            
            bbox = self._calculate_bbox(latitude, longitude, radius_km)
            search = self.catalog.search(collections=["cop-dem-glo-30"], bbox=bbox, limit=1)
            items = list(search.items())
            
            if not items:
                return {"error": "No DEM data available"}
            
            item = planetary_computer.sign(items[0])
            
            with rasterio.open(item.assets["data"].href) as src:
                window = from_bounds(*bbox, src.transform)
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                elevation = src.read(1, window=window).astype(float)
                
                # Convert cell size from degrees to meters
                cell_size_deg = src.res[0]
                cell_size_meters = cell_size_deg * 111320 * np.cos(np.radians(latitude))
                
                dy, dx = np.gradient(elevation, cell_size_meters)
                slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
                
                flat_mask = slope < max_slope_degrees
                flat_pct = float(np.sum(flat_mask) / flat_mask.size * 100)
                
                # Find largest contiguous flat area (simplified)
                suitable = "excellent" if flat_pct > 50 else "good" if flat_pct > 20 else "limited" if flat_pct > 5 else "poor"
                
                result = {
                    "flat_area_percent": round(flat_pct, 1),
                    "slope_threshold_degrees": max_slope_degrees,
                    "suitability_for_landing": suitable,
                    "recommendation": f"{'Abundant' if flat_pct > 30 else 'Some' if flat_pct > 10 else 'Limited'} flat areas available within {radius_km}km radius"
                }
                
                logger.info(f"✅ [TOOL] Flat areas: {flat_pct:.1f}% below {max_slope_degrees}°")
                return result
                
        except Exception as e:
            logger.error(f"❌ [TOOL] Flat area search failed: {e}")
            return {"error": str(e)}
    
    @kernel_function(
        name="analyze_flood_risk",
        description="Analyze flood risk using JRC Global Surface Water historical data. Returns water occurrence percentage (0-100%) indicating how often the area has been covered by water. Use this for permitting to check 100-year flood zones and flood-prone areas."
    )
    async def analyze_flood_risk(
        self,
        latitude: Annotated[float, "Center latitude"],
        longitude: Annotated[float, "Center longitude"],
        radius_km: Annotated[float, "Radius in kilometers"] = 5.0
    ) -> Dict[str, Any]:
        """Analyze flood risk using JRC Global Surface Water occurrence data."""
        try:
            import rasterio
            from rasterio.windows import from_bounds
            import planetary_computer
            import aiohttp
            
            logger.info(f"🌊 [TOOL] analyze_flood_risk at ({latitude:.4f}, {longitude:.4f})")
            
            bbox = self._calculate_bbox(latitude, longitude, radius_km)
            
            # Query JRC Global Surface Water
            search = self.catalog.search(
                collections=["jrc-gsw"],
                bbox=bbox,
                limit=1
            )
            
            items = list(search.items())
            if not items:
                return {"error": "No JRC Global Surface Water data available", "flood_risk": "unknown"}
            
            item = planetary_computer.sign(items[0])
            
            # Use the 'occurrence' asset (0-100% water occurrence)
            if 'occurrence' not in item.assets:
                return {"error": "No occurrence data in JRC-GSW item", "flood_risk": "unknown"}
            
            occurrence_url = item.assets["occurrence"].href
            
            with rasterio.open(occurrence_url) as src:
                window = from_bounds(*bbox, src.transform)
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                if window.width < 1 or window.height < 1:
                    return {"error": "Area too small for analysis", "flood_risk": "unknown"}
                
                occurrence = src.read(1, window=window)
                # JRC-GSW: 0 = never water, 100 = permanent water, 255 = no data
                valid_mask = occurrence <= 100
                if not np.any(valid_mask):
                    return {"error": "No valid water occurrence data", "flood_risk": "unknown"}
                
                valid_data = occurrence[valid_mask]
                
                # Statistics
                mean_occurrence = float(np.mean(valid_data))
                max_occurrence = float(np.max(valid_data))
                pct_ever_flooded = float(np.sum(valid_data > 0) / len(valid_data) * 100)
                pct_frequently_flooded = float(np.sum(valid_data > 25) / len(valid_data) * 100)
                
                # Risk classification
                if max_occurrence > 50 or pct_frequently_flooded > 10:
                    risk_level = "HIGH"
                    permitting_status = "NOT RECOMMENDED"
                    risk_reason = "Significant historical flooding observed"
                elif max_occurrence > 10 or pct_ever_flooded > 20:
                    risk_level = "MODERATE"
                    permitting_status = "CONDITIONAL"
                    risk_reason = "Some historical flooding, mitigation may be required"
                else:
                    risk_level = "LOW"
                    permitting_status = "SUITABLE"
                    risk_reason = "Minimal historical flooding"
                
                result = {
                    "mean_water_occurrence_percent": round(mean_occurrence, 1),
                    "max_water_occurrence_percent": round(max_occurrence, 1),
                    "area_ever_flooded_percent": round(pct_ever_flooded, 1),
                    "area_frequently_flooded_percent": round(pct_frequently_flooded, 1),
                    "flood_risk_level": risk_level,
                    "permitting_status": permitting_status,
                    "risk_reason": risk_reason,
                    "data_source": "JRC Global Surface Water (1984-2021)"
                }
                
                logger.info(f"✅ [TOOL] Flood risk: {risk_level} (max occurrence: {max_occurrence:.0f}%)")
                return result
                
        except Exception as e:
            logger.error(f"❌ [TOOL] Flood risk analysis failed: {e}")
            return {"error": str(e), "flood_risk": "unknown"}
    
    @kernel_function(
        name="analyze_water_proximity",
        description="Calculate distance to nearest water body for setback requirements. Returns estimated minimum distance to water based on JRC Global Surface Water. Use for permitting to verify buffer zones (e.g., 500m from wetlands, 100m from streams)."
    )
    async def analyze_water_proximity(
        self,
        latitude: Annotated[float, "Center latitude of site"],
        longitude: Annotated[float, "Center longitude of site"],
        radius_km: Annotated[float, "Search radius in kilometers"] = 5.0,
        required_setback_meters: Annotated[float, "Required setback distance from water in meters"] = 500.0
    ) -> Dict[str, Any]:
        """Analyze proximity to water bodies for setback requirements."""
        try:
            import rasterio
            from rasterio.windows import from_bounds
            import planetary_computer
            from scipy import ndimage
            
            logger.info(f"💧 [TOOL] analyze_water_proximity at ({latitude:.4f}, {longitude:.4f})")
            
            bbox = self._calculate_bbox(latitude, longitude, radius_km)
            
            # Query JRC Global Surface Water
            search = self.catalog.search(
                collections=["jrc-gsw"],
                bbox=bbox,
                limit=1
            )
            
            items = list(search.items())
            if not items:
                return {"error": "No JRC Global Surface Water data available"}
            
            item = planetary_computer.sign(items[0])
            
            # Use 'occurrence' asset
            if 'occurrence' not in item.assets:
                return {"error": "No occurrence data available"}
            
            occurrence_url = item.assets["occurrence"].href
            
            with rasterio.open(occurrence_url) as src:
                window = from_bounds(*bbox, src.transform)
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                if window.width < 1 or window.height < 1:
                    return {"error": "Area too small for analysis"}
                
                occurrence = src.read(1, window=window)
                
                # Water mask: consider anything with >10% occurrence as water
                water_mask = (occurrence > 10) & (occurrence <= 100)
                
                if not np.any(water_mask):
                    return {
                        "water_detected": False,
                        "nearest_water_meters": "None within search radius",
                        "setback_requirement_meters": required_setback_meters,
                        "setback_satisfied": True,
                        "permitting_status": "SUITABLE",
                        "recommendation": "No significant water bodies detected within analysis area"
                    }
                
                # Calculate distance to nearest water (in pixels, then convert to meters)
                distance_pixels = ndimage.distance_transform_edt(~water_mask)
                
                # Get center pixel (site location)
                center_row = distance_pixels.shape[0] // 2
                center_col = distance_pixels.shape[1] // 2
                
                # Distance at center in pixels
                center_distance_pixels = distance_pixels[center_row, center_col]
                
                # Convert to meters (JRC-GSW is 30m resolution)
                pixel_size_meters = 30.0
                center_distance_meters = center_distance_pixels * pixel_size_meters
                
                # Check setback compliance
                setback_satisfied = center_distance_meters >= required_setback_meters
                
                # Permitting status
                if setback_satisfied:
                    status = "SUITABLE"
                    recommendation = f"Site is {center_distance_meters:.0f}m from nearest water body, exceeds {required_setback_meters:.0f}m requirement"
                else:
                    status = "NOT SUITABLE"
                    recommendation = f"Site is only {center_distance_meters:.0f}m from water, does not meet {required_setback_meters:.0f}m setback requirement"
                
                # Calculate % of area that is water
                water_percent = float(np.sum(water_mask) / water_mask.size * 100)
                
                result = {
                    "water_detected": True,
                    "nearest_water_meters": round(center_distance_meters, 0),
                    "setback_requirement_meters": required_setback_meters,
                    "setback_satisfied": setback_satisfied,
                    "water_area_percent": round(water_percent, 1),
                    "permitting_status": status,
                    "recommendation": recommendation,
                    "data_source": "JRC Global Surface Water (30m resolution)"
                }
                
                logger.info(f"✅ [TOOL] Water proximity: {center_distance_meters:.0f}m, setback {'OK' if setback_satisfied else 'FAILED'}")
                return result
                
        except ImportError:
            return {"error": "scipy not available for distance calculation", "water_proximity": "unknown"}
        except Exception as e:
            logger.error(f"❌ [TOOL] Water proximity analysis failed: {e}")
            return {"error": str(e)}
    
    @kernel_function(
        name="analyze_environmental_sensitivity",
        description="Identify environmentally sensitive areas using ESA WorldCover land classification. Detects wetlands, forests, mangroves, and other protected land types. Use for environmental permitting to check for protected habitats."
    )
    async def analyze_environmental_sensitivity(
        self,
        latitude: Annotated[float, "Center latitude"],
        longitude: Annotated[float, "Center longitude"],
        radius_km: Annotated[float, "Radius in kilometers"] = 5.0
    ) -> Dict[str, Any]:
        """Analyze environmental sensitivity using ESA WorldCover."""
        try:
            import rasterio
            from rasterio.windows import from_bounds
            import planetary_computer
            
            logger.info(f"🌿 [TOOL] analyze_environmental_sensitivity at ({latitude:.4f}, {longitude:.4f})")
            
            bbox = self._calculate_bbox(latitude, longitude, radius_km)
            
            # Query ESA WorldCover
            search = self.catalog.search(
                collections=["esa-worldcover"],
                bbox=bbox,
                limit=1
            )
            
            items = list(search.items())
            if not items:
                return {"error": "No ESA WorldCover data available"}
            
            item = planetary_computer.sign(items[0])
            
            # Use 'map' asset
            if 'map' not in item.assets:
                return {"error": "No land cover map asset available"}
            
            map_url = item.assets["map"].href
            
            with rasterio.open(map_url) as src:
                window = from_bounds(*bbox, src.transform)
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                if window.width < 1 or window.height < 1:
                    return {"error": "Area too small for analysis"}
                
                landcover = src.read(1, window=window)
                total_pixels = landcover.size
                
                # ESA WorldCover class definitions
                # 10 = Tree cover, 20 = Shrubland, 30 = Grassland, 40 = Cropland
                # 50 = Built-up, 60 = Bare/sparse, 70 = Snow/ice, 80 = Permanent water
                # 90 = Herbaceous wetland, 95 = Mangroves, 100 = Moss/lichen
                
                class_counts = {
                    "tree_cover": float(np.sum(landcover == 10) / total_pixels * 100),
                    "shrubland": float(np.sum(landcover == 20) / total_pixels * 100),
                    "grassland": float(np.sum(landcover == 30) / total_pixels * 100),
                    "cropland": float(np.sum(landcover == 40) / total_pixels * 100),
                    "built_up": float(np.sum(landcover == 50) / total_pixels * 100),
                    "bare_sparse": float(np.sum(landcover == 60) / total_pixels * 100),
                    "permanent_water": float(np.sum(landcover == 80) / total_pixels * 100),
                    "herbaceous_wetland": float(np.sum(landcover == 90) / total_pixels * 100),
                    "mangroves": float(np.sum(landcover == 95) / total_pixels * 100),
                }
                
                # Calculate sensitive area percentage
                sensitive_classes = ["tree_cover", "herbaceous_wetland", "mangroves", "permanent_water"]
                sensitive_percent = sum(class_counts[c] for c in sensitive_classes)
                
                # Risk assessment for permitting
                constraints = []
                if class_counts["herbaceous_wetland"] > 5:
                    constraints.append(f"Wetlands ({class_counts['herbaceous_wetland']:.1f}%) - may require wetland mitigation")
                if class_counts["mangroves"] > 1:
                    constraints.append(f"Mangroves ({class_counts['mangroves']:.1f}%) - protected habitat, development restricted")
                if class_counts["tree_cover"] > 30:
                    constraints.append(f"Forest ({class_counts['tree_cover']:.1f}%) - may require deforestation permit")
                if class_counts["permanent_water"] > 10:
                    constraints.append(f"Water bodies ({class_counts['permanent_water']:.1f}%) - setback requirements apply")
                
                if sensitive_percent > 40:
                    sensitivity_level = "HIGH"
                    permitting_status = "NOT RECOMMENDED"
                elif sensitive_percent > 15:
                    sensitivity_level = "MODERATE"
                    permitting_status = "CONDITIONAL"
                else:
                    sensitivity_level = "LOW"
                    permitting_status = "SUITABLE"
                
                # Dominant land cover
                dominant = max(class_counts, key=class_counts.get)
                
                result = {
                    "land_cover_breakdown_percent": {k: round(v, 1) for k, v in class_counts.items() if v > 0.5},
                    "dominant_land_cover": dominant.replace("_", " ").title(),
                    "sensitive_area_percent": round(sensitive_percent, 1),
                    "environmental_sensitivity": sensitivity_level,
                    "permitting_status": permitting_status,
                    "environmental_constraints": constraints if constraints else ["No major environmental constraints identified"],
                    "data_source": "ESA WorldCover 2021 (10m resolution)"
                }
                
                logger.info(f"✅ [TOOL] Environmental sensitivity: {sensitivity_level} ({sensitive_percent:.0f}% sensitive)")
                return result
                
        except Exception as e:
            logger.error(f"❌ [TOOL] Environmental sensitivity analysis failed: {e}")
            return {"error": str(e)}
    
    # NOTE: analyze_screenshot tool has been removed.
    # Visual analysis is now performed automatically BEFORE agent invocation
    # and included in the agent's context as [Visual Analysis of Current Map View].
    # This ensures screenshot analysis always succeeds and is available to the agent.
