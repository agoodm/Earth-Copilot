"""
Enhanced Vision Agent - Semantic Kernel Agent with LLM-Based Tool Selection

This agent answers ALL questions using the best available context with TRUE agentic
tool selection via GPT-4o (not keyword matching).

Available Tools:
1. analyze_screenshot - GPT-4o vision analysis of map screenshot
2. analyze_raster - Quantitative analysis of loaded STAC rasters (elevation, NDVI, etc.)
3. query_knowledge - LLM knowledge for contextual/educational answers
4. compare_locations - Compare features between different areas
5. identify_features - Identify specific geographic features in the view

Design Principles:
- TRUE AGENT: GPT-4o decides which tools to call based on semantic understanding
- Single agent handles ALL question types (replaces contextual, vision, hybrid)
- Synthesizes results from multiple sources into coherent answer
- Maintains session memory for multi-turn conversations
- Uses Semantic Kernel FunctionChoiceBehavior.Auto() for tool selection

Usage:
- All follow-up questions route here
- Questions about visible map → GPT decides: analyze_screenshot
- Quantitative questions (slope, elevation) → GPT decides: analyze_raster
- Educational/factual questions → GPT decides: query_knowledge
- Complex questions → GPT decides: combine multiple tools
"""

from typing import Dict, Any, Optional, List, Annotated
import logging
import os
import base64
import json
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Lazy imports for direct Azure OpenAI calls (for vision)
AzureOpenAI = None

def _load_azure_openai():
    """Lazy load Azure OpenAI SDK for vision calls."""
    global AzureOpenAI
    if AzureOpenAI is None:
        try:
            from openai import AzureOpenAI as _AzureOpenAI
            AzureOpenAI = _AzureOpenAI
        except ImportError as e:
            logging.warning(f"Azure OpenAI SDK not available: {e}")


# ============================================================================
# RASTER POINT SAMPLING UTILITIES
# ============================================================================

async def sample_cog_at_point(cog_url: str, latitude: float, longitude: float, band: int = 1, max_retries: int = 3) -> Dict[str, Any]:
    """
    Sample a Cloud Optimized GeoTIFF (COG) at a specific lat/lng coordinate.
    
    Args:
        cog_url: URL to the COG file
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees  
        band: Band number to sample (1-indexed, default=1)
        max_retries: Maximum number of retries for rate limit errors (409)
    
    Returns:
        Dict with 'value', 'unit', 'crs', 'error' (if any)
    """
    import asyncio
    import time
    
    def _sample_sync(retry_count: int = 0):
        try:
            import rasterio
            from rasterio.session import AWSSession
            import planetary_computer as pc
            
            logging.info(f"📍 sample_cog_at_point: lat={latitude}, lng={longitude}, band={band}")
            logging.info(f"📍 COG URL: {cog_url[:100]}...")
            
            # Sign the URL if it's from Planetary Computer
            if 'blob.core.windows.net' in cog_url:
                try:
                    signed_url = pc.sign(cog_url)
                    logging.info(f"📍 URL signed successfully")
                except Exception as sign_err:
                    logging.debug(f"URL signing skipped: {sign_err}")
                    signed_url = cog_url
            else:
                signed_url = cog_url
            
            # Configure rasterio for cloud access
            env_options = {
                'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
                'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif,.TIF,.tiff,.TIFF',
                'GDAL_HTTP_TIMEOUT': '30',
                'GDAL_HTTP_MAX_RETRY': '3',
            }
            
            with rasterio.Env(**env_options):
                with rasterio.open(signed_url) as src:
                    # Get CRS and transform
                    crs = str(src.crs)
                    logging.info(f"📍 Raster CRS: {crs}, size: {src.width}x{src.height}")
                    logging.info(f"📍 Raster bounds: {src.bounds}")
                    
                    # Transform lat/lng to pixel coordinates
                    from rasterio.warp import transform as transform_coords
                    
                    # Transform from WGS84 to the raster's CRS if needed
                    if src.crs and str(src.crs) != 'EPSG:4326':
                        xs, ys = transform_coords(
                            'EPSG:4326',
                            src.crs,
                            [longitude],
                            [latitude]
                        )
                        x, y = xs[0], ys[0]
                        logging.info(f"📍 Transformed coords: ({longitude}, {latitude}) -> ({x}, {y})")
                    else:
                        x, y = longitude, latitude
                    
                    # NOTE: We skip the strict bounds check because:
                    # 1. STAC bbox is in WGS84 but COG bounds are in native CRS (often UTM)
                    # 2. Reprojection can cause slight differences at edges
                    # 3. If the pin is on visible imagery, the data MUST exist there
                    # 4. We rely on the pixel dimension check below instead
                    
                    # Log the bounds for debugging
                    bounds = src.bounds
                    logging.info(f"📍 Raster bounds ({src.crs}): {bounds}")
                    logging.info(f"📍 Transformed point: ({x}, {y})")
                    
                    # Get row/col from coordinates - rasterio handles this
                    try:
                        row, col = src.index(x, y)
                    except Exception as idx_err:
                        logging.warning(f"📍 src.index() failed: {idx_err}")
                        lon_dir = 'W' if longitude < 0 else 'E'
                        lat_dir = 'N' if latitude >= 0 else 'S'
                        return {
                            'value': None,
                            'error': f'Point ({abs(latitude):.4f}°{lat_dir}, {abs(longitude):.4f}°{lon_dir}) coordinate transform failed',
                            'bounds': list(src.bounds),
                            'crs': crs
                        }
                    
                    logging.info(f"📍 Pixel coordinates: row={row}, col={col}")
                    
                    # Check if point is within raster dimensions
                    if row < 0 or row >= src.height or col < 0 or col >= src.width:
                        logging.warning(f"📍 Pixel ({row}, {col}) outside raster dimensions: {src.height}x{src.width}")
                        lon_dir = 'W' if longitude < 0 else 'E'
                        lat_dir = 'N' if latitude >= 0 else 'S'
                        return {
                            'value': None,
                            'error': f'Point ({abs(latitude):.4f}°{lat_dir}, {abs(longitude):.4f}°{lon_dir}) is outside raster pixel bounds',
                            'bounds': list(src.bounds),
                            'crs': crs
                        }
                    
                    # Read the pixel value - use window for efficiency
                    from rasterio.windows import Window
                    window = Window(col, row, 1, 1)
                    data = src.read(band, window=window)
                    value = float(data[0, 0])
                    logging.info(f"📍 Sampled value: {value}")
                    
                    # Check for nodata
                    nodata = src.nodata
                    if nodata is not None and value == nodata:
                        return {
                            'value': None,
                            'error': 'No data at this location (ocean mask or no coverage)',
                            'nodata_value': nodata,
                            'crs': crs
                        }
                    
                    # Get band description/unit if available
                    description = src.descriptions[band-1] if src.descriptions and len(src.descriptions) >= band else None
                    
                    return {
                        'value': value,
                        'band': band,
                        'description': description,
                        'crs': crs,
                        'pixel_location': {'row': row, 'col': col},
                        'nodata_value': nodata
                    }
                    
        except ImportError as e:
            return {'value': None, 'error': f'rasterio not available: {e}'}
        except rasterio.RasterioIOError as e:
            error_str = str(e)
            logging.error(f"📍 Failed to open COG: {e}")
            # Check for rate limit (409) - retry with exponential backoff
            if '409' in error_str and retry_count < max_retries:
                delay = 2 ** retry_count  # 1s, 2s, 4s
                logging.warning(f"📍 Rate limited (409), retrying in {delay}s (attempt {retry_count + 1}/{max_retries})")
                time.sleep(delay)
                return _sample_sync(retry_count + 1)
            return {'value': None, 'error': f'Could not open COG: {e}'}
        except Exception as e:
            error_str = str(e)
            # Also check for rate limit in generic exceptions
            if '409' in error_str and retry_count < max_retries:
                delay = 2 ** retry_count
                logging.warning(f"📍 Rate limited (409), retrying in {delay}s (attempt {retry_count + 1}/{max_retries})")
                time.sleep(delay)
                return _sample_sync(retry_count + 1)
            return {'value': None, 'error': f'Sampling error: {e}'}
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _sample_sync(0))


async def fetch_stac_item(collection: str, item_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a STAC item from Planetary Computer by collection and item ID.
    
    Args:
        collection: STAC collection ID (e.g., 'noaa-cdr-sea-surface-temperature-whoi')
        item_id: STAC item ID (e.g., 'SEAFLUX-OSB-CDR_V02R00_SST_D20210831_C20211223-7')
    
    Returns:
        STAC item dict with 'assets' or None if failed
    """
    import httpx
    
    stac_url = f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/{collection}/items/{item_id}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(stac_url)
            if response.status_code == 200:
                return response.json()
            else:
                logging.warning(f"Failed to fetch STAC item: {response.status_code}")
                return None
    except Exception as e:
        logging.error(f"Error fetching STAC item: {e}")
        return None


def parse_tile_url(tile_url: str) -> Dict[str, str]:
    """
    Parse a Planetary Computer tile URL to extract collection, item, and asset.
    
    Example URL: https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/...?collection=noaa-cdr-sea-surface-temperature-whoi&item=SEAFLUX-OSB-CDR_V02R00_SST_D20210831_C20211223-7&assets=sea_surface_temperature
    
    Returns:
        {'collection': '...', 'item': '...', 'assets': '...'}
    """
    from urllib.parse import urlparse, parse_qs
    
    parsed = urlparse(tile_url)
    params = parse_qs(parsed.query)
    
    return {
        'collection': params.get('collection', [''])[0],
        'item': params.get('item', [''])[0],
        'assets': params.get('assets', [''])[0]
    }


async def compute_ndvi_statistics(red_url: str, nir_url: str, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Compute NDVI statistics from RED and NIR band COG URLs.
    
    NDVI = (NIR - RED) / (NIR + RED)
    
    Args:
        red_url: URL to the RED band COG (e.g., B04 for HLS/Sentinel-2)
        nir_url: URL to the NIR band COG (e.g., B08/nir08 for HLS/Sentinel-2)
        bbox: Optional bounding box [west, south, east, north] to limit computation area
    
    Returns:
        Dict with min, max, mean, std NDVI values or error
    """
    import asyncio
    import numpy as np
    
    def _compute_sync():
        try:
            import rasterio
            from rasterio.windows import from_bounds
            import planetary_computer as pc
            
            # Sign URLs for Planetary Computer access
            try:
                signed_red = pc.sign(red_url) if 'blob.core.windows.net' in red_url else red_url
                signed_nir = pc.sign(nir_url) if 'blob.core.windows.net' in nir_url else nir_url
            except Exception as sign_err:
                logging.debug(f"URL signing skipped: {sign_err}")
                signed_red, signed_nir = red_url, nir_url
            
            # Configure rasterio for cloud access
            env_options = {
                'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
                'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif,.TIF,.tiff,.TIFF',
                'GDAL_HTTP_TIMEOUT': '30',
                'GDAL_HTTP_MAX_RETRY': '3',
            }
            
            with rasterio.Env(**env_options):
                with rasterio.open(signed_red) as red_src, rasterio.open(signed_nir) as nir_src:
                    # If bbox provided, read only that window; otherwise read overview
                    if bbox:
                        window = from_bounds(*bbox, red_src.transform)
                        red_data = red_src.read(1, window=window).astype(np.float32)
                        nir_data = nir_src.read(1, window=window).astype(np.float32)
                    else:
                        # Read at reduced resolution (overview) for efficiency
                        # Use overview level for faster processing
                        out_shape = (min(512, red_src.height), min(512, red_src.width))
                        red_data = red_src.read(1, out_shape=out_shape).astype(np.float32)
                        nir_data = nir_src.read(1, out_shape=out_shape).astype(np.float32)
                    
                    # Handle nodata
                    red_nodata = red_src.nodata or 0
                    nir_nodata = nir_src.nodata or 0
                    
                    # Create valid mask
                    valid_mask = (red_data != red_nodata) & (nir_data != nir_nodata)
                    valid_mask &= (red_data > 0) | (nir_data > 0)  # At least one non-zero
                    
                    if not np.any(valid_mask):
                        return {'error': 'No valid pixels found in the area'}
                    
                    # Calculate NDVI: (NIR - RED) / (NIR + RED)
                    # Handle division by zero
                    denominator = nir_data + red_data
                    denominator[denominator == 0] = np.nan
                    
                    ndvi = (nir_data - red_data) / denominator
                    
                    # Apply valid mask
                    ndvi_valid = ndvi[valid_mask]
                    
                    # Clip to valid NDVI range [-1, 1]
                    ndvi_valid = np.clip(ndvi_valid, -1, 1)
                    
                    # Remove any remaining NaN
                    ndvi_valid = ndvi_valid[~np.isnan(ndvi_valid)]
                    
                    if len(ndvi_valid) == 0:
                        return {'error': 'No valid NDVI values computed'}
                    
                    # Compute statistics
                    stats = {
                        'min': float(np.min(ndvi_valid)),
                        'max': float(np.max(ndvi_valid)),
                        'mean': float(np.mean(ndvi_valid)),
                        'std': float(np.std(ndvi_valid)),
                        'median': float(np.median(ndvi_valid)),
                        'valid_pixels': int(len(ndvi_valid)),
                        'total_pixels': int(red_data.size)
                    }
                    
                    # Add vegetation classification percentages
                    dense_veg = np.sum(ndvi_valid > 0.6) / len(ndvi_valid) * 100
                    moderate_veg = np.sum((ndvi_valid > 0.2) & (ndvi_valid <= 0.6)) / len(ndvi_valid) * 100
                    sparse_veg = np.sum((ndvi_valid > 0) & (ndvi_valid <= 0.2)) / len(ndvi_valid) * 100
                    non_veg = np.sum(ndvi_valid <= 0) / len(ndvi_valid) * 100
                    
                    stats['classification'] = {
                        'dense_vegetation': round(dense_veg, 1),
                        'moderate_vegetation': round(moderate_veg, 1),
                        'sparse_vegetation': round(sparse_veg, 1),
                        'non_vegetation': round(non_veg, 1)
                    }
                    
                    return stats
                    
        except ImportError as e:
            return {'error': f'rasterio not available: {e}'}
        except Exception as e:
            logging.error(f"NDVI computation error: {e}")
            return {'error': f'NDVI computation failed: {e}'}
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _compute_sync)


logger = logging.getLogger(__name__)


# ============================================================================
# SESSION MEMORY
# ============================================================================

@dataclass
class VisionSession:
    """Stores session state for multi-turn vision conversations."""
    session_id: str
    screenshot_base64: Optional[str] = None
    map_bounds: Optional[Dict[str, float]] = None
    loaded_collections: List[str] = field(default_factory=list)
    tile_urls: List[str] = field(default_factory=list)
    stac_items: List[Dict[str, Any]] = field(default_factory=list)  # STAC features with assets, bbox, properties
    last_analysis: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    chat_history: ChatHistory = field(default_factory=ChatHistory)
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        now = datetime.utcnow()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
    
    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        self.conversation_history.append({"role": role, "content": content})
        self.updated_at = datetime.utcnow()
        # Keep only last 10 turns
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]


# ============================================================================
# VISION AGENT SYSTEM PROMPT
# ============================================================================

VISION_AGENT_INSTRUCTIONS = """You are an intelligent Geospatial Vision Agent for Earth Copilot.

Your role is to answer user questions about the map, satellite imagery, and geographic locations using the tools available to you.

## ⚠️ CRITICAL RULES:
1. **ALWAYS USE TOOLS** - Never give generic advice like "you would need to use software X" or "typically you would access...". YOU have the tools. USE THEM.
2. **NEVER REFUSE** to perform analysis if data is available. Call the appropriate tool.
3. **For ANY numeric value question** (reflectance, temperature, elevation, NDVI, etc.) → Call analyze_raster or sample_raster_value
4. **For "sample" or "at this point/location" queries** → Call sample_raster_value
5. If a tool fails, report the actual error - don't give generic advice

## Available Tools:

### Core Analysis Tools:
1. **analyze_screenshot** - Analyze the current map screenshot using GPT-4o Vision
   - Use when: User asks about visible features, patterns, colors, or "what do you see"
   - Returns: Natural language description of visible imagery
   
2. **analyze_raster** - Get quantitative metrics from loaded raster data
   - Use when: User asks for elevation, slope, NDVI values, reflectance, statistics, measurements, averages
   - metric_type: 'elevation', 'ndvi', 'temperature', 'sst', 'reflectance', or 'general'
   - Returns: Numerical data (min/max/avg elevation, slope degrees, reflectance values, etc.)
   
3. **query_knowledge** - Answer educational or factual questions
   - Use when: User asks "why", "how", "explain", history, or general knowledge
   - Returns: Educational/contextual information

4. **identify_features** - Identify specific geographic features in the view
   - Use when: User asks "what is that river/mountain/city", feature identification
   - Returns: Feature names, classifications, and descriptions

5. **compare_temporal** - Compare current view with historical imagery
   - Use when: User asks about changes over time, before/after, historical comparison
   - Returns: Description of changes detected

### Specialized Raster Analysis Tools:
6. **analyze_vegetation** - Analyze vegetation from MODIS products
   - Use when: Questions about NDVI, LAI, plant productivity (GPP/NPP), forest health
   - analysis_type: 'ndvi', 'lai', 'fpar', 'npp', 'gpp', or 'general'
   - Returns: Vegetation indices, productivity metrics, health assessment

7. **analyze_fire** - Analyze fire activity from MODIS products
   - Use when: Questions about wildfires, thermal anomalies, burn severity
   - analysis_type: 'active', 'thermal', 'burned', or 'general'
   - Returns: Active fire detections, fire intensity, burn area analysis

8. **analyze_land_cover** - Analyze land cover classifications
   - Use when: Questions about land use, urban areas, forest cover, cropland
   - analysis_type: 'classification', 'urban', 'forest', 'agriculture', or 'general'
   - Returns: Land cover types and distributions

9. **analyze_snow** - Analyze snow/ice cover from MODIS products
   - Use when: Questions about snow cover, snow extent, winter conditions
   - analysis_type: 'cover', 'extent', 'albedo', or 'general'
   - Returns: Snow cover percentage, seasonal patterns

10. **analyze_sar** - Analyze radar data from Sentinel-1
    - Use when: Questions about floods, surface changes, works through clouds
    - analysis_type: 'backscatter', 'flood', 'change', or 'general'
    - Returns: SAR analysis, flood detection, surface monitoring

11. **analyze_water** - Analyze surface water from JRC dataset
    - Use when: Questions about lakes, rivers, flooding, water extent
    - analysis_type: 'occurrence', 'seasonality', 'change', or 'general'
    - Returns: Water occurrence, seasonality, water body changes

12. **analyze_biomass** - Analyze above-ground biomass
    - Use when: Questions about forest biomass, carbon stocks, vegetation density
    - analysis_type: 'carbon', 'density', or 'general'
    - Returns: Biomass estimates in tonnes per hectare

### Point Value Extraction (IMPORTANT - Use for specific location queries!):
13. **sample_raster_value** - Extract actual pixel values at a specific location
    - Use when: User asks for the EXACT value at a point (e.g., "what's the temperature here", "SST at this location", "elevation at this spot", "what is the temperature at/on this location")
    - data_type: 'sst', 'temperature', 'elevation', 'ndvi', or 'auto'
    - Returns: The actual numeric value from the raster (e.g., "24.5°C", "1234m")
    - ⚠️ Requires pin location or map center coordinates
    - 🎯 PREFER THIS over analyze_screenshot when user asks about numeric values at a location!

## Tool Selection Guidelines:
- **Visual questions** ("what do you see", "describe", "what's visible") → analyze_screenshot
- **Elevation/terrain** ("what's the elevation", "how steep") → analyze_raster with metric_type='elevation'
- **POINT-SPECIFIC VALUE QUERIES** ("what is the temperature HERE", "temperature AT THIS LOCATION", "SST at this spot", "value at pin", "sample the raster", "extract the value") → **sample_raster_value** (FIRST CHOICE for location-specific numeric values)
- **General temperature/SST overview** ("show temperature pattern", "ocean heat distribution", "explain the colors") → analyze_raster with metric_type='temperature'
- **Exact point values** ("what's the value here", "temperature at this point", "SST at pin") → sample_raster_value
- **Vegetation health** ("NDVI", "plant health", "greenness") → analyze_vegetation
- **Fire activity** ("wildfires", "burn areas", "thermal anomalies") → analyze_fire
- **Land use** ("land cover", "urban areas", "cropland") → analyze_land_cover
- **Snow/ice** ("snow cover", "winter conditions", "percentage snow") → analyze_snow
- **Radar/SAR data** ("SAR backscatter", "Sentinel-1", "ALOS PALSAR", "radar", "VV/VH", "urban water vegetation from SAR") → **analyze_sar**
- **Water/flooding** ("water bodies", "flood", "flooding", "lake extent", "river", "inundation") → **analyze_water**
- **Forest carbon** ("biomass", "carbon stock", "tree density") → analyze_biomass
- **Educational questions** ("what is NDVI", "explain", "history") → query_knowledge
- **Complex questions** → Combine multiple tools and synthesize

## ⚠️ CRITICAL DATA-TO-TOOL MATCHING:
When specific data is loaded, ALWAYS use the matching specialized tool:
- **sentinel-1-rtc, sentinel-1-grd, alos-palsar-mosaic** → MUST USE analyze_sar
- **modis-10a1-061, modis-10a2-061** (snow products) → MUST USE analyze_snow
- **jrc-gsw** (water) → MUST USE analyze_water
- **modis-14a1-061, mtbs** (fire/burn) → MUST USE analyze_fire
- **modis-13q1-061, modis-13a1-061** (vegetation) → MUST USE analyze_vegetation
- **chloris-biomass** → MUST USE analyze_biomass

## Response Guidelines:
1. Always use tools to gather information - don't guess
2. If multiple tools are helpful, call them all
3. Synthesize results into a coherent, natural response
4. Be specific with numbers and measurements when available
5. Reference the actual data you're seeing, not general knowledge
6. Match the tool to the loaded data type for best results

## Context Available:
The session context includes:
- Current screenshot (if available)
- Map bounds (lat/lng)
- Loaded STAC collections (e.g., sentinel-2-l2a, cop-dem-glo-30, modis-13a1-061)
- STAC items with metadata and temporal information
- Conversation history for follow-up questions
"""


# ============================================================================
# VISION AGENT TOOLS (Semantic Kernel Plugin)
# ============================================================================

class VisionAgentTools:
    """
    Tools for the Vision Agent, registered as a Semantic Kernel plugin.
    GPT-4o will decide which tools to call based on the user's question.
    """
    
    def __init__(self, agent_ref: 'EnhancedVisionAgent'):
        """Initialize with reference to parent agent for context access."""
        self._agent = agent_ref
        self._vision_client = None
        # Track tool calls for tracing/debugging
        self._tool_calls: List[Dict[str, Any]] = []
    
    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get list of tool calls made during this session."""
        return self._tool_calls.copy()
    
    def clear_tool_calls(self):
        """Clear tool call history (call before each new query)."""
        self._tool_calls = []
    
    def _log_tool_call(self, tool_name: str, args: Dict[str, Any], result_preview: str = ""):
        """Log a tool call for tracing."""
        import time
        call_record = {
            "tool": tool_name,
            "timestamp": datetime.utcnow().isoformat(),
            "args": args,
            "result_preview": result_preview[:200] if result_preview else ""
        }
        self._tool_calls.append(call_record)
        logger.info(f"🔧 TOOL CALL: {tool_name} | Args: {args} | Preview: {result_preview[:100]}...")
    
    def _get_vision_client(self):
        """Get or create Azure OpenAI client for vision calls."""
        if self._vision_client is None:
            _load_azure_openai()
            if AzureOpenAI is None:
                return None
            
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            self._vision_client = AzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                timeout=120.0
            )
        return self._vision_client
    
    @kernel_function(
        name="analyze_screenshot",
        description="Analyze the current map screenshot using GPT-4o Vision. Use this for visual questions about what's visible on the map, identifying features, describing patterns, colors, or land cover."
    )
    async def analyze_screenshot(
        self,
        question: Annotated[str, "The specific question to answer about the visible imagery"]
    ) -> str:
        """Analyze map screenshot with GPT-4o Vision."""
        logger.info(f"📷 TOOL INVOKED: analyze_screenshot(question='{question[:50]}...')")
        session = self._agent._current_session
        
        # ================================================================
        # 🔍 SCREENSHOT VERIFICATION LOGGING
        # ================================================================
        if session:
            screenshot_data = session.screenshot_base64
            if screenshot_data:
                # Log screenshot details for verification
                size_kb = len(screenshot_data) / 1024
                is_data_url = screenshot_data.startswith('data:image')
                prefix = screenshot_data[:50] if len(screenshot_data) > 50 else screenshot_data
                logger.info(f"📸 SCREENSHOT VERIFIED: size={size_kb:.1f}KB, is_data_url={is_data_url}, prefix='{prefix}...'")
            else:
                logger.warning(f"⚠️ SCREENSHOT MISSING: session exists but screenshot_base64 is None/empty")
        else:
            logger.warning(f"⚠️ SESSION MISSING: _current_session is None")
        
        if not session or not session.screenshot_base64:
            self._log_tool_call("analyze_screenshot", {"question": question}, "No screenshot available")
            return "No screenshot available. The user needs to have a map view loaded."
        
        try:
            client = self._get_vision_client()
            if not client:
                return "Vision analysis unavailable - Azure OpenAI client not initialized."
            
            # Prepare image data
            image_data = session.screenshot_base64
            if image_data.startswith('data:image'):
                image_data = image_data.split(',', 1)[1]
            
            # Build context
            context_parts = []
            if session.map_bounds:
                bounds = session.map_bounds
                context_parts.append(f"Map location: ({bounds.get('center_lat', 'N/A')}, {bounds.get('center_lng', 'N/A')})")
            if session.loaded_collections:
                context_parts.append(f"Data layers: {', '.join(session.loaded_collections)}")
            
            context_str = "\n".join(context_parts) if context_parts else "No additional context"
            
            system_prompt = f"""You are a geospatial imagery analyst. Analyze the satellite/map imagery and answer the question.

Context:
{context_str}

Guidelines:
- Describe visible features clearly (water bodies, vegetation, urban areas, terrain)
- Identify patterns, colors, and their likely meaning
- Be specific about locations and features
- If you can't see something clearly, say so"""

            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            logger.info(f"📷 analyze_screenshot complete ({len(result)} chars)")
            self._log_tool_call("analyze_screenshot", {"question": question, "has_image": True}, result)
            return result
            
        except Exception as e:
            logger.error(f"❌ analyze_screenshot failed: {e}")
            return f"Screenshot analysis failed: {str(e)}"
    
    @kernel_function(
        name="analyze_raster",
        description="Get quantitative metrics from loaded raster data like elevation, slope, NDVI values, or sea surface temperature (SST). Use this for numerical questions about terrain statistics, temperature values, measurements, and calculations. Essential for SST/ocean temperature queries."
    )
    async def analyze_raster(
        self,
        metric_type: Annotated[str, "Type of metric to analyze: 'elevation', 'slope', 'ndvi', 'temperature', 'sst', or 'general'"]
    ) -> str:
        """Analyze raster data for quantitative metrics using actual loaded STAC items."""
        logger.info(f"📊 TOOL INVOKED: analyze_raster(metric_type='{metric_type}')")
        session = self._agent._current_session
        
        if not session or not session.loaded_collections:
            self._log_tool_call("analyze_raster", {"metric_type": metric_type}, "No raster data loaded")
            return "No raster data loaded. The user needs to load satellite imagery first."
        
        try:
            results = []
            collections = session.loaded_collections
            stac_items = session.stac_items if hasattr(session, 'stac_items') else []
            
            # Log what STAC items we have available
            if stac_items:
                logger.info(f"📦 Using {len(stac_items)} loaded STAC items for analysis")
                for item in stac_items[:3]:  # Log first 3
                    logger.info(f"   - Item: {item.get('id')}, Collection: {item.get('collection')}")
            else:
                logger.warning("⚠️ No STAC items in session - will attempt DEM fetch for elevation")
            
            # ================================================================
            # ELEVATION ANALYSIS: Use DEM collection or fetch terrain data
            # ================================================================
            if metric_type in ['elevation', 'general']:
                # Check if we have DEM items loaded
                dem_items = [i for i in stac_items if 'dem' in i.get('collection', '').lower() or 'elevation' in i.get('collection', '').lower()]
                
                if dem_items:
                    # Use actual loaded DEM item
                    item = dem_items[0]
                    results.append(f"**Elevation Data (from {item.get('collection')}):**")
                    results.append(f"- Item ID: {item.get('id')}")
                    if item.get('properties', {}).get('datetime'):
                        results.append(f"- Captured: {item['properties']['datetime'][:10]}")
                    if item.get('bbox'):
                        bbox = item['bbox']
                        results.append(f"- Coverage: {bbox[0]:.2f}°W to {bbox[2]:.2f}°E, {bbox[1]:.2f}°S to {bbox[3]:.2f}°N")
                elif any('dem' in c.lower() for c in collections):
                    # DEM collection loaded but no items - try fetching
                    try:
                        from geoint.raster_data_fetcher import get_raster_fetcher
                        fetcher = get_raster_fetcher()
                        
                        if session.map_bounds:
                            bounds = session.map_bounds
                            terrain_data = await fetcher.fetch_terrain_data(
                                latitude=bounds.get('center_lat', 0),
                                longitude=bounds.get('center_lng', 0),
                                radius_miles=5.0
                            )
                            
                            if terrain_data and terrain_data.get('elevation_stats'):
                                elev = terrain_data['elevation_stats']
                                slope = terrain_data.get('slope_stats', {})
                                
                                results.append(f"**Elevation Statistics:**")
                                results.append(f"- Min: {elev.get('min', 0):.1f}m")
                                results.append(f"- Max: {elev.get('max', 0):.1f}m")
                                results.append(f"- Mean: {elev.get('mean', 0):.1f}m")
                                results.append(f"- Terrain Type: {terrain_data.get('terrain_classification', 'unknown')}")
                                
                                if slope:
                                    results.append(f"\n**Slope Analysis:**")
                                    results.append(f"- Average Slope: {slope.get('mean', 0):.1f}°")
                                    results.append(f"- Max Slope: {slope.get('max', 0):.1f}°")
                    except Exception as e:
                        logger.warning(f"DEM fetch failed: {e}")
            
            # ================================================================
            # SEA SURFACE TEMPERATURE (SST) ANALYSIS
            # ================================================================
            if metric_type in ['temperature', 'sst', 'general']:
                # Find SST/temperature items
                sst_keywords = ['sea-surface-temperature', 'sst', 'temperature-whoi', 'noaa-cdr']
                sst_items = [i for i in stac_items if any(kw in i.get('collection', '').lower() for kw in sst_keywords)]
                
                # Also check if collection name contains temperature
                if not sst_items:
                    sst_items = [i for i in stac_items if 'temperature' in i.get('collection', '').lower()]
                
                if sst_items:
                    item = sst_items[0]
                    props = item.get('properties', {})
                    assets = item.get('assets', {})
                    collection_name = item.get('collection', 'unknown')
                    
                    results.append(f"\n**Sea Surface Temperature Data (from {collection_name}):**")
                    results.append(f"- Item ID: {item.get('id')}")
                    if props.get('datetime'):
                        results.append(f"- Date: {props['datetime'][:10]}")
                    
                    # Get bounding box for location context
                    if item.get('bbox'):
                        bbox = item['bbox']
                        results.append(f"- Coverage: {bbox[0]:.2f}° to {bbox[2]:.2f}°E, {bbox[1]:.2f}° to {bbox[3]:.2f}°N")
                    
                    # SST data is typically in Kelvin (270-310K range)
                    results.append(f"\n**Temperature Information:**")
                    results.append(f"- Data Unit: Kelvin (K)")
                    results.append(f"- Typical Ocean Range: 270K to 310K (-3°C to 37°C)")
                    results.append(f"- Colormap: Turbo (blue=cold, red=warm)")
                    
                    # Check for the sea_surface_temperature asset
                    if 'sea_surface_temperature' in assets:
                        sst_asset = assets['sea_surface_temperature']
                        results.append(f"\n**SST Asset Available:**")
                        results.append(f"- Asset: sea_surface_temperature")
                        if sst_asset.get('href'):
                            results.append(f"- Data accessible for pixel-level analysis")
                    
                    # Provide pin location context if available
                    if session and session.map_bounds:
                        bounds = session.map_bounds
                        # Prefer pin location over map center for point analysis
                        pin_lat = bounds.get('pin_lat') or bounds.get('center_lat')
                        pin_lng = bounds.get('pin_lng') or bounds.get('center_lng')
                        if pin_lat and pin_lng:
                            results.append(f"\n**Pin Location:**")
                            results.append(f"- Coordinates: ({pin_lat:.4f}, {pin_lng:.4f})")
                            results.append(f"- Note: For exact pixel values, raster point sampling would be needed")
                    
                    # Report how many SST items we have
                    if len(sst_items) > 1:
                        results.append(f"\n**Temporal Coverage:**")
                        results.append(f"- {len(sst_items)} SST observations available")
                        dates = [i.get('properties', {}).get('datetime', '')[:10] for i in sst_items[:5] if i.get('properties', {}).get('datetime')]
                        if dates:
                            results.append(f"- Recent dates: {', '.join(dates)}")
                else:
                    # Check if any temperature-related collection is loaded but no items
                    temp_collections = [c for c in collections if any(kw in c.lower() for kw in sst_keywords + ['temperature'])]
                    if temp_collections:
                        results.append(f"\n**Temperature Collections Loaded:**")
                        results.append(f"- Collections: {temp_collections}")
                        results.append(f"- No specific STAC items available in session for pixel extraction")
            
            # ================================================================
            # VEGETATION/NDVI ANALYSIS: Compute actual NDVI from optical imagery
            # ================================================================
            if metric_type in ['ndvi', 'general']:
                # Find optical imagery items (Sentinel-2, Landsat, HLS)
                optical_keywords = ['sentinel-2', 'landsat', 'hls', 's30', 'l30']
                optical_items = [i for i in stac_items if any(kw in i.get('collection', '').lower() for kw in optical_keywords)]
                
                if optical_items:
                    item = optical_items[0]
                    props = item.get('properties', {})
                    assets = item.get('assets', {})
                    collection_name = item.get('collection', 'unknown')
                    
                    results.append(f"\n**Optical Imagery (from {collection_name}):**")
                    results.append(f"- Item ID: {item.get('id')}")
                    if props.get('datetime'):
                        results.append(f"- Captured: {props['datetime'][:10]}")
                    
                    # Report cloud cover
                    cloud_cover = props.get('eo:cloud_cover') or props.get('cloud_cover')
                    if cloud_cover is not None:
                        results.append(f"- Cloud Cover: {cloud_cover:.1f}%")
                    
                    # Identify RED and NIR band asset URLs
                    red_url = None
                    nir_url = None
                    
                    # HLS uses B04 (Red) and B08 (NIR) or Fmask for cloud masking
                    # Landsat uses 'red' and 'nir08'
                    # Sentinel-2 uses B04 and B08
                    
                    # Try HLS/Sentinel-2 naming first
                    if 'B04' in assets:
                        red_url = assets['B04'].get('href')
                    elif 'red' in assets:
                        red_url = assets['red'].get('href')
                    
                    if 'B08' in assets:
                        nir_url = assets['B08'].get('href')
                    elif 'nir08' in assets:
                        nir_url = assets['nir08'].get('href')
                    elif 'B8A' in assets:  # Alternative NIR for Sentinel-2
                        nir_url = assets['B8A'].get('href')
                    
                    if red_url and nir_url:
                        # Compute actual NDVI statistics
                        logger.info(f"📊 Computing NDVI from {collection_name}...")
                        logger.info(f"   RED band: {red_url[:80]}...")
                        logger.info(f"   NIR band: {nir_url[:80]}...")
                        
                        try:
                            # Get bbox from item or session for focused computation
                            bbox = item.get('bbox')
                            
                            ndvi_stats = await compute_ndvi_statistics(red_url, nir_url, bbox=None)  # Use overview for speed
                            
                            if 'error' in ndvi_stats:
                                results.append(f"\n**NDVI Analysis:**")
                                results.append(f"- Error: {ndvi_stats['error']}")
                            else:
                                results.append(f"\n**NDVI Statistics (Computed):**")
                                results.append(f"- Minimum: **{ndvi_stats['min']:.3f}**")
                                results.append(f"- Maximum: **{ndvi_stats['max']:.3f}**")
                                results.append(f"- Mean: **{ndvi_stats['mean']:.3f}**")
                                results.append(f"- Std Dev: {ndvi_stats['std']:.3f}")
                                results.append(f"- Median: {ndvi_stats['median']:.3f}")
                                results.append(f"- Valid Pixels: {ndvi_stats['valid_pixels']:,} / {ndvi_stats['total_pixels']:,}")
                                
                                # Add interpretation
                                mean_ndvi = ndvi_stats['mean']
                                if mean_ndvi > 0.6:
                                    veg_health = "Dense healthy vegetation"
                                elif mean_ndvi > 0.4:
                                    veg_health = "Moderate vegetation"
                                elif mean_ndvi > 0.2:
                                    veg_health = "Sparse/stressed vegetation"
                                elif mean_ndvi > 0:
                                    veg_health = "Minimal vegetation or bare soil"
                                else:
                                    veg_health = "Water, snow, or non-vegetated"
                                
                                results.append(f"\n**Interpretation:**")
                                results.append(f"- Overall: {veg_health}")
                                
                                # Add classification breakdown if available
                                if 'classification' in ndvi_stats:
                                    cls = ndvi_stats['classification']
                                    results.append(f"\n**Land Cover Classification:**")
                                    results.append(f"- Dense Vegetation (NDVI > 0.6): {cls['dense_vegetation']:.1f}%")
                                    results.append(f"- Moderate Vegetation (0.2-0.6): {cls['moderate_vegetation']:.1f}%")
                                    results.append(f"- Sparse Vegetation (0-0.2): {cls['sparse_vegetation']:.1f}%")
                                    results.append(f"- Non-Vegetation (NDVI ≤ 0): {cls['non_vegetation']:.1f}%")
                        except Exception as e:
                            logger.error(f"NDVI computation failed: {e}")
                            results.append(f"\n**NDVI Computation Error:** {str(e)}")
                            # Fall back to capability info
                            results.append(f"\n**Bands Available for NDVI:**")
                            results.append(f"- Red band (B04): Available")
                            results.append(f"- NIR band (B08): Available")
                    else:
                        # No red/nir bands found - report what's available
                        band_names = list(assets.keys()) if assets else []
                        spectral_bands = [b for b in band_names if any(x in b.upper() for x in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'RED', 'NIR', 'GREEN', 'BLUE', 'SWIR'])]
                        
                        if spectral_bands:
                            results.append(f"\n**Available Bands:** {', '.join(spectral_bands[:8])}{'...' if len(spectral_bands) > 8 else ''}")
                            results.append(f"- Note: Could not identify standard RED/NIR bands for NDVI calculation")
                else:
                    # No items but collection mentioned
                    optical_collections = [c for c in collections if any(kw in c.lower() for kw in optical_keywords)]
                    if optical_collections:
                        results.append(f"\n**Vegetation Analysis Available:**")
                        results.append(f"- NDVI can be calculated from: {optical_collections}")
            
            # ================================================================
            # SUMMARY: Report all loaded STAC item metadata
            # ================================================================
            if metric_type == 'general' and stac_items:
                results.append(f"\n**Loaded Imagery Summary:**")
                results.append(f"- Total items: {len(stac_items)}")
                for i, item in enumerate(stac_items[:5]):
                    props = item.get('properties', {})
                    dt = props.get('datetime', 'unknown')[:10] if props.get('datetime') else 'unknown'
                    cloud = props.get('eo:cloud_cover') or props.get('cloud_cover') or 'N/A'
                    cloud_str = f"{cloud:.1f}%" if isinstance(cloud, (int, float)) else cloud
                    results.append(f"  {i+1}. {item.get('id', 'unknown')} ({dt}, cloud: {cloud_str})")
            
            if not results:
                self._log_tool_call("analyze_raster", {"metric_type": metric_type, "collections": collections}, "No data available")
                return f"No {metric_type} data available in loaded collections: {collections}"
            
            result = "\n".join(results)
            self._log_tool_call("analyze_raster", {"metric_type": metric_type, "collections": collections, "stac_items": len(stac_items)}, result)
            return result
            
        except Exception as e:
            logger.error(f"❌ analyze_raster failed: {e}")
            self._log_tool_call("analyze_raster", {"metric_type": metric_type, "error": str(e)}, "Failed")
            return f"Raster analysis failed: {str(e)}"

    # ========================================================================
    # VEGETATION ANALYSIS TOOL
    # ========================================================================
    @kernel_function(
        name="analyze_vegetation",
        description="Analyze vegetation indices from satellite imagery. Supports MODIS vegetation products (pre-computed NDVI/EVI) and optical imagery like HLS, Sentinel-2, Landsat (calculates NDVI from RED/NIR bands). Returns NDVI values and vegetation health assessment."
    )
    async def analyze_vegetation(
        self,
        analysis_type: Annotated[str, "Type: 'ndvi' (greenness), 'lai' (leaf area), 'fpar' (absorbed radiation), 'npp' (net primary productivity), 'gpp' (gross productivity), or 'general'"] = "general"
    ) -> str:
        """Analyze vegetation from MODIS products or calculate NDVI from optical imagery."""
        logger.info(f"🌿 TOOL INVOKED: analyze_vegetation(analysis_type='{analysis_type}')")
        session = self._agent._current_session
        
        if not session or not session.loaded_collections:
            return "No vegetation data loaded. Load MODIS vegetation products or optical imagery (HLS, Sentinel-2, Landsat) first."
        
        try:
            results = []
            stac_items = session.stac_items if hasattr(session, 'stac_items') else []
            collections = session.loaded_collections
            
            # MODIS vegetation product mapping
            veg_products = {
                'modis-13a1-061': {'name': 'MODIS Vegetation Indices 16-Day (500m)', 'metrics': ['NDVI', 'EVI'], 'resolution': '500m'},
                'modis-13q1-061': {'name': 'MODIS Vegetation Indices 16-Day (250m)', 'metrics': ['NDVI', 'EVI'], 'resolution': '250m'},
                'modis-15a2h-061': {'name': 'MODIS LAI/FPAR 8-Day', 'metrics': ['LAI', 'FPAR'], 'resolution': '500m'},
                'modis-15a3h-061': {'name': 'MODIS LAI/FPAR 4-Day', 'metrics': ['LAI', 'FPAR'], 'resolution': '500m'},
                'modis-17a2h-061': {'name': 'MODIS GPP 8-Day', 'metrics': ['GPP', 'PSN'], 'resolution': '500m'},
                'modis-17a2hgf-061': {'name': 'MODIS GPP 8-Day Gap-Filled', 'metrics': ['GPP', 'PSN'], 'resolution': '500m'},
                'modis-17a3hgf-061': {'name': 'MODIS NPP Yearly', 'metrics': ['NPP'], 'resolution': '500m'},
                'modis-16a3gf-061': {'name': 'MODIS Evapotranspiration Yearly', 'metrics': ['ET', 'PET'], 'resolution': '500m'},
            }
            
            # Optical imagery that supports NDVI calculation (RED + NIR bands)
            optical_products = {
                'hls2-l30': {'name': 'Harmonized Landsat Sentinel-2 (Landsat)', 'resolution': '30m', 'red': 'B04', 'nir': 'B05'},
                'hls2-s30': {'name': 'Harmonized Landsat Sentinel-2 (Sentinel)', 'resolution': '30m', 'red': 'B04', 'nir': 'B08'},
                'hls-l30': {'name': 'HLS Landsat 30m', 'resolution': '30m', 'red': 'B04', 'nir': 'B05'},
                'hls-s30': {'name': 'HLS Sentinel 30m', 'resolution': '30m', 'red': 'B04', 'nir': 'B8A'},
                'sentinel-2-l2a': {'name': 'Sentinel-2 L2A', 'resolution': '10m', 'red': 'B04', 'nir': 'B08'},
                'landsat-c2-l2': {'name': 'Landsat Collection 2 L2', 'resolution': '30m', 'red': 'red', 'nir': 'nir08'},
            }
            
            # Find matching MODIS vegetation items
            veg_items = []
            optical_items = []
            for item in stac_items:
                coll = item.get('collection', '').lower()
                if any(vp in coll for vp in veg_products.keys()):
                    veg_items.append(item)
                elif any(op in coll for op in optical_products.keys()):
                    optical_items.append(item)
            
            # Check if we have optical imagery for NDVI calculation
            if optical_items and analysis_type in ['ndvi', 'general']:
                results.append("**🌿 NDVI Analysis from Optical Imagery:**\n")
                results.append("_To calculate NDVI at a specific location, use the `sample_raster_value` tool with data_type='ndvi' after placing a pin._\n")
                
                for item in optical_items[:3]:  # Show up to 3 items
                    coll = item.get('collection', '').lower()
                    props = item.get('properties', {})
                    assets = item.get('assets', {})
                    
                    # Find matching product info
                    product_info = None
                    for key, info in optical_products.items():
                        if key in coll:
                            product_info = info
                            break
                    
                    if product_info:
                        results.append(f"**{product_info['name']}:**")
                        results.append(f"- Item: {item.get('id')}")
                        if props.get('datetime'):
                            results.append(f"- Date: {props['datetime'][:10]}")
                        results.append(f"- Resolution: {product_info['resolution']}")
                        
                        # Check if required bands are available
                        red_key = product_info.get('red')
                        nir_key = product_info.get('nir')
                        has_red = red_key in assets
                        has_nir = nir_key in assets
                        
                        if has_red and has_nir:
                            results.append(f"- ✅ RED band ({red_key}) and NIR band ({nir_key}) available for NDVI calculation")
                        else:
                            missing = []
                            if not has_red:
                                missing.append(f"RED ({red_key})")
                            if not has_nir:
                                missing.append(f"NIR ({nir_key})")
                            results.append(f"- ⚠️ Missing bands: {', '.join(missing)}")
                        
                        if props.get('eo:cloud_cover') is not None:
                            results.append(f"- Cloud Cover: {props['eo:cloud_cover']:.1f}%")
                        
                        results.append("")
                
                results.append("**NDVI Interpretation:**")
                results.append("  - -1.0 to 0.0: Water, bare soil, snow")
                results.append("  - 0.0 to 0.2: Sparse vegetation, urban")
                results.append("  - 0.2 to 0.5: Moderate vegetation, grassland")
                results.append("  - 0.5 to 1.0: Dense vegetation, healthy forest")
                results.append("")
                results.append("_To get the exact NDVI value at your pin location, ask: 'What is the NDVI at this point?'_")
                
                return "\n".join(results)
            
            if veg_items:
                results.append("**🌿 Vegetation Analysis Results:**\n")
                
                for item in veg_items[:3]:  # Analyze up to 3 items
                    coll = item.get('collection', '').lower()
                    props = item.get('properties', {})
                    
                    # Find matching product info
                    product_info = None
                    for key, info in veg_products.items():
                        if key in coll:
                            product_info = info
                            break
                    
                    if product_info:
                        results.append(f"**{product_info['name']}:**")
                        results.append(f"- Item: {item.get('id')}")
                        if props.get('datetime'):
                            results.append(f"- Date: {props['datetime'][:10]}")
                        results.append(f"- Resolution: {product_info['resolution']}")
                        results.append(f"- Available Metrics: {', '.join(product_info['metrics'])}")
                        
                        # Provide interpretation guidance
                        if 'NDVI' in product_info['metrics']:
                            results.append(f"\n**NDVI Interpretation:**")
                            results.append(f"  - -1.0 to 0.0: Water, bare soil, snow")
                            results.append(f"  - 0.0 to 0.2: Sparse vegetation, urban")
                            results.append(f"  - 0.2 to 0.5: Moderate vegetation, grassland")
                            results.append(f"  - 0.5 to 1.0: Dense vegetation, healthy forest")
                        
                        if 'LAI' in product_info['metrics']:
                            results.append(f"\n**LAI (Leaf Area Index) Interpretation:**")
                            results.append(f"  - 0-1: Sparse canopy")
                            results.append(f"  - 1-3: Open canopy")
                            results.append(f"  - 3-6: Closed canopy forest")
                            results.append(f"  - 6+: Dense tropical forest")
                        
                        if 'GPP' in product_info['metrics'] or 'NPP' in product_info['metrics']:
                            results.append(f"\n**Productivity Metrics:**")
                            results.append(f"  - GPP: Total carbon fixed by photosynthesis")
                            results.append(f"  - NPP: Net carbon after plant respiration")
                            results.append(f"  - Units: kg C/m²/period")
                        
                        results.append("")
            else:
                # Check if any vegetation collections are loaded
                veg_collections = [c for c in collections if any(vp in c.lower() for vp in veg_products.keys()) or 'vegetation' in c.lower() or 'ndvi' in c.lower()]
                optical_collections = [c for c in collections if any(op in c.lower() for op in optical_products.keys())]
                if veg_collections:
                    results.append(f"**Vegetation Collections Loaded:** {veg_collections}")
                    results.append("No specific STAC items available for analysis.")
                elif optical_collections:
                    results.append(f"**Optical Imagery Loaded:** {optical_collections}")
                    results.append("You can calculate NDVI from this imagery.")
                    results.append("_To get the NDVI at a specific point, place a pin and ask: 'What is the NDVI at this point?'_")
                else:
                    results.append("No vegetation data loaded. You can use:")
                    results.append("")
                    results.append("**Pre-computed NDVI (MODIS):**")
                    results.append("- modis-13a1-061 or modis-13q1-061 for NDVI/EVI")
                    results.append("- modis-15a2h-061 for LAI/FPAR")
                    results.append("- modis-17a2h-061 for GPP")
                    results.append("")
                    results.append("**Or calculate NDVI from optical imagery:**")
                    results.append("- HLS (hls2-l30, hls2-s30) - 30m resolution")
                    results.append("- Sentinel-2 (sentinel-2-l2a) - 10m resolution")
                    results.append("- Landsat (landsat-c2-l2) - 30m resolution")
            
            return "\n".join(results) if results else "No vegetation analysis available."
            
        except Exception as e:
            logger.error(f"❌ analyze_vegetation failed: {e}")
            return f"Vegetation analysis failed: {str(e)}"

    # ========================================================================
    # FIRE ANALYSIS TOOL
    # ========================================================================
    @kernel_function(
        name="analyze_fire",
        description="Analyze fire activity and burn severity from MODIS fire products. Detects active fires, thermal anomalies, and burned areas."
    )
    async def analyze_fire(
        self,
        analysis_type: Annotated[str, "Type: 'active' (current fires), 'thermal' (thermal anomalies), 'burned' (burn severity), or 'general'"] = "general"
    ) -> str:
        """Analyze fire data from MODIS fire products."""
        logger.info(f"🔥 TOOL INVOKED: analyze_fire(analysis_type='{analysis_type}')")
        session = self._agent._current_session
        
        if not session or not session.loaded_collections:
            return "No fire data loaded. Load MODIS fire products first."
        
        try:
            results = []
            stac_items = session.stac_items if hasattr(session, 'stac_items') else []
            collections = session.loaded_collections
            
            # MODIS fire product mapping
            fire_products = {
                'modis-14a1-061': {'name': 'MODIS Thermal Anomalies/Fire Daily (1km)', 'type': 'active_fire'},
                'modis-14a2-061': {'name': 'MODIS Thermal Anomalies/Fire 8-Day (1km)', 'type': 'active_fire'},
                'modis-64a1-061': {'name': 'MODIS Burned Area Monthly', 'type': 'burned_area'},
                'mtbs': {'name': 'Monitoring Trends in Burn Severity', 'type': 'burn_severity'},
            }
            
            # Find matching fire items
            fire_items = []
            for item in stac_items:
                coll = item.get('collection', '').lower()
                if any(fp in coll for fp in fire_products.keys()):
                    fire_items.append(item)
            
            if fire_items:
                results.append("**🔥 Fire Analysis Results:**\n")
                
                for item in fire_items[:3]:
                    coll = item.get('collection', '').lower()
                    props = item.get('properties', {})
                    
                    product_info = None
                    for key, info in fire_products.items():
                        if key in coll:
                            product_info = info
                            break
                    
                    if product_info:
                        results.append(f"**{product_info['name']}:**")
                        results.append(f"- Item: {item.get('id')}")
                        if props.get('datetime'):
                            results.append(f"- Date: {props['datetime'][:10]}")
                        results.append(f"- Detection Type: {product_info['type'].replace('_', ' ').title()}")
                        
                        if product_info['type'] == 'active_fire':
                            results.append(f"\n**Fire Detection Confidence:**")
                            results.append(f"  - Low: Possible fire, needs verification")
                            results.append(f"  - Nominal: Likely fire activity")
                            results.append(f"  - High: Confirmed thermal anomaly")
                            results.append(f"\n**Fire Radiative Power (FRP):**")
                            results.append(f"  - Indicates fire intensity in MW")
                        
                        if product_info['type'] in ['burned_area', 'burn_severity']:
                            results.append(f"\n**Burn Severity Classes:**")
                            results.append(f"  - Unburned: No fire damage")
                            results.append(f"  - Low: Light surface burn")
                            results.append(f"  - Moderate: Partial canopy damage")
                            results.append(f"  - High: Complete vegetation removal")
                        
                        results.append("")
            else:
                fire_collections = [c for c in collections if any(fp in c.lower() for fp in fire_products.keys()) or 'fire' in c.lower() or 'burn' in c.lower()]
                if fire_collections:
                    results.append(f"**Fire Collections Loaded:** {fire_collections}")
                    results.append("No specific STAC items available for analysis.")
                else:
                    results.append("No fire data loaded. Try loading:")
                    results.append("- modis-14a1-061 for daily fire detection")
                    results.append("- modis-64a1-061 for monthly burned area")
                    results.append("- mtbs for burn severity")
            
            return "\n".join(results) if results else "No fire analysis available."
            
        except Exception as e:
            logger.error(f"❌ analyze_fire failed: {e}")
            return f"Fire analysis failed: {str(e)}"

    # ========================================================================
    # LAND COVER ANALYSIS TOOL
    # ========================================================================
    @kernel_function(
        name="analyze_land_cover",
        description="Analyze land cover and land use classification. Returns land cover types, urban areas, forest cover, and agricultural land percentages."
    )
    async def analyze_land_cover(
        self,
        analysis_type: Annotated[str, "Type: 'classification' (land cover classes), 'urban' (built-up areas), 'forest' (tree cover), 'agriculture' (cropland), or 'general'"] = "general"
    ) -> str:
        """Analyze land cover classifications."""
        logger.info(f"🏘️ TOOL INVOKED: analyze_land_cover(analysis_type='{analysis_type}')")
        session = self._agent._current_session
        
        if not session or not session.loaded_collections:
            return "No land cover data loaded."
        
        try:
            results = []
            stac_items = session.stac_items if hasattr(session, 'stac_items') else []
            collections = session.loaded_collections
            
            # Land cover product mapping
            lc_products = {
                'esa-worldcover': {'name': 'ESA WorldCover 10m', 'resolution': '10m', 'classes': 11},
                'esa-cci-lc': {'name': 'ESA CCI Land Cover', 'resolution': '300m', 'classes': 37},
                'io-lulc': {'name': 'Esri Land Use/Land Cover', 'resolution': '10m', 'classes': 9},
                'io-lulc-9-class': {'name': 'Esri LULC 9-Class', 'resolution': '10m', 'classes': 9},
                'io-lulc-annual-v02': {'name': 'Esri LULC Annual', 'resolution': '10m', 'classes': 9},
                'usda-cdl': {'name': 'USDA Cropland Data Layer', 'resolution': '30m', 'classes': 130},
                'nrcan-landcover': {'name': 'Canada Land Cover', 'resolution': '30m', 'classes': 15},
                'drcog-lulc': {'name': 'Denver Regional Land Use', 'resolution': '1m', 'classes': 8},
                'chesapeake-lc-7': {'name': 'Chesapeake Land Cover 7-Class', 'resolution': '1m', 'classes': 7},
                'chesapeake-lc-13': {'name': 'Chesapeake Land Cover 13-Class', 'resolution': '1m', 'classes': 13},
            }
            
            # Find matching land cover items
            lc_items = []
            for item in stac_items:
                coll = item.get('collection', '').lower()
                if any(lp in coll for lp in lc_products.keys()):
                    lc_items.append(item)
            
            if lc_items:
                results.append("**🏘️ Land Cover Analysis Results:**\n")
                
                for item in lc_items[:3]:
                    coll = item.get('collection', '').lower()
                    props = item.get('properties', {})
                    
                    product_info = None
                    for key, info in lc_products.items():
                        if key in coll:
                            product_info = info
                            break
                    
                    if product_info:
                        results.append(f"**{product_info['name']}:**")
                        results.append(f"- Item: {item.get('id')}")
                        if props.get('datetime'):
                            results.append(f"- Date: {props['datetime'][:10]}")
                        results.append(f"- Resolution: {product_info['resolution']}")
                        results.append(f"- Number of Classes: {product_info['classes']}")
                        
                        # Provide common class descriptions
                        results.append(f"\n**Common Land Cover Classes:**")
                        results.append(f"  - Water: Lakes, rivers, oceans")
                        results.append(f"  - Trees/Forest: Woody vegetation")
                        results.append(f"  - Grassland: Herbaceous vegetation")
                        results.append(f"  - Cropland: Agricultural areas")
                        results.append(f"  - Built-up: Urban, buildings, roads")
                        results.append(f"  - Bare/Sparse: Desert, rock, sand")
                        results.append(f"  - Wetlands: Marshes, swamps")
                        results.append("")
            else:
                lc_collections = [c for c in collections if any(lp in c.lower() for lp in lc_products.keys()) or 'landcover' in c.lower() or 'lulc' in c.lower()]
                if lc_collections:
                    results.append(f"**Land Cover Collections Loaded:** {lc_collections}")
                else:
                    results.append("No land cover data loaded. Try loading:")
                    results.append("- esa-worldcover for global 10m land cover")
                    results.append("- usda-cdl for US cropland classification")
                    results.append("- io-lulc for Esri land use/land cover")
            
            return "\n".join(results) if results else "No land cover analysis available."
            
        except Exception as e:
            logger.error(f"❌ analyze_land_cover failed: {e}")
            return f"Land cover analysis failed: {str(e)}"

    # ========================================================================
    # SNOW/ICE ANALYSIS TOOL
    # ========================================================================
    @kernel_function(
        name="analyze_snow",
        description="Analyze snow and ice cover. USE THIS for snow, ice, glaciers, winter conditions, snow cover percentage, MODIS snow products. Keywords: snow, ice, glacier, winter, frozen, frost, snowpack, snowfall, snow cover, snow extent, cold regions, mountain snow, ice sheet, permafrost."
    )
    async def analyze_snow(
        self,
        analysis_type: Annotated[str, "Type: 'cover' (snow cover %), 'extent' (snow boundary), 'albedo' (reflectance), or 'general'"] = "general"
    ) -> str:
        """Analyze snow and ice from MODIS products."""
        logger.info(f"❄️ TOOL INVOKED: analyze_snow(analysis_type='{analysis_type}')")
        session = self._agent._current_session
        
        if not session or not session.loaded_collections:
            return "No snow data loaded."
        
        try:
            results = []
            stac_items = session.stac_items if hasattr(session, 'stac_items') else []
            collections = session.loaded_collections
            
            # MODIS snow product mapping
            snow_products = {
                'modis-10a1-061': {'name': 'MODIS Snow Cover Daily (500m)', 'temporal': 'daily'},
                'modis-10a2-061': {'name': 'MODIS Snow Cover 8-Day (500m)', 'temporal': '8-day'},
            }
            
            # Find matching snow items
            snow_items = []
            for item in stac_items:
                coll = item.get('collection', '').lower()
                if any(sp in coll for sp in snow_products.keys()):
                    snow_items.append(item)
            
            if snow_items:
                results.append("**❄️ Snow/Ice Analysis Results:**\n")
                
                for item in snow_items[:3]:
                    coll = item.get('collection', '').lower()
                    props = item.get('properties', {})
                    
                    product_info = None
                    for key, info in snow_products.items():
                        if key in coll:
                            product_info = info
                            break
                    
                    if product_info:
                        results.append(f"**{product_info['name']}:**")
                        results.append(f"- Item: {item.get('id')}")
                        if props.get('datetime'):
                            results.append(f"- Date: {props['datetime'][:10]}")
                        results.append(f"- Temporal Resolution: {product_info['temporal']}")
                        
                        results.append(f"\n**Snow Cover Values:**")
                        results.append(f"  - NDSI Snow Cover: 0-100 (% snow)")
                        results.append(f"  - Snow Albedo: Surface reflectance")
                        results.append(f"  - Snow quality flags included")
                        
                        results.append(f"\n**Interpretation:**")
                        results.append(f"  - 0-10%: Snow-free or trace")
                        results.append(f"  - 10-50%: Partial snow cover")
                        results.append(f"  - 50-100%: Significant to complete coverage")
                        results.append("")
            else:
                snow_collections = [c for c in collections if any(sp in c.lower() for sp in snow_products.keys()) or 'snow' in c.lower() or 'ice' in c.lower()]
                if snow_collections:
                    results.append(f"**Snow Collections Loaded:** {snow_collections}")
                else:
                    results.append("No snow data loaded. Try loading:")
                    results.append("- modis-10a1-061 for daily snow cover")
                    results.append("- modis-10a2-061 for 8-day snow cover")
            
            return "\n".join(results) if results else "No snow analysis available."
            
        except Exception as e:
            logger.error(f"❌ analyze_snow failed: {e}")
            return f"Snow analysis failed: {str(e)}"

    # ========================================================================
    # SAR ANALYSIS TOOL
    # ========================================================================
    @kernel_function(
        name="analyze_sar",
        description="Analyze Synthetic Aperture Radar (SAR) data. USE THIS for Sentinel-1, Sentinel-1-rtc, ALOS PALSAR, or any radar imagery. Keywords: SAR, radar, backscatter, VV, VH, HH, HV, polarization, Sentinel-1, ALOS, PALSAR, RTC, flood detection, urban detection, surface monitoring, through clouds, all-weather."
    )
    async def analyze_sar(
        self,
        analysis_type: Annotated[str, "Type: 'backscatter' (radar intensity), 'flood' (water detection), 'change' (temporal change), or 'general'"] = "general"
    ) -> str:
        """Analyze SAR data from Sentinel-1."""
        logger.info(f"📡 TOOL INVOKED: analyze_sar(analysis_type='{analysis_type}')")
        session = self._agent._current_session
        
        if not session or not session.loaded_collections:
            return "No SAR data loaded."
        
        try:
            results = []
            stac_items = session.stac_items if hasattr(session, 'stac_items') else []
            collections = session.loaded_collections
            
            # SAR product mapping
            sar_products = {
                'sentinel-1-grd': {'name': 'Sentinel-1 GRD (Ground Range Detected)', 'type': 'amplitude'},
                'sentinel-1-rtc': {'name': 'Sentinel-1 RTC (Radiometric Terrain Corrected)', 'type': 'calibrated'},
                'alos-palsar-mosaic': {'name': 'ALOS PALSAR Annual Mosaic', 'type': 'L-band'},
            }
            
            # Find matching SAR items
            sar_items = []
            for item in stac_items:
                coll = item.get('collection', '').lower()
                if any(sp in coll for sp in sar_products.keys()):
                    sar_items.append(item)
            
            if sar_items:
                results.append("**📡 SAR Analysis Results:**\n")
                
                for item in sar_items[:3]:
                    coll = item.get('collection', '').lower()
                    props = item.get('properties', {})
                    assets = item.get('assets', {})
                    
                    product_info = None
                    for key, info in sar_products.items():
                        if key in coll:
                            product_info = info
                            break
                    
                    if product_info:
                        results.append(f"**{product_info['name']}:**")
                        results.append(f"- Item: {item.get('id')}")
                        if props.get('datetime'):
                            results.append(f"- Date: {props['datetime'][:10]}")
                        results.append(f"- Product Type: {product_info['type']}")
                        
                        # Check polarization
                        polarizations = []
                        if 'vv' in assets or 'VV' in assets:
                            polarizations.append('VV')
                        if 'vh' in assets or 'VH' in assets:
                            polarizations.append('VH')
                        if polarizations:
                            results.append(f"- Polarizations: {', '.join(polarizations)}")
                        
                        results.append(f"\n**SAR Interpretation:**")
                        results.append(f"  - VV: Sensitive to surface roughness, bare soil")
                        results.append(f"  - VH: Sensitive to volume scattering, vegetation")
                        results.append(f"  - Dark areas: Smooth surfaces, water (specular reflection)")
                        results.append(f"  - Bright areas: Rough surfaces, urban, forests")
                        
                        results.append(f"\n**Applications:**")
                        results.append(f"  - Flood mapping (water appears dark)")
                        results.append(f"  - Ship detection (bright targets on dark sea)")
                        results.append(f"  - Deforestation monitoring")
                        results.append(f"  - Works through clouds!")
                        results.append("")
            else:
                sar_collections = [c for c in collections if any(sp in c.lower() for sp in sar_products.keys()) or 'sar' in c.lower() or 'sentinel-1' in c.lower()]
                if sar_collections:
                    results.append(f"**SAR Collections Loaded:** {sar_collections}")
                else:
                    results.append("No SAR data loaded. Try loading:")
                    results.append("- sentinel-1-grd for Sentinel-1 radar")
                    results.append("- sentinel-1-rtc for terrain-corrected radar")
            
            return "\n".join(results) if results else "No SAR analysis available."
            
        except Exception as e:
            logger.error(f"❌ analyze_sar failed: {e}")
            return f"SAR analysis failed: {str(e)}"

    # ========================================================================
    # WATER ANALYSIS TOOL
    # ========================================================================
    @kernel_function(
        name="analyze_water",
        description="Analyze surface water and flooding. USE THIS for water bodies, floods, lakes, rivers, wetlands, coastal areas, flood extent, water occurrence, inundation. Works with JRC GSW water datasets AND SAR data (Sentinel-1) for flood detection. Keywords: water, flood, flooding, lake, river, reservoir, wetland, ocean, sea, coastal, inundation, water body, surface water, JRC, water extent, SAR flood."
    )
    async def analyze_water(
        self,
        analysis_type: Annotated[str, "Type: 'occurrence' (water frequency), 'seasonality' (seasonal patterns), 'change' (water extent change), 'flood' (SAR flood detection), or 'general'"] = "general"
    ) -> str:
        """Analyze surface water from JRC GSW or SAR data."""
        logger.info(f"💧 TOOL INVOKED: analyze_water(analysis_type='{analysis_type}')")
        session = self._agent._current_session
        
        if not session or not session.loaded_collections:
            return "No water data loaded."
        
        try:
            results = []
            stac_items = session.stac_items if hasattr(session, 'stac_items') else []
            collections = session.loaded_collections
            
            # Water product mapping
            water_products = {
                'jrc-gsw': {'name': 'JRC Global Surface Water', 'resolution': '30m', 'period': '1984-present'},
            }
            
            # SAR products that can detect water/flooding
            sar_products = {
                'sentinel-1-grd': {'name': 'Sentinel-1 GRD', 'type': 'radar'},
                'sentinel-1-rtc': {'name': 'Sentinel-1 RTC', 'type': 'radar'},
            }
            
            # Find matching water items
            water_items = []
            sar_items = []
            for item in stac_items:
                coll = item.get('collection', '').lower()
                if any(wp in coll for wp in water_products.keys()) or 'water' in coll:
                    water_items.append(item)
                elif any(sp in coll for sp in sar_products.keys()):
                    sar_items.append(item)
            
            if water_items:
                results.append("**💧 Surface Water Analysis Results:**\n")
                
                for item in water_items[:3]:
                    coll = item.get('collection', '').lower()
                    props = item.get('properties', {})
                    
                    product_info = water_products.get('jrc-gsw', {'name': 'Water Dataset', 'resolution': 'varies', 'period': 'varies'})
                    
                    results.append(f"**{product_info['name']}:**")
                    results.append(f"- Item: {item.get('id')}")
                    results.append(f"- Resolution: {product_info['resolution']}")
                    results.append(f"- Time Period: {product_info['period']}")
                    
                    results.append(f"\n**Water Occurrence (0-100%):**")
                    results.append(f"  - 0%: Never water (permanent land)")
                    results.append(f"  - 1-25%: Rare flooding")
                    results.append(f"  - 25-75%: Seasonal water")
                    results.append(f"  - 75-100%: Permanent water body")
                    
                    results.append(f"\n**Seasonality Classes:**")
                    results.append(f"  - Permanent: Year-round water")
                    results.append(f"  - Seasonal: Wet season only")
                    results.append(f"  - Ephemeral: Brief flooding events")
                    
                    results.append(f"\n**Change Detection:**")
                    results.append(f"  - New Permanent: Lakes, reservoirs filled")
                    results.append(f"  - Lost Permanent: Dried lakes, drained areas")
                    results.append(f"  - New Seasonal: Increased flooding")
                    results.append("")
            elif sar_items:
                # SAR data can be used for water/flood detection
                results.append("**💧 SAR-Based Water/Flood Analysis:**\n")
                
                for item in sar_items[:3]:
                    coll = item.get('collection', '').lower()
                    props = item.get('properties', {})
                    assets = item.get('assets', {})
                    
                    product_info = None
                    for key, info in sar_products.items():
                        if key in coll:
                            product_info = info
                            break
                    
                    if product_info:
                        results.append(f"**{product_info['name']} - Water Detection:**")
                        results.append(f"- Item: {item.get('id')}")
                        if props.get('datetime'):
                            results.append(f"- Date: {props['datetime'][:10]}")
                        
                        # Check polarization
                        polarizations = []
                        if 'vv' in assets or 'VV' in assets:
                            polarizations.append('VV')
                        if 'vh' in assets or 'VH' in assets:
                            polarizations.append('VH')
                        if polarizations:
                            results.append(f"- Polarizations: {', '.join(polarizations)}")
                        
                        results.append(f"\n**Water Detection in SAR:**")
                        results.append(f"  - **Dark areas = Water**: Smooth water surfaces cause specular reflection")
                        results.append(f"  - Water appears very dark (low backscatter) in SAR imagery")
                        results.append(f"  - VV polarization: Best for calm water detection")
                        results.append(f"  - VH polarization: Better for distinguishing flooded vegetation")
                        
                        results.append(f"\n**Flood Mapping Interpretation:**")
                        results.append(f"  - Compare pre/post event imagery for flood extent")
                        results.append(f"  - Flooded areas show sudden decrease in backscatter")
                        results.append(f"  - Urban flooding may show mixed signals (buildings + water)")
                        results.append(f"  - Works through clouds - ideal for flood emergencies!")
                        
                        results.append(f"\n**Limitations:**")
                        results.append(f"  - Wind roughens water surface, increasing backscatter")
                        results.append(f"  - Shadows in hilly terrain may appear dark like water")
                        results.append(f"  - Smooth roads/runways can also appear dark")
                        results.append("")
            else:
                water_collections = [c for c in collections if 'water' in c.lower() or 'jrc' in c.lower()]
                sar_collections = [c for c in collections if 'sentinel-1' in c.lower() or 'sar' in c.lower()]
                if water_collections:
                    results.append(f"**Water Collections Loaded:** {water_collections}")
                elif sar_collections:
                    results.append(f"**SAR Collections (can detect water):** {sar_collections}")
                else:
                    results.append("No water data loaded. Try loading:")
                    results.append("- jrc-gsw for global surface water mapping")
                    results.append("- sentinel-1-rtc for SAR-based flood detection")
            
            return "\n".join(results) if results else "No water analysis available."
            
        except Exception as e:
            logger.error(f"❌ analyze_water failed: {e}")
            return f"Water analysis failed: {str(e)}"

    # ========================================================================
    # BIOMASS ANALYSIS TOOL
    # ========================================================================
    @kernel_function(
        name="analyze_biomass",
        description="Analyze above-ground biomass from CHLORIS dataset. Returns biomass estimates in tonnes per hectare."
    )
    async def analyze_biomass(
        self,
        analysis_type: Annotated[str, "Type: 'carbon' (carbon stock), 'density' (biomass density), or 'general'"] = "general"
    ) -> str:
        """Analyze biomass from CHLORIS dataset."""
        logger.info(f"🌳 TOOL INVOKED: analyze_biomass(analysis_type='{analysis_type}')")
        session = self._agent._current_session
        
        if not session or not session.loaded_collections:
            return "No biomass data loaded."
        
        try:
            results = []
            stac_items = session.stac_items if hasattr(session, 'stac_items') else []
            collections = session.loaded_collections
            
            # Biomass product mapping
            biomass_products = {
                'chloris-biomass': {'name': 'CHLORIS Above-Ground Biomass', 'resolution': '100m', 'unit': 'Mg/ha'},
                'hgb': {'name': 'Harmonized Global Biomass', 'resolution': 'varies', 'unit': 'Mg/ha'},
            }
            
            # Find matching biomass items
            biomass_items = []
            for item in stac_items:
                coll = item.get('collection', '').lower()
                if any(bp in coll for bp in biomass_products.keys()) or 'biomass' in coll:
                    biomass_items.append(item)
            
            if biomass_items:
                results.append("**🌳 Biomass Analysis Results:**\n")
                
                for item in biomass_items[:3]:
                    coll = item.get('collection', '').lower()
                    props = item.get('properties', {})
                    
                    product_info = None
                    for key, info in biomass_products.items():
                        if key in coll:
                            product_info = info
                            break
                    if not product_info:
                        product_info = {'name': 'Biomass Dataset', 'resolution': 'varies', 'unit': 'Mg/ha'}
                    
                    results.append(f"**{product_info['name']}:**")
                    results.append(f"- Item: {item.get('id')}")
                    if props.get('datetime'):
                        results.append(f"- Date: {props['datetime'][:10]}")
                    results.append(f"- Resolution: {product_info['resolution']}")
                    results.append(f"- Unit: {product_info['unit']} (tonnes per hectare)")
                    
                    results.append(f"\n**Biomass Interpretation:**")
                    results.append(f"  - 0-50 Mg/ha: Grassland, sparse vegetation")
                    results.append(f"  - 50-150 Mg/ha: Woodland, open forest")
                    results.append(f"  - 150-300 Mg/ha: Dense forest")
                    results.append(f"  - 300+ Mg/ha: Tropical rainforest")
                    
                    results.append(f"\n**Carbon Estimation:**")
                    results.append(f"  - Carbon ≈ Biomass × 0.47")
                    results.append(f"  - Critical for climate monitoring")
                    results.append(f"  - Supports REDD+ initiatives")
                    results.append("")
            else:
                biomass_collections = [c for c in collections if 'biomass' in c.lower() or 'chloris' in c.lower() or 'hgb' in c.lower()]
                if biomass_collections:
                    results.append(f"**Biomass Collections Loaded:** {biomass_collections}")
                else:
                    results.append("No biomass data loaded. Try loading:")
                    results.append("- chloris-biomass for above-ground biomass")
            
            return "\n".join(results) if results else "No biomass analysis available."
            
        except Exception as e:
            logger.error(f"❌ analyze_biomass failed: {e}")
            return f"Biomass analysis failed: {str(e)}"

    # ========================================================================
    # RASTER POINT SAMPLING TOOL
    # ========================================================================
    @kernel_function(
        name="sample_raster_value",
        description="Extract the actual pixel/raster value from loaded satellite data at a specific location. Returns the numeric value (e.g., SST in Celsius, elevation in meters, NDVI, reflectance, band values) at the pin/center coordinates. Use this when the user asks for: the raster value, pixel value, band value, NDVI value, temperature, elevation, reflectance, or any numeric data extraction at a point. Works with HLS, Sentinel-2, Landsat, DEM, SST, MODIS, and other raster datasets."
    )
    async def sample_raster_value(
        self,
        data_type: Annotated[str, "Type of data to sample: 'sst', 'temperature', 'elevation', 'ndvi', or 'auto' to detect from loaded collections"] = "auto"
    ) -> str:
        """Sample actual pixel values from COG rasters at the session's pin/center location."""
        logger.info(f"📍 TOOL INVOKED: sample_raster_value(data_type='{data_type}')")
        session = self._agent._current_session
        logger.info(f"📍 TOOL SESSION CHECK: session={session is not None}, id={session.session_id if session else 'N/A'}, stac_items={len(session.stac_items) if session else 'N/A'}")
        
        if not session:
            return "No active session. Please load some data first."
        
        # Get sampling coordinates
        bounds = session.map_bounds if hasattr(session, 'map_bounds') else None
        if not bounds:
            return "No location available. Please set a pin or center the map on a location."
        
        # Get coordinates - prefer pin location, fall back to center
        lat = bounds.get('pin_lat') or bounds.get('center_lat')
        lng = bounds.get('pin_lng') or bounds.get('center_lng')
        
        if lat is None or lng is None:
            return "No coordinates available. Please pin a location on the map."
        
        stac_items = session.stac_items if hasattr(session, 'stac_items') else []
        tile_urls = session.tile_urls if hasattr(session, 'tile_urls') else []
        collections = session.loaded_collections if hasattr(session, 'loaded_collections') else []
        
        logger.info(f"📊 sample_raster_value: stac_items={len(stac_items)}, tile_urls={len(tile_urls)}, collections={collections}")
        
        # Log STAC item details for debugging
        for i, item in enumerate(stac_items[:2]):
            coll = item.get('collection', 'unknown')
            item_id = item.get('id', 'unknown')
            assets = list(item.get('assets', {}).keys())[:5]
            logger.info(f"📊 STAC item {i}: collection={coll}, id={item_id}, assets={assets}")
        
        # If no STAC items but we have tile_urls, fetch STAC items from tile URLs
        if not stac_items and tile_urls:
            logger.info(f"📦 No STAC items, attempting to fetch from {len(tile_urls)} tile URLs...")
            for tile_url in tile_urls[:5]:  # Check first 5 tile URLs
                try:
                    parsed = parse_tile_url(tile_url)
                    if parsed.get('collection') and parsed.get('item'):
                        logger.info(f"📦 Fetching STAC item: {parsed['collection']}/{parsed['item']}")
                        item = await fetch_stac_item(parsed['collection'], parsed['item'])
                        if item:
                            stac_items.append(item)
                            logger.info(f"✅ Fetched STAC item with {len(item.get('assets', {}))} assets")
                            break  # Got one, that's enough
                except Exception as e:
                    logger.warning(f"Failed to parse/fetch tile URL: {e}")
        
        if not stac_items:
            # Fallback: try to construct a sample from collection info
            if collections and lat and lng:
                # For SST, we can try a direct approach using known asset patterns
                for coll in collections:
                    if 'temperature' in coll.lower() or 'sst' in coll.lower():
                        return f"""**Point Sampling at ({lat:.4f}°, {lng:.4f}°):**

The loaded collection is **{coll}** (Sea Surface Temperature data).

Unfortunately, I couldn't access the raw raster data to sample the exact value. The data appears to be in the 0-35°C range based on the colormap.

To get the exact temperature value:
1. The data uses the **rdylbu_r** colormap (blue=cold, red=hot)
2. Looking at the map, estimate the temperature based on the color gradient
3. Blue tones indicate cooler waters (~0-15°C)
4. White/yellow tones indicate moderate temperatures (~15-25°C)  
5. Red/orange tones indicate warmer waters (~25-35°C)

For programmatic access, you can query the STAC API directly at:
`https://planetarycomputer.microsoft.com/api/stac/v1/collections/{coll}`"""
            
            return f"No STAC items available to sample. Available collections: {collections}. Please try loading the data first."
        
        try:
            results = []
            results.append(f"**Point Sampling at ({lat:.4f}°, {lng:.4f}°):**\n")
            
            # Determine which collection to sample based on data_type
            target_items = []
            asset_keys = []  # Which asset to sample
            value_transforms = []  # How to transform/interpret the value
            
            if data_type in ['sst', 'temperature', 'auto']:
                # Find SST items
                sst_keywords = ['sea-surface-temperature', 'sst', 'temperature-whoi', 'noaa-cdr']
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    if any(kw in coll for kw in sst_keywords) or 'temperature' in coll:
                        target_items.append(item)
                        asset_keys.append('sea_surface_temperature')  # Primary SST asset
                        # SST data is in Celsius (per PC documentation)
                        value_transforms.append({
                            'name': 'Sea Surface Temperature',
                            'unit_raw': '°C',
                            'unit_display': '°C',
                            'transform': lambda v: v,  # Already in Celsius
                            'valid_range': (-2, 40)  # Valid Celsius range for SST
                        })
            
            if data_type in ['elevation', 'height', 'lidar', 'hag', '3dep', 'auto'] and not target_items:
                # Find DEM / elevation items (includes 3DEP LiDAR)
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    if 'dem' in coll or 'elevation' in coll or 'cop-dem' in coll or '3dep' in coll or 'lidar' in coll or 'hag' in coll:
                        target_items.append(item)
                        assets = item.get('assets', {})
                        asset_keys.append('data' if 'data' in assets else list(assets.keys())[0] if assets else 'data')
                        # Determine name based on collection type
                        name = 'Height Above Ground' if ('3dep' in coll or 'lidar' in coll or 'hag' in coll) else 'Elevation'
                        value_transforms.append({
                            'name': name,
                            'unit_raw': 'm',
                            'unit_display': 'm',
                            'transform': lambda v: v,
                            'valid_range': (-500, 9000)
                        })
            
            if data_type in ['ndvi', 'auto'] and not target_items:
                # Find optical imagery for NDVI calculation
                optical_keywords = ['sentinel-2', 'landsat', 'hls', 's30', 'l30']
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    if any(kw in coll for kw in optical_keywords):
                        assets = item.get('assets', {})
                        # Check if we have both RED and NIR bands
                        red_key = 'B04' if 'B04' in assets else ('red' if 'red' in assets else None)
                        # NIR bands vary by platform:
                        # - HLS2-S30 (Sentinel): B08 (NIR Broad) or B8A (NIR Narrow)
                        # - HLS2-L30 (Landsat): B05 (NIR Narrow)
                        # - Sentinel-2: B08, nir08
                        nir_key = None
                        for candidate in ['B08', 'B8A', 'B05', 'nir08', 'nir']:
                            if candidate in assets:
                                nir_key = candidate
                                break
                        
                        if red_key and nir_key:
                            logger.info(f"📊 HLS NDVI: Found red={red_key}, nir={nir_key} in collection {coll}")
                            # Store both band keys for NDVI calculation
                            target_items.append(item)
                            asset_keys.append((red_key, nir_key))  # Tuple for NDVI
                            value_transforms.append({
                                'name': 'NDVI',
                                'unit_raw': 'index',
                                'unit_display': '',
                                'is_ndvi': True,  # Special flag for NDVI calculation
                                'valid_range': (-1, 1)
                            })
                        else:
                            # Fallback to just red band reflectance
                            target_items.append(item)
                            asset_keys.append('B04' if 'B04' in assets else 'red')
                            value_transforms.append({
                                'name': 'Red Band Reflectance',
                                'unit_raw': 'reflectance',
                                'unit_display': '',
                                'transform': lambda v: v / 10000 if v and v > 100 else v,
                                'valid_range': (0, 10000)
                            })
            
            # ✅ MTBS Burn Severity
            if data_type in ['burn', 'severity', 'mtbs', 'auto'] and not target_items:
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    if 'mtbs' in coll or 'burn' in coll:
                        assets = item.get('assets', {})
                        target_items.append(item)
                        asset_keys.append('burn-severity' if 'burn-severity' in assets else 'data')
                        value_transforms.append({
                            'name': 'Burn Severity Class',
                            'unit_raw': 'class',
                            'unit_display': '',
                            'transform': lambda v: v,
                            'valid_range': (0, 6),
                            'class_labels': {
                                1: 'Unburned to Low',
                                2: 'Low',
                                3: 'Moderate', 
                                4: 'High',
                                5: 'Increased Greenness (post-fire)',
                                6: 'Non-Processing Area'
                            }
                        })
            
            # ✅ MODIS Fire / Thermal Anomalies
            if data_type in ['fire', 'thermal', 'modis-fire', 'auto'] and not target_items:
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    if 'modis-14' in coll or 'fire' in coll:
                        assets = item.get('assets', {})
                        target_items.append(item)
                        asset_keys.append('FireMask' if 'FireMask' in assets else ('MaxFRP' if 'MaxFRP' in assets else 'data'))
                        value_transforms.append({
                            'name': 'Fire Detection',
                            'unit_raw': 'class',
                            'unit_display': '',
                            'transform': lambda v: v,
                            'valid_range': (0, 9),
                            'class_labels': {
                                0: 'Not processed',
                                1: 'Not processed',
                                2: 'Not processed',
                                3: 'Non-fire water',
                                4: 'Cloud',
                                5: 'Non-fire land',
                                6: 'Unknown',
                                7: 'Low confidence fire',
                                8: 'Nominal confidence fire',
                                9: 'High confidence fire'
                            }
                        })
            
            # ✅ JRC Global Surface Water
            if data_type in ['water', 'occurrence', 'jrc', 'gsw', 'auto'] and not target_items:
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    assets = item.get('assets', {})
                    # Check for JRC-GSW by collection name OR by having occurrence/extent assets
                    is_jrc_data = 'jrc' in coll or 'gsw' in coll or 'surface-water' in coll
                    has_water_assets = 'occurrence' in assets or 'extent' in assets or 'change' in assets or 'seasonality' in assets
                    
                    if is_jrc_data or has_water_assets:
                        # Prefer occurrence, then extent, then any available
                        asset_key = 'occurrence' if 'occurrence' in assets else ('extent' if 'extent' in assets else ('seasonality' if 'seasonality' in assets else 'data'))
                        target_items.append(item)
                        asset_keys.append(asset_key)
                        logger.info(f"📍 Found JRC-GSW item: {item.get('id', 'unknown')}, using asset: {asset_key}")
                        value_transforms.append({
                            'name': 'Water Occurrence',
                            'unit_raw': '%',
                            'unit_display': '%',
                            'transform': lambda v: v,  # Already in percentage (0-100)
                            'valid_range': (0, 100),
                            'interpretation': {
                                (0, 0): 'Never water (0%)',
                                (1, 25): 'Rarely water (1-25%)',
                                (26, 50): 'Sometimes water (26-50%)',
                                (51, 75): 'Often water (51-75%)',
                                (76, 99): 'Usually water (76-99%)',
                                (100, 100): 'Permanent water (100%)'
                            }
                        })
            
            # ✅ MODIS Snow Cover
            if data_type in ['snow', 'ice', 'modis-snow', 'auto'] and not target_items:
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    if 'modis-10' in coll or 'snow' in coll:
                        assets = item.get('assets', {})
                        target_items.append(item)
                        asset_keys.append('NDSI_Snow_Cover' if 'NDSI_Snow_Cover' in assets else 'data')
                        value_transforms.append({
                            'name': 'Snow Cover',
                            'unit_raw': '%',
                            'unit_display': '%',
                            'transform': lambda v: v,
                            'valid_range': (0, 100)
                        })
            
            # ✅ Land Cover / Cropland Data Layer
            if data_type in ['landcover', 'cdl', 'crop', 'auto'] and not target_items:
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    if 'cdl' in coll or 'cropland' in coll or 'usda' in coll or 'land-cover' in coll:
                        assets = item.get('assets', {})
                        target_items.append(item)
                        asset_keys.append('data' if 'data' in assets else list(assets.keys())[0])
                        value_transforms.append({
                            'name': 'Land Cover Class',
                            'unit_raw': 'class',
                            'unit_display': '',
                            'transform': lambda v: v,
                            'valid_range': (0, 255),
                            'is_classification': True
                        })
            
            # ✅ Biomass / Carbon
            if data_type in ['biomass', 'carbon', 'agb', 'auto'] and not target_items:
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    if 'biomass' in coll or 'chloris' in coll or 'carbon' in coll:
                        assets = item.get('assets', {})
                        target_items.append(item)
                        # Check for biomass_wm (web mercator) first, then biomass, then fallbacks
                        if 'biomass_wm' in assets:
                            asset_key = 'biomass_wm'
                        elif 'biomass' in assets:
                            asset_key = 'biomass'
                        elif 'aboveground' in assets:
                            asset_key = 'aboveground'
                        elif 'agb' in assets:
                            asset_key = 'agb'
                        else:
                            asset_key = 'data'
                        asset_keys.append(asset_key)
                        # CHLORIS data is in TOTAL TONNES per pixel (not per hectare)
                        # At 4.6km resolution, each pixel covers ~2116 hectares
                        # Divide by pixel area in hectares to get Mg/ha
                        # GSD = 4633m, so pixel area = 4633 * 4633 / 10000 = 2146 ha
                        pixel_area_ha = 2146  # hectares per pixel
                        value_transforms.append({
                            'name': 'Above-Ground Biomass',
                            'unit_raw': 'tonnes (total per pixel)',
                            'unit_display': 'Mg/ha (tonnes per hectare)',
                            'transform': lambda v, area=pixel_area_ha: round(v / area, 1) if v else v,
                            'valid_range': (0, 500)
                        })
            
            # ✅ MODIS Vegetation (NDVI, LAI, GPP, NPP)
            if data_type in ['vegetation', 'lai', 'gpp', 'npp', 'auto'] and not target_items:
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    if 'modis-13' in coll or 'modis-15' in coll or 'modis-17' in coll:
                        assets = item.get('assets', {})
                        # Determine which product and asset
                        if '250m_16_days_NDVI' in assets:
                            asset_key = '250m_16_days_NDVI'
                            name = 'NDVI'
                            transform = lambda v: v * 0.0001 if v else v  # Scale factor
                            valid_range = (-2000, 10000)
                        elif 'Lai_500m' in assets:
                            asset_key = 'Lai_500m'
                            name = 'Leaf Area Index'
                            transform = lambda v: v * 0.1 if v else v
                            valid_range = (0, 100)
                        elif 'Gpp_500m' in assets:
                            asset_key = 'Gpp_500m'
                            name = 'Gross Primary Productivity'
                            transform = lambda v: v * 0.0001 if v else v
                            valid_range = (0, 65500)
                        elif 'Npp_500m' in assets:
                            asset_key = 'Npp_500m'
                            name = 'Net Primary Productivity'
                            transform = lambda v: v * 0.0001 if v else v
                            valid_range = (0, 65500)
                        else:
                            asset_key = list(assets.keys())[0] if assets else 'data'
                            name = 'MODIS Vegetation'
                            transform = lambda v: v
                            valid_range = None
                        
                        target_items.append(item)
                        asset_keys.append(asset_key)
                        value_transforms.append({
                            'name': name,
                            'unit_raw': 'scaled',
                            'unit_display': '',
                            'transform': transform,
                            'valid_range': valid_range
                        })
            
            # ✅ SAR / Radar backscatter (Sentinel-1, ALOS PALSAR)
            # SAR/Radar data - also handle 'water' when SAR collections are loaded (for flood detection)
            if data_type in ['sar', 'radar', 'backscatter', 'water', 'auto'] and not target_items:
                logger.info(f"🛰️ SAR detection: checking {len(stac_items)} items for SAR data (data_type={data_type})")
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    item_id = item.get('id', 'unknown')[:30]
                    assets = item.get('assets', {})
                    asset_keys_list = list(assets.keys())
                    is_sar_collection = 'sentinel-1' in coll or 'alos-palsar' in coll or 'sar' in coll or 'rtc' in coll
                    logger.info(f"🛰️ Item: {item_id}, collection: {coll}, is_sar: {is_sar_collection}, assets: {asset_keys_list}")
                    if is_sar_collection:
                        # VV and VH are common SAR polarization bands
                        for sar_key in ['vv', 'vh', 'hh', 'hv', 'VV', 'VH', 'HH', 'HV']:
                            if sar_key in assets:
                                target_items.append(item)
                                asset_keys.append(sar_key)
                                value_transforms.append({
                                    'name': f'SAR Backscatter ({sar_key.upper()})',
                                    'unit_raw': 'linear',
                                    'unit_display': 'dB',
                                    'transform': lambda v: 10 * __import__('math').log10(v) if v and v > 0 else v,  # Convert linear to dB
                                    'valid_range': (-30, 10)  # Typical dB range
                                })
                                logger.info(f"✅ Found SAR asset: {sar_key} in {item_id}")
                                break  # Only one band per item (don't duplicate item for VV and VH)
                        # NOTE: Do NOT break here - collect ALL SAR items for proper tile coverage
            
            # ✅ MODIS BRDF / Surface Reflectance (also HLS for surface reflectance queries)
            if data_type in ['reflectance', 'brdf', 'modis-43', 'surface', 'auto'] and not target_items:
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    assets = item.get('assets', {})
                    
                    # Handle MODIS 43A4 BRDF data
                    if 'modis-43' in coll or 'brdf' in coll:
                        for band_num in range(1, 8):
                            band_key = f'Nadir_Reflectance_Band{band_num}'
                            if band_key in assets:
                                target_items.append(item)
                                asset_keys.append(band_key)
                                value_transforms.append({
                                    'name': f'BRDF Reflectance (Band {band_num})',
                                    'unit_raw': 'scaled',
                                    'unit_display': 'reflectance',
                                    'transform': lambda v: v * 0.0001 if v else v,
                                    'valid_range': (0, 10000)
                                })
                                break  # Found asset for this item, move to next item
                        # Continue to collect ALL BRDF items for maximum tile coverage
                    
                    # Handle HLS surface reflectance (hls2-l30, hls2-s30)
                    elif 'hls' in coll or 'hls2' in coll or 's30' in coll or 'l30' in coll:
                        # HLS uses B01-B12 for Sentinel, B01-B07 for Landsat
                        # Sample a representative reflectance band - check ALL common HLS bands
                        for band_key in ['B04', 'B03', 'B02', 'B05', 'B06', 'B8A', 'B08', 'B07', 'B01', 'B11', 'B12']:
                            if band_key in assets:
                                target_items.append(item)
                                asset_keys.append(band_key)
                                value_transforms.append({
                                    'name': f'Surface Reflectance ({band_key})',
                                    'unit_raw': 'scaled',
                                    'unit_display': 'reflectance',
                                    'transform': lambda v: v * 0.0001 if v else v,  # HLS scale factor
                                    'valid_range': (0, 10000)
                                })
                                break
                        # Also try lowercase band names for this item if uppercase didn't match
                        if item not in target_items:  # Only if we didn't already add this item
                            for band_key in ['b04', 'b03', 'b02', 'b05', 'b06', 'red', 'green', 'blue', 'nir']:
                                if band_key in assets:
                                    target_items.append(item)
                                    asset_keys.append(band_key)
                                    value_transforms.append({
                                        'name': f'Surface Reflectance ({band_key})',
                                        'unit_raw': 'scaled',
                                        'unit_display': 'reflectance',
                                        'transform': lambda v: v * 0.0001 if v else v,
                                        'valid_range': (0, 10000)
                                    })
                                    break  # Found asset for this item
                        # Continue to collect ALL HLS items for maximum tile coverage
            
            # ✅ 3DEP LiDAR Height Above Ground
            if data_type in ['lidar', 'hag', 'height', '3dep', 'auto'] and not target_items:
                for item in stac_items:
                    coll = item.get('collection', '').lower()
                    if '3dep' in coll or 'lidar' in coll or 'hag' in coll:
                        assets = item.get('assets', {})
                        target_items.append(item)
                        asset_keys.append('data' if 'data' in assets else list(assets.keys())[0])
                        value_transforms.append({
                            'name': 'Height Above Ground',
                            'unit_raw': 'm',
                            'unit_display': 'm',
                            'transform': lambda v: v,
                            'valid_range': (0, 500)  # Most structures under 500m
                        })
                        # Continue to collect ALL 3DEP/LiDAR items for maximum tile coverage
            
            # ✅ Generic fallback for any remaining collections (works for ANY data_type)
            # If specific handlers above didn't find matching assets, try generic raster detection
            if not target_items and stac_items:
                # Use the first available STAC item with any asset
                logger.info(f"📍 Generic fallback: checking {len(stac_items)} STAC items for raster data (data_type={data_type})")
                for item in stac_items:
                    assets = item.get('assets', {})
                    coll = item.get('collection', 'unknown')
                    logger.info(f"📍 Checking item: {item.get('id', 'unknown')}, collection: {coll}, assets: {list(assets.keys())}")
                    # Find any image/tiff asset - expanded list includes common STAC asset keys
                    for asset_key, asset_info in assets.items():
                        asset_type = asset_info.get('type', '') if isinstance(asset_info, dict) else ''
                        # Match by type OR by common asset key names
                        is_raster = asset_type.startswith('image/tiff') or asset_type.startswith('application/x-geotiff')
                        is_common_key = asset_key.lower() in ['data', 'visual', 'default', 'occurrence', 'extent', 'change', 'seasonality', 'recurrence', 'transitions']
                        if is_raster or is_common_key:
                            target_items.append(item)
                            asset_keys.append(asset_key)
                            value_transforms.append({
                                'name': f'{coll} - {asset_key}',
                                'unit_raw': 'raw',
                                'unit_display': '',
                                'transform': lambda v: v,
                                'valid_range': None  # Unknown range
                            })
                            break  # Found asset for this item, move to next item
                    # Continue to collect ALL raster items for maximum tile coverage
            
            if not target_items:
                return f"No {data_type} data loaded to sample. Available collections: {session.loaded_collections}"
            
            # ============================================================================
            # ============================================================================
            # TILE SELECTION: Try all available tiles, prioritizing those that contain the pin
            # NOTE: We no longer strictly filter by bbox because:
            # 1. Users may pan/zoom the map beyond the original search bbox
            # 2. TiTiler renders tiles dynamically based on viewport
            # 3. If user dropped a pin on visible imagery, data should exist there
            # We'll try tiles containing the pin first, then try others as fallback
            # ============================================================================
            def point_in_bbox(lat: float, lng: float, bbox: list) -> bool:
                """Check if point (lat, lng) is within bbox [west, south, east, north]."""
                if not bbox or len(bbox) < 4:
                    return True  # No bbox = assume it could contain the point
                west, south, east, north = bbox[0], bbox[1], bbox[2], bbox[3]
                return west <= lng <= east and south <= lat <= north
            
            # Sort items: tiles containing the pin come first, then others
            items_with_pin = []
            items_without_pin = []
            asset_keys_with_pin = []
            asset_keys_without_pin = []
            transforms_with_pin = []
            transforms_without_pin = []
            
            for item, asset_key, transform_info in zip(target_items, asset_keys, value_transforms):
                item_bbox = item.get('bbox')
                item_id = item.get('id', 'unknown')[:40]
                if point_in_bbox(lat, lng, item_bbox):
                    items_with_pin.append(item)
                    asset_keys_with_pin.append(asset_key)
                    transforms_with_pin.append(transform_info)
                    logger.info(f"✅ Tile {item_id} contains pin ({lat:.4f}, {lng:.4f}), bbox={item_bbox}")
                else:
                    items_without_pin.append(item)
                    asset_keys_without_pin.append(asset_key)
                    transforms_without_pin.append(transform_info)
                    logger.info(f"⚠️ Tile {item_id} bbox doesn't contain pin ({lat:.4f}, {lng:.4f}), will try as fallback")
            
            # Combine: tiles with pin first, then tiles without (as fallback)
            filtered_items = items_with_pin + items_without_pin
            filtered_asset_keys = asset_keys_with_pin + asset_keys_without_pin
            filtered_transforms = transforms_with_pin + transforms_without_pin
            
            if not filtered_items:
                return f"No tiles available to sample. Please load satellite data first using a STAC search query."
            
            logger.info(f"📍 Tile ordering: {len(items_with_pin)} tiles contain pin, {len(items_without_pin)} tiles as fallback")
            
            # ============================================================================
            # FALLBACK STAC SEARCH: If no tiles contain pin, search for tiles at pin location
            # This handles cases where the original AOI search didn't return tiles for the pin
            # ============================================================================
            if not items_with_pin and items_without_pin:
                logger.warning(f"⚠️ No loaded tiles contain pin ({lat:.4f}, {lng:.4f}). Attempting fallback STAC search...")
                
                # Get the collection from the first available item
                first_item = items_without_pin[0]
                collection_id = first_item.get('collection', '')
                
                if collection_id:
                    try:
                        import aiohttp
                        import planetary_computer as pc
                        
                        # Create a small bbox around the pin (0.5 degree buffer for DEM tiles which are 1°x1°)
                        buffer = 0.5
                        pin_bbox = [lng - buffer, lat - buffer, lng + buffer, lat + buffer]
                        
                        logger.info(f"🔍 Fallback STAC search: collection={collection_id}, pin_bbox={pin_bbox}")
                        
                        stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
                        search_body = {
                            "collections": [collection_id],
                            "bbox": pin_bbox,
                            "limit": 5
                        }
                        
                        async with aiohttp.ClientSession() as http_session:
                            async with http_session.post(
                                stac_url,
                                json=search_body,
                                headers={"Content-Type": "application/json"},
                                timeout=aiohttp.ClientTimeout(total=30)
                            ) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    fallback_features = result.get("features", [])
                                    logger.info(f"✅ Fallback STAC search returned {len(fallback_features)} features")
                                    
                                    # Process fallback features and find ones containing the pin
                                    for feature in fallback_features:
                                        feature_bbox = feature.get('bbox')
                                        if point_in_bbox(lat, lng, feature_bbox):
                                            # Sign the asset URLs with Planetary Computer token
                                            try:
                                                signed_feature = pc.sign(feature)
                                            except Exception as sign_err:
                                                logger.warning(f"Could not sign feature: {sign_err}")
                                                signed_feature = feature
                                            
                                            # Find appropriate asset for this collection
                                            assets = signed_feature.get('assets', {})
                                            asset_key = None
                                            transform_info = None
                                            
                                            # Detect asset and transform based on collection
                                            if 'dem' in collection_id.lower() or 'cop-dem' in collection_id.lower():
                                                asset_key = 'data'
                                                transform_info = {
                                                    'name': 'Elevation',
                                                    'unit_raw': 'm',
                                                    'unit_display': 'meters',
                                                    'transform': lambda v: v,
                                                    'valid_range': (-500, 9000)
                                                }
                                            elif asset_key is None:
                                                # Generic fallback
                                                for key in ['data', 'visual', 'default']:
                                                    if key in assets:
                                                        asset_key = key
                                                        transform_info = {
                                                            'name': f'{collection_id}',
                                                            'unit_raw': 'raw',
                                                            'unit_display': '',
                                                            'transform': lambda v: v,
                                                            'valid_range': None
                                                        }
                                                        break
                                            
                                            if asset_key and asset_key in assets:
                                                # Insert at the beginning so these tiles are tried first
                                                filtered_items.insert(0, signed_feature)
                                                filtered_asset_keys.insert(0, asset_key)
                                                filtered_transforms.insert(0, transform_info)
                                                logger.info(f"✅ Added fallback tile: {signed_feature.get('id', 'unknown')[:40]} containing pin")
                                    
                                    # Update counts for logging
                                    items_with_pin = [it for it in filtered_items if point_in_bbox(lat, lng, it.get('bbox'))]
                                    logger.info(f"📍 After fallback: {len(items_with_pin)} tiles now contain pin")
                                else:
                                    logger.warning(f"⚠️ Fallback STAC search failed: {response.status}")
                                    
                    except Exception as fallback_err:
                        logger.error(f"❌ Fallback STAC search error: {fallback_err}")
            
            # Sample each target item (now only tiles containing the pin)
            sampled_count = 0
            tiles_tried = 0
            nodata_tiles = []  # Track tiles that had nodata (cloud/ocean mask)
            max_successful_samples = 2  # Stop after 2 successful samples to avoid rate limiting
            for i, (item, asset_key, transform_info) in enumerate(zip(filtered_items[:5], filtered_asset_keys, filtered_transforms)):  # Try up to 5 tiles
                # Early exit if we have enough successful samples
                if sampled_count >= max_successful_samples:
                    logger.info(f"✅ Got {sampled_count} successful samples, stopping early to avoid rate limiting")
                    break
                
                # Add small delay between tile requests to avoid 409 rate limiting
                if i > 0:
                    await asyncio.sleep(0.5)  # 500ms delay between tiles
                
                props = item.get('properties', {})
                assets = item.get('assets', {})
                collection = item.get('collection', 'unknown')
                item_id = item.get('id', 'unknown')[:30]
                tiles_tried += 1
                
                # Check if this is an NDVI calculation (asset_key is a tuple)
                if transform_info.get('is_ndvi') and isinstance(asset_key, tuple):
                    red_key, nir_key = asset_key
                    red_url = assets.get(red_key, {}).get('href') if red_key in assets else None
                    nir_url = assets.get(nir_key, {}).get('href') if nir_key in assets else None
                    
                    if red_url and nir_url:
                        # Sample both bands at the point
                        logger.info(f"📍 Sampling NDVI at ({lat}, {lng}) from {collection} tile {i+1}...")
                        red_result = await sample_cog_at_point(red_url, lat, lng)
                        nir_result = await sample_cog_at_point(nir_url, lat, lng)
                        
                        # If point is outside bounds for this tile, try the next one
                        if red_result.get('error') and 'outside' in str(red_result.get('error', '')).lower():
                            logger.info(f"📍 Point outside tile bounds for {collection}, will try next tile...")
                            continue  # Try next STAC item
                        elif nir_result.get('error') and 'outside' in str(nir_result.get('error', '')).lower():
                            logger.info(f"📍 Point outside tile bounds for {collection}, will try next tile...")
                            continue  # Try next STAC item
                        
                        # If nodata (cloud/ocean mask), track and try next tile
                        if red_result.get('error') and ('no data' in str(red_result.get('error', '')).lower() or 'nodata' in str(red_result.get('error', '')).lower()):
                            nodata_tiles.append(f"{item_id} (cloud/ocean mask)")
                            logger.info(f"📍 NoData at location in tile {item_id}, trying next tile...")
                            continue  # Try next STAC item
                        elif nir_result.get('error') and ('no data' in str(nir_result.get('error', '')).lower() or 'nodata' in str(nir_result.get('error', '')).lower()):
                            nodata_tiles.append(f"{item_id} (cloud/ocean mask)")
                            logger.info(f"📍 NoData at location in tile {item_id}, trying next tile...")
                            continue  # Try next STAC item
                        
                        if red_result.get('error'):
                            results.append(f"**NDVI ({collection}):**")
                            results.append(f"- Error sampling RED band: {red_result['error']}")
                        elif nir_result.get('error'):
                            results.append(f"**NDVI ({collection}):**")
                            results.append(f"- Error sampling NIR band: {nir_result['error']}")
                        elif red_result.get('value') is not None and nir_result.get('value') is not None:
                            red_val = float(red_result['value'])
                            nir_val = float(nir_result['value'])
                            
                            # Calculate NDVI = (NIR - RED) / (NIR + RED)
                            if (nir_val + red_val) != 0:
                                ndvi = (nir_val - red_val) / (nir_val + red_val)
                                # Clip to valid range
                                ndvi = max(-1, min(1, ndvi))
                                
                                results.append(f"**NDVI at Pin Location ({collection}):**")
                                results.append(f"- RED band ({red_key}): {red_val:.0f}")
                                results.append(f"- NIR band ({nir_key}): {nir_val:.0f}")
                                results.append(f"- **NDVI Value: {ndvi:.3f}**")
                                
                                # Interpretation
                                if ndvi > 0.6:
                                    interp = "Dense, healthy vegetation"
                                elif ndvi > 0.4:
                                    interp = "Moderate vegetation"
                                elif ndvi > 0.2:
                                    interp = "Sparse or stressed vegetation"
                                elif ndvi > 0:
                                    interp = "Minimal vegetation, bare soil"
                                else:
                                    interp = "Water, snow, or non-vegetated surface"
                                results.append(f"- Interpretation: {interp}")
                                
                                if props.get('datetime'):
                                    results.append(f"- Date: {props['datetime'][:10]}")
                                
                                sampled_count += 1
                            else:
                                results.append(f"**NDVI ({collection}):** Division by zero (both bands = 0)")
                        else:
                            results.append(f"**NDVI ({collection}):** Could not sample band values")
                    else:
                        results.append(f"**NDVI ({collection}):** Missing RED or NIR band URLs")
                    
                    results.append("")
                    continue
                
                # Standard single-band sampling
                # Find the asset URL
                cog_url = None
                
                # Try the specified asset key first
                if asset_key in assets:
                    cog_url = assets[asset_key].get('href')
                    logger.info(f"📍 Found asset '{asset_key}' with href ending in ...{cog_url[-50:] if cog_url else 'None'}")
                else:
                    logger.info(f"📍 Asset key '{asset_key}' not found in assets: {list(assets.keys())}")
                
                # Try common fallback asset names
                if not cog_url:
                    for fallback_key in ['data', 'visual', 'default', list(assets.keys())[0] if assets else None]:
                        if fallback_key and fallback_key in assets:
                            asset_info = assets[fallback_key]
                            # Check by type, or href extension, or just use it if it's our explicitly requested key
                            if asset_info.get('type', '').startswith('image/tiff') or asset_info.get('href', '').endswith('.tif'):
                                cog_url = asset_info.get('href')
                                logger.info(f"📍 Using fallback asset '{fallback_key}'")
                                break
                
                if not cog_url:
                    results.append(f"**{collection}:** No COG asset found for sampling")
                    continue
                
                # Sample the COG
                sample_result = await sample_cog_at_point(cog_url, lat, lng)
                
                if sample_result.get('error'):
                    error_msg = str(sample_result.get('error', '')).lower()
                    
                    # If point is outside this tile's bounds, skip and try next tile
                    if 'outside' in error_msg and 'coverage' in error_msg:
                        logger.info(f"📍 Point outside tile bounds for {collection} tile {i+1}, trying next tile...")
                        continue  # Try next STAC item
                    
                    # If nodata (cloud/ocean mask), track and try next tile
                    if 'no data' in error_msg or 'nodata' in error_msg:
                        nodata_tiles.append(f"{item_id} (cloud/ocean mask)")
                        logger.info(f"📍 NoData at location in tile {item_id}, trying next tile...")
                        continue  # Try next STAC item
                    
                    # For other errors, report them
                    results.append(f"**{transform_info['name']} ({collection}):**")
                    results.append(f"- Error: {sample_result['error']}")
                elif sample_result.get('value') is not None:
                    raw_value = sample_result['value']
                    
                    # Apply transform
                    try:
                        display_value = transform_info['transform'](raw_value)
                    except:
                        display_value = raw_value
                    
                    # Check if value is in valid range
                    valid_range = transform_info.get('valid_range')
                    in_range = True
                    if valid_range:
                        in_range = valid_range[0] <= raw_value <= valid_range[1]
                    
                    results.append(f"**{transform_info['name']} ({collection}):**")
                    
                    # Check if this is categorical data with class labels
                    class_labels = transform_info.get('class_labels')
                    if class_labels:
                        # Categorical data (burn severity, fire detection, etc.)
                        int_value = int(round(raw_value))
                        class_name = class_labels.get(int_value, f'Unknown class {int_value}')
                        results.append(f"- Class Value: **{int_value}**")
                        results.append(f"- Classification: **{class_name}**")
                        
                        # Add interpretation for burn severity
                        if 'burn' in transform_info['name'].lower() or 'severity' in transform_info['name'].lower():
                            if int_value == 4:
                                results.append(f"- 🔥 This indicates severe burn damage with complete vegetation mortality")
                            elif int_value == 3:
                                results.append(f"- 🟠 Moderate burn with partial vegetation damage")
                            elif int_value == 2:
                                results.append(f"- 🟡 Low severity burn with light surface damage")
                            elif int_value == 1:
                                results.append(f"- 🟢 Unburned or very low impact")
                        
                        # Add interpretation for fire detection
                        if 'fire' in transform_info['name'].lower():
                            if int_value >= 8:
                                results.append(f"- 🔥🔥 Active fire detected with high confidence!")
                            elif int_value == 7:
                                results.append(f"- 🔥 Possible fire activity detected")
                            else:
                                results.append(f"- ✅ No fire detected at this location")
                    else:
                        # Non-categorical numeric data
                        if transform_info['unit_raw'] != transform_info['unit_display']:
                            results.append(f"- Raw Value: {raw_value:.2f} {transform_info['unit_raw']}")
                            results.append(f"- Converted: **{display_value:.2f} {transform_info['unit_display']}**")
                        else:
                            results.append(f"- Value: **{display_value:.2f} {transform_info['unit_display']}**")
                        
                        # Add interpretation for water occurrence
                        if 'water' in transform_info['name'].lower() or 'occurrence' in transform_info['name'].lower():
                            pct = float(raw_value)
                            if pct == 0:
                                results.append(f"- 🏜️ This location is never covered by water")
                            elif pct < 25:
                                results.append(f"- 💧 This location is rarely covered by water ({pct:.0f}% of observations)")
                            elif pct < 50:
                                results.append(f"- 💧💧 This location is sometimes covered by water ({pct:.0f}% of observations)")
                            elif pct < 75:
                                results.append(f"- 🌊 This location is often covered by water ({pct:.0f}% of observations)")
                            elif pct < 100:
                                results.append(f"- 🌊🌊 This location is usually covered by water ({pct:.0f}% of observations)")
                            else:
                                results.append(f"- 🌊🌊🌊 This is permanent water (100% of observations)")
                        
                        # Add interpretation for snow cover
                        if 'snow' in transform_info['name'].lower():
                            pct = float(raw_value)
                            if pct == 0:
                                results.append(f"- ☀️ No snow cover at this location")
                            elif pct < 25:
                                results.append(f"- ❄️ Light snow cover ({pct:.0f}%)")
                            elif pct < 75:
                                results.append(f"- ❄️❄️ Moderate snow cover ({pct:.0f}%)")
                            else:
                                results.append(f"- ❄️❄️❄️ Heavy snow cover ({pct:.0f}%)")
                    
                    if props.get('datetime'):
                        results.append(f"- Date: {props['datetime'][:10]}")
                    
                    if not in_range and not class_labels:
                        results.append(f"- ⚠️ Value outside typical range {valid_range}")
                    
                    sampled_count += 1
                else:
                    results.append(f"**{transform_info['name']}:** No value returned")
                
                results.append("")
            
            if sampled_count == 0:
                # All tiles that contained the pin had issues (nodata/cloud mask)
                ndvi_items_count = sum(1 for tf in filtered_transforms if tf.get('is_ndvi'))
                if ndvi_items_count > 0:
                    if nodata_tiles:
                        results.append(f"\n⚠️ **NDVI calculation failed** - All {tiles_tried} tiles at this location have cloud/ocean mask.")
                        results.append(f"\n**Masked tiles:** {', '.join(nodata_tiles[:3])}")
                        results.append("\n**Possible solutions:**")
                        results.append("1. Search for a **different date** with less cloud cover")
                        results.append("2. If near coast, move pin **further inland** to avoid ocean mask")
                    else:
                        results.append(f"\n⚠️ **Sampling failed** for {tiles_tried} tiles at this location.")
                        results.append("\n**Possible causes:**")
                        results.append("- COG access issues or server timeout")
                        results.append("- Data format incompatibility")
                else:
                    results.append("\n⚠️ Could not extract pixel values. This may be due to:")
                    results.append("- Data is masked (e.g., cloud mask, land mask for SST)")
                    results.append("- COG access issues")
            
            self._log_tool_call("sample_raster_value", {
                "data_type": data_type,
                "lat": lat,
                "lng": lng,
                "samples": sampled_count,
                "tiles_tried": len(filtered_items),
                "tiles_total": len(target_items)
            }, f"Sampled {sampled_count} values from {len(filtered_items)} tiles")
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"❌ sample_raster_value failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Raster sampling failed: {str(e)}"

    @kernel_function(
        name="query_knowledge",
        description="Answer educational or factual questions about geography, satellite data, or scientific concepts. Use this for 'why', 'how', 'explain', history questions, or general knowledge."
    )
    async def query_knowledge(
        self,
        question: Annotated[str, "The educational or factual question to answer"]
    ) -> str:
        """Query LLM knowledge base for educational answers."""
        logger.info(f"📚 TOOL INVOKED: query_knowledge(question='{question[:50]}...')")
        session = self._agent._current_session
        
        try:
            client = self._get_vision_client()
            if not client:
                self._log_tool_call("query_knowledge", {"question": question}, "Client not initialized")
                return "Knowledge query unavailable - Azure OpenAI client not initialized."
            
            # Build context
            context_parts = []
            if session and session.map_bounds:
                bounds = session.map_bounds
                context_parts.append(f"User is viewing: ({bounds.get('center_lat', 'N/A')}, {bounds.get('center_lng', 'N/A')})")
            if session and session.loaded_collections:
                context_parts.append(f"Loaded datasets: {', '.join(session.loaded_collections)}")
            
            context_str = "\n".join(context_parts) if context_parts else "No map context"
            
            system_prompt = f"""You are a knowledgeable geospatial expert. Answer the question using your knowledge.

Current Context:
{context_str}

Guidelines:
- Provide accurate, educational answers
- Include relevant geographic, scientific, or historical facts
- If the question relates to the current map location, incorporate that context
- Be concise but informative"""

            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=800,
                temperature=0.5
            )
            
            result = response.choices[0].message.content
            logger.info(f"🧠 query_knowledge complete ({len(result)} chars)")
            self._log_tool_call("query_knowledge", {"question": question}, result)
            return result
            
        except Exception as e:
            logger.error(f"❌ query_knowledge failed: {e}")
            self._log_tool_call("query_knowledge", {"question": question, "error": str(e)}, "Failed")
            return f"Knowledge query failed: {str(e)}"
    
    @kernel_function(
        name="identify_features",
        description="Identify specific geographic features visible on the map such as rivers, mountains, cities, or landmarks. Use this when the user asks 'what is that' or wants to identify a specific feature."
    )
    async def identify_features(
        self,
        feature_type: Annotated[str, "Type of feature to identify: 'water', 'mountain', 'city', 'road', 'vegetation', or 'any'"]
    ) -> str:
        """Identify geographic features in the current view."""
        logger.info(f"🔍 TOOL INVOKED: identify_features(feature_type='{feature_type}')")
        session = self._agent._current_session
        
        if not session or not session.screenshot_base64:
            self._log_tool_call("identify_features", {"feature_type": feature_type}, "No map view available")
            return "No map view available to identify features."
        
        try:
            client = self._get_vision_client()
            if not client:
                self._log_tool_call("identify_features", {"feature_type": feature_type}, "Client not initialized")
                return "Feature identification unavailable - Azure OpenAI client not initialized."
            
            image_data = session.screenshot_base64
            if image_data.startswith('data:image'):
                image_data = image_data.split(',', 1)[1]
            
            # Build location context
            location_hint = ""
            if session.map_bounds:
                bounds = session.map_bounds
                location_hint = f"Approximate location: ({bounds.get('center_lat', 'N/A')}, {bounds.get('center_lng', 'N/A')})"
            
            prompt = f"""Identify {feature_type} features visible in this satellite/map image.

{location_hint}

For each feature identified, provide:
1. Feature name (if recognizable)
2. Feature type (river, lake, mountain, city, etc.)
3. Notable characteristics

Be specific and confident only about features you can clearly identify."""

            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            logger.info(f"🔍 identify_features complete ({len(result)} chars)")
            self._log_tool_call("identify_features", {"feature_type": feature_type, "has_image": True}, result)
            return result
            
        except Exception as e:
            logger.error(f"❌ identify_features failed: {e}")
            self._log_tool_call("identify_features", {"feature_type": feature_type, "error": str(e)}, "Failed")
            return f"Feature identification failed: {str(e)}"
    
    @kernel_function(
        name="compare_temporal",
        description="Compare satellite imagery between two different time periods to detect changes in surface reflectance, vegetation, or other metrics. Samples actual pixel values and calculates quantitative change (absolute + percentage). Use for questions like 'how did reflectance change between 01/2020 and 01/2024', 'before and after analysis', or 'compare NDVI over time'."
    )
    async def compare_temporal(
        self,
        location: Annotated[str, "The location to analyze (e.g., 'Athens', 'Miami Beach', 'Amazon rainforest')"],
        time_period_1: Annotated[str, "First time period (e.g., '01/2020', 'June 2025', '2020', 'January 2024')"],
        time_period_2: Annotated[str, "Second time period (e.g., '01/2024', 'December 2025', 'now')"],
        analysis_focus: Annotated[str, "What to compare: 'surface reflectance', 'vegetation', 'ndvi', 'urban development', 'water levels', 'snow cover', or 'general'"] = "surface reflectance"
    ) -> str:
        """
        Compare temporal changes by executing two STAC queries and analyzing differences.
        
        This tool:
        1. Parses the two time periods into STAC datetime ranges
        2. Resolves the location to a bbox
        3. Selects the appropriate collection based on analysis focus
        4. Executes two STAC queries (same location/collection, different dates)
        5. Uses GPT-4o vision to compare the two result sets
        """
        logger.info(f"⏳ TOOL INVOKED: compare_temporal(location='{location}', t1='{time_period_1}', t2='{time_period_2}', focus='{analysis_focus}')")
        session = self._agent._current_session
        
        try:
            # ================================================================
            # STEP 1: Parse time periods into STAC datetime format
            # ================================================================
            datetime_1 = self._parse_time_period_to_stac(time_period_1)
            datetime_2 = self._parse_time_period_to_stac(time_period_2)
            
            if not datetime_1 or not datetime_2:
                return f"Could not parse time periods: '{time_period_1}' and '{time_period_2}'. Please use formats like 'June 2025', '2020', or 'January-March 2024'."
            
            logger.info(f"📅 Parsed time periods: {datetime_1} vs {datetime_2}")
            
            # ================================================================
            # STEP 2: Select collection based on analysis focus
            # ================================================================
            collection = self._select_collection_for_analysis(analysis_focus)
            logger.info(f"📚 Selected collection: {collection}")
            
            # ================================================================
            # STEP 3: Resolve location to bbox using geocoding
            # ================================================================
            bbox = await self._resolve_location_to_bbox(location)
            if not bbox:
                return f"Could not resolve location: '{location}'. Please provide a valid city, region, or coordinate."
            
            logger.info(f"📍 Resolved bbox: {bbox}")
            
            # Calculate center point for reflectance sampling
            center_lng = (bbox[0] + bbox[2]) / 2
            center_lat = (bbox[1] + bbox[3]) / 2
            
            # ================================================================
            # STEP 4: Execute two STAC queries in parallel
            # ================================================================
            logger.info(f"🔍 Executing parallel STAC queries for temporal comparison...")
            
            query_1, query_2 = await asyncio.gather(
                self._execute_stac_query(collection, bbox, datetime_1, limit=5),
                self._execute_stac_query(collection, bbox, datetime_2, limit=5)
            )
            
            if not query_1.get("features") and not query_2.get("features"):
                return f"No imagery found for {location} in either time period. Try a different location or date range."
            
            # ================================================================
            # STEP 5: Generate comparison analysis with reflectance sampling
            # ================================================================
            comparison_result = await self._analyze_temporal_comparison(
                location=location,
                collection=collection,
                time_period_1=time_period_1,
                time_period_2=time_period_2,
                query_1_results=query_1,
                query_2_results=query_2,
                analysis_focus=analysis_focus,
                lat=center_lat,
                lng=center_lng
            )
            
            self._log_tool_call("compare_temporal", {
                "location": location,
                "time_period_1": time_period_1,
                "time_period_2": time_period_2,
                "collection": collection,
                "results_t1": len(query_1.get("features", [])),
                "results_t2": len(query_2.get("features", []))
            }, comparison_result[:200])
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"❌ compare_temporal failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._log_tool_call("compare_temporal", {"error": str(e)}, "Failed")
            return f"Temporal comparison failed: {str(e)}"
    
    def _parse_time_period_to_stac(self, time_period: str) -> Optional[str]:
        """
        Parse a natural language time period into STAC datetime format.
        
        Examples:
        - "June 2025" → "2025-06-01/2025-06-30"
        - "2020" → "2020-01-01/2020-12-31"
        - "January-March 2024" → "2024-01-01/2024-03-31"
        - "now" → current month
        """
        import re
        from datetime import datetime
        
        time_period_lower = time_period.lower().strip()
        
        # Handle "now" or "current"
        if time_period_lower in ["now", "current", "today", "present"]:
            now = datetime.now()
            start = now.replace(day=1)
            # End of current month
            if now.month == 12:
                end = now.replace(year=now.year + 1, month=1, day=1)
            else:
                end = now.replace(month=now.month + 1, day=1)
            return f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        
        # Month name to number mapping
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        # Pattern: "Month Year" (e.g., "June 2025")
        month_year_pattern = r'(\w+)\s+(\d{4})'
        match = re.search(month_year_pattern, time_period_lower)
        if match:
            month_str, year_str = match.groups()
            month = month_map.get(month_str)
            if month:
                year = int(year_str)
                # Get last day of month
                if month == 12:
                    last_day = 31
                else:
                    from calendar import monthrange
                    last_day = monthrange(year, month)[1]
                return f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}"
        
        # Pattern: MM/YYYY (e.g., "01/2020", "12/2025")
        mm_yyyy_pattern = r'^(\d{1,2})/(\d{4})$'
        match = re.match(mm_yyyy_pattern, time_period.strip())
        if match:
            month, year = int(match.group(1)), int(match.group(2))
            if 1 <= month <= 12:
                from calendar import monthrange
                last_day = monthrange(year, month)[1]
                return f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}"
        
        # Pattern: Just year (e.g., "2020")
        year_pattern = r'^(\d{4})$'
        match = re.match(year_pattern, time_period.strip())
        if match:
            year = int(match.group(1))
            return f"{year}-01-01/{year}-12-31"
        
        # Pattern: "Month-Month Year" (e.g., "January-March 2024")
        range_pattern = r'(\w+)\s*[-–to]\s*(\w+)\s+(\d{4})'
        match = re.search(range_pattern, time_period_lower)
        if match:
            start_month_str, end_month_str, year_str = match.groups()
            start_month = month_map.get(start_month_str)
            end_month = month_map.get(end_month_str)
            if start_month and end_month:
                year = int(year_str)
                from calendar import monthrange
                last_day = monthrange(year, end_month)[1]
                return f"{year}-{start_month:02d}-01/{year}-{end_month:02d}-{last_day:02d}"
        
        # Fallback: try to extract year
        year_match = re.search(r'(\d{4})', time_period)
        if year_match:
            year = int(year_match.group(1))
            return f"{year}-01-01/{year}-12-31"
        
        return None
    
    def _select_collection_for_analysis(self, analysis_focus: str) -> str:
        """Select the best STAC collection based on analysis focus."""
        focus_lower = analysis_focus.lower()
        
        # Mapping of analysis focus to best collection
        collection_map = {
            "vegetation": "hls",  # HLS for NDVI analysis
            "ndvi": "hls",
            "surface reflectance": "hls",
            "reflectance": "hls",
            "urban development": "sentinel-2-l2a",  # Higher resolution for urban
            "urban": "sentinel-2-l2a",
            "water levels": "jrc-gsw",  # JRC Global Surface Water
            "water": "sentinel-2-l2a",
            "snow cover": "modis-snow",
            "snow": "modis-snow",
            "fire": "modis-fire",
            "wildfire": "modis-fire",
            "general": "sentinel-2-l2a",  # Default to Sentinel-2
        }
        
        return collection_map.get(focus_lower, "sentinel-2-l2a")
    
    async def _resolve_location_to_bbox(self, location: str) -> Optional[List[float]]:
        """Resolve a location name to a bounding box using geocoding."""
        try:
            # Try to import and use the location resolver from the main app
            from location_resolver import get_location_resolver
            resolver = get_location_resolver()
            
            result = await resolver.resolve(location)
            if result and result.get("bbox"):
                return result["bbox"]
            
            # Fallback: Use session context if available
            session = self._agent._current_session
            if session and session.map_bounds:
                bounds = session.map_bounds
                return [
                    bounds.get("west", -180),
                    bounds.get("south", -90),
                    bounds.get("east", 180),
                    bounds.get("north", 90)
                ]
            
            return None
        except Exception as e:
            logger.warning(f"⚠️ Location resolution failed: {e}")
            return None
    
    async def _execute_stac_query(
        self,
        collection: str,
        bbox: List[float],
        datetime_range: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Execute a STAC query and return results."""
        try:
            import aiohttp
            
            # Map collection aliases to actual STAC collection IDs
            collection_aliases = {
                "hls": "hls-l30-v2.0",  # Harmonized Landsat Sentinel
                "sentinel-2": "sentinel-2-l2a",
                "landsat": "landsat-c2-l2",
                "modis-snow": "modis-10A1-061",
                "modis-fire": "modis-14A1-061",
            }
            
            stac_collection = collection_aliases.get(collection.lower(), collection)
            
            # Build STAC search request
            search_body = {
                "collections": [stac_collection],
                "bbox": bbox,
                "datetime": datetime_range,
                "limit": limit,
                "sortby": [{"field": "datetime", "direction": "desc"}]
            }
            
            # Add cloud cover filter for optical data
            if stac_collection in ["sentinel-2-l2a", "hls-l30-v2.0", "landsat-c2-l2"]:
                search_body["query"] = {
                    "eo:cloud_cover": {"lt": 30}
                }
            
            logger.info(f"🔍 STAC Query: {stac_collection}, bbox={bbox}, datetime={datetime_range}")
            
            # Execute search against Planetary Computer
            stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    stac_url,
                    json=search_body,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        features = result.get("features", [])
                        logger.info(f"✅ STAC returned {len(features)} features for {datetime_range}")
                        return result
                    else:
                        logger.warning(f"⚠️ STAC search failed: {response.status}")
                        return {"features": [], "error": f"Status {response.status}"}
                        
        except Exception as e:
            logger.error(f"❌ STAC query failed: {e}")
            return {"features": [], "error": str(e)}
    
    async def _analyze_temporal_comparison(
        self,
        location: str,
        collection: str,
        time_period_1: str,
        time_period_2: str,
        query_1_results: Dict[str, Any],
        query_2_results: Dict[str, Any],
        analysis_focus: str,
        lat: float = None,
        lng: float = None
    ) -> str:
        """Use GPT-4o to analyze the differences between two time periods with actual reflectance sampling."""
        try:
            # Extract metadata from results
            features_1 = query_1_results.get("features", [])
            features_2 = query_2_results.get("features", [])
            
            # Build summary of available data
            summary_1 = self._summarize_stac_results(features_1, time_period_1)
            summary_2 = self._summarize_stac_results(features_2, time_period_2)
            
            # ================================================================
            # REFLECTANCE SAMPLING: Sample actual pixel values for comparison
            # ================================================================
            reflectance_comparison = ""
            
            if lat is not None and lng is not None and features_1 and features_2:
                logger.info(f"📊 Sampling reflectance at ({lat}, {lng}) for temporal comparison...")
                
                # Determine which bands to sample based on analysis focus
                if 'reflectance' in analysis_focus.lower() or 'surface' in analysis_focus.lower():
                    bands = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
                elif 'vegetation' in analysis_focus.lower() or 'ndvi' in analysis_focus.lower():
                    bands = ['B04', 'B08']  # Red, NIR for NDVI
                else:
                    bands = ['B02', 'B03', 'B04', 'B08']  # Default: all visible + NIR
                
                # Sample both time periods
                sample_1 = await self._sample_reflectance_at_point(features_1, lat, lng, bands)
                sample_2 = await self._sample_reflectance_at_point(features_2, lat, lng, bands)
                
                if sample_1.get('values') and sample_2.get('values'):
                    reflectance_lines = [
                        "",
                        "### 📊 Quantitative Reflectance Comparison",
                        "",
                        f"**Sampling Location:** ({lat:.4f}°, {lng:.4f}°)",
                        "",
                        f"#### {time_period_1} (Scene: {sample_1.get('date', 'N/A')})",
                    ]
                    
                    # Show values for period 1
                    for band, data in sample_1['values'].items():
                        reflectance_lines.append(f"- {band}: {data['reflectance']:.4f} (raw: {data['raw']:.0f})")
                    
                    reflectance_lines.append(f"")
                    reflectance_lines.append(f"#### {time_period_2} (Scene: {sample_2.get('date', 'N/A')})")
                    
                    # Show values for period 2
                    for band, data in sample_2['values'].items():
                        reflectance_lines.append(f"- {band}: {data['reflectance']:.4f} (raw: {data['raw']:.0f})")
                    
                    # Calculate and show changes
                    reflectance_lines.append("")
                    reflectance_lines.append("#### Change Analysis")
                    
                    common_bands = set(sample_1['values'].keys()) & set(sample_2['values'].keys())
                    for band in sorted(common_bands):
                        val_1 = sample_1['values'][band]['reflectance']
                        val_2 = sample_2['values'][band]['reflectance']
                        abs_change = val_2 - val_1
                        pct_change = ((val_2 - val_1) / val_1 * 100) if val_1 != 0 else 0
                        
                        change_direction = "↑" if abs_change > 0 else "↓" if abs_change < 0 else "→"
                        reflectance_lines.append(
                            f"- {band}: {abs_change:+.4f} ({pct_change:+.1f}%) {change_direction}"
                        )
                    
                    # Calculate NDVI if we have Red and NIR
                    if 'B04' in common_bands and 'B08' in common_bands:
                        red_1, nir_1 = sample_1['values']['B04']['reflectance'], sample_1['values']['B08']['reflectance']
                        red_2, nir_2 = sample_2['values']['B04']['reflectance'], sample_2['values']['B08']['reflectance']
                        
                        ndvi_1 = (nir_1 - red_1) / (nir_1 + red_1) if (nir_1 + red_1) != 0 else 0
                        ndvi_2 = (nir_2 - red_2) / (nir_2 + red_2) if (nir_2 + red_2) != 0 else 0
                        ndvi_change = ndvi_2 - ndvi_1
                        
                        reflectance_lines.append("")
                        reflectance_lines.append("#### NDVI Change")
                        reflectance_lines.append(f"- {time_period_1} NDVI: {ndvi_1:.3f}")
                        reflectance_lines.append(f"- {time_period_2} NDVI: {ndvi_2:.3f}")
                        reflectance_lines.append(f"- **Change: {ndvi_change:+.3f}**")
                        
                        # Interpretation
                        if ndvi_change > 0.1:
                            reflectance_lines.append("- Interpretation: Significant vegetation increase (greening)")
                        elif ndvi_change > 0.02:
                            reflectance_lines.append("- Interpretation: Moderate vegetation increase")
                        elif ndvi_change < -0.1:
                            reflectance_lines.append("- Interpretation: Significant vegetation decrease (browning/loss)")
                        elif ndvi_change < -0.02:
                            reflectance_lines.append("- Interpretation: Moderate vegetation decrease")
                        else:
                            reflectance_lines.append("- Interpretation: Stable vegetation conditions")
                    
                    reflectance_comparison = "\n".join(reflectance_lines)
                elif sample_1.get('error') or sample_2.get('error'):
                    reflectance_comparison = f"\n\n*Note: Could not sample reflectance - {sample_1.get('error', sample_2.get('error', 'unknown error'))}*"
            
            # ================================================================
            # GPT-4o Analysis
            # ================================================================
            client = self._get_vision_client()
            analysis = ""
            
            if client:
                # Create analysis prompt
                system_prompt = f"""You are a geospatial analyst comparing satellite imagery between two time periods.

Location: {location}
Collection: {collection}
Analysis Focus: {analysis_focus}

Time Period 1 ({time_period_1}):
{summary_1}

Time Period 2 ({time_period_2}):
{summary_2}

Based on the available imagery metadata and your knowledge:
1. Describe what changes would typically be observable between these time periods
2. Explain what the satellite data would show for this analysis focus
3. If there are seasonal differences, explain what those mean
4. Suggest what specific features or indices to examine for detailed analysis

Be specific to the location and time periods. If one period has no data, note that and explain why."""

                deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
                
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Compare {analysis_focus} changes in {location} between {time_period_1} and {time_period_2}."}
                    ],
                    max_tokens=1000,
                    temperature=0.5
                )
                
                analysis = response.choices[0].message.content
            
            # Format the response
            result_parts = [
                f"## Temporal Comparison: {location}",
                f"**Collection:** {collection}",
                f"**Time Periods:** {time_period_1} vs {time_period_2}",
                f"**Analysis Focus:** {analysis_focus}",
                "",
                f"### Data Availability",
                f"- {time_period_1}: {len(features_1)} scenes found",
                f"- {time_period_2}: {len(features_2)} scenes found",
            ]
            
            # Add reflectance comparison if available
            if reflectance_comparison:
                result_parts.append(reflectance_comparison)
            
            if analysis:
                result_parts.append("")
                result_parts.append("### Expert Analysis")
                result_parts.append(analysis)
            
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"❌ Temporal analysis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._format_basic_comparison(
                location, time_period_1, time_period_2,
                query_1_results, query_2_results, analysis_focus
            )
    
    def _summarize_stac_results(self, features: List[Dict], time_period: str) -> str:
        """Create a text summary of STAC results."""
        if not features:
            return f"No imagery available for {time_period}"
        
        summaries = []
        for f in features[:3]:  # Top 3 results
            props = f.get("properties", {})
            datetime_str = props.get("datetime", "Unknown date")
            cloud_cover = props.get("eo:cloud_cover", "N/A")
            summaries.append(f"  - {datetime_str[:10]} (cloud: {cloud_cover}%)")
        
        return f"Found {len(features)} scenes:\n" + "\n".join(summaries)
    
    async def _sample_reflectance_at_point(
        self,
        features: List[Dict],
        lat: float,
        lng: float,
        bands: List[str] = None
    ) -> Dict[str, Any]:
        """
        Sample surface reflectance values from STAC features at a specific point.
        
        Args:
            features: List of STAC feature dicts
            lat: Latitude
            lng: Longitude
            bands: List of band names to sample (default: ['B02', 'B03', 'B04', 'B08'] for RGB+NIR)
        
        Returns:
            Dict with 'values' (band→value), 'date', 'item_id', 'error' (if any)
        """
        if not features:
            return {'error': 'No features available', 'values': {}}
        
        # Default bands for surface reflectance comparison (Sentinel-2 / HLS)
        if bands is None:
            bands = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
        
        # Try to sample from the first available feature with good data
        for feature in features[:3]:
            try:
                assets = feature.get('assets', {})
                props = feature.get('properties', {})
                item_id = feature.get('id', 'unknown')
                datetime_str = props.get('datetime', '')[:10]
                
                sampled_values = {}
                
                for band in bands:
                    # Handle different band naming conventions
                    band_keys_to_try = [
                        band,  # e.g., 'B04'
                        band.lower(),  # e.g., 'b04'
                        f'{band.lower()}_sr',  # e.g., 'b04_sr' (surface reflectance)
                        band.replace('B', 'B0')[:3],  # e.g., 'B04' from 'B4'
                    ]
                    
                    asset_url = None
                    for key in band_keys_to_try:
                        if key in assets:
                            asset_url = assets[key].get('href')
                            break
                    
                    if not asset_url:
                        continue
                    
                    # Sample the COG at the point
                    result = await sample_cog_at_point(asset_url, lat, lng)
                    
                    if result.get('value') is not None:
                        # Apply typical scale factor for surface reflectance (0.0001)
                        raw_value = result['value']
                        scaled_value = raw_value * 0.0001 if raw_value > 100 else raw_value
                        sampled_values[band] = {
                            'raw': raw_value,
                            'scaled': scaled_value,
                            'reflectance': scaled_value  # 0-1 range
                        }
                
                if sampled_values:
                    return {
                        'values': sampled_values,
                        'date': datetime_str,
                        'item_id': item_id,
                        'cloud_cover': props.get('eo:cloud_cover', 'N/A')
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to sample feature: {e}")
                continue
        
        return {'error': 'Could not sample any features', 'values': {}}

    def _format_basic_comparison(
        self,
        location: str,
        time_period_1: str,
        time_period_2: str,
        query_1_results: Dict[str, Any],
        query_2_results: Dict[str, Any],
        analysis_focus: str
    ) -> str:
        """Format a basic comparison without GPT analysis."""
        features_1 = query_1_results.get("features", [])
        features_2 = query_2_results.get("features", [])
        
        result_parts = [
            f"## Temporal Comparison: {location}",
            f"**Time Periods:** {time_period_1} vs {time_period_2}",
            f"**Analysis Focus:** {analysis_focus}",
            "",
            f"### {time_period_1}",
            self._summarize_stac_results(features_1, time_period_1),
            "",
            f"### {time_period_2}",
            self._summarize_stac_results(features_2, time_period_2),
            "",
            "### Next Steps",
            "To complete the comparison:",
            "1. Load imagery from each time period",
            "2. Use the map to visually compare the scenes",
            "3. Or use raster analysis for quantitative metrics (NDVI, etc.)"
        ]
        
        return "\n".join(result_parts)
    
# ============================================================================
# ENHANCED VISION AGENT (Semantic Kernel-based)
# ============================================================================

class EnhancedVisionAgent:
    """
    TRUE AGENTIC Vision Agent using Semantic Kernel.
    
    GPT-4o decides which tools to call based on the user's question,
    rather than using brittle keyword matching. This enables:
    - Semantic understanding of user intent
    - Multi-tool orchestration for complex questions
    - Better handling of ambiguous queries
    
    Available Tools:
    - analyze_screenshot: Visual analysis of map imagery
    - analyze_raster: Quantitative metrics (elevation, NDVI, etc.)
    - query_knowledge: Factual/educational answers
    - identify_features: Identify geographic features
    - compare_temporal: Temporal change analysis
    """
    
    def __init__(self):
        """Initialize the enhanced vision agent with Semantic Kernel."""
        self.sessions: Dict[str, VisionSession] = {}
        self.memory_ttl = timedelta(minutes=30)
        self._kernel: Optional[Kernel] = None
        self._agent: Optional[ChatCompletionAgent] = None
        self._tools: Optional[VisionAgentTools] = None
        self._initialized = False
        self._current_session: Optional[VisionSession] = None
        
        logger.info("✅ EnhancedVisionAgent initialized (Semantic Kernel mode)")
    
    def _ensure_initialized(self):
        """Lazy initialization of Semantic Kernel and agent."""
        if self._initialized:
            return
        
        try:
            # Create Semantic Kernel instance
            self._kernel = Kernel()
            
            # Configure Azure OpenAI service
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            use_managed_identity = os.getenv("AZURE_OPENAI_USE_MANAGED_IDENTITY", "").lower() == "true"
            
            if use_managed_identity or not api_key:
                # Use Managed Identity
                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
                service = AzureChatCompletion(
                    deployment_name=deployment,
                    endpoint=endpoint,
                    api_version=api_version,
                    ad_token_provider=token_provider
                )
            else:
                # Use API key
                service = AzureChatCompletion(
                    deployment_name=deployment,
                    endpoint=endpoint,
                    api_version=api_version,
                    api_key=api_key
                )
            
            self._kernel.add_service(service)
            
            # Create and register tools plugin
            self._tools = VisionAgentTools(self)
            self._kernel.add_plugin(self._tools, "vision_tools")
            
            # Create agent with function calling enabled
            self._agent = ChatCompletionAgent(
                kernel=self._kernel,
                name="VisionAgent",
                instructions=VISION_AGENT_INSTRUCTIONS,
                execution_settings=AzureChatPromptExecutionSettings(
                    function_choice_behavior=FunctionChoiceBehavior.Auto()
                )
            )
            
            self._initialized = True
            logger.info("✅ Semantic Kernel agent initialized with function calling")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Semantic Kernel: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Mark as initialized but with fallback mode
            # Create a basic tools instance even without the full agent
            self._initialized = True
            if self._tools is None:
                self._tools = VisionAgentTools(self)
            logger.warning("⚠️ Vision agent running in fallback mode (direct tool calls only)")
    
    def get_or_create_session(self, session_id: str) -> VisionSession:
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = VisionSession(session_id=session_id)
            logger.info(f"📝 Created new vision session: {session_id}")
        return self.sessions[session_id]
    
    def update_session(
        self,
        session_id: str,
        screenshot_base64: Optional[str] = None,
        map_bounds: Optional[Dict[str, float]] = None,
        collections: Optional[List[str]] = None,
        tile_urls: Optional[List[str]] = None,
        stac_items: Optional[List[Dict[str, Any]]] = None
    ):
        """Update session with new context."""
        session = self.get_or_create_session(session_id)
        
        if screenshot_base64 is not None:
            session.screenshot_base64 = screenshot_base64
        if map_bounds is not None:
            session.map_bounds = map_bounds
        if collections is not None:
            session.loaded_collections = collections
        if tile_urls is not None:
            session.tile_urls = tile_urls
        if stac_items is not None:
            session.stac_items = stac_items
        
        session.updated_at = datetime.utcnow()
    
    async def analyze(
        self,
        user_query: str,
        session_id: str = "default",
        imagery_base64: Optional[str] = None,
        map_bounds: Optional[Dict[str, float]] = None,
        collections: Optional[List[str]] = None,
        tile_urls: Optional[List[str]] = None,
        stac_items: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point: Analyze using Semantic Kernel agent.
        
        GPT-4o decides which tools to call based on the user's question.
        This replaces the old keyword-based tool selection with true agentic behavior.
        
        Args:
            user_query: User's natural language question
            session_id: Session identifier for memory
            imagery_base64: Optional screenshot from frontend
            map_bounds: Geographic bounds of the view
            collections: List of loaded STAC collection IDs
            tile_urls: List of loaded tile URLs
            conversation_history: Conversation history from the current map session
            
        Returns:
            {
                "response": "Agent's response",
                "tools_used": ["tool1", "tool2"],
                "confidence": 0.9,
                "session_id": session_id
            }
        """
        try:
            logger.info(f"🤖 EnhancedVisionAgent (SK) analyzing: '{user_query}'")
            
            # ================================================================
            # 🔍 DETAILED INPUT LOGGING FOR DEBUGGING
            # ================================================================
            logger.info(f"📥 ANALYZE INPUTS: stac_items={len(stac_items) if stac_items else 0}, tile_urls={len(tile_urls) if tile_urls else 0}, collections={collections}")
            if stac_items:
                for i, item in enumerate(stac_items[:2]):
                    logger.info(f"📥 STAC item {i}: id={item.get('id', 'unknown')}, collection={item.get('collection', 'unknown')}, assets={list(item.get('assets', {}).keys())[:5]}")
            
            # ================================================================
            # 🔍 SCREENSHOT RECEIPT VERIFICATION
            # ================================================================
            if imagery_base64:
                size_kb = len(imagery_base64) / 1024
                logger.info(f"📸 VISION AGENT RECEIVED SCREENSHOT: {size_kb:.1f}KB")
            else:
                logger.warning(f"⚠️ VISION AGENT: No screenshot received in analyze() call")
            
            # Ensure agent is initialized
            self._ensure_initialized()
            
            # Clear tool call history for fresh tracking
            if self._tools:
                self._tools.clear_tool_calls()
            
            # Get or update session
            session = self.get_or_create_session(session_id)
            
            if imagery_base64:
                session.screenshot_base64 = imagery_base64
            if map_bounds:
                session.map_bounds = map_bounds
            if collections:
                session.loaded_collections = collections
            if tile_urls:
                session.tile_urls = tile_urls
            if stac_items:
                session.stac_items = stac_items
                logger.info(f"📦 STAC items stored in session: {len(stac_items)} items")
            
            # Populate conversation history from frontend if provided
            if conversation_history:
                for turn in conversation_history:
                    role = turn.get('role', 'user')
                    content = turn.get('content', '')
                    if content and content not in [t.get('content') for t in session.conversation_history]:
                        session.add_turn(role, content)
            
            # Set current session for tools to access
            self._current_session = session
            logger.info(f"🔗 _current_session set: id={session.session_id}, stac_items={len(session.stac_items)}, collections={session.loaded_collections}")
            
            # Add context to the query for better tool selection
            context_parts = []
            if session.screenshot_base64:
                context_parts.append("A map screenshot is available for visual analysis.")
            if session.loaded_collections:
                context_parts.append(f"Loaded data: {', '.join(session.loaded_collections)}")
                # Add explicit tool recommendations based on loaded collections
                tool_recommendations = []
                collections_lower = [c.lower() for c in session.loaded_collections]
                for coll in collections_lower:
                    if 'sentinel-1' in coll or 'alos-palsar' in coll or 'sar' in coll or 'rtc' in coll:
                        tool_recommendations.append("SAR/radar data detected → USE analyze_sar tool")
                    if 'modis-10' in coll or 'snow' in coll:
                        tool_recommendations.append("Snow data detected → USE analyze_snow tool")
                    if 'jrc-gsw' in coll or 'water' in coll:
                        tool_recommendations.append("Water data detected → USE analyze_water tool")
                    if 'modis-14' in coll or 'fire' in coll or 'mtbs' in coll:
                        tool_recommendations.append("Fire data detected → USE analyze_fire tool")
                    if 'modis-13' in coll or 'vegetation' in coll or 'ndvi' in coll:
                        tool_recommendations.append("Vegetation data detected → USE analyze_vegetation tool")
                    if 'dem' in coll or 'cop-dem' in coll or 'alos-dem' in coll or '3dep' in coll or 'lidar' in coll:
                        tool_recommendations.append("Elevation/DEM data detected → USE analyze_raster or sample_raster_value tool")
                if tool_recommendations:
                    context_parts.append("**TOOL RECOMMENDATIONS: " + "; ".join(set(tool_recommendations)) + "**")
            if session.map_bounds:
                bounds = session.map_bounds
                context_parts.append(f"Location: ({bounds.get('center_lat', 'N/A')}, {bounds.get('center_lng', 'N/A')})")
            
            context_hint = "\n".join(context_parts) if context_parts else "No map context available."
            
            # Build augmented query
            augmented_query = f"""User Question: {user_query}

Available Context:
{context_hint}

Use the appropriate tools to answer the user's question. If multiple tools are helpful, call them all."""
            
            # Add to chat history
            session.chat_history.add_user_message(augmented_query)
            
            # Invoke the agent with function calling
            tools_used = []
            response_text = ""
            
            # ================================================================
            # 🎯 FORCED TOOL SELECTION: Override LLM tool selection for specific data types
            # When we have clear data-to-tool mappings, call tools directly instead of
            # relying on GPT-4o's unreliable automatic function calling
            # ================================================================
            forced_tool_used = False
            
            # 🎯 PRIORITY -1: Frontend analysis_type hint OVERRIDES all other routing
            # This ensures GetStartedButton queries are routed to the correct tools:
            # - 'raster' → sample_raster_value (Column 2a: Raster Analysis)
            # - 'screenshot' → analyze_screenshot (Column 2b: Image Analysis)
            analysis_type = kwargs.get('analysis_type')
            logger.info(f"🎯 DEBUG: kwargs={list(kwargs.keys())}, analysis_type={analysis_type}, self._tools={self._tools is not None}")
            if analysis_type:
                logger.info(f"🎯 Frontend analysis_type hint received: {analysis_type}")
            
            # 📊 RASTER ANALYSIS: Force sample_raster_value tool for raster queries
            if analysis_type == 'raster' and self._tools:
                logger.info(f"🎯 FORCED: Using sample_raster_value (frontend hint: raster)")
                logger.info(f"🎯 DEBUG: session.loaded_collections={session.loaded_collections}, session.stac_items={len(session.stac_items) if session.stac_items else 0}")
                # Auto-detect data type based on loaded collections
                data_type = 'auto'
                if session.loaded_collections:
                    coll_str = ' '.join(session.loaded_collections).lower()
                    if 'hls' in coll_str or 's30' in coll_str or 'l30' in coll_str:
                        data_type = 'ndvi'
                    elif 'dem' in coll_str or '3dep' in coll_str or 'lidar' in coll_str:
                        data_type = 'elevation'
                    elif 'modis-14' in coll_str or 'mtbs' in coll_str:
                        data_type = 'fire'
                    elif 'jrc-gsw' in coll_str:
                        data_type = 'water'
                    elif 'modis-10' in coll_str:
                        data_type = 'snow'
                    elif 'sentinel-1' in coll_str or 'palsar' in coll_str:
                        data_type = 'sar'
                    elif 'sst' in coll_str or 'temperature' in coll_str:
                        data_type = 'sst'
                    elif 'biomass' in coll_str or 'chloris' in coll_str:
                        data_type = 'biomass'
                    elif 'cdl' in coll_str or 'cropland' in coll_str:
                        data_type = 'cdl'
                logger.info(f"🎯 Auto-detected data_type: {data_type} from collections: {session.loaded_collections}")
                response_text = await self._tools.sample_raster_value(data_type)
                tools_used = ["sample_raster_value"]
                forced_tool_used = True
            
            # 📸 SCREENSHOT ANALYSIS: Force analyze_screenshot tool for image queries  
            elif analysis_type == 'screenshot' and session.screenshot_base64 and self._tools:
                logger.info(f"🎯 FORCED: Using analyze_screenshot (frontend hint: screenshot)")
                response_text = await self._tools.analyze_screenshot(user_query)
                tools_used = ["analyze_screenshot"]
                forced_tool_used = True
            elif not forced_tool_used and self._tools and session.loaded_collections:
                collections_lower = [c.lower() for c in session.loaded_collections]
                query_lower = user_query.lower()
                
                # Define collection patterns once for reuse
                sar_collections = ['sentinel-1', 'alos-palsar', 'sentinel-1-rtc', 'sar', 'rtc']
                snow_collections = ['modis-10', 'snow']
                water_collections = ['jrc-gsw', 'water', 'flood']
                fire_collections = ['modis-14', 'mtbs', 'fire', 'burn']
                optical_collections = ['hls', 'sentinel-2', 'landsat', 's30', 'l30']
                
                # Helper: check if any collection matches patterns
                def has_collection(patterns):
                    return any(any(p in coll for p in patterns) for coll in collections_lower)
                
                # ============================================================
                # PRIORITY 0: Point-specific value queries → sample_raster_value
                # This MUST come first to extract actual raster values at pin locations
                # We want ALL GetStartedButton.tsx vision queries to use this tool
                # ============================================================
                
                # Keywords that indicate user wants actual raster VALUES (not visual analysis)
                value_keywords = [
                    # Location-specific
                    'value at', 'at this location', 'at this point', 'at the pin', 
                    'at this pin', 'at this spot', 'at this pixel', 'at this ocean',
                    'at this building', 'at this forest', 'at this field',
                    # Action words for sampling
                    'extract', 'sample', 'what is the', 'what are the',
                    # Data types to extract
                    'pixel value', 'raster value', 'band value', 'reflectance value',
                    'backscatter value', 'polarization value',
                    # Specific measurements
                    'in meters', 'in celsius', 'in db', 'in kg', 'in tonnes',
                    'in mw', 'percentage', 'classification value', 'class value',
                    'ndvi value', 'evi value', 'ndsi value', 'npp value',
                    'elevation value', 'temperature value', 'biomass value',
                    'severity value', 'occurrence value', 'height above ground',
                    # Question patterns for specific values
                    'what is the exact', 'what is the fire confidence',
                    'what is the water occurrence', 'what is the sea surface',
                    'what is the net primary', 'what is the burn severity',
                    'what is the alos dem', 'what is the sar backscatter',
                    'what crop type', 'decode the cdl',
                    # Comparison/calculation requests
                    'calculate vegetation index', 'compare to',
                ]
                
                is_value_query = any(kw in query_lower for kw in value_keywords)
                
                # If user asks for ANY specific value, use sample_raster_value
                # This handles ALL GetStartedButton.tsx vision queries
                if is_value_query:
                    # Determine the best data_type parameter based on query/collection
                    data_type = 'auto'
                    if 'ndvi' in query_lower or 'vegetation index' in query_lower:
                        data_type = 'ndvi'
                    elif 'temperature' in query_lower or 'sst' in query_lower or 'celsius' in query_lower:
                        data_type = 'sst'
                    elif 'elevation' in query_lower or 'dem' in query_lower or 'height' in query_lower:
                        data_type = 'elevation'
                    elif 'burn' in query_lower or 'severity' in query_lower or 'mtbs' in query_lower:
                        data_type = 'burn'
                    elif 'fire' in query_lower or 'frp' in query_lower or 'thermal' in query_lower:
                        data_type = 'fire'
                    elif 'water' in query_lower or 'occurrence' in query_lower or 'flood' in query_lower:
                        data_type = 'water'
                    elif 'snow' in query_lower or 'ndsi' in query_lower:
                        data_type = 'snow'
                    elif 'sar' in query_lower or 'backscatter' in query_lower or 'vv' in query_lower or 'vh' in query_lower:
                        data_type = 'sar'
                    elif 'biomass' in query_lower or 'chloris' in query_lower:
                        data_type = 'biomass'
                    elif 'crop' in query_lower or 'cdl' in query_lower:
                        data_type = 'cdl'
                    elif 'npp' in query_lower or 'productivity' in query_lower:
                        data_type = 'npp'
                    elif 'reflectance' in query_lower or 'brdf' in query_lower:
                        data_type = 'reflectance'
                    
                    logger.info(f"🎯 FORCED TOOL: Value query detected → sample_raster_value('{data_type}')")
                    response_text = await self._tools.sample_raster_value(data_type)
                    tools_used = ["sample_raster_value"]
                    forced_tool_used = True
                
                # ============================================================
                # PRIORITY 0.5: CATCH-ALL for ANY vision query with STAC data
                # If we have loaded STAC items and haven't matched keywords above,
                # STILL use sample_raster_value with auto-detection
                # This ensures ALL GetStartedButton.tsx vision queries use raster sampling
                # ============================================================
                if not forced_tool_used and session.stac_items:
                    logger.info(f"🎯 FORCED TOOL: STAC data loaded ({len(session.stac_items)} items) → sample_raster_value('auto')")
                    response_text = await self._tools.sample_raster_value('auto')
                    tools_used = ["sample_raster_value"]
                    forced_tool_used = True
                
                # ============================================================
                # PRIORITY 1: Check for SAR data with WATER/FLOOD keywords FIRST
                # (Only if not already handled by value query above)
                # (SAR can detect water, so water queries on SAR data → analyze_water)
                # ============================================================
                water_keywords = ['water', 'flood', 'flooding', 'lake', 'river', 'bodies', 'inundation', 'wetland']
                if not forced_tool_used and has_collection(sar_collections):
                    if any(kw in query_lower for kw in water_keywords):
                        logger.info("🎯 FORCED TOOL: SAR with water query → calling analyze_water directly")
                        response_text = await self._tools.analyze_water("general")
                        tools_used = ["analyze_water"]
                        forced_tool_used = True
                
                # ============================================================
                # PRIORITY 2: Check for snow data → force analyze_snow
                # ============================================================
                if not forced_tool_used:
                    snow_keywords = ['snow', 'ice', 'cover', 'percentage', 'frozen', 'winter', 'glacier']
                    if has_collection(snow_collections):
                        if any(kw in query_lower for kw in snow_keywords) or 'analyze' in query_lower:
                            logger.info("🎯 FORCED TOOL: Snow data detected → calling analyze_snow directly")
                            response_text = await self._tools.analyze_snow("general")
                            tools_used = ["analyze_snow"]
                            forced_tool_used = True
                
                # ============================================================
                # PRIORITY 3: Check for SAR/radar data → force analyze_sar
                # (Moved after water check - water queries on SAR data handled above)
                # ============================================================
                if not forced_tool_used:
                    sar_keywords = ['sar', 'radar', 'backscatter', 'urban', 'land cover', 'classify', 'polarization', 'vv', 'vh', 'hh', 'hv']
                    if has_collection(sar_collections):
                        if any(kw in query_lower for kw in sar_keywords) or 'analyze' in query_lower:
                            logger.info("🎯 FORCED TOOL: SAR data detected → calling analyze_sar directly")
                            response_text = await self._tools.analyze_sar("general")
                            tools_used = ["analyze_sar"]
                            forced_tool_used = True
                
                # ============================================================
                # PRIORITY 4: Check for water data → force analyze_water
                # ============================================================
                if not forced_tool_used:
                    if has_collection(water_collections):
                        if any(kw in query_lower for kw in water_keywords) or 'analyze' in query_lower:
                            logger.info("🎯 FORCED TOOL: Water data detected → calling analyze_water directly")
                            response_text = await self._tools.analyze_water("general")
                            tools_used = ["analyze_water"]
                            forced_tool_used = True
                
                # ============================================================
                # PRIORITY 5: Check for fire data → force analyze_fire
                # ============================================================
                if not forced_tool_used:
                    fire_keywords = ['fire', 'burn', 'thermal', 'hotspot', 'wildfire', 'anomaly', 'active']
                    if has_collection(fire_collections):
                        if any(kw in query_lower for kw in fire_keywords) or 'analyze' in query_lower:
                            logger.info("🎯 FORCED TOOL: Fire data detected → calling analyze_fire directly")
                            response_text = await self._tools.analyze_fire("general")
                            tools_used = ["analyze_fire"]
                            forced_tool_used = True
                
                # ============================================================
                # PRIORITY 6: Check for vegetation data → force analyze_vegetation
                # Note: Point-specific NDVI queries are handled in PRIORITY 0
                # This handles general vegetation analysis (health, productivity, etc.)
                # ============================================================
                if not forced_tool_used:
                    veg_collections = ['modis-13', 'modis-17', 'ndvi', 'vegetation', 'hls', 'landsat', 'sentinel-2']
                    veg_keywords = ['vegetation', 'greenness', 'health', 'productivity', 'npp', 'lai', 'fpar', 'analyze vegetation']
                    # For general NDVI analysis (not point-specific), also use analyze_vegetation
                    ndvi_general = 'ndvi' in query_lower and not is_point_query
                    if has_collection(veg_collections):
                        if any(kw in query_lower for kw in veg_keywords) or ndvi_general:
                            logger.info("🎯 FORCED TOOL: Vegetation data detected → calling analyze_vegetation directly")
                            response_text = await self._tools.analyze_vegetation("general")
                            tools_used = ["analyze_vegetation"]
                            forced_tool_used = True
            
            # If no forced tool was used, proceed with normal agent invocation
            if not forced_tool_used and self._agent:
                logger.info("🤖 Using Semantic Kernel agent with function calling")
                try:
                    # Use the agent to process the query
                    async for message in self._agent.invoke(session.chat_history):
                        response_text = str(message.content)
                        session.chat_history.add_assistant_message(response_text)
                        logger.info(f"🤖 Agent response: {response_text[:200]}...")
                        
                        # Track which tools were called (from function call metadata)
                        # Note: Semantic Kernel tracks this internally
                        
                except Exception as agent_error:
                    logger.warning(f"Agent invoke failed: {agent_error}, falling back to direct tool call")
                    # Fallback: call query_knowledge directly
                    response_text = await self._tools.query_knowledge(user_query)
                    tools_used = ["query_knowledge"]
            elif not forced_tool_used:
                # No agent available and no forced tool, use fallback with direct tool calls
                logger.warning("⚠️ Agent not available, using direct tool fallback")
                if self._tools:
                    # Determine which tool to use based on query keywords
                    query_lower = user_query.lower()
                    
                    # ✅ PRIORITY 1: Point-specific value queries → sample_raster_value
                    point_keywords = ['sample', 'at this point', 'at this location', 'at the pin', 'value at', 'here', 'at this spot', 'extract the value', 'pixel value']
                    if any(kw in query_lower for kw in point_keywords):
                        logger.info("🔧 Fallback: Using sample_raster_value for point query")
                        response_text = await self._tools.sample_raster_value('auto')
                        tools_used = ["sample_raster_value"]
                    # ✅ PRIORITY 2: Fire/burn queries → analyze_fire
                    elif any(kw in query_lower for kw in ['fire', 'wildfire', 'burn', 'thermal', 'hotspot', 'frp', 'flames', 'blaze', 'conflagration']):
                        logger.info("🔧 Fallback: Using analyze_fire")
                        response_text = await self._tools.analyze_fire("general")
                        tools_used = ["analyze_fire"]
                    # ✅ PRIORITY 3: Vegetation queries → analyze_vegetation
                    elif any(kw in query_lower for kw in ['vegetation', 'ndvi', 'lai', 'fpar', 'npp', 'gpp', 'greenness', 'plant health', 'forest health', 'productivity']):
                        logger.info("🔧 Fallback: Using analyze_vegetation")
                        response_text = await self._tools.analyze_vegetation("general")
                        tools_used = ["analyze_vegetation"]
                    # ✅ PRIORITY 4: Snow/ice queries → analyze_snow
                    elif any(kw in query_lower for kw in ['snow', 'ice', 'glacial', 'winter', 'frozen', 'frost', 'snowpack', 'snowfall']):
                        logger.info("🔧 Fallback: Using analyze_snow")
                        response_text = await self._tools.analyze_snow("general")
                        tools_used = ["analyze_snow"]
                    # ✅ PRIORITY 5: Water queries → analyze_water
                    elif any(kw in query_lower for kw in ['water', 'lake', 'river', 'flood', 'reservoir', 'wetland', 'ocean', 'sea']):
                        logger.info("🔧 Fallback: Using analyze_water")
                        response_text = await self._tools.analyze_water("general")
                        tools_used = ["analyze_water"]
                    # ✅ PRIORITY 6: Land cover queries → analyze_land_cover
                    elif any(kw in query_lower for kw in ['land cover', 'land use', 'urban', 'city', 'cropland', 'agriculture', 'classification', 'built-up']):
                        logger.info("🔧 Fallback: Using analyze_land_cover")
                        response_text = await self._tools.analyze_land_cover("general")
                        tools_used = ["analyze_land_cover"]
                    # ✅ PRIORITY 7: SAR/radar queries → analyze_sar
                    elif any(kw in query_lower for kw in ['sar', 'radar', 'sentinel-1', 'backscatter', 'polarization', 'through clouds']):
                        logger.info("🔧 Fallback: Using analyze_sar")
                        response_text = await self._tools.analyze_sar("general")
                        tools_used = ["analyze_sar"]
                    # ✅ PRIORITY 8: Biomass/carbon queries → analyze_biomass
                    elif any(kw in query_lower for kw in ['biomass', 'carbon', 'carbon stock', 'tree density', 'agb', 'above-ground']):
                        logger.info("🔧 Fallback: Using analyze_biomass")
                        response_text = await self._tools.analyze_biomass("general")
                        tools_used = ["analyze_biomass"]
                    # ✅ PRIORITY 9: Quantitative/statistics queries → analyze_raster
                    elif any(kw in query_lower for kw in ['elevation', 'slope', 'height', 'temperature', 'statistics', 'calculate', 'compute', 'measure', 'average', 'mean', 'reflectance', 'band', 'range', 'min', 'max']):
                        logger.info("🔧 Fallback: Using analyze_raster")
                        response_text = await self._tools.analyze_raster("general")
                        tools_used = ["analyze_raster"]
                    # ✅ PRIORITY 10: Visual analysis questions → analyze_screenshot
                    elif session.screenshot_base64 and any(kw in query_lower for kw in ['see', 'visible', 'show', 'map', 'image', 'screenshot', 'what is', 'describe', 'identify', 'look']):
                        logger.info("🔧 Fallback: Using analyze_screenshot")
                        response_text = await self._tools.analyze_screenshot(user_query)
                        tools_used = ["analyze_screenshot"]
                    else:
                        # Default to knowledge query
                        logger.info("🔧 Fallback: Using query_knowledge")
                        response_text = await self._tools.query_knowledge(user_query)
                        tools_used = ["query_knowledge"]
                else:
                    response_text = "Vision agent tools not initialized. Please check Azure OpenAI configuration."
            
            # Store in session memory
            session.last_analysis = response_text
            session.add_turn("user", user_query)
            session.add_turn("assistant", response_text)
            
            # ================================================================
            # TOOL USAGE TRACKING: Capture which tools were called
            # ================================================================
            # Get tool call history from the tools plugin
            # BUT only if we didn't already set tools_used via forced tool path
            tool_calls = []
            if not forced_tool_used and self._tools:
                tool_calls = self._tools.get_tool_calls()
                tools_used = [tc["tool"] for tc in tool_calls]
            
            # Log summary for debugging
            if tools_used:
                logger.info(f"🔧 VISION AGENT TOOLS USED: {tools_used}")
                for tc in tool_calls:
                    logger.info(f"   └── {tc['tool']}: {tc.get('result_preview', '')[:80]}...")
            else:
                logger.info("ℹ️ Vision Agent: No explicit tool calls tracked (agent_auto mode)")
                tools_used = ["agent_auto"]
            
            logger.info(f"✅ Vision analysis complete ({len(response_text)} chars)")
            
            return {
                "response": response_text,
                "analysis": response_text,  # Alias for compatibility
                "tools_used": tools_used,
                "tool_calls": tool_calls,  # Detailed tool call history for tracing
                "confidence": 0.9 if response_text else 0.5,
                "session_id": session_id,
                "agent_mode": "semantic_kernel",
                "context": {
                    "has_screenshot": bool(session.screenshot_base64),
                    "collections": session.loaded_collections,
                    "map_bounds": session.map_bounds
                }
            }
            
        except Exception as e:
            logger.error(f"❌ EnhancedVisionAgent error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "response": f"I encountered an error: {str(e)}",
                "error": str(e),
                "confidence": 0.0
            }
    
    def cleanup_old_sessions(self, max_age_minutes: int = 30):
        """Remove sessions older than max_age_minutes."""
        now = datetime.utcnow()
        expired = [
            sid for sid, session in self.sessions.items()
            if (now - session.updated_at).total_seconds() > max_age_minutes * 60
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"🗑️ Cleaned up expired vision session: {sid}")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_enhanced_vision_agent: Optional[EnhancedVisionAgent] = None


def get_enhanced_vision_agent() -> EnhancedVisionAgent:
    """Get the singleton EnhancedVisionAgent instance."""
    global _enhanced_vision_agent
    if _enhanced_vision_agent is None:
        _enhanced_vision_agent = EnhancedVisionAgent()
    return _enhanced_vision_agent


# Backwards compatibility alias
def get_vision_agent() -> EnhancedVisionAgent:
    """Backwards compatibility alias for get_enhanced_vision_agent."""
    return get_enhanced_vision_agent()


class VisionAgent(EnhancedVisionAgent):
    """Backwards compatibility class alias."""
    pass
