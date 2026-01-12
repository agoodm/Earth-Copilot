#!/usr/bin/env python3
"""
Local Raster Sampling Test Script

This script tests the Vision Agent's raster sampling capabilities LOCALLY
without requiring a deployed backend. This allows rapid iteration and debugging.

USAGE:
    cd earth-copilot/container-app
    .venv\Scripts\python.exe tests\test_local_raster_sampling.py

REQUIREMENTS:
    pip install rasterio planetary-computer httpx pystac-client
"""

import asyncio
import sys
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# TEST CASES
# ============================================================================

@dataclass
class RasterTestCase:
    """Test case for raster sampling."""
    id: int
    name: str
    collection: str
    location_name: str
    latitude: float
    longitude: float
    expected_data_type: str  # ndvi, temperature, elevation, etc.
    expected_asset_key: str  # B04, data, sea_surface_temperature, etc.
    description: str

TEST_CASES = [
    RasterTestCase(
        id=1,
        name="HLS Athens NDVI",
        collection="hls2-l30",
        location_name="Athens, Greece (land)",
        latitude=37.9838,  # Athens city center
        longitude=23.7275,
        expected_data_type="ndvi",
        expected_asset_key="B04,B05",  # RED and NIR
        description="Test NDVI calculation from HLS Landsat data"
    ),
    RasterTestCase(
        id=2,
        name="HLS Sentinel NDVI",
        collection="hls2-s30",
        location_name="Paris, France (land)",
        latitude=48.8566,
        longitude=2.3522,
        expected_data_type="ndvi",
        expected_asset_key="B04,B08",  # RED and NIR (Sentinel)
        description="Test NDVI calculation from HLS Sentinel data"
    ),
    RasterTestCase(
        id=3,
        name="MODIS Vegetation Ukraine",
        collection="modis-13Q1-061",
        location_name="Kyiv, Ukraine",
        latitude=50.4501,
        longitude=30.5234,
        expected_data_type="ndvi",
        expected_asset_key="250m_16_days_NDVI",
        description="Test MODIS NDVI product"
    ),
    RasterTestCase(
        id=4,
        name="Copernicus DEM Grand Canyon",
        collection="cop-dem-glo-30",
        location_name="Grand Canyon, Arizona",
        latitude=36.0544,
        longitude=-112.1401,
        expected_data_type="elevation",
        expected_asset_key="data",
        description="Test elevation extraction from Copernicus DEM"
    ),
    RasterTestCase(
        id=5,
        name="ALOS DEM Galapagos",
        collection="alos-dem",
        location_name="Tomas de Berlanga, Galapagos",
        latitude=-0.8675,
        longitude=-91.1334,
        expected_data_type="elevation",
        expected_asset_key="data",
        description="Test elevation extraction from ALOS DEM"
    ),
    RasterTestCase(
        id=6,
        name="MODIS Fire California",
        collection="modis-14A1-061",
        location_name="California Central Valley",
        latitude=36.7783,
        longitude=-119.4179,
        expected_data_type="fire",
        expected_asset_key="FireMask",
        description="Test fire mask value extraction"
    ),
    RasterTestCase(
        id=7,
        name="JRC Surface Water Bangladesh",
        collection="jrc-gsw",
        location_name="Bangladesh Sundarbans",
        latitude=22.0000,
        longitude=89.0000,
        expected_data_type="water",
        expected_asset_key="occurrence",
        description="Test water occurrence extraction"
    ),
    RasterTestCase(
        id=8,
        name="MODIS Snow Quebec",
        collection="modis-10A1-061",
        location_name="Quebec City, Canada",
        latitude=46.8139,
        longitude=-71.2080,
        expected_data_type="snow",
        expected_asset_key="NDSI_Snow_Cover",
        description="Test snow cover extraction"
    ),
]


# ============================================================================
# STAC UTILITIES
# ============================================================================

async def search_stac_items(collection: str, lat: float, lng: float, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for STAC items covering a specific location."""
    import urllib.request
    import json
    
    # Build STAC search query
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    
    # Create a small bbox around the point
    delta = 0.1  # ~10km
    bbox = [lng - delta, lat - delta, lng + delta, lat + delta]
    
    query = {
        "collections": [collection],
        "bbox": bbox,
        "limit": limit,
        "sortby": [{"field": "datetime", "direction": "desc"}]
    }
    
    try:
        data = json.dumps(query).encode('utf-8')
        req = urllib.request.Request(stac_url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('features', [])
    except Exception as e:
        logger.error(f"STAC search error: {e}")
        return []


async def fetch_stac_item(collection: str, item_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a specific STAC item."""
    import urllib.request
    import json
    
    url = f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/{collection}/items/{item_id}"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        logger.error(f"Error fetching item: {e}")
        return None


def sign_url(url: str) -> str:
    """Sign a Planetary Computer URL."""
    try:
        import planetary_computer as pc
        return pc.sign(url)
    except Exception as e:
        logger.warning(f"Failed to sign URL: {e}")
        return url


# ============================================================================
# RASTER SAMPLING
# ============================================================================

def sample_cog_at_point_sync(cog_url: str, latitude: float, longitude: float, band: int = 1) -> Dict[str, Any]:
    """
    Sample a COG at a specific point (synchronous version for testing).
    """
    try:
        import rasterio
        from rasterio.warp import transform as transform_coords
        from rasterio.windows import Window
        
        logger.info(f"📍 Sampling COG at ({latitude:.4f}, {longitude:.4f})")
        logger.info(f"📍 URL: {cog_url[:100]}...")
        
        # Sign the URL if needed
        signed_url = sign_url(cog_url) if 'blob.core.windows.net' in cog_url else cog_url
        
        env_options = {
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif,.TIF,.tiff,.TIFF',
            'GDAL_HTTP_TIMEOUT': '30',
            'GDAL_HTTP_MAX_RETRY': '3',
        }
        
        with rasterio.Env(**env_options):
            with rasterio.open(signed_url) as src:
                crs = str(src.crs)
                logger.info(f"📍 Raster CRS: {crs}, size: {src.width}x{src.height}")
                logger.info(f"📍 Raster bounds: {src.bounds}")
                
                # Transform coordinates if needed
                if src.crs and str(src.crs) != 'EPSG:4326':
                    xs, ys = transform_coords('EPSG:4326', src.crs, [longitude], [latitude])
                    x, y = xs[0], ys[0]
                    logger.info(f"📍 Transformed: ({longitude}, {latitude}) -> ({x}, {y})")
                else:
                    x, y = longitude, latitude
                
                # Check bounds
                bounds = src.bounds
                if x < bounds.left or x > bounds.right or y < bounds.bottom or y > bounds.top:
                    return {
                        'value': None,
                        'error': f'Point outside tile bounds',
                        'bounds': [bounds.left, bounds.bottom, bounds.right, bounds.top],
                        'crs': crs
                    }
                
                # Get pixel coordinates
                row, col = src.index(x, y)
                logger.info(f"📍 Pixel: row={row}, col={col}")
                
                if row < 0 or row >= src.height or col < 0 or col >= src.width:
                    return {'value': None, 'error': 'Pixel outside raster dimensions'}
                
                # Read value
                window = Window(col, row, 1, 1)
                data = src.read(band, window=window)
                value = float(data[0, 0])
                logger.info(f"📍 ✅ Value: {value}")
                
                # Check nodata
                if src.nodata is not None and value == src.nodata:
                    return {'value': None, 'error': 'NoData at this location', 'nodata_value': src.nodata}
                
                return {
                    'value': value,
                    'band': band,
                    'crs': crs,
                    'success': True
                }
                
    except Exception as e:
        logger.error(f"📍 ❌ Sampling error: {e}")
        return {'value': None, 'error': str(e)}


def calculate_ndvi(red_value: float, nir_value: float) -> Tuple[float, str]:
    """Calculate NDVI and provide interpretation."""
    if (nir_value + red_value) == 0:
        return 0.0, "Division by zero"
    
    ndvi = (nir_value - red_value) / (nir_value + red_value)
    ndvi = max(-1, min(1, ndvi))  # Clip to valid range
    
    if ndvi > 0.6:
        interp = "Dense, healthy vegetation"
    elif ndvi > 0.4:
        interp = "Moderate vegetation"
    elif ndvi > 0.2:
        interp = "Sparse or stressed vegetation"
    elif ndvi > 0:
        interp = "Minimal vegetation, bare soil"
    else:
        interp = "Water, snow, or non-vegetated"
    
    return ndvi, interp


# ============================================================================
# TEST RUNNER
# ============================================================================

async def run_test(test: RasterTestCase) -> Dict[str, Any]:
    """Run a single test case."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST {test.id}: {test.name}")
    logger.info(f"Location: {test.location_name} ({test.latitude}, {test.longitude})")
    logger.info(f"Collection: {test.collection}")
    logger.info(f"{'='*60}")
    
    result = {
        'test_id': test.id,
        'test_name': test.name,
        'collection': test.collection,
        'location': test.location_name,
        'status': 'unknown',
        'stac_items': 0,
        'value': None,
        'error': None
    }
    
    # Step 1: Search for STAC items
    logger.info("📡 Step 1: Searching for STAC items...")
    items = await search_stac_items(test.collection, test.latitude, test.longitude)
    result['stac_items'] = len(items)
    
    if not items:
        result['status'] = 'no_stac_items'
        result['error'] = f"No STAC items found for {test.collection} at location"
        logger.error(f"❌ {result['error']}")
        return result
    
    logger.info(f"✅ Found {len(items)} STAC items")
    
    # Step 2: Get the first item and check its assets
    item = items[0]
    item_id = item.get('id', 'unknown')
    assets = item.get('assets', {})
    
    logger.info(f"📦 Step 2: Checking assets for item: {item_id}")
    logger.info(f"   Available assets: {list(assets.keys())}")
    
    result['item_id'] = item_id
    result['available_assets'] = list(assets.keys())
    
    # Step 3: Sample the raster
    logger.info("🎯 Step 3: Sampling raster value...")
    
    if test.expected_data_type == 'ndvi':
        # Need to sample RED and NIR bands
        expected_keys = test.expected_asset_key.split(',')
        red_key = expected_keys[0] if len(expected_keys) > 0 else None
        nir_key = expected_keys[1] if len(expected_keys) > 1 else None
        
        # Find actual keys in assets
        actual_red = red_key if red_key in assets else ('B04' if 'B04' in assets else None)
        actual_nir = nir_key if nir_key in assets else None
        
        # Check for NIR alternatives
        if not actual_nir:
            for candidate in ['B05', 'B08', 'B8A', 'nir08', 'nir', '250m_16_days_NDVI']:
                if candidate in assets:
                    actual_nir = candidate
                    break
        
        # If MODIS NDVI, just sample the pre-computed NDVI band
        if '250m_16_days_NDVI' in assets:
            logger.info("📊 Using pre-computed MODIS NDVI product")
            ndvi_url = assets['250m_16_days_NDVI'].get('href')
            if ndvi_url:
                sample_result = sample_cog_at_point_sync(ndvi_url, test.latitude, test.longitude)
                if sample_result.get('value') is not None:
                    # MODIS NDVI is scaled by 10000
                    raw_value = sample_result['value']
                    ndvi = raw_value / 10000.0 if abs(raw_value) > 1 else raw_value
                    result['value'] = ndvi
                    result['raw_value'] = raw_value
                    result['status'] = 'success'
                    logger.info(f"✅ MODIS NDVI: {ndvi:.4f} (raw: {raw_value})")
                else:
                    result['status'] = 'sampling_failed'
                    result['error'] = sample_result.get('error')
                    logger.error(f"❌ {result['error']}")
            return result
        
        if not actual_red or not actual_nir:
            result['status'] = 'missing_bands'
            result['error'] = f"Missing bands: RED={actual_red}, NIR={actual_nir}"
            logger.error(f"❌ {result['error']}")
            return result
        
        logger.info(f"📊 Sampling RED ({actual_red}) and NIR ({actual_nir}) bands...")
        
        red_url = assets.get(actual_red, {}).get('href')
        nir_url = assets.get(actual_nir, {}).get('href')
        
        if not red_url or not nir_url:
            result['status'] = 'missing_urls'
            result['error'] = "Missing asset URLs for RED or NIR bands"
            logger.error(f"❌ {result['error']}")
            return result
        
        red_result = sample_cog_at_point_sync(red_url, test.latitude, test.longitude)
        nir_result = sample_cog_at_point_sync(nir_url, test.latitude, test.longitude)
        
        if red_result.get('value') is not None and nir_result.get('value') is not None:
            ndvi, interp = calculate_ndvi(red_result['value'], nir_result['value'])
            result['value'] = ndvi
            result['red_value'] = red_result['value']
            result['nir_value'] = nir_result['value']
            result['interpretation'] = interp
            result['status'] = 'success'
            logger.info(f"✅ RED: {red_result['value']}, NIR: {nir_result['value']}")
            logger.info(f"✅ NDVI: {ndvi:.4f} - {interp}")
        else:
            result['status'] = 'sampling_failed'
            result['error'] = red_result.get('error') or nir_result.get('error')
            result['red_error'] = red_result.get('error')
            result['nir_error'] = nir_result.get('error')
            logger.error(f"❌ Sampling failed: RED={red_result.get('error')}, NIR={nir_result.get('error')}")
    
    else:
        # Single band sampling
        asset_key = test.expected_asset_key
        if asset_key not in assets:
            # Try to find a suitable asset
            for candidate in [asset_key, 'data', 'visual', 'default']:
                if candidate in assets:
                    asset_key = candidate
                    break
        
        if asset_key not in assets:
            result['status'] = 'missing_asset'
            result['error'] = f"Asset '{test.expected_asset_key}' not found"
            logger.error(f"❌ {result['error']}")
            return result
        
        cog_url = assets[asset_key].get('href')
        if not cog_url:
            result['status'] = 'missing_url'
            result['error'] = f"No href for asset '{asset_key}'"
            return result
        
        sample_result = sample_cog_at_point_sync(cog_url, test.latitude, test.longitude)
        
        if sample_result.get('value') is not None:
            result['value'] = sample_result['value']
            result['status'] = 'success'
            logger.info(f"✅ Value: {sample_result['value']}")
        else:
            result['status'] = 'sampling_failed'
            result['error'] = sample_result.get('error')
            logger.error(f"❌ {result['error']}")
    
    return result


async def main():
    """Run all test cases."""
    print("\n" + "="*70)
    print(" 🧪 LOCAL RASTER SAMPLING TEST SUITE")
    print("="*70)
    print("\nThis tests the Vision Agent's ability to sample raster data WITHOUT")
    print("requiring a deployed backend. Use this for rapid debugging.\n")
    
    # Check dependencies
    try:
        import rasterio
        import planetary_computer
        print("✅ Dependencies: rasterio, planetary-computer available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install rasterio planetary-computer httpx")
        return
    
    results = []
    
    # Run tests
    for test in TEST_CASES:
        try:
            result = await run_test(test)
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Test {test.id} crashed: {e}")
            results.append({
                'test_id': test.id,
                'test_name': test.name,
                'status': 'crashed',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*70)
    print(" 📊 TEST SUMMARY")
    print("="*70)
    
    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"\n✅ Passed: {len(success)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if success:
        print("\n🎯 SUCCESSFUL TESTS:")
        for r in success:
            value = r.get('value', 'N/A')
            if isinstance(value, float):
                value = f"{value:.4f}"
            print(f"   #{r['test_id']} {r['test_name']}: {value}")
    
    if failed:
        print("\n⚠️ FAILED TESTS:")
        for r in failed:
            print(f"   #{r['test_id']} {r['test_name']}: {r['status']} - {r.get('error', 'Unknown error')}")
    
    print("\n" + "="*70)
    print(" 🔧 DEBUGGING TIPS")
    print("="*70)
    print("""
If tests fail:
1. Check if STAC items exist for the collection/location
2. Verify the expected asset keys match what's in STAC
3. Ensure the pin location is within tile bounds
4. Check if data is available for the current date

For HLS NDVI specifically:
- HLS2-L30 (Landsat): RED=B04, NIR=B05
- HLS2-S30 (Sentinel): RED=B04, NIR=B08 or B8A

Common issues:
- 'No STAC items': Collection may not have data for that location
- 'Point outside bounds': Pin is in ocean/outside tile coverage
- 'NoData': Valid tile but cloud mask or ocean mask at that pixel
""")


if __name__ == "__main__":
    asyncio.run(main())
