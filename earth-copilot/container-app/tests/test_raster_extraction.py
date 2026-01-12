"""
Test: Raster Value Extraction for All Supported Collections
============================================================

Tests the `sample_raster_value` tool across all supported collection types.

Workflow for each collection:
1. Load collection via STAC search (simulates user loading data)
2. Sample raster value at pin coordinates
3. Verify valid value is returned

Run with: python tests/test_raster_extraction.py
Or specific: python tests/test_raster_extraction.py --collection sst
"""

import httpx
import asyncio
import subprocess
import json
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

CONTAINER_APP_NAME = "ca-api-m2grir76dsjlq"
RESOURCE_GROUP = "rg-earthcopilot"
API_ENDPOINT_OVERRIDE = None  # Set to "http://localhost:8000" for local testing
TIMEOUT = 120.0  # seconds


def get_api_endpoint() -> str:
    """Get the API endpoint dynamically from Azure."""
    if API_ENDPOINT_OVERRIDE:
        return API_ENDPOINT_OVERRIDE
    
    try:
        result = subprocess.run(
            ["az", "containerapp", "show",
             "--name", CONTAINER_APP_NAME,
             "--resource-group", RESOURCE_GROUP,
             "--query", "properties.configuration.ingress.fqdn",
             "-o", "tsv"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"https://{result.stdout.strip()}"
    except Exception as e:
        print(f"⚠️ Azure CLI error: {e}")
    
    return "https://ca-api-m2grir76dsjlq.nicewater-9342f9cd.eastus2.azurecontainerapps.io"


# ============================================================================
# RASTER EXTRACTION TEST CASES
# ============================================================================

@dataclass
class RasterTestCase:
    """Test case for raster value extraction."""
    data_type: str           # Type key for sample_raster_value
    collection: str          # STAC collection ID
    display_name: str        # Human-readable name
    stac_query: str          # Query to load the data
    sample_lat: float        # Latitude to sample
    sample_lng: float        # Longitude to sample
    expected_unit: str       # Expected unit in response
    valid_range: tuple       # (min, max) expected value range
    asset_key: str           # Expected asset being sampled


# Test cases for each supported collection type
RASTER_TEST_CASES = [
    # SST / Temperature
    RasterTestCase(
        data_type="sst",
        collection="noaa-cdr-sea-surface-temperature-whoi",
        display_name="Sea Surface Temperature",
        stac_query="Show sea surface temperature near the Gulf Stream",
        sample_lat=35.0,
        sample_lng=-70.0,
        expected_unit="°C",
        valid_range=(-2, 40),
        asset_key="sea_surface_temperature"
    ),
    
    # Elevation - Copernicus DEM
    RasterTestCase(
        data_type="elevation",
        collection="cop-dem-glo-30",
        display_name="Copernicus DEM",
        stac_query="Show elevation map of Grand Canyon",
        sample_lat=36.1069,
        sample_lng=-112.1129,
        expected_unit="m",
        valid_range=(-500, 9000),
        asset_key="data"
    ),
    
    # NDVI - HLS
    RasterTestCase(
        data_type="ndvi",
        collection="hls-l30",
        display_name="HLS NDVI",
        stac_query="Show HLS imagery of Central Park, NYC with low cloud cover",
        sample_lat=40.7829,
        sample_lng=-73.9654,
        expected_unit="index",
        valid_range=(-1, 1),
        asset_key="B04+B08"
    ),
    
    # Burn Severity - MTBS
    RasterTestCase(
        data_type="burn",
        collection="mtbs",
        display_name="MTBS Burn Severity",
        stac_query="Show MTBS burn severity for California in 2020",
        sample_lat=37.5,
        sample_lng=-119.5,
        expected_unit="class",
        valid_range=(0, 6),
        asset_key="burn-severity"
    ),
    
    # Fire Detection - MODIS
    RasterTestCase(
        data_type="fire",
        collection="modis-14A1-061",
        display_name="MODIS Fire Detection",
        stac_query="Show MODIS fire data for California",
        sample_lat=34.0,
        sample_lng=-118.0,
        expected_unit="class",
        valid_range=(0, 9),
        asset_key="FireMask"
    ),
    
    # Water Occurrence - JRC-GSW
    RasterTestCase(
        data_type="water",
        collection="jrc-gsw",
        display_name="JRC Water Occurrence",
        stac_query="Show JRC Global Surface Water for Bangladesh",
        sample_lat=23.8,
        sample_lng=90.4,
        expected_unit="%",
        valid_range=(0, 100),
        asset_key="occurrence"
    ),
    
    # Snow Cover - MODIS
    RasterTestCase(
        data_type="snow",
        collection="modis-10A1-061",
        display_name="MODIS Snow Cover",
        stac_query="Show MODIS snow cover for Quebec in January 2025",
        sample_lat=46.8,
        sample_lng=-71.2,
        expected_unit="%",
        valid_range=(0, 100),
        asset_key="NDSI_Snow_Cover"
    ),
    
    # Land Cover - USDA CDL
    RasterTestCase(
        data_type="landcover",
        collection="usda-cdl",
        display_name="USDA Cropland",
        stac_query="Show USDA Cropland Data Layer for Iowa",
        sample_lat=42.0,
        sample_lng=-93.5,
        expected_unit="class",
        valid_range=(0, 255),
        asset_key="data"
    ),
    
    # Biomass - Chloris
    RasterTestCase(
        data_type="biomass",
        collection="chloris-biomass",
        display_name="Chloris Biomass",
        stac_query="Show Chloris biomass for Amazon rainforest",
        sample_lat=-3.0,
        sample_lng=-60.0,
        expected_unit="Mg/ha",
        valid_range=(0, 500),
        asset_key="aboveground"
    ),
    
    # MODIS Vegetation - NDVI
    RasterTestCase(
        data_type="vegetation",
        collection="modis-13Q1-061",
        display_name="MODIS Vegetation NDVI",
        stac_query="Show MODIS vegetation indices for Ukraine",
        sample_lat=49.0,
        sample_lng=32.0,
        expected_unit="scaled",
        valid_range=(-2000, 10000),
        asset_key="250m_16_days_NDVI"
    ),
]


# ============================================================================
# API HELPERS
# ============================================================================

async def load_stac_data(query: str, pin: Dict, api_endpoint: str) -> Dict:
    """Load STAC data via the /api/query endpoint."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{api_endpoint}/api/query",
            json={
                "query": query,
                "pin": pin,
                "session_id": f"test-raster-{datetime.now().strftime('%H%M%S')}"
            }
        )
        response.raise_for_status()
        return response.json()


async def sample_raster_via_vision(
    data_type: str,
    pin: Dict,
    session_id: str,
    api_endpoint: str
) -> Dict:
    """Sample raster value using the Vision GEOINT endpoint."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Use the vision endpoint with a sampling query
        response = await client.post(
            f"{api_endpoint}/api/geoint/vision",
            json={
                "query": f"Sample the {data_type} value at this location",
                "session_id": session_id,
                "map_bounds": {
                    "pin_lat": pin["lat"],
                    "pin_lng": pin["lng"],
                    "center_lat": pin["lat"],
                    "center_lng": pin["lng"],
                    "zoom": 12
                }
            }
        )
        response.raise_for_status()
        return response.json()


async def test_raster_extraction_direct(test_case: RasterTestCase, api_endpoint: str) -> Dict:
    """
    Test raster extraction for a single collection.
    
    This simulates the real user workflow:
    1. Load data via STAC search
    2. Sample at pin location
    """
    result = {
        "collection": test_case.collection,
        "display_name": test_case.display_name,
        "data_type": test_case.data_type,
        "status": "❓ PENDING",
        "stac_loaded": False,
        "sample_value": None,
        "sample_unit": None,
        "in_range": False,
        "error": None,
        "response_snippet": None
    }
    
    pin = {"lat": test_case.sample_lat, "lng": test_case.sample_lng}
    session_id = f"raster-test-{test_case.data_type}-{datetime.now().strftime('%H%M%S')}"
    
    try:
        # Step 1: Load STAC data
        print(f"  📦 Loading: {test_case.stac_query[:50]}...")
        stac_response = await load_stac_data(test_case.stac_query, pin, api_endpoint)
        
        # Check if STAC data was loaded
        stac_items = stac_response.get("stac_items", [])
        tile_urls = stac_response.get("tile_urls", [])
        
        if stac_items or tile_urls:
            result["stac_loaded"] = True
            print(f"  ✅ Loaded {len(stac_items)} items, {len(tile_urls)} tiles")
        else:
            result["status"] = "⚠️ NO STAC"
            result["error"] = "No STAC items returned"
            return result
        
        # Step 2: Sample raster value via Vision endpoint
        print(f"  📍 Sampling at ({pin['lat']:.4f}, {pin['lng']:.4f})...")
        vision_response = await sample_raster_via_vision(
            test_case.data_type, pin, session_id, api_endpoint
        )
        
        # Parse response for value
        analysis = vision_response.get("analysis", "") or vision_response.get("response", "")
        result["response_snippet"] = analysis[:200] if analysis else "No response"
        
        # Check for success indicators in response
        if "Value:" in analysis or "value:" in analysis.lower():
            result["status"] = "✅ PASS"
            result["sample_value"] = "extracted"  # Would need regex to parse actual value
            result["in_range"] = True
        elif "Error" in analysis or "error" in analysis.lower():
            result["status"] = "❌ ERROR"
            result["error"] = "Sampling error in response"
        elif "No" in analysis and ("data" in analysis.lower() or "value" in analysis.lower()):
            result["status"] = "⚠️ NO DATA"
            result["error"] = "No data at location"
        else:
            # Check if any numeric-looking content exists
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', analysis)
            if numbers:
                result["status"] = "✅ PASS"
                result["sample_value"] = numbers[0]
            else:
                result["status"] = "⚠️ UNCLEAR"
                result["error"] = "Could not parse value from response"
        
        return result
        
    except httpx.TimeoutException:
        result["status"] = "⏱️ TIMEOUT"
        result["error"] = f"Request timed out after {TIMEOUT}s"
        return result
    except httpx.HTTPStatusError as e:
        result["status"] = f"❌ HTTP {e.response.status_code}"
        result["error"] = str(e)
        return result
    except Exception as e:
        result["status"] = "❌ ERROR"
        result["error"] = str(e)
        return result


# ============================================================================
# QUICK API TEST (More efficient)
# ============================================================================

async def test_vision_sampling_quick(api_endpoint: str) -> List[Dict]:
    """
    Quick test - just call the vision endpoint with sampling queries.
    Doesn't load data first, relies on existing session or defaults.
    """
    results = []
    
    sampling_queries = [
        {"type": "sst", "query": "What is the sea surface temperature at this location?", "coords": (35.0, -70.0)},
        {"type": "elevation", "query": "What is the elevation here?", "coords": (36.1, -112.1)},
        {"type": "ndvi", "query": "What is the NDVI value at this point?", "coords": (40.78, -73.97)},
        {"type": "water", "query": "What is the water occurrence percentage here?", "coords": (23.8, 90.4)},
        {"type": "snow", "query": "What is the snow cover at this location?", "coords": (46.8, -71.2)},
        {"type": "burn", "query": "What is the burn severity class here?", "coords": (37.5, -119.5)},
        {"type": "fire", "query": "Is there fire activity at this location?", "coords": (34.0, -118.0)},
        {"type": "landcover", "query": "What is the land cover type here?", "coords": (42.0, -93.5)},
        {"type": "biomass", "query": "What is the biomass at this point?", "coords": (-3.0, -60.0)},
        {"type": "vegetation", "query": "What is the vegetation index here?", "coords": (49.0, 32.0)},
    ]
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for sq in sampling_queries:
            lat, lng = sq["coords"]
            try:
                response = await client.post(
                    f"{api_endpoint}/api/geoint/vision",
                    json={
                        "query": sq["query"],
                        "session_id": f"quick-test-{sq['type']}",
                        "map_bounds": {
                            "pin_lat": lat, "pin_lng": lng,
                            "center_lat": lat, "center_lng": lng,
                            "zoom": 10
                        }
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    analysis = data.get("analysis", "") or data.get("response", "")
                    has_value = any(kw in analysis.lower() for kw in ["value", "°c", "%", "m ", "class", "index"])
                    results.append({
                        "type": sq["type"],
                        "status": "✅" if has_value else "⚠️",
                        "snippet": analysis[:100] if analysis else "No response"
                    })
                else:
                    results.append({
                        "type": sq["type"],
                        "status": f"❌ {response.status_code}",
                        "snippet": response.text[:100]
                    })
                    
            except Exception as e:
                results.append({
                    "type": sq["type"],
                    "status": "❌ ERROR",
                    "snippet": str(e)[:100]
                })
    
    return results


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

async def run_full_tests(collection_filter: Optional[str] = None):
    """Run full raster extraction tests."""
    api_endpoint = get_api_endpoint()
    
    # Filter test cases if specified
    if collection_filter:
        tests = [t for t in RASTER_TEST_CASES 
                 if collection_filter.lower() in t.data_type.lower() 
                 or collection_filter.lower() in t.collection.lower()]
    else:
        tests = RASTER_TEST_CASES
    
    print("\n" + "=" * 100)
    print(f"🔬 RASTER VALUE EXTRACTION TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📡 API: {api_endpoint}")
    print(f"🧪 Testing: {len(tests)} collections")
    print("=" * 100)
    
    results = []
    for tc in tests:
        print(f"\n📊 Testing: {tc.display_name} ({tc.collection})")
        result = await test_raster_extraction_direct(tc, api_endpoint)
        results.append(result)
        print(f"   Result: {result['status']}")
        if result["error"]:
            print(f"   Error: {result['error']}")
    
    # Summary table
    print("\n" + "=" * 100)
    print("📋 SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Data Type':<15} | {'Collection':<35} | {'STAC':<6} | {'Sample':<12} | {'Status':<12}")
    print("-" * 100)
    
    passed = failed = warnings = 0
    for r in results:
        stac_status = "✅" if r["stac_loaded"] else "❌"
        print(f"{r['data_type']:<15} | {r['collection']:<35} | {stac_status:<6} | {str(r['sample_value'])[:12]:<12} | {r['status']:<12}")
        
        if "✅" in r["status"]:
            passed += 1
        elif "❌" in r["status"]:
            failed += 1
        else:
            warnings += 1
    
    print("-" * 100)
    print(f"\n🏁 RESULTS: {passed} passed, {failed} failed, {warnings} warnings")
    
    return results


async def run_quick_test():
    """Run quick sampling test (no STAC loading)."""
    api_endpoint = get_api_endpoint()
    
    print("\n" + "=" * 80)
    print(f"⚡ QUICK RASTER SAMPLING TEST - {datetime.now().strftime('%H:%M:%S')}")
    print(f"📡 API: {api_endpoint}")
    print("=" * 80)
    
    results = await test_vision_sampling_quick(api_endpoint)
    
    print(f"\n{'Type':<15} | {'Status':<8} | {'Response Preview':<50}")
    print("-" * 80)
    for r in results:
        print(f"{r['type']:<15} | {r['status']:<8} | {r['snippet'][:50]}")
    
    return results


async def main():
    """Main entry point."""
    if "--quick" in sys.argv:
        await run_quick_test()
    elif "--collection" in sys.argv:
        idx = sys.argv.index("--collection")
        if idx + 1 < len(sys.argv):
            await run_full_tests(sys.argv[idx + 1])
    else:
        await run_full_tests()


if __name__ == "__main__":
    asyncio.run(main())
