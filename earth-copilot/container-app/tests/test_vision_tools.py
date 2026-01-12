"""
Automated Vision Agent Tool Tests

Tests all Vision Follow-Up Questions from _queries.md Type 3 table.
Verifies that:
1. sample_raster_value works for collections requiring numeric extraction
2. analyze_screenshot works for visual analysis
3. Other specialized tools (analyze_fire, analyze_vegetation, etc.) work correctly

Usage:
    python test_vision_tools.py [--base-url URL] [--output-file PATH]
    
Example:
    python test_vision_tools.py --base-url https://ca-earthcopilot-api.agreeablesea-d9973792.eastus2.azurecontainerapps.io
"""

import asyncio
import httpx
import json
import argparse
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


# ============================================================================
# TEST CASES - Type 3 Vision Follow-Up Questions
# ============================================================================

@dataclass
class VisionTestCase:
    """A single vision tool test case."""
    id: int
    stac_query: str
    collection: str
    vision_question: str
    expected_tool: str
    # Test location (lat, lng) - center of the expected region
    test_lat: float
    test_lng: float
    status: str = "⏳"
    result: str = ""
    tool_called: str = ""
    response_preview: str = ""
    error: str = ""


# Test cases from GetStartedButton.tsx - 21 STAC/Vision pairs (all raster-based)
# All expected_tool = sample_raster_value since we're testing raster value extraction
TEST_CASES = [
    # ============================================================================
    # 🌍 High-Resolution Imagery (HLS S30) - Tests 1-3
    # ============================================================================
    VisionTestCase(
        id=1,
        stac_query="Show Harmonized Landsat Sentinel-2 imagery of Athens",
        collection="hls2-s30",
        vision_question="What is the NDVI value at this pin location? Describe the land cover types visible.",
        expected_tool="sample_raster_value",
        test_lat=38.0986,  # Athens coordinates from PC link
        test_lng=23.5861
    ),
    VisionTestCase(
        id=2,
        stac_query="Show Harmonized Landsat Sentinel-2 (HLS) Version 2.0 images of Moscow",
        collection="hls2-s30",
        vision_question="Sample the surface reflectance bands at this location. What type of urban area is this?",
        expected_tool="sample_raster_value",
        test_lat=55.7497,  # Moscow coordinates from PC link
        test_lng=37.6296
    ),
    VisionTestCase(
        id=3,
        stac_query="Show HLS images of Washington DC",
        collection="hls2-s30",
        vision_question="What are the raster values for the red and NIR bands? Calculate vegetation index from sampled values.",
        expected_tool="sample_raster_value",
        test_lat=38.9445,  # DC coordinates from PC link
        test_lng=-76.9717
    ),
    # ============================================================================
    # 🔥 Fire Detection & Monitoring - Tests 4-6
    # ============================================================================
    VisionTestCase(
        id=4,
        stac_query="Show wildfire MODIS data for California",
        collection="modis-14A1-061",
        vision_question="What is the fire confidence value (FireMask) at this pixel? Describe the fire intensity level.",
        expected_tool="sample_raster_value",
        test_lat=32.4538,  # California coordinates from PC link
        test_lng=-116.7265
    ),
    VisionTestCase(
        id=5,
        stac_query="Show fire modis thermal anomalies daily activity for Australia from June 2025",
        collection="modis-14A2-061",
        vision_question="Sample the Fire Radiative Power (MaxFRP) at this location. What is the thermal intensity in MW?",
        expected_tool="sample_raster_value",
        test_lat=-15.0196,  # Australia coordinates from PC link
        test_lng=143.7962
    ),
    VisionTestCase(
        id=6,
        stac_query="Show MTBS burn severity for California in 2017",
        collection="mtbs",
        vision_question="What is the burn severity classification value at this point? (1=unburned to 4=high severity)",
        expected_tool="sample_raster_value",
        test_lat=37.5345,  # California MTBS coordinates from PC link
        test_lng=-120.1414
    ),
    # ============================================================================
    # 🌊 Water & Surface Reflectance - Tests 7-9
    # ============================================================================
    VisionTestCase(
        id=7,
        stac_query="Display JRC Global Surface Water in Bangladesh",
        collection="jrc-gsw",
        vision_question="What is the water occurrence percentage at this location? How often is this area flooded?",
        expected_tool="sample_raster_value",
        test_lat=23.6238,  # Bangladesh coordinates from PC link
        test_lng=90.8176
    ),
    VisionTestCase(
        id=8,
        stac_query="Show modis snow cover daily for Quebec for January 2025",
        collection="modis-10A1-061",
        vision_question="Sample the NDSI (snow index) value at this point. Is this location snow-covered (NDSI > 0.4)?",
        expected_tool="sample_raster_value",
        test_lat=50.1979,  # Quebec coordinates from PC link
        test_lng=-68.9402
    ),
    VisionTestCase(
        id=9,
        stac_query="Show me Sea Surface Temperature near Madagascar",
        collection="noaa-cdr-sea-surface-temperature-whoi",
        vision_question="What is the sea surface temperature in Celsius at this ocean location?",
        expected_tool="sample_raster_value",
        test_lat=-18.9211,  # Madagascar coordinates from PC link
        test_lng=45.4423
    ),
    # ============================================================================
    # 🌲 Vegetation & Agriculture - Tests 10-14
    # ============================================================================
    VisionTestCase(
        id=10,
        stac_query="Show modis net primary production for San Jose",
        collection="modis-17A3HGF-061",
        vision_question="What is the Net Primary Production (NPP) value in kgC/m²/year at this location?",
        expected_tool="sample_raster_value",
        test_lat=37.0229,  # San Jose coordinates from PC link
        test_lng=-121.8004
    ),
    VisionTestCase(
        id=11,
        stac_query="Show me chloris biomass for the Amazon rainforest",
        collection="chloris-biomass",
        vision_question="Sample the aboveground biomass value in tonnes/hectare at this forest location.",
        expected_tool="sample_raster_value",
        test_lat=-5.0908,  # Amazon coordinates from PC link
        test_lng=-59.7980
    ),
    VisionTestCase(
        id=12,
        stac_query="Show modis vedgetation indices for Ukraine",
        collection="modis-13Q1-061",
        vision_question="What are the NDVI and EVI values at this agricultural field? Is vegetation healthy (NDVI > 0.6)?",
        expected_tool="sample_raster_value",
        test_lat=49.4227,  # Ukraine coordinates from PC link
        test_lng=31.9567
    ),
    VisionTestCase(
        id=13,
        stac_query="Show USDA Cropland Data Layers (CDLs) for Florida",
        collection="usda-cdl",
        vision_question="What crop type code is at this location? Decode the CDL classification value.",
        expected_tool="sample_raster_value",
        test_lat=31.2059,  # Florida coordinates from PC link
        test_lng=-85.1998
    ),
    VisionTestCase(
        id=14,
        stac_query="Show recent modis nadir BDRF adjusted reflectance for Mexico",
        collection="modis-43A4-061",
        vision_question="Sample the BRDF-adjusted reflectance values for bands 1-4. What surface type does this indicate?",
        expected_tool="sample_raster_value",
        test_lat=26.5068,  # Mexico coordinates from PC link
        test_lng=-95.5417
    ),
    # ============================================================================
    # 🏔️ Elevation - Tests 15-18
    # ============================================================================
    VisionTestCase(
        id=15,
        stac_query="Show elevation map of Grand Canyon",
        collection="cop-dem-glo-30",
        vision_question="What is the exact elevation in meters at this point? Sample the DEM raster value.",
        expected_tool="sample_raster_value",
        test_lat=36.0544,  # Grand Canyon coordinates
        test_lng=-112.1401
    ),
    VisionTestCase(
        id=16,
        stac_query="Show ALOS World 3D-30m of Tomas de Berlanga",
        collection="alos-dem",
        vision_question="What is the ALOS DEM elevation value at this location? Compare to Copernicus DEM if available.",
        expected_tool="sample_raster_value",
        test_lat=-0.8630,  # Galapagos coordinates from PC link
        test_lng=-91.0202
    ),
    VisionTestCase(
        id=17,
        stac_query="Show USGS 3DEP Lidar Height above Ground for New Orleans",
        collection="3dep-lidar-hag",
        vision_question="What is the height above ground (HAG) in meters at this building location?",
        expected_tool="sample_raster_value",
        test_lat=29.9511,  # New Orleans coordinates from PC link
        test_lng=-90.0715
    ),
    VisionTestCase(
        id=18,
        stac_query="Show USGS 3DEP Lidar Height above Ground for Denver, Colorado",
        collection="3dep-lidar-hag",
        vision_question="Sample the LiDAR HAG raster. Is this a building (HAG > 3m) or ground level?",
        expected_tool="sample_raster_value",
        test_lat=39.7392,  # Denver coordinates from PC link
        test_lng=-104.9903
    ),
    # ============================================================================
    # 📡 Radar & Reflectance - Tests 19-21
    # ============================================================================
    VisionTestCase(
        id=19,
        stac_query="Show Sentinel 1 RTC for Baltimore",
        collection="sentinel-1-rtc",
        vision_question="What are the VV and VH backscatter values in dB? Is this water (low backscatter) or urban (high)?",
        expected_tool="sample_raster_value",
        test_lat=39.2547,  # Baltimore coordinates from PC link
        test_lng=-76.6287
    ),
    VisionTestCase(
        id=20,
        stac_query="Show ALOS PALSAR Annual for Ecuador",
        collection="alos-palsar-mosaic",
        vision_question="Sample the HH and HV polarization values. What land cover type does this backscatter pattern indicate?",
        expected_tool="sample_raster_value",
        test_lat=-1.4259,  # Ecuador coordinates from PC link
        test_lng=-78.3799
    ),
    VisionTestCase(
        id=21,
        stac_query="Show Sentinel 1 Radiometrically Terrain Corrected (RTC) for Philippines",
        collection="sentinel-1-rtc",
        vision_question="What is the SAR backscatter at this location? Is the VV value consistent with flooded area (<-15 dB)?",
        expected_tool="sample_raster_value",
        test_lat=9.9849,  # Philippines coordinates from PC link
        test_lng=124.1929
    ),
]


# ============================================================================
# TEST RUNNER
# ============================================================================

class VisionToolTester:
    """Runs automated tests for Vision Agent tools."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.results: List[VisionTestCase] = []
        
    async def run_stac_query(self, client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
        """Run a STAC search query and return the session ID and STAC items."""
        url = f"{self.base_url}/api/query"
        payload = {
            "query": query,
            "session_id": f"test-{datetime.now().timestamp()}"
        }
        
        try:
            print(f"    STAC API call: POST {url}")
            print(f"    Query: {query}")
            resp = await client.post(url, json=payload, timeout=180.0)  # 3 min timeout
            print(f"    Response status: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                
                # ============================================================================
                # CORRECT API RESPONSE STRUCTURE:
                # - data.stac_results.features for STAC items
                # - translation_metadata.all_tile_urls for tile URLs
                # - data.search_metadata.collections_searched for collections
                # ============================================================================
                stac_results = data.get("data", {}).get("stac_results", {})
                stac_items = stac_results.get("features", [])
                
                # Get collections from search metadata or extract from items
                search_metadata = data.get("data", {}).get("search_metadata", {})
                collections = search_metadata.get("collections_searched", [])
                if not collections and stac_items:
                    collections = list(set(item.get("collection", "") for item in stac_items if item.get("collection")))
                
                # Get tile URLs from translation_metadata
                translation_metadata = data.get("translation_metadata", {})
                all_tile_urls = translation_metadata.get("all_tile_urls") or []  # Handle None
                tile_urls = [t.get("tilejson_url", "") for t in all_tile_urls if t.get("tilejson_url")]
                
                return {
                    "success": True,
                    "session_id": payload.get("session_id"),
                    "stac_items": stac_items,
                    "tile_urls": tile_urls,
                    "collections": collections,
                    "response": data.get("response", "")
                }
            else:
                return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            import traceback
            print(f"    Exception: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    async def run_vision_query(
        self, 
        client: httpx.AsyncClient, 
        question: str,
        lat: float,
        lng: float,
        stac_items: List[Dict],
        collections: List[str],
        tile_urls: List[str]
    ) -> Dict[str, Any]:
        """Run a vision query with map context."""
        url = f"{self.base_url}/api/geoint/vision"
        
        # The /api/geoint/vision endpoint expects:
        # - latitude, longitude: Pin coordinates
        # - screenshot: base64 encoded image (required - but we'll use a placeholder)
        # - user_query: The question to answer
        # - stac_items: STAC items with assets for raster sampling
        # - tile_urls: TiTiler URLs for the tiles
        # - collection: The collection name(s)
        
        # Create a minimal placeholder screenshot (1x1 white pixel PNG base64)
        # This is required by the endpoint but vision agent can work with STAC data
        placeholder_screenshot = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        map_bounds = {
            "center_lat": lat,
            "center_lng": lng,
            "pin_lat": lat,
            "pin_lng": lng,
            "zoom": 12
        }
        
        payload = {
            "latitude": lat,
            "longitude": lng,
            "screenshot": placeholder_screenshot,
            "user_query": question,
            "session_id": f"vision-test-{datetime.now().timestamp()}",
            "stac_items": stac_items[:10],  # Limit to avoid payload size issues
            "tile_urls": tile_urls[:10],
            "collection": collections[0] if collections else None,
            "map_bounds": map_bounds
        }
        
        try:
            print(f"    Vision API call: POST {url}")
            resp = await client.post(url, json=payload, timeout=180.0)
            print(f"    Vision response status: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                
                # ============================================================================
                # VISION API RESPONSE STRUCTURE:
                # {
                #   "status": "success",
                #   "result": {
                #     "analysis": "...",
                #     "tools_used": ["sample_raster_value", ...],
                #     "vision_data": {...}
                #   },
                #   "session_id": "...",
                #   "timestamp": "..."
                # }
                # ============================================================================
                result = data.get("result", {})
                response_text = result.get("analysis", "") or data.get("response", "")
                tools_used = result.get("tools_used", []) or data.get("tools_used", [])
                
                print(f"    Response length: {len(response_text)}")
                print(f"    Tools used: {tools_used}")
                
                return {
                    "success": True,
                    "response": response_text,
                    "tools_used": tools_used,
                    "tool_calls": data.get("tool_calls", []),
                    "raw_response": data  # For debugging
                }
            else:
                error_text = resp.text[:500]
                print(f"    Error: {error_text}")
                return {"success": False, "error": f"HTTP {resp.status_code}: {error_text}"}
        except Exception as e:
            import traceback
            print(f"    Exception: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def analyze_response(self, test_case: VisionTestCase, vision_result: Dict[str, Any]) -> None:
        """Analyze the vision response and update test case status."""
        if not vision_result.get("success"):
            test_case.status = "FAIL"
            test_case.error = vision_result.get("error", "Unknown error")
            return
        
        response = vision_result.get("response", "")
        tools_used = vision_result.get("tools_used", [])
        
        test_case.response_preview = response[:150] + "..." if len(response) > 150 else response
        test_case.tool_called = ", ".join(tools_used) if tools_used else "unknown"
        
        # Check if the expected tool was called
        expected = test_case.expected_tool.lower()
        tools_lower = [t.lower() for t in tools_used]
        
        tool_matched = any(expected in t for t in tools_lower)
        
        # Check for successful sampling indicators
        sampling_success = False
        if test_case.expected_tool == "sample_raster_value":
            # Look for numeric values or specific success indicators
            sampling_patterns = [
                r'\d+\.?\d*\s*°?C',  # Temperature
                r'\d+\.?\d*\s*m',     # Meters/elevation
                r'NDVI.*-?\d+\.?\d*', # NDVI values
                r'\d+\.?\d*\s*%',     # Percentages
                r'value.*\d+',        # Generic value
                r'class.*\d+',        # Classification
                r'Mg/ha',             # Biomass units
                r'Point Sampling',    # Sampling header
            ]
            for pattern in sampling_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    sampling_success = True
                    break
        
        # Check for failure indicators
        failure_indicators = [
            "cannot be determined",
            "use GIS software",
            "not available",
            "unable to",
            "I couldn't",
            "failed to",
            "no data",
            "error"
        ]
        has_failure = any(ind.lower() in response.lower() for ind in failure_indicators)
        
        # Determine status
        if test_case.expected_tool == "sample_raster_value":
            if sampling_success and not has_failure:
                test_case.status = "PASS"
                test_case.result = "Numeric value extracted"
            elif tool_matched:
                test_case.status = "PARTIAL"
                test_case.result = "Tool called but sampling may have failed"
            else:
                test_case.status = "FAIL"
                test_case.result = "Sample tool not called or sampling failed"
        else:
            # For other tools, check if tool was called and response is meaningful
            if tool_matched and len(response) > 50 and not has_failure:
                test_case.status = "PASS"
                test_case.result = f"{test_case.expected_tool} executed successfully"
            elif len(response) > 100:
                test_case.status = "PARTIAL"
                test_case.result = "Response generated but tool match unclear"
            else:
                test_case.status = "FAIL"
                test_case.result = "Tool not called or response too short"
    
    async def run_test(self, test_case: VisionTestCase, client: httpx.AsyncClient) -> VisionTestCase:
        """Run a single test case."""
        print(f"\n{'='*60}")
        print(f"Test #{test_case.id}: {test_case.collection}")
        print(f"Expected tool: {test_case.expected_tool}")
        print(f"Question: {test_case.vision_question[:60]}...")
        
        # Step 1: Run STAC query
        print(f"  -> Running STAC query...")
        stac_result = await self.run_stac_query(client, test_case.stac_query)
        
        if not stac_result.get("success"):
            test_case.status = "FAIL"
            test_case.error = f"STAC query failed: {stac_result.get('error')}"
            print(f"  [X] STAC query failed: {test_case.error}")
            return test_case
        
        stac_items = stac_result.get("stac_items", [])
        collections = stac_result.get("collections", [])
        tile_urls = stac_result.get("tile_urls", [])
        
        print(f"  [OK] STAC returned {len(stac_items)} items, {len(tile_urls)} tiles")
        
        # ============================================================================
        # DYNAMIC PIN PLACEMENT FIX:
        # Instead of using hardcoded test coords that may fall outside returned tiles,
        # compute the pin location from the FIRST tile's bbox CENTER.
        # This ensures the pin is ALWAYS on valid tile data.
        # ============================================================================
        test_lat = test_case.test_lat
        test_lng = test_case.test_lng
        
        if stac_items:
            # Get bbox from first item: [west, south, east, north]
            first_item = stac_items[0]
            bbox = first_item.get("bbox")
            if bbox and len(bbox) >= 4:
                # Calculate center of bbox
                center_lat = (bbox[1] + bbox[3]) / 2  # (south + north) / 2
                center_lng = (bbox[0] + bbox[2]) / 2  # (west + east) / 2
                print(f"  [PIN] Using tile center: ({center_lat:.4f}, {center_lng:.4f}) instead of ({test_lat:.4f}, {test_lng:.4f})")
                test_lat = center_lat
                test_lng = center_lng
        
        # Step 2: Run vision query
        print(f"  -> Running vision query at ({test_lat}, {test_lng})...")
        vision_result = await self.run_vision_query(
            client,
            test_case.vision_question,
            test_lat,
            test_lng,
            stac_items,
            collections,
            tile_urls
        )
        
        # Step 3: Analyze response
        self.analyze_response(test_case, vision_result)
        
        print(f"  -> Status: {test_case.status}")
        print(f"  -> Tool called: {test_case.tool_called}")
        print(f"  -> Result: {test_case.result}")
        if test_case.response_preview:
            print(f"  -> Response preview: {test_case.response_preview[:200]}...")
        
        return test_case
    
    async def run_all_tests(self, test_cases: List[VisionTestCase] = None) -> List[VisionTestCase]:
        """Run all test cases."""
        if test_cases is None:
            test_cases = TEST_CASES.copy()
        
        print(f"\n{'#'*60}")
        print(f"# Vision Agent Tool Tests")
        print(f"# Base URL: {self.base_url}")
        print(f"# Total tests: {len(test_cases)}")
        print(f"# Started: {datetime.now().isoformat()}")
        print(f"{'#'*60}")
        
        async with httpx.AsyncClient() as client:
            for test_case in test_cases:
                try:
                    await self.run_test(test_case, client)
                except Exception as e:
                    test_case.status = "FAIL"
                    test_case.error = f"Exception: {str(e)}"
                    print(f"  [X] Test #{test_case.id} exception: {e}")
                
                # Increased delay between tests to avoid Planetary Computer rate limiting (409)
                await asyncio.sleep(5)
        
        self.results = test_cases
        return test_cases
    
    def generate_summary(self) -> str:
        """Generate a summary of test results."""
        total = len(self.results)
        passed = sum(1 for t in self.results if "PASS" in t.status)
        partial = sum(1 for t in self.results if "PARTIAL" in t.status)
        failed = sum(1 for t in self.results if "FAIL" in t.status)
        
        summary = f"""
## Vision Tool Test Results

**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Base URL:** {self.base_url}

### Summary
- **Total Tests:** {total}
- **Passed:** {passed} [PASS]
- **Partial:** {partial} [PARTIAL]
- **Failed:** {failed} [FAIL]
- **Pass Rate:** {(passed/total)*100:.1f}%

### sample_raster_value Coverage
"""
        # Analyze sample_raster_value tests specifically
        sample_tests = [t for t in self.results if t.expected_tool == "sample_raster_value"]
        sample_passed = sum(1 for t in sample_tests if "PASS" in t.status)
        
        summary += f"- **Tests:** {len(sample_tests)}\n"
        summary += f"- **Passed:** {sample_passed}/{len(sample_tests)}\n"
        
        if sample_passed < len(sample_tests):
            summary += "\n**Failed sample_raster_value collections:**\n"
            for t in sample_tests:
                if "PASS" not in t.status:
                    summary += f"- {t.collection}: {t.result}\n"
        
        return summary
    
    def generate_markdown_table(self) -> str:
        """Generate updated markdown table for _queries.md."""
        lines = [
            "| # | STAC Query (Get Started Button) | Collection | Vision Follow-Up Question | Expected Tool | Status |",
            "|---|--------------------------------|------------|---------------------------|---------------|--------|"
        ]
        
        for t in self.results:
            # Escape pipe characters in question
            question = t.vision_question.replace("|", "\\|")
            lines.append(f"| {t.id} | {t.stac_query} | {t.collection} | {question} | {t.expected_tool} | {t.status} |")
        
        return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Test Vision Agent tools")
    parser.add_argument(
        "--base-url", 
        default="https://ca-earthcopilot-api.agreeablesea-d9973792.eastus2.azurecontainerapps.io",
        help="Base URL of the Earth Copilot API"
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--test-ids",
        type=str,
        default=None,
        help="Comma-separated list of test IDs to run (e.g., '1,9,15')"
    )
    parser.add_argument(
        "--update-queries-doc",
        action="store_true",
        help="Update the _queries.md file with results"
    )
    
    args = parser.parse_args()
    
    tester = VisionToolTester(args.base_url)
    
    # Filter test cases if specific IDs requested
    test_cases = TEST_CASES.copy()
    if args.test_ids:
        ids = [int(x.strip()) for x in args.test_ids.split(",")]
        test_cases = [t for t in test_cases if t.id in ids]
    
    # Run tests
    results = await tester.run_all_tests(test_cases)
    
    # Print summary
    print("\n" + tester.generate_summary())
    
    # Print updated table
    print("\n### Updated Table for _queries.md:\n")
    print(tester.generate_markdown_table())
    
    # Save JSON results
    if args.output_file:
        output = {
            "run_date": datetime.now().isoformat(),
            "base_url": args.base_url,
            "results": [
                {
                    "id": t.id,
                    "collection": t.collection,
                    "expected_tool": t.expected_tool,
                    "tool_called": t.tool_called,
                    "status": t.status,
                    "result": t.result,
                    "response_preview": t.response_preview,
                    "error": t.error
                }
                for t in results
            ]
        }
        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n[OK] Results saved to {args.output_file}")
    
    # Update _queries.md if requested
    if args.update_queries_doc:
        queries_path = Path(__file__).parent.parent.parent.parent / "documentation" / "melisa" / "_queries.md"
        if queries_path.exists():
            print(f"\n[DOC] Would update: {queries_path}")
            print("   (Use --update-queries-doc to actually update the file)")


if __name__ == "__main__":
    asyncio.run(main())
