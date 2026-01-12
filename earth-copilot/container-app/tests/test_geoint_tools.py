"""
Test: GEOINT Agent Tools
========================

Automated tests for all GEOINT module tools to verify:
1. Tool execution (direct function calls or API)
2. Raster sampling works correctly
3. Correct data sources are used
4. Results are structured properly

Run with: python tests/test_geoint_tools.py
Results saved to: tests/results/geoint_tools_results.json

Modules Tested:
- Vision Agent: analyze_screenshot, analyze_raster, sample_raster_value, etc.
- Terrain Agent: get_elevation_analysis, get_slope_analysis, analyze_flood_risk, etc.
- Mobility Agent: analyze_mobility (with raster sampling)
- Comparison Agent: compare_temporal
"""

import asyncio
import json
import os
import sys
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import httpx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# CONFIGURATION
# ============================================================================

CONTAINER_APP_NAME = "ca-api-m2grir76dsjlq"
RESOURCE_GROUP = "rg-earthcopilot"
API_ENDPOINT_OVERRIDE = None  # Set to "http://localhost:8000" for local testing
TIMEOUT = 120.0  # seconds (longer for raster operations)


def get_api_endpoint() -> str:
    """Get the API endpoint dynamically from Azure or use override."""
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
        print(f"⚠️ Azure CLI failed: {e}")
    
    return "https://ca-api-m2grir76dsjlq.nicewater-9342f9cd.eastus2.azurecontainerapps.io"


# ============================================================================
# TEST CASE DEFINITIONS
# ============================================================================

@dataclass
class ToolTestCase:
    """Test case for a tool invocation."""
    module: str           # Vision, Terrain, Mobility, Comparison
    tool_name: str        # Tool function name
    query: str            # Natural language query that triggers this tool
    test_location: Dict[str, float]  # {"lat": x, "lon": y}
    expected_fields: List[str]       # Fields that should be in response
    description: str      # What we're testing


# Test locations with known characteristics
TEST_LOCATIONS = {
    "grand_canyon": {"lat": 36.0544, "lon": -112.1401, "name": "Grand Canyon, AZ"},
    "amazon": {"lat": -3.4653, "lon": -62.2159, "name": "Amazon Rainforest"},
    "sahara": {"lat": 23.4162, "lon": 25.6628, "name": "Sahara Desert"},
    "alps": {"lat": 46.8182, "lon": 8.2275, "name": "Swiss Alps"},
    "florida_coast": {"lat": 25.7617, "lon": -80.1918, "name": "Miami, FL"},
    "la_fires": {"lat": 34.0522, "lon": -118.2437, "name": "Los Angeles, CA"},
    "alaska": {"lat": 64.2008, "lon": -149.4937, "name": "Alaska"},
    "bangladesh": {"lat": 23.6850, "lon": 90.3563, "name": "Bangladesh"},
    "madagascar": {"lat": -18.7669, "lon": 46.8691, "name": "Madagascar"},
    "ukraine": {"lat": 48.3794, "lon": 31.1656, "name": "Ukraine"},
}


# ============================================================================
# VISION MODULE TESTS
# ============================================================================

VISION_TESTS = [
    ToolTestCase(
        module="Vision",
        tool_name="analyze_screenshot",
        query="What do you see in this satellite image?",
        test_location=TEST_LOCATIONS["grand_canyon"],
        expected_fields=["analysis", "features"],
        description="GPT-4o vision analysis of map screenshot"
    ),
    ToolTestCase(
        module="Vision",
        tool_name="sample_raster_value",
        query="What is the elevation at Grand Canyon?",
        test_location=TEST_LOCATIONS["grand_canyon"],
        expected_fields=["value", "collection", "unit"],
        description="Sample elevation from cop-dem-glo-30"
    ),
    ToolTestCase(
        module="Vision",
        tool_name="analyze_vegetation",
        query="Analyze vegetation health in the Amazon",
        test_location=TEST_LOCATIONS["amazon"],
        expected_fields=["ndvi", "vegetation_status"],
        description="NDVI calculation from Sentinel-2"
    ),
    ToolTestCase(
        module="Vision",
        tool_name="analyze_fire",
        query="Are there any active fires near Los Angeles?",
        test_location=TEST_LOCATIONS["la_fires"],
        expected_fields=["fire_detected", "confidence"],
        description="MODIS thermal anomaly detection"
    ),
    ToolTestCase(
        module="Vision",
        tool_name="analyze_water",
        query="Show water coverage in Bangladesh",
        test_location=TEST_LOCATIONS["bangladesh"],
        expected_fields=["water_coverage", "source"],
        description="JRC Global Surface Water analysis"
    ),
    ToolTestCase(
        module="Vision",
        tool_name="analyze_snow",
        query="What's the snow cover in Alaska?",
        test_location=TEST_LOCATIONS["alaska"],
        expected_fields=["snow_coverage", "source"],
        description="MODIS snow cover detection"
    ),
    ToolTestCase(
        module="Vision",
        tool_name="analyze_land_cover",
        query="What is the land cover type in the Sahara?",
        test_location=TEST_LOCATIONS["sahara"],
        expected_fields=["land_cover_class", "source"],
        description="ESA WorldCover classification"
    ),
    ToolTestCase(
        module="Vision",
        tool_name="analyze_biomass",
        query="Estimate biomass in the Amazon rainforest",
        test_location=TEST_LOCATIONS["amazon"],
        expected_fields=["biomass", "unit"],
        description="Chloris biomass estimation"
    ),
    ToolTestCase(
        module="Vision",
        tool_name="analyze_sar",
        query="Show SAR backscatter for Miami coast",
        test_location=TEST_LOCATIONS["florida_coast"],
        expected_fields=["backscatter", "polarization"],
        description="Sentinel-1 SAR analysis"
    ),
]


# ============================================================================
# TERRAIN MODULE TESTS
# ============================================================================

TERRAIN_TESTS = [
    ToolTestCase(
        module="Terrain",
        tool_name="get_elevation_analysis",
        query="Analyze elevation profile of the Swiss Alps",
        test_location=TEST_LOCATIONS["alps"],
        expected_fields=["min_elevation", "max_elevation", "mean_elevation"],
        description="Copernicus DEM elevation statistics"
    ),
    ToolTestCase(
        module="Terrain",
        tool_name="get_slope_analysis",
        query="Calculate slope steepness at Grand Canyon",
        test_location=TEST_LOCATIONS["grand_canyon"],
        expected_fields=["avg_slope", "max_slope", "terrain_classification"],
        description="Slope calculation from DEM gradient"
    ),
    ToolTestCase(
        module="Terrain",
        tool_name="get_aspect_analysis",
        query="What direction do the slopes face in the Alps?",
        test_location=TEST_LOCATIONS["alps"],
        expected_fields=["dominant_aspect", "aspect_distribution"],
        description="Aspect (compass direction) from DEM"
    ),
    ToolTestCase(
        module="Terrain",
        tool_name="get_terrain_ruggedness",
        query="How rugged is the terrain at Grand Canyon?",
        test_location=TEST_LOCATIONS["grand_canyon"],
        expected_fields=["ruggedness_index", "classification"],
        description="Terrain Ruggedness Index calculation"
    ),
    ToolTestCase(
        module="Terrain",
        tool_name="get_viewshed_analysis",
        query="What can be seen from this viewpoint in the Alps?",
        test_location=TEST_LOCATIONS["alps"],
        expected_fields=["visible_area", "max_view_distance"],
        description="Viewshed visibility analysis"
    ),
    ToolTestCase(
        module="Terrain",
        tool_name="analyze_flood_risk",
        query="Is this location in a flood zone?",
        test_location=TEST_LOCATIONS["bangladesh"],
        expected_fields=["flood_risk_level", "permitting_status"],
        description="JRC-GSW flood zone analysis"
    ),
    ToolTestCase(
        module="Terrain",
        tool_name="analyze_water_proximity",
        query="How far is the nearest water body?",
        test_location=TEST_LOCATIONS["florida_coast"],
        expected_fields=["distance_to_water", "meets_setback"],
        description="Distance to water body calculation"
    ),
    ToolTestCase(
        module="Terrain",
        tool_name="analyze_environmental_sensitivity",
        query="Are there wetlands or sensitive areas here?",
        test_location=TEST_LOCATIONS["amazon"],
        expected_fields=["sensitivity_level", "land_cover_types"],
        description="ESA WorldCover wetland/forest detection"
    ),
]


# ============================================================================
# MOBILITY MODULE TESTS
# ============================================================================

MOBILITY_TESTS = [
    ToolTestCase(
        module="Mobility",
        tool_name="analyze_mobility",
        query="Analyze terrain mobility in all directions",
        test_location=TEST_LOCATIONS["grand_canyon"],
        expected_fields=["north", "south", "east", "west", "data_sources"],
        description="Full mobility assessment with raster sampling"
    ),
    ToolTestCase(
        module="Mobility",
        tool_name="analyze_mobility",
        query="Can vehicles traverse this terrain?",
        test_location=TEST_LOCATIONS["sahara"],
        expected_fields=["directional_analysis", "summary"],
        description="Desert terrain mobility (should be mostly GO)"
    ),
    ToolTestCase(
        module="Mobility",
        tool_name="analyze_mobility",
        query="Mobility analysis for dense forest area",
        test_location=TEST_LOCATIONS["amazon"],
        expected_fields=["directional_analysis", "summary"],
        description="Forest mobility (vegetation/water constraints)"
    ),
]


# ============================================================================
# COMPARISON MODULE TESTS
# ============================================================================

COMPARISON_TESTS = [
    ToolTestCase(
        module="Comparison",
        tool_name="compare_temporal",
        query="Compare surface reflectance in Miami between 01/2020 and 01/2025",
        test_location=TEST_LOCATIONS["florida_coast"],
        expected_fields=["before", "after", "change_analysis"],
        description="Temporal comparison with dual STAC queries"
    ),
    ToolTestCase(
        module="Comparison",
        tool_name="compare_temporal",
        query="How did vegetation change in Ukraine from 2021 to 2024?",
        test_location=TEST_LOCATIONS["ukraine"],
        expected_fields=["before", "after", "change_analysis"],
        description="NDVI temporal comparison"
    ),
]


# ============================================================================
# TEST EXECUTION
# ============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    module: str
    tool_name: str
    description: str
    test_location: str
    status: str  # PASS, FAIL, ERROR, SKIP
    duration_ms: float
    response_preview: str
    expected_fields_found: List[str]
    expected_fields_missing: List[str]
    error_message: Optional[str] = None
    raster_sampled: bool = False
    data_sources: List[str] = None


async def run_tool_test(client: httpx.AsyncClient, endpoint: str, test: ToolTestCase) -> TestResult:
    """Execute a single tool test via the API."""
    start_time = datetime.now()
    
    try:
        # Build the request
        payload = {
            "query": test.query,
            "center_lat": test.test_location["lat"],
            "center_lng": test.test_location["lon"],
            "screenshot_base64": None,  # Some tests may need this
            "history": [],
            "selected_module": test.module.lower()
        }
        
        # Call the chat endpoint
        response = await client.post(
            f"{endpoint}/chat",
            json=payload,
            timeout=TIMEOUT
        )
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        if response.status_code != 200:
            return TestResult(
                module=test.module,
                tool_name=test.tool_name,
                description=test.description,
                test_location=test.test_location.get("name", str(test.test_location)),
                status="ERROR",
                duration_ms=duration_ms,
                response_preview=response.text[:500],
                expected_fields_found=[],
                expected_fields_missing=test.expected_fields,
                error_message=f"HTTP {response.status_code}"
            )
        
        data = response.json()
        response_text = str(data)
        
        # Check for expected fields in response
        found_fields = []
        missing_fields = []
        for field in test.expected_fields:
            if field.lower() in response_text.lower():
                found_fields.append(field)
            else:
                missing_fields.append(field)
        
        # Detect if raster sampling occurred
        raster_indicators = ["sampled", "pixel", "COG", "rasterio", "elevation:", "ndvi:", "backscatter"]
        raster_sampled = any(ind.lower() in response_text.lower() for ind in raster_indicators)
        
        # Extract data sources if mentioned
        data_sources = []
        source_keywords = ["Sentinel-1", "Sentinel-2", "MODIS", "Copernicus", "JRC", "ESA", "Landsat", "Chloris"]
        for src in source_keywords:
            if src.lower() in response_text.lower():
                data_sources.append(src)
        
        status = "PASS" if len(missing_fields) == 0 else "PARTIAL" if len(found_fields) > 0 else "FAIL"
        
        return TestResult(
            module=test.module,
            tool_name=test.tool_name,
            description=test.description,
            test_location=test.test_location.get("name", str(test.test_location)),
            status=status,
            duration_ms=duration_ms,
            response_preview=response_text[:500],
            expected_fields_found=found_fields,
            expected_fields_missing=missing_fields,
            raster_sampled=raster_sampled,
            data_sources=data_sources
        )
        
    except asyncio.TimeoutError:
        return TestResult(
            module=test.module,
            tool_name=test.tool_name,
            description=test.description,
            test_location=test.test_location.get("name", str(test.test_location)),
            status="TIMEOUT",
            duration_ms=TIMEOUT * 1000,
            response_preview="",
            expected_fields_found=[],
            expected_fields_missing=test.expected_fields,
            error_message=f"Timeout after {TIMEOUT}s"
        )
    except Exception as e:
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        return TestResult(
            module=test.module,
            tool_name=test.tool_name,
            description=test.description,
            test_location=test.test_location.get("name", str(test.test_location)),
            status="ERROR",
            duration_ms=duration_ms,
            response_preview="",
            expected_fields_found=[],
            expected_fields_missing=test.expected_fields,
            error_message=str(e)
        )


async def run_all_tests() -> Dict[str, Any]:
    """Run all GEOINT tool tests."""
    endpoint = get_api_endpoint()
    print(f"🎯 Testing endpoint: {endpoint}")
    print("=" * 80)
    
    all_tests = VISION_TESTS + TERRAIN_TESTS + MOBILITY_TESTS + COMPARISON_TESTS
    results: List[TestResult] = []
    
    async with httpx.AsyncClient() as client:
        for i, test in enumerate(all_tests, 1):
            print(f"\n[{i}/{len(all_tests)}] {test.module} | {test.tool_name}")
            print(f"    📍 {test.test_location.get('name', 'Unknown')}")
            print(f"    📝 {test.description}")
            
            result = await run_tool_test(client, endpoint, test)
            results.append(result)
            
            # Print status
            status_emoji = {
                "PASS": "✅",
                "PARTIAL": "⚠️",
                "FAIL": "❌",
                "ERROR": "💥",
                "TIMEOUT": "⏰",
                "SKIP": "⏭️"
            }
            print(f"    {status_emoji.get(result.status, '❓')} {result.status} ({result.duration_ms:.0f}ms)")
            
            if result.raster_sampled:
                print(f"    📊 Raster sampling: YES")
            if result.data_sources:
                print(f"    🛰️ Data sources: {', '.join(result.data_sources)}")
            if result.error_message:
                print(f"    ⚠️ Error: {result.error_message}")
    
    # Generate summary
    summary = generate_summary(results)
    
    # Save results
    output = {
        "test_run": {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "total_tests": len(results)
        },
        "summary": summary,
        "results": [asdict(r) for r in results]
    }
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, "geoint_tools_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return output


def generate_summary(results: List[TestResult]) -> Dict[str, Any]:
    """Generate test summary statistics."""
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    partial = sum(1 for r in results if r.status == "PARTIAL")
    failed = sum(1 for r in results if r.status == "FAIL")
    errors = sum(1 for r in results if r.status in ["ERROR", "TIMEOUT"])
    raster_tests = sum(1 for r in results if r.raster_sampled)
    
    by_module = {}
    for r in results:
        if r.module not in by_module:
            by_module[r.module] = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
        if r.status == "PASS":
            by_module[r.module]["pass"] += 1
        elif r.status == "PARTIAL":
            by_module[r.module]["partial"] += 1
        elif r.status == "FAIL":
            by_module[r.module]["fail"] += 1
        else:
            by_module[r.module]["error"] += 1
    
    avg_duration = sum(r.duration_ms for r in results) / total if total > 0 else 0
    
    return {
        "total": total,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "errors": errors,
        "pass_rate": f"{(passed / total * 100):.1f}%" if total > 0 else "0%",
        "raster_sampling_detected": raster_tests,
        "avg_duration_ms": round(avg_duration, 0),
        "by_module": by_module
    }


def print_summary(summary: Dict[str, Any]):
    """Print formatted summary."""
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {summary['total']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"⚠️ Partial: {summary['partial']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"💥 Errors: {summary['errors']}")
    print(f"📈 Pass Rate: {summary['pass_rate']}")
    print(f"📊 Raster Sampling Detected: {summary['raster_sampling_detected']} tests")
    print(f"⏱️ Avg Duration: {summary['avg_duration_ms']}ms")
    
    print("\n📦 By Module:")
    for module, stats in summary["by_module"].items():
        total_module = stats["pass"] + stats["partial"] + stats["fail"] + stats["error"]
        print(f"  {module}: {stats['pass']}/{total_module} passed")


if __name__ == "__main__":
    print("🧪 GEOINT Tools Test Suite")
    print("=" * 80)
    
    output = asyncio.run(run_all_tests())
    print_summary(output["summary"])
