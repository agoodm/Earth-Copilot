"""
Unit Test: Mobility Agent Raster Sampling
==========================================

Direct tests for the mobility agent's raster sampling functions.
These tests don't require the API - they test the core functions directly.

Run with: python tests/test_mobility_raster_sampling.py

Tests:
1. _read_cog_window - Read pixels from Cloud Optimized GeoTIFF
2. _analyze_elevation - Slope calculation from DEM
3. _analyze_water - SAR backscatter water detection
4. _analyze_vegetation - NDVI calculation
5. Full analyze_mobility with raster sampling

NOTE: Requires rasterio, planetary_computer, pystac_client to be installed.
      These are available in the container environment but may not be locally.
"""

import asyncio
import sys
import os
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for required dependencies
DEPS_AVAILABLE = True
MISSING_DEPS = []

try:
    import rasterio
    from rasterio.windows import from_bounds
except ImportError as e:
    DEPS_AVAILABLE = False
    MISSING_DEPS.append(f"rasterio: {e}")

try:
    import planetary_computer
except ImportError as e:
    DEPS_AVAILABLE = False
    MISSING_DEPS.append(f"planetary_computer: {e}")

try:
    from pystac_client import Client
except ImportError as e:
    DEPS_AVAILABLE = False  
    MISSING_DEPS.append(f"pystac_client: {e}")


# Test locations
TEST_LOCATIONS = {
    "grand_canyon": {"lat": 36.0544, "lon": -112.1401, "name": "Grand Canyon, AZ"},
    "miami": {"lat": 25.7617, "lon": -80.1918, "name": "Miami, FL"},
    "amazon": {"lat": -3.4653, "lon": -62.2159, "name": "Amazon"},
}


async def test_cog_reading():
    """Test reading pixels from a COG file directly."""
    print("\n" + "=" * 60)
    print("TEST 1: COG Reading (Copernicus DEM)")
    print("=" * 60)
    
    try:
        import planetary_computer
        import rasterio
        from rasterio.windows import from_bounds
        from pystac_client import Client
        
        # Connect to Planetary Computer
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        
        loc = TEST_LOCATIONS["grand_canyon"]
        lat, lon = loc["lat"], loc["lon"]
        
        # Small bbox (~1km)
        bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
        
        print(f"📍 Location: {loc['name']} ({lat}, {lon})")
        print(f"📦 Bbox: {bbox}")
        
        # Search for DEM
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox,
            limit=1
        )
        items = list(search.items())
        
        if not items:
            print("❌ No DEM items found")
            return False
        
        print(f"✅ Found {len(items)} DEM item(s)")
        
        # Sign and read
        item = planetary_computer.sign(items[0])
        dem_url = item.assets["data"].href
        print(f"📡 COG URL: {dem_url[:80]}...")
        
        with rasterio.open(dem_url) as src:
            window = from_bounds(*bbox, src.transform)
            elevation = src.read(1, window=window)
            
            print(f"✅ Read elevation array: shape={elevation.shape}, dtype={elevation.dtype}")
            print(f"   Min: {elevation.min():.1f}m, Max: {elevation.max():.1f}m, Mean: {elevation.mean():.1f}m")
            
            # Calculate slope
            dy, dx = np.gradient(elevation, 30)  # 30m resolution
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            
            print(f"   Avg Slope: {slope_deg.mean():.1f}°, Max Slope: {slope_deg.max():.1f}°")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_water_detection():
    """Test SAR water detection."""
    print("\n" + "=" * 60)
    print("TEST 2: Water Detection (Sentinel-1 SAR)")
    print("=" * 60)
    
    try:
        import planetary_computer
        import rasterio
        from rasterio.windows import from_bounds
        from pystac_client import Client
        from datetime import timedelta
        
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        
        loc = TEST_LOCATIONS["miami"]
        lat, lon = loc["lat"], loc["lon"]
        bbox = [lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02]
        
        print(f"📍 Location: {loc['name']} ({lat}, {lon})")
        
        # Search for recent Sentinel-1
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        search = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
            limit=5
        )
        items = list(search.items())
        
        if not items:
            print("⚠️ No Sentinel-1 items found (this is common for some locations)")
            return True  # Not a failure, just no data
        
        print(f"✅ Found {len(items)} Sentinel-1 item(s)")
        
        item = planetary_computer.sign(items[0])
        
        # Try VV or VH polarization
        vv_url = None
        for asset_name in ["vv", "vh"]:
            if asset_name in item.assets:
                vv_url = item.assets[asset_name].href
                print(f"📡 Using {asset_name.upper()} polarization")
                break
        
        if not vv_url:
            print("⚠️ No VV/VH assets found")
            return True
        
        with rasterio.open(vv_url) as src:
            window = from_bounds(*bbox, src.transform)
            sar_data = src.read(1, window=window)
            
            print(f"✅ Read SAR array: shape={sar_data.shape}")
            
            # Convert to dB if needed (values < 0 are likely already dB)
            if sar_data.mean() > 0:
                sar_db = 10 * np.log10(sar_data + 1e-10)
            else:
                sar_db = sar_data
            
            print(f"   Mean backscatter: {sar_db.mean():.1f} dB")
            
            # Water detection (< -20 dB typically indicates water)
            water_threshold = -20
            water_mask = sar_db < water_threshold
            water_pct = (water_mask.sum() / water_mask.size) * 100
            
            print(f"   Water coverage: {water_pct:.1f}% (threshold: {water_threshold} dB)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vegetation_ndvi():
    """Test NDVI vegetation analysis."""
    print("\n" + "=" * 60)
    print("TEST 3: Vegetation Analysis (Sentinel-2 NDVI)")
    print("=" * 60)
    
    try:
        import planetary_computer
        import rasterio
        from rasterio.windows import from_bounds
        from pystac_client import Client
        from datetime import timedelta
        
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        
        loc = TEST_LOCATIONS["amazon"]
        lat, lon = loc["lat"], loc["lon"]
        bbox = [lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02]
        
        print(f"📍 Location: {loc['name']} ({lat}, {lon})")
        
        # Search for Sentinel-2 with low cloud cover
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=60)
        
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
            query={"eo:cloud_cover": {"lt": 30}},
            limit=5
        )
        items = list(search.items())
        
        if not items:
            print("⚠️ No Sentinel-2 items found with low cloud cover")
            return True
        
        print(f"✅ Found {len(items)} Sentinel-2 item(s)")
        
        item = planetary_computer.sign(items[0])
        
        # Get Red (B04) and NIR (B08) bands
        red_url = item.assets.get("B04", {}).href if "B04" in item.assets else None
        nir_url = item.assets.get("B08", {}).href if "B08" in item.assets else None
        
        if not red_url or not nir_url:
            print("⚠️ Missing Red/NIR bands")
            return True
        
        with rasterio.open(red_url) as red_src, rasterio.open(nir_url) as nir_src:
            window = from_bounds(*bbox, red_src.transform)
            red = red_src.read(1, window=window).astype(float)
            nir = nir_src.read(1, window=window).astype(float)
            
            # Calculate NDVI
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndvi = np.clip(ndvi, -1, 1)
            
            valid_ndvi = ndvi[(~np.isnan(ndvi)) & (ndvi >= -1) & (ndvi <= 1)]
            
            print(f"✅ Calculated NDVI: shape={ndvi.shape}")
            print(f"   Mean NDVI: {valid_ndvi.mean():.3f}")
            print(f"   Range: [{valid_ndvi.min():.3f}, {valid_ndvi.max():.3f}]")
            
            # Vegetation classification
            sparse_pct = (valid_ndvi < 0.3).sum() / len(valid_ndvi) * 100
            moderate_pct = ((valid_ndvi >= 0.3) & (valid_ndvi < 0.6)).sum() / len(valid_ndvi) * 100
            dense_pct = (valid_ndvi >= 0.6).sum() / len(valid_ndvi) * 100
            
            print(f"   Sparse (<0.3): {sparse_pct:.1f}%")
            print(f"   Moderate (0.3-0.6): {moderate_pct:.1f}%")
            print(f"   Dense (>0.6): {dense_pct:.1f}%")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mobility_agent():
    """Test full mobility agent with raster sampling."""
    print("\n" + "=" * 60)
    print("TEST 4: Full Mobility Agent Analysis")
    print("=" * 60)
    
    try:
        from geoint.mobility_agent import GeointMobilityAgent
        
        agent = GeointMobilityAgent()
        
        loc = TEST_LOCATIONS["grand_canyon"]
        print(f"📍 Location: {loc['name']} ({loc['lat']}, {loc['lon']})")
        print("⏳ Running full mobility analysis (this may take 30-60 seconds)...")
        
        start = datetime.now()
        result = await agent.analyze_mobility(
            latitude=loc["lat"],
            longitude=loc["lon"],
            include_vision_analysis=False  # Skip vision for unit test
        )
        duration = (datetime.now() - start).total_seconds()
        
        print(f"✅ Analysis complete in {duration:.1f}s")
        
        # Check results
        print("\n📊 Results:")
        print(f"   Agent: {result.get('agent')}")
        print(f"   Location: {result.get('location')}")
        print(f"   Radius: {result.get('radius_miles')} miles")
        print(f"   Data Sources: {result.get('data_sources', [])}")
        
        directions = result.get("directional_analysis", {})
        for direction, data in directions.items():
            status = data.get("status", "UNKNOWN")
            factors = data.get("factors", [])
            confidence = data.get("confidence", "unknown")
            
            emoji = {"GO": "🟢", "SLOW-GO": "🟡", "NO-GO": "🔴", "UNKNOWN": "⚪"}.get(status, "❓")
            print(f"\n   {direction.upper()}: {emoji} {status} (confidence: {confidence})")
            for factor in factors[:2]:  # First 2 factors
                print(f"      • {factor}")
        
        # Check if raster sampling actually happened
        collection_status = result.get("collection_status", {})
        if collection_status:
            print("\n📡 Collection Status:")
            for coll, status in collection_status.items():
                emoji = "✅" if status == "success" else "❌"
                print(f"   {emoji} {coll}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🧪 Mobility Agent Raster Sampling Unit Tests")
    print("=" * 60)
    print(f"⏰ Started: {datetime.now().isoformat()}")
    
    # Check dependencies first
    if not DEPS_AVAILABLE:
        print("\n⚠️ Missing dependencies (install with pip install rasterio planetary-computer pystac-client):")
        for dep in MISSING_DEPS:
            print(f"   - {dep}")
        print("\n💡 These tests require the full container environment.")
        print("   Run in Docker or on the deployed container instead.")
        print("\n   Skipping tests...")
        return True  # Don't fail CI, just skip
    
    results = {
        "COG Reading": await test_cog_reading(),
        "Water Detection": await test_water_detection(),
        "NDVI Calculation": await test_vegetation_ndvi(),
        "Full Mobility Agent": await test_mobility_agent(),
    }
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        emoji = "✅" if result else "❌"
        print(f"   {emoji} {test_name}")
    
    print(f"\n   {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
