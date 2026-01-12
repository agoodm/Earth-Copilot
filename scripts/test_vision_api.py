#!/usr/bin/env python3
"""
Quick API Test for Vision Agent NDVI

Test the deployed Vision Agent API directly without the UI.
This allows rapid testing of NDVI extraction capabilities.

USAGE:
    python scripts/test_vision_api.py

REQUIREMENTS:
    - urllib only (standard library)
"""

import json
import urllib.request
import urllib.error
import ssl
from typing import Dict, Any

# Configuration
API_BASE_URL = "https://ca-earthcopilot-api.agreeablesea-d9973792.eastus2.azurecontainerapps.io"

# Test cases: known land locations with good satellite coverage
TEST_CASES = [
    {
        "name": "Athens Land (Parthenon area - VERIFIED ON LAND)",
        "lat": 37.9715,  # On land, near Parthenon - verified coordinates
        "lng": 23.7267,  # This is EAST of the coast, definitely on land
        "collection": "hls2-l30",
        "question": "What is the NDVI value at this location?"
    },
    {
        "name": "Athens Urban (Syntagma Square - VERIFIED ON LAND)",
        "lat": 37.9755,  # Syntagma Square - heart of Athens
        "lng": 23.7348,  # Well inland
        "collection": "hls2-l30",
        "question": "Sample the NDVI value at this pin location."
    },
    {
        "name": "Ukraine Farm (Kyiv region)", 
        "lat": 50.4501,
        "lng": 30.5234,
        "collection": "modis-13Q1-061",
        "question": "What is the NDVI value here?"
    },
    {
        "name": "Grand Canyon Elevation",
        "lat": 36.1070,
        "lng": -112.1130,
        "collection": "cop-dem-glo-30",
        "question": "What is the elevation at this location?"
    }
]


def search_stac(collection: str, lat: float, lng: float, limit: int = 5) -> Dict[str, Any]:
    """Search for STAC items at a location."""
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    
    delta = 0.1
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
        
        # Create SSL context that doesn't verify (for testing)
        ctx = ssl.create_default_context()
        
        with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except Exception as e:
        return {"error": str(e)}


def call_vision_api(lat: float, lng: float, question: str, collection: str, stac_items: list) -> Dict[str, Any]:
    """Call the Vision API endpoint."""
    url = f"{API_BASE_URL}/api/geoint/vision"
    
    payload = {
        "message": question,
        "latitude": lat,
        "longitude": lng,
        "collection": collection,
        "stac_items": stac_items[:3],  # First 3 items
        "include_screenshot": False  # Skip screenshot for this test
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        ctx = ssl.create_default_context()
        
        with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else "No body"
        return {"error": f"HTTP {e.code}: {error_body[:500]}"}
    except Exception as e:
        return {"error": str(e)}


def run_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print(" 🧪 VISION API DIRECT TEST")
    print("="*70)
    print(f"\nAPI: {API_BASE_URL}")
    print("\nThis tests the Vision Agent API directly without UI.\n")
    
    for i, test in enumerate(TEST_CASES):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {test['name']}")
        print(f"Location: ({test['lat']}, {test['lng']})")
        print(f"Collection: {test['collection']}")
        print(f"Question: {test['question']}")
        print("="*60)
        
        # Step 1: Search for STAC items
        print("\n📡 Step 1: Searching STAC...")
        stac_result = search_stac(test['collection'], test['lat'], test['lng'])
        
        if 'error' in stac_result:
            print(f"❌ STAC search failed: {stac_result['error']}")
            continue
        
        features = stac_result.get('features', [])
        print(f"✅ Found {len(features)} STAC items")
        
        if not features:
            print("❌ No STAC items found for this location")
            continue
        
        # Show first item's assets
        first_item = features[0]
        assets = list(first_item.get('assets', {}).keys())[:10]
        print(f"   Item: {first_item.get('id', 'unknown')}")
        print(f"   Assets: {assets}")
        
        # Step 2: Call Vision API
        print("\n🎯 Step 2: Calling Vision API...")
        vision_result = call_vision_api(
            test['lat'], 
            test['lng'], 
            test['question'],
            test['collection'],
            features
        )
        
        if 'error' in vision_result:
            print(f"❌ Vision API failed: {vision_result['error']}")
            continue
        
        # Parse response
        result = vision_result.get('result', {})
        response_text = result.get('response', str(result))
        tools_used = result.get('tools_called', [])
        
        print(f"✅ Vision API responded")
        print(f"   Tools used: {tools_used}")
        print(f"\n   Response preview:")
        # Show first 500 chars
        preview = response_text[:500] if len(response_text) > 500 else response_text
        for line in preview.split('\n')[:10]:
            print(f"   {line}")
        
        # Check for success indicators
        if 'NDVI' in response_text.upper() and any(x in response_text for x in ['0.', '-0.']):
            print("\n✅ SUCCESS: NDVI value appears to be extracted!")
        elif 'elevation' in response_text.lower() and any(c.isdigit() for c in response_text):
            print("\n✅ SUCCESS: Elevation value appears to be extracted!")
        elif 'error' in response_text.lower() or 'outside' in response_text.lower():
            print("\n⚠️ PARTIAL: Response contains error/outside warning")
        else:
            print("\n❓ UNKNOWN: Check response manually")
    
    print("\n" + "="*70)
    print(" 📊 TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_tests()
