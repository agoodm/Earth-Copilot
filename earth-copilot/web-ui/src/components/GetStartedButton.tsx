// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import React, { useState } from 'react';
import './GetStartedButton.css';

interface ExampleQuery {
  query: string;
  description: string;
  dataset: string;
  pc_link?: string;
  rasterQuery?: string; // Step 2a: Raster sampling query (sample_raster_value)
  screenshotQuery?: string; // Step 2b: Screenshot analysis query (analyze_screenshot)
}

interface TerrainQuery {
  location: string;
  setupQuery: string;
  question: string;
  expectedTools: string[];
}

interface MobilityQuery {
  location: string;
  setupQuery: string;
  question: string;
  analysisType: string;
}

interface QueryCategory {
  category: string;
  icon: string;
  examples: ExampleQuery[];
}

interface GetStartedButtonProps {
  onQuerySelect?: (query: string) => void;
}

const GetStartedButton: React.FC<GetStartedButtonProps> = ({ onQuerySelect }) => {
  const [showModal, setShowModal] = useState(false);
  const [activeColumn, setActiveColumn] = useState<'stac' | 'vision'>('stac'); // For mobile toggle
  const [activeTab, setActiveTab] = useState<'stac' | 'terrain' | 'mobility'>('stac'); // Main tab navigation

  // Terrain Module Test Queries (Type 4)
  const terrainQueries: TerrainQuery[] = [
    {
      location: "Grand Canyon, Arizona",
      setupQuery: "Show DEM elevation map of Grand Canyon",
      question: "What is the elevation range and slope distribution at this location?",
      expectedTools: ["get_elevation_analysis", "get_slope_analysis"]
    },
    {
      location: "Mount Rainier, Washington",
      setupQuery: "Go to Mount Rainier",
      question: "Is this location suitable for a construction permit? Analyze terrain constraints including slope, flood risk, and flat areas.",
      expectedTools: ["get_slope_analysis", "analyze_flood_risk", "find_flat_areas"]
    },
    {
      location: "Houston, Texas",
      setupQuery: "Show HLS imagery of Houston",
      question: "Analyze the flood risk and environmental sensitivity for this site. What is the permitting recommendation?",
      expectedTools: ["analyze_flood_risk", "analyze_environmental_sensitivity"]
    },
    {
      location: "Denver, Colorado",
      setupQuery: "Show 3DEP LiDAR for Denver",
      question: "Which direction do the slopes face? What is the sun exposure rating for solar panel installation?",
      expectedTools: ["get_aspect_analysis", "get_elevation_analysis"]
    },
    {
      location: "Everglades, Florida",
      setupQuery: "Display JRC Global Surface Water in Florida",
      question: "Is this area suitable for a solar farm? Check flat areas, water proximity, and setback requirements.",
      expectedTools: ["find_flat_areas", "analyze_water_proximity", "get_slope_analysis"]
    }
  ];

  // Mobility Module Test Queries (Type 5)
  const mobilityQueries: MobilityQuery[] = [
    {
      location: "Baltimore, Maryland",
      setupQuery: "Show Sentinel 1 RTC for Baltimore",
      question: "Analyze mobility conditions for ground vehicles in all directions (N/S/E/W). What are the GO/SLOW-GO/NO-GO zones?",
      analysisType: "Directional Analysis"
    },
    {
      location: "Amazon Rainforest, Brazil",
      setupQuery: "Show chloris biomass for Amazon",
      question: "Can military vehicles traverse this forested area? Analyze vegetation density and trafficability.",
      analysisType: "Vegetation Obstacles"
    },
    {
      location: "Grand Canyon, Arizona",
      setupQuery: "Show DEM elevation map of Grand Canyon",
      question: "What is the mobility assessment for off-road vehicles near the canyon rim? Identify slope-based GO/NO-GO zones.",
      analysisType: "Slope Constraints"
    },
    {
      location: "Bangladesh Flood Plains",
      setupQuery: "Display JRC Global Surface Water in Bangladesh",
      question: "Identify water obstacles and river crossing challenges. Can wheeled vehicles cross during monsoon season?",
      analysisType: "Water Obstacles"
    },
    {
      location: "California Fire Zone",
      setupQuery: "Show wildfire MODIS data for California",
      question: "Can emergency vehicles safely access this area with active fires? Identify safe corridors and evacuation routes.",
      analysisType: "Fire Hazards"
    }
  ];

  const exampleQueries: QueryCategory[] = [
    {
      category: "🌍 High-Resolution Imagery",
      icon: "🛰️",
      examples: [
        {
          query: "Show Harmonized Landsat Sentinel-2 imagery of Athens",
          description: "30m resolution harmonized imagery combining Landsat-8/9 and Sentinel-2",
          dataset: "HLS S30",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=23.5861%2C38.0986&z=9.81&v=2&d=hls2-s30&m=Most+recent+%28low+cloud%29&r=Natural+color&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "What is the NDVI value at this pin location?",
          screenshotQuery: "Describe what you see in this satellite image. What land cover types are visible?"
        },
        {
          query: "Show Harmonized Landsat Sentinel-2 (HLS) Version 2.0 images of Moscow",
          description: "30m resolution harmonized imagery combining Landsat-8/9 and Sentinel-2",
          dataset: "HLS S30",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=37.6296%2C55.7497&z=11.88&v=2&d=sentinel-2&m=Most+recent+%28low+cloud%29&r=Natural+color&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "Sample the surface reflectance bands at this location.",
          screenshotQuery: "What urban features can you identify in this image of Moscow?"
        },
        {
          query: "Show HLS images of Washington DC",
          description: "30m resolution harmonized imagery combining Landsat-8/9 and Sentinel-2",
          dataset: "HLS S30",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-76.9717%2C38.9445&z=10.07&v=2&d=hls2-s30&s=false%3A%3A100%3A%3Atrue&ae=0&sr=desc&m=Most+recent+%28low+cloud%29&r=Natural+color",
          rasterQuery: "What are the raster values for the red and NIR bands?",
          screenshotQuery: "Describe the urban layout and green spaces visible in Washington DC."
        }
      ]
    },
    {
      category: "🔥 Fire Detection & Monitoring",
      icon: "🔥",
      examples: [
        {
          query: "Show wildfire MODIS data for California",
          description: "1km thermal anomalies and fire detection, daily updates. Note: Zoom level 10+ required.",
          dataset: "MODIS 14A1",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-116.7265%2C32.4538&z=9.44&v=2&d=modis-14A1-061&m=Most+recent&r=Confidence+of+fire%2C+daily&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "What is the fire confidence value (FireMask) at this pixel?",
          screenshotQuery: "Can you see any active fire hotspots or burn scars in this thermal imagery?"
        },
        {
          query: "Show fire modis thermal anomalies daily activity for Australia from June 2025",
          description: "1km thermal anomalies and fire locations, 8-day composite. Note: Zoom level 10+ required.",
          dataset: "MODIS 14A2",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=143.7962%2C-15.0196&z=7.84&v=2&d=modis-14A2-061&m=Most+recent&r=Confidence+of+fire+8-day&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "Sample the Fire Radiative Power (MaxFRP) at this location.",
          screenshotQuery: "Describe the fire activity patterns visible in this thermal anomaly map."
        },
        {
          query: "Show MTBS burn severity for California in 2017",
          description: "30m burn severity assessment for large fires (1984-2018)",
          dataset: "MTBS",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-120.1414%2C37.5345&z=7.00&v=2&d=mtbs&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0&m=2017&r=Burn+severity",
          rasterQuery: "What is the burn severity classification value at this point?",
          screenshotQuery: "What burn severity patterns do you see? Where are the most severely burned areas?"
        }
      ]
    },
    {
      category: "🌊 Water & Surface Reflectance",
      icon: "💧",
      examples: [
        {
          query: "Display JRC Global Surface Water in Bangladesh",
          description: "30m water occurrence mapping from 1984-2021",
          dataset: "JRC Global Surface Water",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=90.8176%2C23.6238&z=7.48&v=2&d=jrc-gsw&m=Most+recent&r=Water+occurrence&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "What is the water occurrence percentage at this location?",
          screenshotQuery: "Describe the water bodies and flood patterns visible in this water occurrence map."
        },
        {
          query: "Show modis snow cover daily for Quebec for January 2025",
          description: "500m snow cover and NDSI, daily updates",
          dataset: "MODIS 10A1",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-68.9402%2C50.1979&z=6.60&v=2&d=modis-10A1-061&m=Most+recent&r=Normalized+difference+snow+index+%28NDSI%29+daily&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "Sample the NDSI (snow index) value at this point.",
          screenshotQuery: "What snow coverage patterns do you see? Are there any snow-free areas?"
        },
        {
          query: "Show me Sea Surface Temperature near Madagascar",
          description: "0.25° resolution daily sea surface temperature (PC collection missing)",
          dataset: "NOAA CDR Sea Surface Temperature",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=45.4423%2C-18.9211&z=8.39&v=2&d=noaa-cdr-sea-surface-temperature-whoi&m=Most+recent&r=Sea+surface+temperature&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "What is the sea surface temperature in Celsius at this ocean location?",
          screenshotQuery: "Describe the ocean temperature gradients visible. Where are the warmest/coldest waters?"
        }
      ]
    },
    {
      category: "🌲 Vegetation & Agriculture",
      icon: "🌿",
      examples: [
        {
          query: "Show modis net primary production for San Jose",
          description: "500m net primary productivity, yearly composite",
          dataset: "MODIS 17A3HGF",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-121.8004%2C37.0229&z=8.41&v=2&d=modis-17A3HGF-061&m=Most+recent&r=Net+primary+productivity+gap-filled+yearly+%28kgC%2Fm%C2%B2%29&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "What is the Net Primary Production (NPP) value in kgC/m²/year at this location?",
          screenshotQuery: "What vegetation productivity patterns do you see? Where is vegetation most productive?"
        },
        {
          query: "Show me chloris biomass for the Amazon rainforest",
          description: "30m aboveground woody biomass estimates",
          dataset: "Chloris Biomass",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-59.7980%2C-5.0908&z=4.89&v=2&d=chloris-biomass&m=All+years&r=Aboveground+Biomass+%28tonnes%29&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "Sample the aboveground biomass value in tonnes/hectare at this forest location.",
          screenshotQuery: "Describe the biomass distribution visible. Where are the highest carbon stocks?"
        },
        {
          query: "Show modis vedgetation indices for Ukraine",
          description: "250m NDVI and EVI vegetation indices, 16-day composite",
          dataset: "MODIS 13Q1",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=31.9567%2C49.4227&z=9.82&v=2&d=modis-13Q1-061&m=Most+recent&r=Normalized+difference+vegetation+index+%28NDVI%29+16-day&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "What are the NDVI and EVI values at this agricultural field?",
          screenshotQuery: "Describe the vegetation health patterns. Which agricultural areas look most productive?"
        },
        {
          query: "Show USDA Cropland Data Layers (CDLs) for Florida",
          description: "30m crop-specific land cover classification",
          dataset: "USDA CDL",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-85.1998%2C31.2059&z=10.11&v=2&d=usda-cdl&m=Most+recent+cropland&r=Default&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "What crop type code is at this location?",
          screenshotQuery: "What crop types and land use patterns can you identify in this agricultural map?"
        },
        {
          query: "Show recent modis nadir BDRF adjusted reflectance for Mexico",
          description: "500m nadir BRDF-adjusted reflectance, daily",
          dataset: "MODIS 43A4",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-95.5417%2C26.5068&z=6.84&v=2&d=modis-43A4-061&m=Most+recent&r=Natural+color&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "Sample the BRDF-adjusted reflectance values for bands 1-4.",
          screenshotQuery: "Describe the surface types visible in this reflectance image. Any notable features?"
        }
      ]
    },
    {
      category: "🏔️ Elevation & Buildings",
      icon: "⛰️",
      examples: [
        {
          query: "Show elevation map of Grand Canyon",
          description: "30m Copernicus Digital Elevation Model",
          dataset: "COP-DEM GLO-30",
          rasterQuery: "What is the exact elevation in meters at this point?",
          screenshotQuery: "Describe the terrain features visible. Where are the canyon walls and rim?"
        },
        {
          query: "Show ALOS World 3D-30m of Tomas de Berlanga",
          description: "30m ALOS World 3D digital surface model",
          dataset: "ALOS World 3D-30m",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-91.0202%2C-0.8630&z=14.22&v=2&d=alos-dem&m=Most+recent&r=Hillshade&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "What is the ALOS DEM elevation value at this location?",
          screenshotQuery: "What volcanic and island terrain features can you identify in the Galapagos?"
        },
        {
          query: "Show USGS 3DEP Lidar Height above Ground for New Orleans",
          description: "High-resolution lidar-derived height above ground",
          dataset: "USGS 3DEP Lidar HAG",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-90.0715%2C29.9511&z=12.83&v=2&d=3dep-lidar-hag&m=Most+recent&r=Height+Above+Ground&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "What is the height above ground (HAG) in meters at this building location?",
          screenshotQuery: "What building heights and urban structures are visible in this LiDAR data?"
        },
        {
          query: "Show USGS 3DEP Lidar Height above Ground for Denver, Colorado",
          description: "High-resolution lidar-derived height above ground",
          dataset: "USGS 3DEP Lidar HAG",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-104.9903%2C39.7392&z=12.70&v=2&d=3dep-lidar-hag&m=Most+recent&r=Height+Above+Ground&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "Sample the LiDAR HAG raster. Is this a building (HAG > 3m) or ground level?",
          screenshotQuery: "Describe the building distribution and vegetation heights visible in Denver."
        }
      ]
    },
    {
      category: "📡 Radar & Reflectance",
      icon: "⛰️",
      examples: [
        {
          query: "Show Sentinel 1 RTC for Baltimore",
          description: "10m SAR backscatter radiometrically terrain corrected",
          dataset: "Sentinel-1 RTC",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-76.6287%2C39.2547&z=12.18&v=2&d=sentinel-1-rtc&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0&m=Most+recent+-+VV%2C+VH&r=VV%2C+VH+False-color+composite",
          rasterQuery: "What are the VV and VH backscatter values in dB?",
          screenshotQuery: "What surface types can you identify from the radar backscatter? Urban vs water areas?"
        },
        {
          query: "Show ALOS PALSAR Annual for Ecuador",
          description: "25m L-band SAR annual mosaic",
          dataset: "ALOS PALSAR Annual Mosaic",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=-78.3799%2C-1.4259&z=9.04&v=2&d=alos-palsar-mosaic&m=All+years&r=False-color+composite&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "Sample the HH and HV polarization values.",
          screenshotQuery: "Describe the land cover patterns visible in this L-band radar image of Ecuador."
        },
        {
          query: "Show Sentinel 1 Radiometrically Terrain Corrected (RTC) for Philippines",
          description: "10m SAR backscatter radiometrically terrain corrected",
          dataset: "Sentinel-1 RTC",
          pc_link: "https://planetarycomputer.microsoft.com/explore?c=124.1929%2C9.9849&z=7.94&v=2&d=sentinel-1-rtc&m=Most+recent+-+VV%2C+VH&r=VV%2C+VH+False-color+composite&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0",
          rasterQuery: "What is the SAR backscatter at this location?",
          screenshotQuery: "What flood or water patterns can you identify in this radar imagery of the Philippines?"
        }
      ]
    }
  ];

  // Handler for STAC search queries (Step 1) - clears all GEOINT sessions
  const handleStacQueryClick = (query: string) => {
    console.log('🚀 GetStartedButton: STAC query clicked:', query);
    
    // Close the modal
    setShowModal(false);
    
    // If onQuerySelect callback provided (from Landing Page), use it
    if (onQuerySelect) {
      console.log('✅ GetStartedButton: Using onQuerySelect callback');
      onQuerySelect(query);
      return;
    }
    
    // Dispatch STAC query event - this clears all GEOINT sessions
    setTimeout(() => {
      console.log('📤 GetStartedButton: Dispatching earthcopilot-stac-query event (clears sessions)');
      const event = new CustomEvent('earthcopilot-stac-query', { 
        detail: { query, clearSessions: true },
        bubbles: true,
        composed: true
      });
      window.dispatchEvent(event);
    }, 150);
  };

  // Handler for Raster Analysis queries (Step 2a) - uses sample_raster_value tool
  // REQUIRES: Vision module ON + pin dropped + STAC data loaded
  const handleRasterQueryClick = (query: string) => {
    console.log('🚀 GetStartedButton: Raster query clicked:', query);
    
    // Close the modal
    setShowModal(false);
    
    // Dispatch query event with raster analysis type hint and validation requirements
    setTimeout(() => {
      console.log('📤 GetStartedButton: Dispatching earthcopilot-query event (raster)');
      const event = new CustomEvent('earthcopilot-query', { 
        detail: { 
          query, 
          analysisType: 'raster',
          requiresVision: true,  // Must have Vision module ON
          requiresPin: true,     // Must have a pin dropped
          requiresStacData: true // Must have STAC tiles loaded (Step 1 completed)
        },
        bubbles: true,
        composed: true
      });
      window.dispatchEvent(event);
    }, 150);
  };

  // Handler for Image/Screenshot Analysis queries (Step 2b) - uses analyze_screenshot tool
  // REQUIRES: Vision module ON + pin dropped + STAC data loaded
  const handleScreenshotQueryClick = (query: string) => {
    console.log('🚀 GetStartedButton: Screenshot query clicked:', query);
    
    // Close the modal
    setShowModal(false);
    
    // Dispatch query event with screenshot analysis type hint and validation requirements
    setTimeout(() => {
      console.log('📤 GetStartedButton: Dispatching earthcopilot-query event (screenshot)');
      const event = new CustomEvent('earthcopilot-query', { 
        detail: { 
          query, 
          analysisType: 'screenshot',
          requiresVision: true,  // Must have Vision module ON
          requiresPin: true,     // Must have a pin dropped  
          requiresStacData: true // Must have STAC tiles loaded (Step 1 completed)
        },
        bubbles: true,
        composed: true
      });
      window.dispatchEvent(event);
    }, 150);
  };

  // Handler for Terrain/Mobility queries (generic vision queries without specific tool hints)
  const handleVisionQueryClick = (query: string) => {
    console.log('🚀 GetStartedButton: Vision query clicked:', query);
    
    // Close the modal
    setShowModal(false);
    
    // Dispatch query event without specific analysis type
    setTimeout(() => {
      console.log('📤 GetStartedButton: Dispatching earthcopilot-query event (vision)');
      const event = new CustomEvent('earthcopilot-query', { 
        detail: { query },
        bubbles: true,
        composed: true
      });
      window.dispatchEvent(event);
    }, 150);
  };

  return (
    <>
      <div
        onClick={() => setShowModal(true)}
        className="get-started-button"
        title="Get Started Guide"
      >
        <span className="get-started-button-label">🔭 Get Started</span>
      </div>

      {showModal && (
        <div className="get-started-modal-overlay" onClick={() => setShowModal(false)}>
          <div className="get-started-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="get-started-modal-header">
              <h2>🚀 Get Started with Earth Copilot</h2>
              <button 
                onClick={() => setShowModal(false)} 
                className="get-started-modal-close"
                title="Close"
              >
                ×
              </button>
            </div>

            <div className="get-started-modal-body">
              {/* Example Queries - Two Column Layout */}
              <section className="get-started-section">
                <h3>📝 Try These Example Queries</h3>
                
                <div className="instructions-box">
                  <p className="instruction-step">
                    <strong>Step 1:</strong> Click the <button className="copy-query-btn" style={{cursor: 'default', pointerEvents: 'none'}}>▶️ Go</button> button for a query in the STAC Search column to load satellite data on the map.
                  </p>
                  <p className="instruction-step">
                    <strong>Step 2:</strong> After the map loads, click the Vision module <span className="geoint-icon-inline" title="Geointelligence Modules"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg></span> and drop a pin at the desired location. Then click <button className="copy-query-btn" style={{cursor: 'default', pointerEvents: 'none'}}>▶️ Go</button> on the corresponding vision query to analyze the underlying data at those coordinates.
                  </p>
                </div>

                {/* Tab Navigation for different module types */}
                <div className="module-tabs">
                  <button 
                    className={`module-tab ${activeTab === 'stac' ? 'active' : ''}`}
                    onClick={() => setActiveTab('stac')}
                  >
                    Vision
                  </button>
                  <button 
                    className="module-tab disabled"
                    disabled
                    title="Coming Soon"
                  >
                    Terrain Analysis
                    <span className="coming-soon-badge">Coming Soon</span>
                  </button>
                  <button 
                    className="module-tab disabled"
                    disabled
                    title="Coming Soon"
                  >
                    Mobility Assessment
                    <span className="coming-soon-badge">Coming Soon</span>
                  </button>
                </div>

                {/* STAC + Vision Tab Content */}
                {activeTab === 'stac' && (
                  <>
                    {/* Mobile Column Toggle */}
                    <div className="column-toggle-mobile">
                      <button 
                        className={`toggle-btn ${activeColumn === 'stac' ? 'active' : ''}`}
                        onClick={() => setActiveColumn('stac')}
                      >
                        🛰️ Step 1: STAC Search
                      </button>
                      <button 
                        className={`toggle-btn ${activeColumn === 'vision' ? 'active' : ''}`}
                        onClick={() => setActiveColumn('vision')}
                      >
                        🔭 Step 2: Vision Module
                      </button>
                    </div>

                    {/* Two Column Headers */}
                    <div className="three-column-header">
                      <div className="column-header stac-header">
                        <div className="column-header-row">
                          <span className="column-title">Step 1: STAC Search</span>
                        </div>
                      </div>
                      <div className="column-header vision-module-header">
                        <div className="column-header-row">
                          <span className="column-title vision-module-title">Step 2: Vision Module</span>
                        </div>
                        <div className="vision-sub-headers">
                          <div className="sub-header raster-sub">
                            <span className="sub-header-title">Raster Analysis</span>
                          </div>
                          <div className="sub-header screenshot-sub">
                            <span className="sub-header-title">Image Analysis</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {exampleQueries.map((category) => (
                      <div key={category.category} className="example-category">
                        <h4 className="category-title">{category.category}</h4>
                        
                        {/* Row-based layout: each example pair is a row with 3 columns */}
                        <div className="example-rows">
                          {category.examples.map((example, index) => (
                            <div key={`row-${index}`} className="example-row three-col">
                              {/* STAC Search Card */}
                              <div className={`example-card stac-card ${activeColumn === 'stac' ? 'active' : ''}`}>
                                <div className="example-query">
                                  <strong>"{example.query}"</strong>
                                </div>
                                <div className="example-description">
                                  {example.description}
                                </div>
                                <div className="example-meta">
                                  <span className="example-dataset">📊 {example.dataset}</span>
                                  <div className="example-buttons">
                                    <button
                                      className="copy-query-btn"
                                      onClick={() => handleStacQueryClick(example.query)}
                                      title="Run this query in Earth Copilot"
                                    >
                                      ▶️ Go
                                    </button>
                                    {example.pc_link && (
                                      <button
                                        className="pc-explorer-btn"
                                        onClick={() => window.open(example.pc_link, '_blank')}
                                        title="View in Planetary Computer Explorer"
                                      >
                                        🌍 PC
                                      </button>
                                    )}
                                  </div>
                                </div>
                              </div>
                              
                              {/* Raster Analysis Card (Step 2a) */}
                              <div className={`example-card raster-card ${activeColumn === 'vision' ? 'active' : ''}`}>
                                <div className="example-query">
                                  <strong>"{example.rasterQuery || 'Coming soon...'}"</strong>
                                </div>
                                <div className="example-meta">
                                  <div className="example-buttons">
                                    <button
                                      className="copy-query-btn"
                                      onClick={() => handleRasterQueryClick(example.rasterQuery || '')}
                                      title="Run this Raster query"
                                      disabled={!example.rasterQuery}
                                    >
                                      ▶️ Go
                                    </button>
                                  </div>
                                </div>
                              </div>
                              
                              {/* Image Analysis Card (Step 2b) */}
                              <div className={`example-card screenshot-card ${activeColumn === 'vision' ? 'active' : ''}`}>
                                <div className="example-query">
                                  <strong>"{example.screenshotQuery || 'Coming soon...'}"</strong>
                                </div>
                                <div className="example-meta">
                                  <div className="example-buttons">
                                    <button
                                      className="copy-query-btn"
                                      onClick={() => handleScreenshotQueryClick(example.screenshotQuery || '')}
                                      title="Run this Image Analysis query"
                                      disabled={!example.screenshotQuery}
                                    >
                                      ▶️ Go
                                    </button>
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </>
                )}

                {/* Terrain Analysis Tab Content */}
                {activeTab === 'terrain' && (
                  <div className="terrain-queries-section">
                    <p className="section-description">
                      Test the Terrain Agent tools: elevation analysis, slope calculation, aspect analysis, flat area detection, flood risk, and environmental sensitivity.
                    </p>
                    <div className="terrain-queries-grid">
                      {terrainQueries.map((query, index) => (
                        <div key={`terrain-${index}`} className="example-card terrain-card">
                          <div className="query-location">📍 {query.location}</div>
                          <div className="setup-query">
                            <span className="query-label">1. Setup:</span>
                            <strong>"{query.setupQuery}"</strong>
                            <button
                              className="copy-query-btn"
                              onClick={() => handleStacQueryClick(query.setupQuery)}
                              title="Load the map first"
                            >
                              ▶️ Go
                            </button>
                          </div>
                          <div className="terrain-question">
                            <span className="query-label">2. Analyze:</span>
                            <strong>"{query.question}"</strong>
                            <button
                              className="copy-query-btn"
                              onClick={() => handleVisionQueryClick(query.question)}
                              title="Run terrain analysis"
                            >
                              ▶️ Go
                            </button>
                          </div>
                          <div className="expected-tools">
                            <span className="tools-label">Expected Tools:</span>
                            {query.expectedTools.map((tool, i) => (
                              <span key={i} className="tool-tag">{tool}</span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Mobility Assessment Tab Content */}
                {activeTab === 'mobility' && (
                  <div className="mobility-queries-section">
                    <p className="section-description">
                      Test the Mobility Agent: GO/SLOW-GO/NO-GO classification using Sentinel-1 SAR, DEM slope, NDVI vegetation, and MODIS fire data.
                    </p>
                    <div className="mobility-queries-grid">
                      {mobilityQueries.map((query, index) => (
                        <div key={`mobility-${index}`} className="example-card mobility-card">
                          <div className="query-location">📍 {query.location}</div>
                          <div className="analysis-type">
                            <span className={`analysis-badge ${query.analysisType.toLowerCase().replace(/\s+/g, '-')}`}>
                              {query.analysisType}
                            </span>
                          </div>
                          <div className="setup-query">
                            <span className="query-label">1. Setup:</span>
                            <strong>"{query.setupQuery}"</strong>
                            <button
                              className="copy-query-btn"
                              onClick={() => handleStacQueryClick(query.setupQuery)}
                              title="Load the map first"
                            >
                              ▶️ Go
                            </button>
                          </div>
                          <div className="mobility-question">
                            <span className="query-label">2. Analyze:</span>
                            <strong>"{query.question}"</strong>
                            <button
                              className="copy-query-btn"
                              onClick={() => handleVisionQueryClick(query.question)}
                              title="Run mobility analysis"
                            >
                              ▶️ Go
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Pro Tip - at bottom */}
                <div className="zoom-tip" style={{ marginTop: '24px' }}>
                  <span className="zoom-tip-icon">🔍</span>
                  <div className="zoom-tip-content">
                    <strong>Pro Tip:</strong> Some satellite collections (especially MODIS fire data) only display tiles at deeper zoom levels. 
                    Try zooming to <strong>level 10+</strong> and panning around the map to see all available tiles. Gray tiles represent clouds.
                  </div>
                </div>
              </section>

              {/* Learn More */}
              <section className="get-started-section learn-more-section">
                <h3>📚 Want to Learn More?</h3>
                <div className="learn-more-grid">
                  <div className="learn-more-card">
                    <div className="learn-more-icon">📊</div>
                    <h4>Browse STAC Collections</h4>
                    <p>
                      Explore all 68 available datasets, their parameters, and visualization options
                    </p>
                    <button 
                      className="learn-more-btn"
                      onClick={() => {
                        setShowModal(false);
                        // Trigger the STAC Data button click
                        setTimeout(() => {
                          const stacButton = document.querySelector('.stac-info-button') as HTMLButtonElement;
                          if (stacButton) stacButton.click();
                        }, 100);
                      }}
                    >
                      View STAC Collections →
                    </button>
                  </div>

                  <div className="learn-more-card">
                    <div className="learn-more-icon">💚</div>
                    <h4>Check System Health</h4>
                    <p>
                      Monitor backend connectivity, API status, and service availability
                    </p>
                    <button 
                      className="learn-more-btn"
                      onClick={() => {
                        setShowModal(false);
                        setTimeout(() => {
                          const healthButton = document.querySelector('.health-button') as HTMLButtonElement;
                          if (healthButton) healthButton.click();
                        }, 100);
                      }}
                    >
                      View System Status →
                    </button>
                  </div>
                </div>
              </section>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default GetStartedButton;
