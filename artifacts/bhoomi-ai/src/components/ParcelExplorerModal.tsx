import { useState, useEffect, useMemo } from 'react';
import { X, Map as MapIcon, Table2, Search, Download, FileJson } from 'lucide-react';
import { generateMicroParcels, ParcelProperties } from '../utils/parcelGenerator';
import { MicroParcelMap } from './MicroParcelMap';
import { ParcelInventoryTable } from './ParcelInventoryTable';
import type { FeatureCollection, Polygon } from 'geojson';

interface ParcelExplorerModalProps {
  isOpen: boolean;
  onClose: () => void;
  regionData: {
    name: string;
    code: string;
    bounds?: [[number, number], [number, number]];
  };
}

export function ParcelExplorerModal({ isOpen, onClose, regionData }: ParcelExplorerModalProps) {
  const [viewMode, setViewMode] = useState<'map' | 'table'>('map');
  const [searchQuery, setSearchQuery] = useState('');
  
  // Memoize generated parcels so it doesn't regenerate on every render
  const generatedGeoData = useMemo(() => {
    if (!isOpen) return null;
    
    // Default to a central India bounding box if none provided
    const bounds = regionData.bounds || [[18.0, 72.0], [28.0, 85.0]];
    
    return generateMicroParcels(
      bounds as [[number, number], [number, number]], 
      regionData.code || 'IN', 
      300
    );
  }, [regionData.code, regionData.bounds, isOpen]);

  // Handle escape key to close
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const exportCSV = () => {
    if (!generatedGeoData) return;
    const headers = ['ULPIN', 'Owner Name', 'Area (sq.m)', 'Status', 'Confidence'];
    const rows = generatedGeoData.features.map(f => {
      const p = f.properties;
      return `${p.ulpin},"${p.owner_name}",${p.area_sqm},${p.ownership_status},${p.confidence_score}`;
    });
    
    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `parcels_${regionData.code.toLowerCase()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportGeoJSON = () => {
    if (!generatedGeoData) return;
    const blob = new Blob([JSON.stringify(generatedGeoData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `parcels_${regionData.code.toLowerCase()}.geojson`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="parcel-explorer-overlay" role="dialog" aria-modal="true">
      <div className="parcel-explorer-container">
        
        {/* Header */}
        <header className="explorer-header">
          <div>
            <div className="explorer-eyebrow">MICRO-CADASTRAL EXPLORER</div>
            <h2 className="explorer-title">{regionData.name} Jurisdiction</h2>
          </div>
          <button className="explorer-close" onClick={onClose} aria-label="Close Explorer">
            <X size={24} />
          </button>
        </header>

        {/* Toolbar */}
        <div className="explorer-toolbar">
          <div className="explorer-view-toggle">
            <button 
              className={`toggle-btn ${viewMode === 'map' ? 'active' : ''}`}
              onClick={() => setViewMode('map')}
            >
              <MapIcon size={15} /> Map View
            </button>
            <button 
              className={`toggle-btn ${viewMode === 'table' ? 'active' : ''}`}
              onClick={() => setViewMode('table')}
            >
              <Table2 size={15} /> Tabular Inventory
            </button>
          </div>

          <div className="explorer-actions">
            <div className="search-box">
              <Search size={15} className="search-icon" />
              <input 
                type="text" 
                placeholder="Search ULPIN or Owner..." 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            <button className="export-btn" onClick={exportCSV} title="Download CSV">
              <Download size={15} /> CSV
            </button>
            <button className="export-btn" onClick={exportGeoJSON} title="Download GeoJSON">
              <FileJson size={15} /> GeoJSON
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="explorer-content">
          {viewMode === 'map' ? (
            <MicroParcelMap 
              bounds={regionData.bounds || [[18.0, 72.0], [28.0, 85.0]]} 
              geoData={generatedGeoData} 
            />
          ) : (
            <ParcelInventoryTable 
              geoData={generatedGeoData} 
              searchQuery={searchQuery}
            />
          )}
        </div>

      </div>
    </div>
  );
}
