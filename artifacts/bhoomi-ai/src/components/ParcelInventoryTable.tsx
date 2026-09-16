import { useState } from 'react';
import { FeatureCollection, Feature, Polygon } from 'geojson';
import { ParcelProperties } from '../utils/parcelGenerator';
import { ArrowUpDown, Search } from 'lucide-react';

interface ParcelInventoryTableProps {
  geoData: FeatureCollection<Polygon, ParcelProperties> | null;
  searchQuery: string;
}

type SortKey = 'ulpin' | 'area_sqm' | 'owner_name' | 'confidence_score' | 'ownership_status';

export function ParcelInventoryTable({ geoData, searchQuery }: ParcelInventoryTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('ulpin');
  const [sortAsc, setSortAsc] = useState(true);

  if (!geoData || !geoData.features) {
    return <div className="p-4 text-center text-slate-500">No data available.</div>;
  }

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  // owner_name/ownership_status are genuinely null when geocadastra has no
  // real value for them -- matched as empty strings here, not skipped
  // with a crash the way `.toLowerCase()` on null used to.
  const filteredFeatures = geoData.features.filter((f) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    const props = f.properties;
    return (
      props.ulpin.toLowerCase().includes(q) ||
      (props.owner_name ?? '').toLowerCase().includes(q) ||
      (props.ownership_status ?? '').toLowerCase().includes(q)
    );
  });

  const sortedFeatures = [...filteredFeatures].sort((a, b) => {
    const aVal = a.properties[sortKey] ?? '';
    const bVal = b.properties[sortKey] ?? '';

    if (aVal < bVal) return sortAsc ? -1 : 1;
    if (aVal > bVal) return sortAsc ? 1 : -1;
    return 0;
  });

  const getStatusColor = (status: string | null) => {
    if (status === 'Disputed') return 'status-disputed';
    if (status === 'Government') return 'status-gov';
    if (status === null) return 'status-unknown';
    return 'status-private';
  };

  return (
    <div className="table-wrap" style={{ margin: 0, height: '100%', overflowY: 'auto' }}>
      <table className="data-table" style={{ width: '100%', textAlign: 'left' }}>
        <thead style={{ position: 'sticky', top: 0, background: '#f8fafc', zIndex: 1 }}>
          <tr>
            <th onClick={() => handleSort('ulpin')} className="sortable-th">
              ULPIN <ArrowUpDown size={12} />
            </th>
            <th onClick={() => handleSort('owner_name')} className="sortable-th">
              Owner Name <ArrowUpDown size={12} />
            </th>
            <th onClick={() => handleSort('area_sqm')} className="sortable-th">
              Area (sq.m) <ArrowUpDown size={12} />
            </th>
            <th onClick={() => handleSort('ownership_status')} className="sortable-th">
              Status <ArrowUpDown size={12} />
            </th>
            <th onClick={() => handleSort('confidence_score')} className="sortable-th">
              Confidence <ArrowUpDown size={12} />
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedFeatures.map((f, i) => (
            <tr key={f.properties.ulpin + i}>
              <td style={{ fontFamily: 'var(--app-font-mono)', fontSize: '12px', color: 'var(--navy)' }}>
                {f.properties.ulpin}
              </td>
              <td>{f.properties.owner_name ?? <span style={{ color: '#94a3b8' }}>Not tracked</span>}</td>
              <td style={{ fontFamily: 'var(--app-font-mono)', fontSize: '12px' }}>
                {f.properties.area_sqm.toLocaleString()}
              </td>
              <td>
                <span className={`status-pill ${getStatusColor(f.properties.ownership_status)}`}>
                  {f.properties.ownership_status ?? 'Not tracked'}
                </span>
              </td>
              <td>
                {f.properties.confidence_score == null ? (
                  <span style={{ fontSize: '11px', color: '#94a3b8' }}>Not tracked</span>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <div style={{ width: '40px', height: '4px', background: '#e2e8f0', borderRadius: '2px' }}>
                      <div
                        style={{
                          width: `${f.properties.confidence_score}%`,
                          height: '100%',
                          background: '#059669',
                          borderRadius: '2px',
                        }}
                      />
                    </div>
                    <span style={{ fontSize: '11px', color: '#64748b' }}>{f.properties.confidence_score}%</span>
                  </div>
                )}
              </td>
            </tr>
          ))}
          {sortedFeatures.length === 0 && (
            <tr>
              <td colSpan={5} style={{ textAlign: 'center', padding: '30px', color: '#64748b' }}>
                No parcels found matching "{searchQuery}"
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
