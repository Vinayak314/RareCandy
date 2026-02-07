import React from 'react';
import NetworkGraph from './components/NetworkGraph';
import { mockGraphData } from './data/mockData';
import './styles/theme.css';

function App() {
  return (
    <div className="dashboard-container" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header style={{ padding: '1rem 2rem', borderBottom: '2px solid var(--color-primary-red)' }}>
        <h1>Financial Network Simulation</h1>
        <p style={{ fontWeight: 'bold' }}>Systemic Risk & Liquidity Dashboard</p>
      </header>

      <main style={{ flex: 1, padding: '1rem', display: 'flex', gap: '1rem' }}>
        <div className="graph-panel" style={{ flex: 3, border: '1px solid #ddd', borderRadius: '8px', overflow: 'hidden', minHeight: '400px' }}>
          <NetworkGraph data={mockGraphData} />
        </div>

        <div className="metrics-panel" style={{ flex: 1, padding: '1rem', backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #ddd' }}>
          <h2>Network Stats</h2>
          <div style={{ marginBottom: '1rem' }}>
            <strong>Active Nodes:</strong> {mockGraphData.nodes.length}
          </div>
          <div style={{ marginBottom: '1rem' }}>
            <strong>Total Exposures:</strong> {mockGraphData.links.length}
          </div>
          <hr style={{ borderColor: 'var(--color-primary-red)' }} />
          <h3>Controls</h3>
          <button style={{
            backgroundColor: 'var(--color-primary-red)',
            color: 'white',
            border: 'none',
            padding: '0.5rem 1rem',
            cursor: 'pointer',
            fontWeight: 'bold',
            marginTop: '1rem',
            width: '100%'
          }}>
            Refresh Simulation
          </button>
        </div>
      </main>
    </div>
  );
}

export default App;
