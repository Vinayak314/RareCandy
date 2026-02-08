import React, { useState, useEffect } from 'react';
import NetworkGraph from './components/NetworkGraph';
import ControlPanel from './components/ControlPanel';
import SimulationResults from './components/SimulationResults';
import { getBanks, getNetwork, getStocks, simulateBankShock, simulateStockShock, resetSimulation, getMlStatus, getMarginRequirements, regenerateMargins } from './api';
import './styles/theme.css';
import './App.css';

function App() {
  const [banks, setBanks] = useState([]);
  const [stocks, setStocks] = useState([]);
  const [networkData, setNetworkData] = useState({ nodes: [], links: [] });
  const [simResult, setSimResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState(null);
  
  // ML Model State
  const [mlStatus, setMlStatus] = useState({ ml_available: false, ml_margins_enabled: false });
  const [marginInfo, setMarginInfo] = useState({ total_margin_B: 0, margins: [] });

  // Load initial data
  useEffect(() => {
    async function loadData() {
      try {
        setInitializing(true);
        setError(null);
        const [banksData, networkDataRes, stocksData, mlStatusData, marginData] = await Promise.all([
          getBanks(),
          getNetwork(),
          getStocks(),
          getMlStatus().catch(() => ({ ml_available: false, ml_margins_enabled: false })),
          getMarginRequirements().catch(() => ({ total_margin_B: 0, margins: [] })),
        ]);
        setBanks(banksData);
        setNetworkData(networkDataRes);
        setStocks(stocksData);
        setMlStatus(mlStatusData);
        setMarginInfo(marginData);
      } catch (e) {
        setError('Failed to connect to backend. Make sure the Flask server is running on port 5000.');
        console.error(e);
      } finally {
        setInitializing(false);
      }
    }
    loadData();
  }, []);

  const handleSimulate = async (config) => {
    try {
      setLoading(true);
      setError(null);
      let result;
      if (config.type === 'bank') {
        result = await simulateBankShock(config.bank, config.shock_pct, config.failure_threshold);
      } else {
        result = await simulateStockShock(config.shocks, config.failure_threshold);
      }

      setSimResult(result);

      // Update network graph with post-simulation state
      if (result.network) {
        setNetworkData(result.network);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async (useMlMargins = true) => {
    try {
      setLoading(true);
      setError(null);
      setSimResult(null);
      await resetSimulation(useMlMargins);
      const [banksData, networkDataRes, stocksData, mlStatusData, marginData] = await Promise.all([
        getBanks(),
        getNetwork(),
        getStocks(),
        getMlStatus().catch(() => ({ ml_available: false, ml_margins_enabled: false })),
        getMarginRequirements().catch(() => ({ total_margin_B: 0, margins: [] })),
      ]);
      setBanks(banksData);
      setNetworkData(networkDataRes);
      setStocks(stocksData);
      setMlStatus(mlStatusData);
      setMarginInfo(marginData);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleMlMargins = async () => {
    try {
      setLoading(true);
      setError(null);
      const newUseMl = !mlStatus.ml_margins_enabled;
      await regenerateMargins(newUseMl);
      const [banksData, mlStatusData, marginData] = await Promise.all([
        getBanks(),
        getMlStatus().catch(() => ({ ml_available: false, ml_margins_enabled: false })),
        getMarginRequirements().catch(() => ({ total_margin_B: 0, margins: [] })),
      ]);
      setBanks(banksData);
      setMlStatus(mlStatusData);
      setMarginInfo(marginData);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (initializing) {
    return (
      <div className="loading-screen">
        <h1>Financial Network Simulation</h1>
        <p>Connecting to simulation backend...</p>
        <div className="spinner" />
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <h1>Financial Network Simulation</h1>
          <p>Systemic Risk &amp; Contagion Dashboard</p>
        </div>
        <div className="header-right">
          <span className="stat">{banks.length} Banks</span>
          <span className="stat">{stocks.length} Stocks</span>
          <span className="stat">{networkData.links.length} Exposures</span>
          <span className="stat margin-stat" title={`Total Margin: $${marginInfo.total_margin_B?.toFixed(1) || 0}B`}>
            ${marginInfo.total_margin_B?.toFixed(0) || 0}B Margin
          </span>
          {mlStatus.ml_available && (
            <button 
              className={`ml-toggle-btn ${mlStatus.ml_margins_enabled ? 'ml-active' : ''}`}
              onClick={handleToggleMlMargins}
              disabled={loading}
              title={mlStatus.ml_margins_enabled ? 'Using ML-based margins' : 'Using random margins'}
            >
              {mlStatus.ml_margins_enabled ? '🤖 ML' : '🎲 Random'}
            </button>
          )}
          <button className="reset-btn" onClick={() => handleReset(mlStatus.ml_margins_enabled)} disabled={loading}>
            Reset
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner">{error}</div>
      )}

      {/* Main Layout */}
      <main className="main-layout">
        {/* Left: Controls */}
        <aside className="sidebar-left">
          <ControlPanel
            banks={banks}
            stocks={stocks}
            onSimulate={handleSimulate}
            loading={loading}
            mlStatus={mlStatus}
            onToggleMlMargins={handleToggleMlMargins}
            marginInfo={marginInfo}
          />
        </aside>

        {/* Center: Network Graph */}
        <section className="graph-section">
          <NetworkGraph data={networkData} />
          <div className="graph-legend">
            <span className="legend-item"><span className="dot green" /> Healthy (&gt;60)</span>
            <span className="legend-item"><span className="dot orange" /> Stressed (30-60)</span>
            <span className="legend-item"><span className="dot red" /> Critical / Failed (&lt;30)</span>
          </div>
        </section>

        {/* Right: Results */}
        <aside className="sidebar-right">
          {simResult ? (
            <SimulationResults result={simResult} />
          ) : (
            <div className="placeholder-panel">
              <h2>Results</h2>
              <p>Configure a shock and run the simulation to see contagion effects.</p>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}

export default App;
