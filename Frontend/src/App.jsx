import React, { useState, useEffect } from 'react';
import NetworkGraph from './components/NetworkGraph';
import ControlPanel from './components/ControlPanel';
import SimulationResults from './components/SimulationResults';
import { getBanks, getNetwork, getStocks, simulateBankShock, simulateStockShock, resetSimulation } from './api';
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

  // Load initial data
  useEffect(() => {
    async function loadData() {
      try {
        setInitializing(true);
        setError(null);
        const [banksData, networkDataRes, stocksData] = await Promise.all([
          getBanks(),
          getNetwork(),
          getStocks(),
        ]);
        setBanks(banksData);
        setNetworkData(networkDataRes);
        setStocks(stocksData);
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

  const handleReset = async () => {
    try {
      setLoading(true);
      setError(null);
      setSimResult(null);
      await resetSimulation();
      const [banksData, networkDataRes, stocksData] = await Promise.all([
        getBanks(),
        getNetwork(),
        getStocks(),
      ]);
      setBanks(banksData);
      setNetworkData(networkDataRes);
      setStocks(stocksData);
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
          <button className="reset-btn" onClick={handleReset} disabled={loading}>
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
