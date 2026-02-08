import React, { useState, useEffect } from 'react';
import NetworkGraph from './components/NetworkGraph';
import ControlPanel from './components/ControlPanel';
import SimulationResults from './components/SimulationResults';
import ForecastResults from './components/ForecastResults';
import { getBanks, getNetwork, getStocks, getAllStocks, simulateBankShock, simulateStockShock, resetSimulation } from './api';
import './styles/theme.css';
import './App.css';

function App() {
  const [banks, setBanks] = useState([]);
  const [stocks, setStocks] = useState([]);
  const [allStocks, setAllStocks] = useState([]);  // All 965 stocks for shock selection
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
        const [banksData, networkDataRes, stocksData, allStocksData] = await Promise.all([
          getBanks(),
          getNetwork(),
          getStocks(),
          getAllStocks(),
        ]);
        setBanks(banksData);
        setNetworkData(networkDataRes);
        setStocks(stocksData);
        setAllStocks(allStocksData);
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

      const isForecast = config.simulationType === 'forecast';

      if (config.type === 'bank') {
        if (isForecast) {
          // Import dynamically to avoid issues
          const { forecastBankShock } = await import('./api');
          result = await forecastBankShock(config.bank, config.shock_pct, config.failure_threshold);
        } else {
          result = await simulateBankShock(config.bank, config.shock_pct, config.failure_threshold);
        }
      } else {
        if (isForecast) {
          const { forecastStockShock } = await import('./api');
          result = await forecastStockShock(config.shocks, config.failure_threshold);
        } else {
          result = await simulateStockShock(config.shocks, config.failure_threshold);
        }
      }

      setSimResult(result);

      // Update network graph with post-simulation state (only for instant simulations)
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
      const [banksData, networkDataRes, stocksData, allStocksData] = await Promise.all([
        getBanks(),
        getNetwork(),
        getStocks(),
        getAllStocks(),
      ]);
      setBanks(banksData);
      setNetworkData(networkDataRes);
      setStocks(stocksData);
      setAllStocks(allStocksData);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (initializing) {
    return (
      <div className="loading-screen">
        <div className="loading-brand">
          <h1 className="brand-title">MARGIN CALL</h1>
          <p className="brand-tagline">Spanking the Banking System</p>
        </div>
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
          <h1 className="brand-title">MARGIN CALL</h1>
          <p className="brand-tagline">Spanking the Banking System <span className="tagline-separator">|</span> <span className="tagline-subtitle">Systemic Risk & Contagion Dashboard</span></p>
        </div>
        <div className="header-right">
          <span className="stat">{banks.length} Banks</span>
          <span className="stat">{allStocks.length} Stocks Available</span>
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
            allStocks={allStocks}
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
            // Check if it's a forecast result (has forecasts property)
            simResult.forecasts ? (
              <ForecastResults result={simResult} />
            ) : (
              <SimulationResults result={simResult} />
            )
          ) : (
            <div className="placeholder-panel">
              <h2>Results</h2>
              <p>Configure a shock and run the simulation to see contagion effects.</p>
              <p style={{ fontSize: '0.75rem', color: '#888', marginTop: '0.5rem' }}>
                <strong>Instant:</strong> Immediate shock impact<br />
                <strong>Forecast:</strong> Projects impact over 1mo, 3mo, 6mo, 1yr
              </p>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}

export default App;
