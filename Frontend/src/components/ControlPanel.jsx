import React, { useState, useMemo, useRef, useEffect } from 'react';

export default function ControlPanel({ banks, stocks, allStocks = [], onSimulate, loading }) {
  const [mode, setMode] = useState('bank'); // 'bank' | 'stock'
  const [simulationType, setSimulationType] = useState('instant'); // 'instant' | 'forecast'
  const [selectedBank, setSelectedBank] = useState('');
  const [bankShockPct, setBankShockPct] = useState(50);
  const [stockShocks, setStockShocks] = useState({});
  const [threshold, setThreshold] = useState(20);

  // Stock search state
  const [stockSearch, setStockSearch] = useState('');
  const [showStockDropdown, setShowStockDropdown] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowStockDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Filter allStocks based on search
  const filteredStocks = useMemo(() => {
    if (!stockSearch.trim()) return allStocks.slice(0, 50); // Show first 50 by default
    const search = stockSearch.toUpperCase();
    return allStocks.filter(s =>
      s.ticker.toUpperCase().includes(search)
    ).slice(0, 50); // Limit to 50 results
  }, [allStocks, stockSearch]);

  // Get selected stocks for display
  const selectedStockTickers = Object.keys(stockShocks).filter(t => stockShocks[t] > 0);

  const handleStockShockChange = (ticker, value) => {
    setStockShocks(prev => {
      const next = { ...prev };
      if (value === '' || Number(value) === 0) {
        delete next[ticker];
      } else {
        next[ticker] = Number(value);
      }
      return next;
    });
  };

  const addStockToShock = (ticker) => {
    if (!stockShocks[ticker]) {
      setStockShocks(prev => ({ ...prev, [ticker]: 20 })); // Default 20% shock
    }
    setStockSearch('');
    setShowStockDropdown(false);
  };

  const removeStockFromShock = (ticker) => {
    setStockShocks(prev => {
      const next = { ...prev };
      delete next[ticker];
      return next;
    });
  };

  const handleRun = () => {
    if (mode === 'bank') {
      if (!selectedBank) return;
      onSimulate({
        type: 'bank',
        simulationType,
        bank: selectedBank,
        shock_pct: bankShockPct,
        failure_threshold: threshold
      });
    } else {
      const activeShocks = Object.fromEntries(
        Object.entries(stockShocks).filter(([, v]) => v > 0)
      );
      if (Object.keys(activeShocks).length === 0) return;
      onSimulate({
        type: 'stock',
        simulationType,
        shocks: activeShocks,
        failure_threshold: threshold
      });
    }
  };

  return (
    <div className="control-panel">
      <h2>Shock Controls</h2>

      {/* Mode Toggle */}
      <div className="mode-toggle">
        <button
          className={mode === 'bank' ? 'active' : ''}
          onClick={() => setMode('bank')}
        >
          Bank Shock
        </button>
        <button
          className={mode === 'stock' ? 'active' : ''}
          onClick={() => setMode('stock')}
        >
          Stock Shock
        </button>
      </div>

      {/* Simulation Type Toggle */}
      <div className="simulation-type-toggle">
        <label>Simulation Type</label>
        <div className="sim-type-buttons">
          <button
            className={simulationType === 'instant' ? 'active' : ''}
            onClick={() => setSimulationType('instant')}
          >
            ⚡ Instant
          </button>
          <button
            className={simulationType === 'forecast' ? 'active' : ''}
            onClick={() => setSimulationType('forecast')}
          >
            📈 Forecast
          </button>
        </div>
        <div className="sim-type-description">
          {simulationType === 'instant'
            ? 'Immediate shock impact simulation'
            : 'Projects impact over 1mo, 3mo, 6mo, 1yr'}
        </div>
      </div>

      {/* Bank Shock Mode */}
      {mode === 'bank' && (
        <div className="shock-config">
          <label>Target Bank</label>
          <select value={selectedBank} onChange={e => setSelectedBank(e.target.value)}>
            <option value="">Select bank...</option>
            {banks.map(b => (
              <option key={b.id} value={b.id}>
                {b.id} — ${b.total_assets.toFixed(0)}B
              </option>
            ))}
          </select>

          <label>Shock Severity: {bankShockPct}%</label>
          <input
            type="range" min={5} max={95} step={5}
            value={bankShockPct}
            onChange={e => setBankShockPct(Number(e.target.value))}
          />
          <div className="range-labels">
            <span>5%</span><span>50%</span><span>95%</span>
          </div>
        </div>
      )}

      {/* Stock Shock Mode */}
      {mode === 'stock' && (
        <div className="shock-config">
          <label>Search & Add Stocks ({allStocks.length} available)</label>
          <div className="stock-search-container" ref={dropdownRef}>
            <input
              type="text"
              placeholder="Search by ticker (e.g., AAPL, MSFT)..."
              value={stockSearch}
              onChange={e => {
                setStockSearch(e.target.value);
                setShowStockDropdown(true);
              }}
              onFocus={() => setShowStockDropdown(true)}
              className="stock-search-input"
            />
            {showStockDropdown && filteredStocks.length > 0 && (
              <div className="stock-dropdown">
                {filteredStocks.map(s => (
                  <div
                    key={s.ticker}
                    className={`stock-dropdown-item ${stockShocks[s.ticker] ? 'selected' : ''}`}
                    onClick={() => addStockToShock(s.ticker)}
                  >
                    <span className="ticker">{s.ticker}</span>
                    <span className="price">${s.price.toFixed(2)}</span>
                    {stockShocks[s.ticker] && <span className="check">✓</span>}
                  </div>
                ))}
                {filteredStocks.length === 50 && (
                  <div className="stock-dropdown-hint">
                    Type to narrow results (showing first 50)
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Selected Stocks with shock values */}
          {selectedStockTickers.length > 0 && (
            <div className="selected-stocks">
              <label>Selected Stocks ({selectedStockTickers.length})</label>
              <div className="stock-shock-list">
                {selectedStockTickers.map(ticker => {
                  const stock = allStocks.find(s => s.ticker === ticker);
                  return (
                    <div key={ticker} className="stock-shock-row">
                      <span className="ticker">{ticker}</span>
                      <span className="price">${stock?.price.toFixed(2) || '?'}</span>
                      <input
                        type="number" min={1} max={95} step={5}
                        value={stockShocks[ticker] || ''}
                        onChange={e => handleStockShockChange(ticker, e.target.value)}
                      />
                      <span className="pct">%</span>
                      <button
                        className="remove-btn"
                        onClick={() => removeStockFromShock(ticker)}
                        title="Remove"
                      >
                        ×
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {selectedStockTickers.length === 0 && (
            <div className="stock-hint">
              Search and add stocks above to apply shocks
            </div>
          )}
        </div>
      )}

      {/* Threshold */}
      <div className="threshold-config">
        <label>Failure Threshold: {threshold}</label>
        <input
          type="range" min={5} max={50} step={5}
          value={threshold}
          onChange={e => setThreshold(Number(e.target.value))}
        />
      </div>

      {/* Run Button */}
      <button className="run-btn" onClick={handleRun} disabled={loading}>
        {loading ? 'Simulating...' : 'Run Simulation'}
      </button>
    </div>
  );
}
