import React, { useState } from 'react';

/**
 * Displays forecast simulation results for multiple time horizons
 * (1 month, 3 months, 6 months, 1 year)
 */
export default function ForecastResults({ result }) {
    const [selectedPeriod, setSelectedPeriod] = useState('1_month');

    // Define preferred order
    const ORDERED_PERIODS = ['1_month', '3_months', '6_months', '1_year'];

    if (!result || !result.forecasts) {
        return null;
    }

    const forecasts = result.forecasts;
    // Sort periods based on predefined order
    const periods = Object.keys(forecasts).sort((a, b) => {
        return ORDERED_PERIODS.indexOf(a) - ORDERED_PERIODS.indexOf(b);
    });

    const currentForecast = forecasts[selectedPeriod];
    const isBankShock = result.shock_type === 'BANK_SHOCK_FORECAST';

    // Helper to downsample timeline data for visualization
    const downsampleTimeline = (timeline, maxPoints = 60) => {
        if (!timeline || timeline.length <= maxPoints) return timeline;

        const step = Math.ceil(timeline.length / maxPoints);
        return timeline.filter((_, index) => index % step === 0);
    };

    const displayTimeline = currentForecast ? downsampleTimeline(currentForecast.health_timeline) : [];

    // Calculate trend indicators
    const getTrendClass = (value, baseline = 50) => {
        if (value >= 70) return 'safe';
        if (value >= 40) return 'warning';
        return 'danger';
    };

    return (
        <div className="forecast-results">
            <h2>📈 Forecast Results</h2>

            {/* Shock Summary */}
            <div className="forecast-shock-summary">
                {isBankShock ? (
                    <div className="shock-info">
                        <strong>Bank Shock:</strong> {result.target_bank} at -{result.shock_percent}%
                    </div>
                ) : (
                    <div className="shock-info">
                        <strong>Stock Shock:</strong> {result.num_stocks_shocked} stock(s) affected
                    </div>
                )}
            </div>

            {/* Period Selector */}
            <div className="period-selector">
                {periods.map(period => {
                    const f = forecasts[period];
                    return (
                        <button
                            key={period}
                            className={`period-btn ${selectedPeriod === period ? 'active' : ''} ${f.total_failures > 0 ? 'has-failures' : ''}`}
                            onClick={() => setSelectedPeriod(period)}
                        >
                            <div className="period-label">{f.period_label}</div>
                            <div className="period-days">{f.days} days</div>
                            {f.total_failures > 0 && (
                                <div className="period-failures">⚠️ {f.total_failures}</div>
                            )}
                        </button>
                    );
                })}
            </div>

            {/* Selected Period Details */}
            {currentForecast && (
                <div className="forecast-details">
                    <h3>{currentForecast.period_label} Forecast</h3>

                    {/* Key Metrics */}
                    <div className="forecast-metrics">
                        <div className={`metric-card ${getTrendClass(currentForecast.final_avg_health)}`}>
                            <div className="metric-value">{currentForecast.final_avg_health.toFixed(1)}%</div>
                            <div className="metric-label">System Health</div>
                        </div>
                        <div className={`metric-card ${currentForecast.total_failures > 0 ? 'danger' : 'safe'}`}>
                            <div className="metric-value">{currentForecast.total_failures}</div>
                            <div className="metric-label">Bank Failures</div>
                        </div>
                        <div className={`metric-card ${currentForecast.ccp_payoff_B < -10 ? 'danger' : currentForecast.ccp_payoff_B < 0 ? 'warning' : 'safe'}`}>
                            <div className="metric-value">${currentForecast.ccp_payoff_B.toFixed(1)}B</div>
                            <div className="metric-label">CCP Impact</div>
                        </div>
                    </div>

                    {/* Bank Shock Specific: Target Bank Health */}
                    {isBankShock && currentForecast.final_target_health !== undefined && (
                        <div className="target-bank-health">
                            <label>Target Bank Health Evolution</label>
                            <div className="health-bar-full">
                                <div
                                    className="health-bar-fill"
                                    style={{
                                        width: `${Math.max(0, Math.min(100, currentForecast.final_target_health))}%`,
                                        backgroundColor: currentForecast.final_target_health < 20 ? '#D91A25' :
                                            currentForecast.final_target_health < 50 ? '#FF9800' : '#4CAF50'
                                    }}
                                />
                            </div>
                            <div className="health-value">{currentForecast.final_target_health.toFixed(1)}%</div>
                        </div>
                    )}

                    {/* Stock Projections (for stock shocks) */}
                    {!isBankShock && currentForecast.stock_projections && Object.keys(currentForecast.stock_projections).length > 0 && (
                        <div className="stock-projections">
                            <h4>Stock Price Projections</h4>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Ticker</th>
                                        <th>Initial</th>
                                        <th>Shocked</th>
                                        <th>Projected</th>
                                        <th>Recovery</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(currentForecast.stock_projections).map(([ticker, proj]) => (
                                        <tr key={ticker}>
                                            <td className="ticker">{ticker}</td>
                                            <td>${proj.initial}</td>
                                            <td className="shocked">${proj.shocked}</td>
                                            <td>${proj.projected_mean}</td>
                                            <td className={proj.recovery_pct >= 0 ? 'positive' : 'negative'}>
                                                {proj.recovery_pct >= 0 ? '+' : ''}{proj.recovery_pct.toFixed(1)}%
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {/* Failed Banks List */}
                    {currentForecast.failed_banks && currentForecast.failed_banks.length > 0 && (
                        <div className="failed-banks-forecast">
                            <h4>Failed Banks</h4>
                            <div className="failed-tags">
                                {currentForecast.failed_banks.map(bank => (
                                    <span key={bank} className="failed-tag">{bank}</span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Timeline Summary */}
                    {currentForecast.health_timeline && currentForecast.health_timeline.length > 0 && (
                        <div className="timeline-summary">
                            <h4>Health Evolution</h4>
                            <div
                                className="mini-chart"
                                style={{ gap: displayTimeline.length > 30 ? '1px' : '2px' }}
                            >
                                {displayTimeline.map((point, idx) => (
                                    <div
                                        key={idx}
                                        className="chart-bar"
                                        style={{
                                            height: `${point.avg_system_health}%`,
                                            backgroundColor: point.avg_system_health < 40 ? '#D91A25' :
                                                point.avg_system_health < 60 ? '#FF9800' : '#4CAF50',
                                            minWidth: displayTimeline.length > 30 ? '2px' : '4px'
                                        }}
                                        title={`Day ${point.day}: ${point.avg_system_health.toFixed(1)}%`}
                                    />
                                ))}
                            </div>
                            <div className="chart-labels">
                                <span>Day 0</span>
                                <span>Day {currentForecast.days}</span>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Compare All Periods Summary */}
            <div className="period-comparison">
                <h4>All Periods Comparison</h4>
                <table className="comparison-table">
                    <thead>
                        <tr>
                            <th>Period</th>
                            <th>Health</th>
                            <th>Failures</th>
                            <th>CCP Impact</th>
                        </tr>
                    </thead>
                    <tbody>
                        {periods.map(period => {
                            const f = forecasts[period];
                            return (
                                <tr key={period} className={selectedPeriod === period ? 'selected' : ''}>
                                    <td>{f.period_label}</td>
                                    <td className={getTrendClass(f.final_avg_health)}>
                                        {f.final_avg_health.toFixed(1)}%
                                    </td>
                                    <td className={f.total_failures > 0 ? 'danger' : ''}>
                                        {f.total_failures}
                                    </td>
                                    <td className={f.ccp_payoff_B < -10 ? 'danger' : f.ccp_payoff_B < 0 ? 'warning' : ''}>
                                        ${f.ccp_payoff_B.toFixed(1)}B
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
