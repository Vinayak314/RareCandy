import React, { useRef, useEffect, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const NetworkGraph = ({ data, onNodeClick }) => {
  const fgRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const containerRef = useRef(null);

  useEffect(() => {
    const update = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };
    window.addEventListener('resize', update);
    update();
    return () => window.removeEventListener('resize', update);
  }, []);

  const nodeColor = useCallback((node) => {
    if (node.failed) return '#D91A25';
    if (node.health === undefined) return '#2C2C2C';
    if (node.health >= 60) return '#4CAF50';
    if (node.health >= 30) return '#FF9800';
    return '#D91A25';
  }, []);

  const paintNode = useCallback((node, ctx, globalScale) => {
    const r = node.val ? Math.sqrt(node.val) * 2 : 5;
    const fontSize = Math.max(10 / globalScale, 2);

    // Glow effect for failed banks
    if (node.failed) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 4, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(217, 26, 37, 0.3)';
      ctx.fill();
    }

    // Main circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = nodeColor(node);
    ctx.fill();

    // Border
    ctx.strokeStyle = node.failed ? '#D91A25' : '#555';
    ctx.lineWidth = node.failed ? 2 : 0.5;
    ctx.stroke();

    // Label
    ctx.font = `${node.failed ? 'bold ' : ''}${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#1A1A1A';
    ctx.fillText(node.name, node.x, node.y + r + fontSize);
  }, [nodeColor]);

  const handleNodeClick = useCallback((node) => {
    if (onNodeClick) onNodeClick(node);
    if (fgRef.current) {
      fgRef.current.centerAt(node.x, node.y, 400);
      fgRef.current.zoom(3, 400);
    }
  }, [onNodeClick]);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '500px' }}>
      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={data}
        nodeLabel={node =>
          `${node.name}\nAssets: $${(node.total_assets || 0).toFixed(1)}B\nHealth: ${(node.health || 0).toFixed(1)}`
        }
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        linkWidth={link => Math.max(0.5, Math.sqrt(link.value || 1) * 0.5)}
        linkColor={() => 'rgba(100,100,100,0.3)'}
        nodeCanvasObject={paintNode}
        onNodeClick={handleNodeClick}
        backgroundColor="#F4EFEC"
        cooldownTicks={100}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
      />
    </div>
  );
};

export default NetworkGraph;
