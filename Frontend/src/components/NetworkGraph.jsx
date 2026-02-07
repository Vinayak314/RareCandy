import React, { useRef, useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const NetworkGraph = ({ data }) => {
    const fgRef = useRef();
    const [containerDimensions, setContainerDimensions] = useState({ width: 800, height: 600 });
    const containerRef = useRef(null);

    useEffect(() => {
        // Responsive graph container
        const updateDimensions = () => {
            if (containerRef.current) {
                setContainerDimensions({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight
                });
            }
        };

        window.addEventListener('resize', updateDimensions);
        updateDimensions();

        return () => window.removeEventListener('resize', updateDimensions);
    }, []);

    return (
        <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '500px' }}>
            <ForceGraph2D
                ref={fgRef}
                width={containerDimensions.width}
                height={containerDimensions.height}
                graphData={data}
                nodeLabel="name"
                nodeAutoColorBy="group"
                linkDirectionalParticles={2}
                linkDirectionalParticleSpeed={d => d.value * 0.001}
                nodeCanvasObject={(node, ctx, globalScale) => {
                    const label = node.name;
                    const fontSize = 12 / globalScale;
                    ctx.font = `${fontSize}px Sans-Serif`;
                    const textWidth = ctx.measureText(label).width;
                    const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2); // some padding

                    // Draw Node Circle
                    ctx.beginPath();
                    const r = node.val ? Math.sqrt(node.val) * 2 : 5;
                    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);

                    // Color based on group (using CSS variables roughly)
                    // Since canvas can't read CSS vars easily here without helper, hardcode theme colors for now or use node color
                    if (node.id === 'CCP') {
                        ctx.fillStyle = '#D91A25'; // Primary Red
                    } else {
                        ctx.fillStyle = '#2C2C2C'; // Dark
                    }
                    ctx.fill();

                    // Draw Label
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillStyle = '#000';
                    ctx.fillText(label, node.x, node.y + r + fontSize);
                }}
                backgroundColor="#F4EFEC" // Match theme bg
            />
        </div>
    );
};

export default NetworkGraph;
