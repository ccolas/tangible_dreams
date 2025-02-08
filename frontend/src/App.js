import React, { useState, useEffect } from 'react';

const Node = ({ id, type, label, position, nodes, connections, onConnect, onStateChange, gain1: initialGain1, gain2: initialGain2 }) => {
  const [gain1, setGain1] = useState(initialGain1 || 0.5);
  const [gain2, setGain2] = useState(initialGain2 || 0.5);
  const [activation, setActivation] = useState('tanh');

  const handleGain1Change = (value) => {
    setGain1(value);
    onStateChange({
      id,
      type,
      gain1: value,
      gain2,
      activation
    });
  };

  const handleGain2Change = (value) => {
    setGain2(value);
    onStateChange({
      id,
      type,
      gain1,
      gain2: value,
      activation
    });
  };

  useEffect(() => {
      onStateChange && onStateChange({
          id, 
          type,
          gain1,
          gain2,
          activation
      });
  }, [gain1, gain2, activation, id, type, onStateChange]);

  // Update parent when state changes
  useEffect(() => {
    onStateChange && onStateChange({
      id,
      type,
      gain1,
      gain2,
      activation
    });
  }, [gain1, gain2, activation]);

  // Get available outputs (all nodes that come before this one)
  const availableOutputs = nodes.filter(n => n.id < id);

  return (
    <div
      className="absolute p-3 bg-gray-800 border border-gray-600 rounded-lg shadow-lg"
      style={{
        left: position.x,
        top: position.y,
        width: '200px'
      }}
    >
      <div className="text-sm font-semibold mb-2 text-gray-300">
        {type} Node {id} {label ? `(${label})` : ''}
      </div>

      {/* Input selection dropdowns */}
      {type !== 'input' && (
        <div className="space-y-2">
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Input 1</label>
            <select
              className="w-full bg-gray-700 text-xs p-1 rounded border border-gray-600 text-gray-300"
              value={connections.find(c => c.to === id && c.port === 'input1')?.from || ''}
              onChange={(e) => onConnect(parseInt(e.target.value), id, 'input1')}
            >
              <option value="">Select input</option>
              {availableOutputs.map(node => (
                <option key={node.id} value={node.id}>
                  {node.type} Node {node.id} {node.label ? `(${node.label})` : ''}
                </option>
              ))}
            </select>
            {type === 'middle' && (
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={gain1}
                onChange={(e) => setGain1(parseFloat(e.target.value))}
                className="w-full"
              />
            )}
          </div>

          <div className="space-y-1">
            <label className="text-xs text-gray-400">Input 2</label>
            <select
              className="w-full bg-gray-700 text-xs p-1 rounded border border-gray-600 text-gray-300"
              value={connections.find(c => c.to === id && c.port === 'input2')?.from || ''}
              onChange={(e) => onConnect(parseInt(e.target.value), id, 'input2')}
            >
              <option value="">Select input</option>
              {availableOutputs.map(node => (
                <option key={node.id} value={node.id}>
                  {node.type} Node {node.id} {node.label ? `(${node.label})` : ''}
                </option>
              ))}
            </select>
            {type === 'middle' && (
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={gain2}
                onChange={(e) => setGain2(parseFloat(e.target.value))}
                className="w-full"
              />
            )}
          </div>
        </div>
      )}

      {type === 'middle' && (
        <div className="mt-2">
          <select
            value={activation}
            onChange={(e) => setActivation(e.target.value)}
            className="w-full bg-gray-700 text-xs p-1 rounded border border-gray-600 text-gray-300"
          >
            <option value="tanh">tanh</option>
            <option value="sigmoid">sigmoid</option>
            <option value="relu">ReLU</option>
          </select>
        </div>
      )}
    </div>
  );
};

const OutputVisualization = ({ nodes, connections }) => {
  const [imageData, setImageData] = useState(null);

  useEffect(() => {
    const computeOutput = async () => {
      try {
        // Prepare the network state
        const networkState = {
          nodes: nodes.map(node => ({
            id: node.id,
            type: node.type,
            gain1: node.gain1 || 0.5,
            gain2: node.gain2 || 0.5,
            activation: node.activation || 'tanh'
          })),
          connections: connections.map(conn => ({
            from_id: conn.from,
            to: conn.to,
            port: conn.port
          }))
        };

        const response = await fetch('http://localhost:8000/compute', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(networkState)
        });

        const data = await response.json();

        // Create canvas and draw the image
        const width = 200;
        const height = 200;
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(width, height);

        for(let y = 0; y < height; y++) {
          for(let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            imageData.data[idx] = data.r[y][x] * 255;     // R
            imageData.data[idx + 1] = data.g[y][x] * 255; // G
            imageData.data[idx + 2] = data.b[y][x] * 255; // B
            imageData.data[idx + 3] = 255;                 // A
          }
        }

        ctx.putImageData(imageData, 0, 0);
        setImageData(canvas.toDataURL());
      } catch (error) {
        console.error('Error computing CPPN output:', error);
      }
    };

    computeOutput();
  }, [nodes, connections]);

  return (
    <div className="absolute left-4 bottom-4 bg-gray-800 border border-gray-600 rounded-lg p-2">
      <div className="text-sm font-semibold mb-2 text-gray-300">Output</div>
      {imageData && (
        <img
          src={imageData}
          alt="CPPN Output"
          className="w-[200px] h-[200px] rounded"
        />
      )}
    </div>
  );
};


function App() {
  const [nodes, setNodes] = useState([
      { id: 1, type: 'input', position: { x: 100, y: 50 }, label: 'X', gain1: 0.5, gain2: 0.5, activation: 'tanh' },
      { id: 2, type: 'input', position: { x: 100, y: 200 }, label: 'Y', gain1: 0.5, gain2: 0.5, activation: 'tanh' },
      { id: 3, type: 'input', position: { x: 100, y: 350 }, label: 'R', gain1: 0.5, gain2: 0.5, activation: 'tanh' },
      { id: 4, type: 'middle', position: { x: 350, y: 50 }, gain1: 0.5, gain2: 0.5, activation: 'tanh' },
      { id: 5, type: 'middle', position: { x: 350, y: 350 }, gain1: 0.5, gain2: 0.5, activation: 'tanh' },
      { id: 6, type: 'middle', position: { x: 600, y: 50 }, gain1: 0.5, gain2: 0.5, activation: 'tanh' },
      { id: 7, type: 'middle', position: { x: 600, y: 350 }, gain1: 0.5, gain2: 0.5, activation: 'tanh' },
      { id: 8, type: 'output', position: { x: 850, y: 50 }, label: 'R', gain1: 0.5, gain2: 0.5, activation: 'tanh' },
      { id: 9, type: 'output', position: { x: 850, y: 200 }, label: 'G', gain1: 0.5, gain2: 0.5, activation: 'tanh' },
      { id: 10, type: 'output', position: { x: 850, y: 350 }, label: 'B', gain1: 0.5, gain2: 0.5, activation: 'tanh' }
  ]);

  const [connections, setConnections] = useState([]);

  const handleConnect = (fromId, toId, inputPort) => {
    if (!fromId) {
      // If empty selection, remove connection
      setConnections(connections.filter(
        conn => !(conn.to === toId && conn.port === inputPort)
      ));
      return;
    }

    const existingConnection = connections.find(
      conn => conn.to === toId && conn.port === inputPort
    );

    if (existingConnection) {
      // Replace existing connection
      setConnections(connections.map(conn =>
        (conn.to === toId && conn.port === inputPort)
          ? { from: fromId, to: toId, port: inputPort }
          : conn
      ));
    } else {
      // Add new connection
      setConnections([
        ...connections,
        { from: fromId, to: toId, port: inputPort }
      ]);
    }
  };

    const handleNodeStateChange = (nodeState) => {
    setNodes(nodes.map(node =>
        node.id === nodeState.id
            ? { ...node, ...nodeState }
            : node
    ));
  };
  return (
    <div className="min-h-screen bg-gray-900 p-8">
      <h1 className="text-2xl font-bold mb-8 text-gray-100">CPPN Interface</h1>
      <div className="relative w-[1100px] h-[800px] border border-gray-700 rounded-lg bg-gray-950 overflow-hidden">
        {nodes.map(node => (
            <Node
                key={node.id}
                {...node}
                nodes={nodes}
                connections={connections}
                onConnect={handleConnect}
                onStateChange={handleNodeStateChange}
            />
        ))}
        <OutputVisualization nodes={nodes} connections={connections} />
      </div>
    </div>
  );
}

export default App;