from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Connection(BaseModel):
    from_id: int
    to: int
    port: str


class Node(BaseModel):
    id: int
    type: str
    gain1: float = 0.5
    gain2: float = 0.5
    activation: str = "tanh"


class NetworkState(BaseModel):
    nodes: List[Node]
    connections: List[Connection]


def activation_function(x: np.ndarray, func_type: str) -> np.ndarray:
    if func_type == "tanh":
        return np.tanh(x)
    elif func_type == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif func_type == "relu":
        return np.maximum(0, x)
    return x


@app.post("/compute")
async def compute_output(state: NetworkState):
    # Create coordinate grid
    size = 200
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Distance from center
    R = np.sqrt(X ** 2 + Y ** 2)

    # Store intermediate outputs
    outputs = {
        1: X,  # X input
        2: Y,  # Y input
        3: R,  # R input
    }

    # Process each node in order
    for node in sorted(state.nodes, key=lambda x: x.id):
        if node.type == 'middle' or node.type == 'output':
            # Get input connections
            input1_conn = next((c for c in state.connections if c.to == node.id and c.port == 'input1'), None)
            input2_conn = next((c for c in state.connections if c.to == node.id and c.port == 'input2'), None)

            # Get input values
            input1 = outputs[input1_conn.from_id] if input1_conn else np.zeros_like(X)
            input2 = outputs[input2_conn.from_id] if input2_conn else np.zeros_like(X)

            # Compute weighted sum
            result = node.gain1 * input1 + node.gain2 * input2

            # Apply activation function for middle nodes
            if node.type == 'middle':
                result = activation_function(result, node.activation)

            outputs[node.id] = result

    # Get RGB outputs
    r = outputs.get(8, np.zeros_like(X))
    g = outputs.get(9, np.zeros_like(X))
    b = outputs.get(10, np.zeros_like(X))

    # Normalize to [0, 1] range
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)

    return {
        "r": r.tolist(),
        "g": g.tolist(),
        "b": b.tolist()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)