#!/bin/bash

# Stop existing processes
pkill -9 -f "server.main"
pkill -9 -f "worker.mock_worker"
sleep 1

# Clean logs
rm -f server.log worker.log

# Start server
export UV_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
nohup uv run python -m server.main > server.log 2>&1 &
echo "Server started (PID: $!)"

# Wait for server to be ready
sleep 2

# Start worker
nohup uv run python -m worker.mock_worker --name mock-worker-1 --server-url http://localhost:7000 --port 7001 > worker.log 2>&1 &
echo "Worker started (PID: $!)"

sleep 2
echo "Done! Server: http://0.0.0.0:7000/ui"
