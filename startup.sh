#!/bin/bash
port=${PORT:-8000}
streamlit run vehicledamage2.py --server.port $port --server.address 0.0.0.0