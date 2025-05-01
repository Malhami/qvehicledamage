#!/bin/bash

# Install necessary dependencies
apt-get update && apt-get install -y libgl1-mesa-glx

# Run the Streamlit app
streamlit run vehicledamage2.py --server.port=$PORT --server.address=0.0.0.0
