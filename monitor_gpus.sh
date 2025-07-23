#!/bin/bash
"""
GPU monitoring script - Monitor all 8 GPUs during training
Shows real-time GPU utilization to verify parallel training
"""

set -e

echo "=========================================="
echo "Multi-GPU Training Monitor"
echo "=========================================="
echo "Monitoring all 8 GPUs for parallel utilization"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Function to display GPU status
monitor_gpus() {
    while true; do
        clear
        echo "🔥 GPU Training Monitor - $(date)"
        echo "==========================================="
        
        # Display GPU utilization
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
        while IFS=',' read -r gpu_id name gpu_util mem_util mem_used mem_total temp power; do
            # Clean up whitespace
            gpu_id=$(echo $gpu_id | xargs)
            name=$(echo $name | xargs)
            gpu_util=$(echo $gpu_util | xargs)
            mem_util=$(echo $mem_util | xargs)
            temp=$(echo $temp | xargs)
            power=$(echo $power | xargs)
            
            # Color coding based on utilization
            if [ $gpu_util -gt 80 ]; then
                status="🔥 ACTIVE "
            elif [ $gpu_util -gt 20 ]; then
                status="⚡ WORKING"
            else
                status="💤 IDLE   "
            fi
            
            printf "GPU %s: %s | GPU: %2s%% | Mem: %2s%% | Temp: %2s°C | Power: %3sW\n" \
                   "$gpu_id" "$status" "$gpu_util" "$mem_util" "$temp" "$power"
        done
        
        echo ""
        echo "Expected for PARALLEL training:"
        echo "✅ All 8 GPUs should show >80% utilization"
        echo "✅ Each GPU should have similar memory usage"
        echo "❌ If only GPU 0 is active = SEQUENTIAL (bad)"
        echo ""
        echo "Agent allocation:"
        echo "  Agent 0 -> GPU 0    Agent 4 -> GPU 4"
        echo "  Agent 1 -> GPU 1    Agent 5 -> GPU 5"  
        echo "  Agent 2 -> GPU 2    Agent 6 -> GPU 6"
        echo "  Agent 3 -> GPU 3    Agent 7 -> GPU 7"
        
        sleep 2
    done
}

# Start monitoring
monitor_gpus
