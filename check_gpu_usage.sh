#!/bin/bash

# Safe GPU usage checker - only shows your own processes
echo "=== GPU Usage Checker (Your Processes Only) ==="
echo "Current user: $(whoami)"
echo "Time: $(date)"
echo ""

# Check GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r gpu_id name mem_used mem_total util; do
    echo "GPU $gpu_id ($name): ${mem_used}MB/${mem_total}MB used, ${util}% utilization"
done

echo ""
echo "Your processes using GPU:"

# Get your username
USERNAME=$(whoami)

# Check each GPU for processes
for gpu_id in {0..7}; do
    # Get processes on this GPU
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | while IFS=, read -r pid process_name mem_used; do
        if [ ! -z "$pid" ] && [ "$pid" != "pid" ]; then
            # Check if this process belongs to you
            process_owner=$(ps -p $pid -o user= 2>/dev/null)
            if [ "$process_owner" = "$USERNAME" ]; then
                echo "  GPU $gpu_id: PID $pid ($process_name) - ${mem_used}MB"
            fi
        fi
    done
done

echo ""
echo "All your Python processes:"
ps -u $USERNAME | grep python | grep -v grep

echo ""
echo "=== End of Report ===" 