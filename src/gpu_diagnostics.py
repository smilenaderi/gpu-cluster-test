"""
GPU Cluster Diagnostics - Comprehensive Infrastructure Analysis

This script performs detailed diagnostics of GPU cluster infrastructure:
- GPU hardware detection and capabilities
- CUDA/cuDNN/NCCL versions
- Network topology and bandwidth
- Memory hierarchy (GPU, CPU, shared)
- PCIe topology and P2P capabilities
- InfiniBand/RoCE detection
- Multi-node communication testing
- Performance benchmarks

Usage:
    Single node:    python gpu_diagnostics.py
    Multi-GPU:      torchrun --nproc_per_node=8 gpu_diagnostics.py
    Multi-node:     torchrun --nnodes=2 --nproc_per_node=8 gpu_diagnostics.py
"""

import os
import sys
import socket
import platform
import subprocess
from datetime import datetime
import torch
import torch.distributed as dist


def print_header(title, char="="):
    """Print formatted section header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def print_subheader(title):
    """Print formatted subsection header."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def run_command(cmd, shell=True, capture=True):
    """Run shell command and return output."""
    try:
        if capture:
            result = subprocess.run(
                cmd, shell=shell, capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else None
        else:
            subprocess.run(cmd, shell=shell, timeout=10)
            return True
    except Exception as e:
        return None


def get_system_info():
    """Collect system information."""
    print_header("SYSTEM INFORMATION")
    
    print(f"Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hostname:         {socket.gethostname()}")
    print(f"Platform:         {platform.platform()}")
    print(f"Architecture:     {platform.machine()}")
    print(f"Processor:        {platform.processor() or 'N/A'}")
    print(f"Python Version:   {sys.version.split()[0]}")
    print(f"Python Path:      {sys.executable}")
    
    # CPU info - detailed
    print_subheader("CPU Information")
    cpu_info = run_command("lscpu | grep 'Model name' | cut -d':' -f2")
    if cpu_info:
        print(f"Model:            {cpu_info.strip()}")
    
    cpu_count = os.cpu_count()
    print(f"Total Cores:      {cpu_count}")
    
    # CPU details
    cpu_details = run_command("lscpu | grep -E '(Socket|Core|Thread|CPU MHz|Cache)'")
    if cpu_details:
        print(cpu_details)
    
    # Memory info - detailed
    print_subheader("Memory Information")
    mem_info = run_command("free -h")
    if mem_info:
        print(mem_info)
    
    # Disk info
    print_subheader("Disk Information")
    disk_info = run_command("df -h | grep -E '(Filesystem|/dev/|/shared)'")
    if disk_info:
        print(disk_info)
    
    # Kernel and OS
    print_subheader("Operating System")
    kernel = run_command("uname -r")
    if kernel:
        print(f"Kernel Version:   {kernel}")
    
    os_release = run_command("cat /etc/os-release 2>/dev/null | grep -E '(PRETTY_NAME|VERSION_ID)'")
    if os_release:
        print(os_release)
    
    # System uptime
    uptime = run_command("uptime")
    if uptime:
        print(f"Uptime:           {uptime}")
    
    # Load average
    load_avg = run_command("cat /proc/loadavg")
    if load_avg:
        print(f"Load Average:     {load_avg}")


def get_gpu_info():
    """Collect GPU hardware information."""
    print_header("GPU HARDWARE INFORMATION")
    
    # Check nvidia-smi first
    print_subheader("NVIDIA Driver & GPU Detection")
    nvidia_smi_check = run_command("nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader")
    if nvidia_smi_check:
        print("✅ nvidia-smi available:")
        print(nvidia_smi_check)
    else:
        print("❌ nvidia-smi not available or no GPUs detected")
        print("   Possible causes:")
        print("   - NVIDIA drivers not installed")
        print("   - No GPU hardware present")
        print("   - GPU not visible in container (missing --gpus flag)")
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA is not available in PyTorch!")
        print("   - Check if NVIDIA drivers are installed: nvidia-smi")
        print("   - Check if PyTorch was built with CUDA support")
        print(f"   - PyTorch version: {torch.__version__}")
        print(f"   - CUDA compiled version: {torch.version.cuda}")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"\n✅ CUDA Available:  Yes")
    print(f"GPU Count:         {gpu_count}")
    print()
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  ├─ UUID:                  {run_command(f'nvidia-smi --query-gpu=uuid --format=csv,noheader -i {i}') or 'N/A'}")
        print(f"  ├─ Compute Capability:    {props.major}.{props.minor}")
        print(f"  ├─ Total Memory:          {props.total_memory / 1024**3:.2f} GB")
        print(f"  ├─ Multi-Processors:      {props.multi_processor_count}")
        print(f"  ├─ Max Threads/Block:     {props.max_threads_per_block}")
        print(f"  ├─ Max Threads/MP:        {props.max_threads_per_multi_processor}")
        print(f"  ├─ Warp Size:             {props.warp_size}")
        print(f"  ├─ L2 Cache Size:         {props.l2_cache_size / 1024**2:.2f} MB")
        print(f"  ├─ Clock Rate:            {props.clock_rate / 1000:.0f} MHz")
        print(f"  ├─ Memory Clock Rate:     {run_command(f'nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits -i {i}') or 'N/A'} MHz")
        print(f"  ├─ Memory Bus Width:      {run_command(f'nvidia-smi --query-gpu=memory.bus --format=csv,noheader -i {i}') or 'N/A'} bits")
        print(f"  ├─ PCIe Link Gen:         {run_command(f'nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader -i {i}') or 'N/A'}")
        print(f"  ├─ PCIe Link Width:       x{run_command(f'nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader -i {i}') or 'N/A'}")
        print(f"  ├─ Power Limit:           {run_command(f'nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i {i}') or 'N/A'} W")
        print(f"  ├─ Temperature:           {run_command(f'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i {i}') or 'N/A'} °C")
        print(f"  └─ GPU Utilization:       {run_command(f'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i {i}') or 'N/A'} %")
        print()
    
    # Additional GPU details
    print_subheader("GPU Detailed Query")
    gpu_details = run_command("nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit --format=csv")
    if gpu_details:
        print(gpu_details)


def get_cuda_info():
    """Collect CUDA/cuDNN/NCCL version information."""
    print_header("CUDA ECOSYSTEM VERSIONS")
    
    # PyTorch version
    print(f"PyTorch Version:   {torch.__version__}")
    print(f"PyTorch Path:      {torch.__file__}")
    
    # CUDA version
    if torch.cuda.is_available():
        print(f"CUDA Available:    ✅ Yes")
        print(f"CUDA Version:      {torch.version.cuda}")
        print(f"cuDNN Version:     {torch.backends.cudnn.version()}")
        print(f"cuDNN Enabled:     {torch.backends.cudnn.enabled}")
        print(f"cuDNN Benchmark:   {torch.backends.cudnn.benchmark}")
        print(f"cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
        
        # NCCL version
        try:
            nccl_version = torch.cuda.nccl.version()
            print(f"NCCL Version:      {nccl_version[0]}.{nccl_version[1]}.{nccl_version[2]}")
            print(f"NCCL Available:    ✅ Yes")
        except Exception as e:
            print(f"NCCL Version:      ❌ Not available ({e})")
        
        # CUDA device count
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device:    {torch.cuda.current_device() if torch.cuda.device_count() > 0 else 'N/A'}")
    else:
        print(f"CUDA Available:    ❌ No")
        print(f"CUDA Compiled:     {torch.version.cuda}")
    
    # NVIDIA driver version
    print_subheader("NVIDIA Driver Information")
    driver_version = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1")
    if driver_version:
        print(f"NVIDIA Driver:     {driver_version}")
    else:
        print(f"NVIDIA Driver:     ❌ Not available")
    
    # CUDA runtime from nvidia-smi
    cuda_runtime = run_command("nvidia-smi | grep 'CUDA Version' | awk '{print $9}'")
    if cuda_runtime:
        print(f"CUDA Runtime:      {cuda_runtime}")
    
    # Full nvidia-smi output
    print_subheader("nvidia-smi Full Output")
    nvidia_smi_full = run_command("nvidia-smi")
    if nvidia_smi_full:
        print(nvidia_smi_full)
    else:
        print("❌ nvidia-smi not available")
    
    # CUDA libraries
    print_subheader("CUDA Libraries")
    cuda_libs = run_command("ldconfig -p | grep -E '(cuda|nccl|cudnn)' | head -20")
    if cuda_libs:
        print(cuda_libs)
    else:
        print("No CUDA libraries found in ldconfig cache")


def get_gpu_utilization():
    """Get current GPU utilization."""
    print_header("GPU UTILIZATION & MEMORY")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        
        # Memory info
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        print(f"  ├─ Memory Allocated:  {mem_allocated:.2f} GB")
        print(f"  ├─ Memory Reserved:   {mem_reserved:.2f} GB")
        print(f"  ├─ Memory Total:      {mem_total:.2f} GB")
        print(f"  └─ Memory Free:       {mem_total - mem_reserved:.2f} GB")
    
    # nvidia-smi output
    print_subheader("nvidia-smi Output")
    nvidia_smi = run_command("nvidia-smi", capture=False)


def get_pcie_topology():
    """Get PCIe topology and P2P capabilities."""
    print_header("PCIe TOPOLOGY & P2P CAPABILITIES")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    gpu_count = torch.cuda.device_count()
    
    # P2P access matrix
    print("P2P Access Matrix (can_device_access_peer):")
    print("     ", end="")
    for j in range(gpu_count):
        print(f"GPU{j:2d} ", end="")
    print()
    
    for i in range(gpu_count):
        print(f"GPU{i:2d}", end="")
        for j in range(gpu_count):
            if i == j:
                access = " --  "
            else:
                try:
                    access = torch.cuda.can_device_access_peer(i, j)
                    access = " ✅  " if access else " ❌  "
                except:
                    access = " ?   "
            print(access, end="")
        print()
    
    # PCIe topology from nvidia-smi
    print_subheader("PCIe Topology (nvidia-smi topo -m)")
    run_command("nvidia-smi topo -m", capture=False)


def get_network_info():
    """Get network configuration."""
    print_header("NETWORK CONFIGURATION")
    
    # Hostname and IP
    hostname = socket.gethostname()
    print(f"Hostname:          {hostname}")
    print(f"FQDN:              {socket.getfqdn()}")
    
    try:
        ip_address = socket.gethostbyname(hostname)
        print(f"Primary IP:        {ip_address}")
    except:
        print(f"Primary IP:        Unable to resolve")
    
    # All IP addresses
    try:
        all_ips = socket.gethostbyname_ex(hostname)[2]
        print(f"All IPs:           {', '.join(all_ips)}")
    except:
        pass
    
    # Network interfaces - detailed
    print_subheader("Network Interfaces (Brief)")
    ifconfig_brief = run_command("ip -br addr show")
    if ifconfig_brief:
        print(ifconfig_brief)
    
    print_subheader("Network Interfaces (Detailed)")
    ifconfig_detail = run_command("ip addr show")
    if ifconfig_detail:
        print(ifconfig_detail)
    else:
        # Fallback to ifconfig
        ifconfig_detail = run_command("ifconfig -a")
        if ifconfig_detail:
            print(ifconfig_detail)
    
    # Network routes
    print_subheader("Network Routes")
    routes = run_command("ip route show")
    if routes:
        print(routes)
    
    # DNS configuration
    print_subheader("DNS Configuration")
    dns_config = run_command("cat /etc/resolv.conf 2>/dev/null | grep -v '^#' | grep -v '^$'")
    if dns_config:
        print(dns_config)
    
    # Network statistics
    print_subheader("Network Interface Statistics")
    net_stats = run_command("ip -s link show")
    if net_stats:
        print(net_stats)
    
    # InfiniBand detection
    print_subheader("InfiniBand/RDMA Detection")
    
    # Check for IB devices
    ib_stat = run_command("ibstat 2>/dev/null")
    if ib_stat:
        print("✅ InfiniBand detected:")
        print(ib_stat)
    else:
        print("❌ InfiniBand not detected (ibstat not available or no devices)")
    
    # Check ibv_devices
    ibv_devices = run_command("ibv_devices 2>/dev/null")
    if ibv_devices and "device" in ibv_devices.lower():
        print("\n✅ RDMA devices (ibv_devices):")
        print(ibv_devices)
    else:
        print("\n❌ No RDMA devices found (ibv_devices)")
    
    # Check ibv_devinfo
    ibv_devinfo = run_command("ibv_devinfo 2>/dev/null")
    if ibv_devinfo:
        print("\n✅ RDMA device info (ibv_devinfo):")
        print(ibv_devinfo)
    
    # Check for RDMA devices in /dev
    rdma_devices = run_command("ls -la /dev/infiniband/ 2>/dev/null")
    if rdma_devices:
        print(f"\n✅ RDMA devices in /dev/infiniband/:")
        print(rdma_devices)
    else:
        print("\n❌ No RDMA devices in /dev/infiniband/")
    
    # Check for RoCE
    roce_info = run_command("show_gids 2>/dev/null")
    if roce_info:
        print("\n✅ RoCE GID table:")
        print(roce_info)
    
    # Network bandwidth test tools
    print_subheader("Network Testing Tools")
    tools = ["iperf3", "netperf", "qperf", "perftest"]
    for tool in tools:
        if run_command(f"which {tool}"):
            print(f"✅ {tool:15s} available")
        else:
            print(f"❌ {tool:15s} not found")
    
    # TCP/UDP settings
    print_subheader("TCP/UDP Kernel Settings")
    tcp_settings = [
        "net.core.rmem_max",
        "net.core.wmem_max",
        "net.ipv4.tcp_rmem",
        "net.ipv4.tcp_wmem",
        "net.core.netdev_max_backlog",
        "net.ipv4.tcp_congestion_control",
    ]
    for setting in tcp_settings:
        value = run_command(f"sysctl {setting} 2>/dev/null")
        if value:
            print(f"  {value}")
    
    # Check connectivity to common ports
    print_subheader("Network Connectivity Check")
    if "MASTER_ADDR" in os.environ:
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ.get("MASTER_PORT", "29500")
        print(f"Testing connection to master: {master_addr}:{master_port}")
        
        # Ping test
        ping_result = run_command(f"ping -c 3 -W 2 {master_addr} 2>&1")
        if ping_result and "0% packet loss" in ping_result:
            print(f"  ✅ Ping to {master_addr}: Success")
        else:
            print(f"  ❌ Ping to {master_addr}: Failed")
        
        # Port test
        port_test = run_command(f"timeout 2 bash -c 'cat < /dev/null > /dev/tcp/{master_addr}/{master_port}' 2>&1")
        if port_test is not None:
            print(f"  ✅ Port {master_port} on {master_addr}: Open")
        else:
            print(f"  ❌ Port {master_port} on {master_addr}: Closed or filtered")
    else:
        print("MASTER_ADDR not set - skipping master connectivity test")


def get_distributed_info():
    """Get distributed training configuration."""
    print_header("DISTRIBUTED TRAINING CONFIGURATION")
    
    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ.get("MASTER_ADDR", "N/A")
        master_port = os.environ.get("MASTER_PORT", "N/A")
        
        print(f"✅ Distributed Mode:  Enabled")
        print(f"Rank:              {rank}")
        print(f"Local Rank:        {local_rank}")
        print(f"World Size:        {world_size}")
        print(f"Master Address:    {master_addr}")
        print(f"Master Port:       {master_port}")
        
        # Initialize process group if not already done
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend="nccl")
                print(f"Backend:           {dist.get_backend()}")
                print(f"✅ Process group initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize process group: {e}")
        else:
            print(f"Backend:           {dist.get_backend()}")
            print(f"✅ Process group already initialized")
    else:
        print("❌ Distributed Mode:  Not enabled")
        print("   Run with: torchrun --nproc_per_node=N script.py")


def test_gpu_compute():
    """Test GPU compute capability with simple operations."""
    print_header("GPU COMPUTE TEST")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping compute test")
        return
    
    try:
        device = torch.device("cuda:0")
        
        # Matrix multiplication test
        print("Testing matrix multiplication...")
        size = 4096
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Timed test
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        c = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        tflops = (2 * size**3) / (elapsed_ms * 1e9)
        
        print(f"  ├─ Matrix size:       {size}x{size}")
        print(f"  ├─ Time:              {elapsed_ms:.2f} ms")
        print(f"  └─ Performance:       {tflops:.2f} TFLOPS")
        
        # Memory bandwidth test
        print("\nTesting memory bandwidth...")
        size_mb = 1024
        data = torch.randn(size_mb * 1024 * 1024 // 4, device=device)
        
        start.record()
        data_copy = data.clone()
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        bandwidth_gb = (size_mb / 1024) / (elapsed_ms / 1000)
        
        print(f"  ├─ Data size:         {size_mb} MB")
        print(f"  ├─ Time:              {elapsed_ms:.2f} ms")
        print(f"  └─ Bandwidth:         {bandwidth_gb:.2f} GB/s")
        
        print("\n✅ GPU compute test passed")
        
    except Exception as e:
        print(f"❌ GPU compute test failed: {e}")


def test_nccl_communication():
    """Test NCCL communication if in distributed mode."""
    print_header("NCCL COMMUNICATION TEST")
    
    if not dist.is_initialized():
        print("❌ Distributed mode not initialized - skipping NCCL test")
        print("   Run with: torchrun --nproc_per_node=N script.py")
        return
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    try:
        # Test all_reduce
        print(f"[Rank {rank}] Testing all_reduce...")
        tensor = torch.tensor([float(rank)], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(world_size))
        
        if abs(tensor.item() - expected) < 1e-6:
            print(f"[Rank {rank}] ✅ all_reduce passed (result: {tensor.item():.0f}, expected: {expected})")
        else:
            print(f"[Rank {rank}] ❌ all_reduce failed (result: {tensor.item():.0f}, expected: {expected})")
        
        dist.barrier()
        
        # Test broadcast
        if rank == 0:
            print(f"[Rank {rank}] Testing broadcast...")
        
        tensor = torch.tensor([42.0], device=device) if rank == 0 else torch.zeros(1, device=device)
        dist.broadcast(tensor, src=0)
        
        if abs(tensor.item() - 42.0) < 1e-6:
            print(f"[Rank {rank}] ✅ broadcast passed (result: {tensor.item():.0f})")
        else:
            print(f"[Rank {rank}] ❌ broadcast failed (result: {tensor.item():.0f})")
        
        dist.barrier()
        
        if rank == 0:
            print("\n✅ NCCL communication test passed")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ NCCL communication test failed: {e}")


def get_environment_variables():
    """Display relevant environment variables."""
    print_header("ENVIRONMENT VARIABLES")
    
    env_vars = [
        # CUDA/GPU
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_HOME",
        "CUDA_PATH",
        # NCCL
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_DEBUG_FILE",
        "NCCL_IB_DISABLE",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_SOCKET_IFNAME",
        "NCCL_P2P_LEVEL",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_TIMEOUT",
        "NCCL_BLOCKING_WAIT",
        "NCCL_ASYNC_ERROR_HANDLING",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_CROSS_NIC",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_MIN_NCHANNELS",
        "NCCL_MAX_NCHANNELS",
        # PyTorch Distributed
        "TORCH_DISTRIBUTED_DEBUG",
        "TORCH_CPP_LOG_LEVEL",
        "TORCH_SHOW_CPP_STACKTRACES",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "NODE_RANK",
        # Slurm
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS_PER_NODE",
        "SLURM_NODEID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_TASK_PID",
        # Container
        "ENROOT_CONTAINER_NAME",
        "SINGULARITY_CONTAINER",
        "DOCKER_CONTAINER",
        # Other
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "LD_LIBRARY_PATH",
    ]
    
    print_subheader("CUDA & GPU Variables")
    for var in env_vars[:5]:
        value = os.environ.get(var, "Not set")
        print(f"{var:35s} = {value}")
    
    print_subheader("NCCL Variables")
    for var in env_vars[5:25]:
        value = os.environ.get(var, "Not set")
        print(f"{var:35s} = {value}")
    
    print_subheader("PyTorch Distributed Variables")
    for var in env_vars[25:33]:
        value = os.environ.get(var, "Not set")
        print(f"{var:35s} = {value}")
    
    print_subheader("Slurm Variables")
    for var in env_vars[33:45]:
        value = os.environ.get(var, "Not set")
        print(f"{var:35s} = {value}")
    
    print_subheader("Other Variables")
    for var in env_vars[45:]:
        value = os.environ.get(var, "Not set")
        if var == "LD_LIBRARY_PATH" and value != "Not set":
            # Truncate long paths
            paths = value.split(":")
            if len(paths) > 5:
                value = ":".join(paths[:3]) + f":... ({len(paths)} paths total)"
        print(f"{var:35s} = {value}")


def get_slurm_info():
    """Get Slurm job information if running under Slurm."""
    print_header("SLURM INFORMATION")
    
    if "SLURM_JOB_ID" not in os.environ:
        print("❌ Not running under Slurm")
        return
    
    print("✅ Running under Slurm")
    print()
    
    slurm_vars = {
        "Job ID": "SLURM_JOB_ID",
        "Job Name": "SLURM_JOB_NAME",
        "Partition": "SLURM_JOB_PARTITION",
        "Node List": "SLURM_JOB_NODELIST",
        "Num Nodes": "SLURM_JOB_NUM_NODES",
        "Num Tasks": "SLURM_NTASKS",
        "Tasks per Node": "SLURM_NTASKS_PER_NODE",
        "CPUs per Task": "SLURM_CPUS_PER_TASK",
        "GPUs per Node": "SLURM_GPUS_PER_NODE",
        "Node ID": "SLURM_NODEID",
        "Local ID": "SLURM_LOCALID",
        "Proc ID": "SLURM_PROCID",
        "Submit Dir": "SLURM_SUBMIT_DIR",
        "Submit Host": "SLURM_SUBMIT_HOST",
    }
    
    for label, var in slurm_vars.items():
        value = os.environ.get(var, "N/A")
        print(f"{label:20s}: {value}")
    
    # Get job details from scontrol
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        print_subheader(f"Slurm Job Details (scontrol show job {job_id})")
        job_details = run_command(f"scontrol show job {job_id}")
        if job_details:
            print(job_details)
    
    # Get node details
    print_subheader("Slurm Node Information")
    node_info = run_command("scontrol show node $SLURM_NODELIST")
    if node_info:
        print(node_info)


def get_container_info():
    """Detect and display container information."""
    print_header("CONTAINER INFORMATION")
    
    # Check for various container runtimes
    is_container = False
    
    # Docker
    if os.path.exists("/.dockerenv"):
        print("✅ Running in Docker container")
        is_container = True
        docker_info = run_command("cat /proc/1/cgroup | grep docker")
        if docker_info:
            print(f"   Docker cgroup: {docker_info.split('/')[-1][:20]}...")
    
    # Singularity/Apptainer
    if "SINGULARITY_CONTAINER" in os.environ:
        print(f"✅ Running in Singularity/Apptainer container")
        print(f"   Container: {os.environ['SINGULARITY_CONTAINER']}")
        is_container = True
    
    # Enroot
    if "ENROOT_CONTAINER_NAME" in os.environ:
        print(f"✅ Running in Enroot container")
        print(f"   Container: {os.environ['ENROOT_CONTAINER_NAME']}")
        is_container = True
    
    # Generic container check
    if not is_container:
        cgroup_check = run_command("cat /proc/1/cgroup 2>/dev/null | grep -E '(docker|lxc|kubepods)'")
        if cgroup_check:
            print("✅ Running in container (detected via cgroup)")
            is_container = True
    
    if not is_container:
        print("❌ Not running in a container (or container not detected)")
    
    # Container mounts
    print_subheader("Mount Points")
    mounts = run_command("mount | grep -E '(shared|workspace|project|home)' | head -20")
    if mounts:
        print(mounts)
    else:
        print("No special mounts detected")
    
    # Check for GPU device access
    print_subheader("GPU Device Access in Container")
    gpu_devices = run_command("ls -la /dev/nvidia* 2>/dev/null")
    if gpu_devices:
        print("✅ NVIDIA devices accessible:")
        print(gpu_devices)
    else:
        print("❌ No NVIDIA devices found in /dev/")
        print("   Container may not have GPU access (missing --gpus flag)")


def get_process_info():
    """Get process and resource information."""
    print_header("PROCESS INFORMATION")
    
    print(f"Process ID (PID):  {os.getpid()}")
    print(f"Parent PID:        {os.getppid()}")
    print(f"User:              {os.getenv('USER', 'unknown')}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Process limits
    print_subheader("Process Limits (ulimit)")
    limits = run_command("ulimit -a")
    if limits:
        print(limits)
    
    # Current processes
    print_subheader("Python/PyTorch Processes")
    processes = run_command("ps aux | grep -E '(python|torch)' | grep -v grep | head -10")
    if processes:
        print(processes)
    
    # GPU processes
    print_subheader("GPU Processes")
    gpu_processes = run_command("nvidia-smi pmon -c 1 2>/dev/null || nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
    if gpu_processes:
        print(gpu_processes)
    else:
        print("No GPU processes detected or nvidia-smi not available")


def main():
    """Run all diagnostics."""
    print_header("GPU CLUSTER DIAGNOSTICS", "=")
    print("Comprehensive infrastructure analysis for GPU clusters")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all diagnostic functions
    get_system_info()
    get_process_info()
    get_container_info()
    get_gpu_info()
    get_cuda_info()
    get_gpu_utilization()
    get_pcie_topology()
    get_network_info()
    get_distributed_info()
    get_slurm_info()
    get_environment_variables()
    test_gpu_compute()
    test_nccl_communication()
    
    # Final summary
    print_header("DIAGNOSTIC SUMMARY")
    
    cuda_ok = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_ok else 0
    dist_ok = dist.is_initialized() if "RANK" in os.environ else False
    
    print(f"CUDA Available:        {'✅ Yes' if cuda_ok else '❌ No'}")
    print(f"GPU Count:             {gpu_count}")
    print(f"Distributed Mode:      {'✅ Enabled' if dist_ok else '❌ Disabled'}")
    
    if dist_ok:
        print(f"World Size:            {dist.get_world_size()}")
        print(f"Backend:               {dist.get_backend()}")
    
    # Check for common issues
    print_subheader("Common Issues Check")
    issues = []
    
    if not cuda_ok:
        issues.append("❌ CUDA not available - check drivers and PyTorch installation")
    
    if cuda_ok and gpu_count == 0:
        issues.append("❌ No GPUs detected - check CUDA_VISIBLE_DEVICES")
    
    if "RANK" in os.environ and not dist_ok:
        issues.append("❌ Distributed mode expected but not initialized")
    
    nvidia_smi = run_command("nvidia-smi --query-gpu=name --format=csv,noheader")
    if not nvidia_smi:
        issues.append("❌ nvidia-smi not working - driver or GPU access issue")
    
    if os.path.exists("/.dockerenv") or "SINGULARITY_CONTAINER" in os.environ:
        gpu_dev = run_command("ls /dev/nvidia0 2>/dev/null")
        if not gpu_dev:
            issues.append("❌ Container detected but no GPU devices in /dev/ - missing --gpus flag?")
    
    if issues:
        print("\n".join(issues))
    else:
        print("✅ No common issues detected")
    
    print()
    print("=" * 80)
    print("Diagnostics complete!")
    print("=" * 80)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
