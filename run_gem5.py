import m5
from m5.objects import *
import argparse

parser = argparse.ArgumentParser(description="Run RISC-V binary in gem5")
parser.add_argument("--cmd", type=str, required=True, help="Path to the RISC-V ELF binary")
args = parser.parse_args()

# --- 1. System Setup ---
system = System()
# Upgraded to 3GHz
system.clk_domain = SrcClockDomain(clock='3GHz', voltage_domain=VoltageDomain())
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('4GB')] 

# --- 2. The CPU Upgrade (Superscalar Out-of-Order) ---
system.cpu = TimingSimpleCPU()

system.cpu.isa = [RiscvISA(vlen=128)]

# --- 3. Cache Architecture Definitions ---
class L1ICache(Cache):
    size = '64kB'
    assoc = 4
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 4
    tgts_per_mshr = 20

class L1DCache(Cache):
    size = '64kB'
    assoc = 4
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 16
    tgts_per_mshr = 20

class L2Cache(Cache):
    size = '2MB'
    assoc = 8
    tag_latency = 12
    data_latency = 12
    response_latency = 12
    mshrs = 32
    tgts_per_mshr = 12

# --- 4. Instantiate and Connect Caches ---
system.cpu.icache = L1ICache()
system.cpu.dcache = L1DCache()

# Connect the CPU to the L1 caches
system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port

# Create an L2 crossbar to connect the L1s to the L2
system.l2bus = L2XBar()

# Connect the L1 caches to the L2 bus
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

# Instantiate the L2 cache and connect it to the L2 bus
system.l2cache = L2Cache()
system.l2cache.cpu_side = system.l2bus.mem_side_ports

# Create the main system memory bus
system.membus = SystemXBar()

# Connect the L2 cache to the main memory bus
system.l2cache.mem_side = system.membus.cpu_side_ports

system.cpu.createInterruptController()

# --- 5. Main Memory Upgrade ---
system.mem_ctrl = MemCtrl()
# Upgraded to DDR4
system.mem_ctrl.dram = DDR4_2400_8x8() 
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports
system.system_port = system.membus.cpu_side_ports

# --- 6. Workload Execution ---
process = Process()
process.cmd = [args.cmd]
system.cpu.workload = process
system.cpu.createThreads()

system.workload = SEWorkload.init_compatible(args.cmd)

root = Root(full_system=False, system=system)
m5.instantiate()

print(f"--- Starting Beast Mode gem5 simulation for {args.cmd} ---")
exit_event = m5.simulate()
print(f"--- Simulation finished @ tick {m5.curTick()} because {exit_event.getCause()} ---")
