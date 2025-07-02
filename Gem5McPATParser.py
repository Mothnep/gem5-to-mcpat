"""
[usage]:
python3 Gem5ToMcPAT-Parser.py -c ../m5out/config.json -s ../m5out/stats.txt -t template.xml

# Tested
python 3.6.9
python 3.8.5

"""
import argparse
import sys
import json
import re
from xml.etree import ElementTree as ET
from xml.dom import minidom
import copy
import types
import logging

# Global debug flag
DEBUG = False

def debug_print(*args, **kwargs):
    """Print debug messages only if DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Gem5 to McPAT parser")

    parser.add_argument(
        '--config', '-c', type=str, required=True,
        metavar='PATH',
        help="Input config.json from Gem5 output.")
    parser.add_argument(
        '--stats', '-s', type=str, required=True,
        metavar='PATH',
        help="Input stats.txt from Gem5 output.")
    parser.add_argument(
        '--template', '-t', type=str, required=True,
        metavar='PATH',
        help="Template XML file")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default="mcpat-in.xml",
        metavar='PATH',
        help="Output file for McPAT input in XML format (default: mcpat-in.xml)")
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help="Enable verbose/debug output")

    return parser


class PIParser(ET.TreeBuilder):
    def __init__(self, *args, **kwargs):
        # call init of superclass and pass args and kwargs
        super(PIParser, self).__init__(*args, **kwargs)

        self.CommentHandler = self.comment
        self.ProcessingInstructionHandler = self.pi
        # Remove this line that was adding the document wrapper:
        # self.start("document", {})

    def close(self):
        # Remove this line that was closing the document wrapper:
        # self.end("document")
        return ET.TreeBuilder.close(self)

    def comment(self, data):
        self.start(ET.Comment, {})
        self.data(data)
        self.end(ET.Comment)

    def pi(self, target, data):
        self.start(ET.PI, {})
        self.data(target + " " + data)
        self.end(ET.PI)


def parse(source):
    parser = ET.XMLParser(target=PIParser())
    return ET.parse(source, parser=parser)


def readStatsFile(statsFile): #Works fine
    global stats
    stats = {}
    F = open(statsFile)
    ignores = re.compile(r'^---|^$')
    statLine = re.compile(
        r'([a-zA-Z0-9_\.:-]+)\s+([-+]?[0-9]+\.[0-9]+|[-+]?[0-9]+|nan|inf)')
    count = 0
    for line in F:
        # ignore empty lines and lines starting with "---"
        if not ignores.match(line):
            count += 1
            statKind = statLine.match(line).group(1)
            statValue = statLine.match(line).group(2)
            if statValue == 'nan':
                logging.warning("%s is nan. Setting it to 0" % statKind)
                statValue = '0'
            stats[statKind] = statValue
    F.close()
    #print(stats)


def readConfigFile(configFile):
    global config
    F = open(configFile)
    config = json.load(F)
    
    # Print the loaded JSON content
    # print("Loaded config JSON:")
    #print(json.dumps(config, indent=4))  # Pretty-print the JSON content
    
    F.close()


def readMcpatFile(templateFile):
    global templateMcpat
    templateMcpat = parse(templateFile)
    
    # Print the parsed XML tree
    #print("Parsed McPAT template:")
    #ET.dump(templateMcpat)



def updateComponentParameters(component, component_id, param_mappings, stat_mappings=None):
    """Generic function to update component parameters and stats"""
    debug_print(f"Updating {component_id} component")
    
    for child in component:
        if child.tag == 'param':
            param_name = child.get("name")
            if param_name in param_mappings:
                value = param_mappings[param_name]
                if value is not None:
                    child.set("value", str(value))
                    debug_print(f"Updated {component_id} {param_name} to {value}")
        elif child.tag == 'stat' and stat_mappings:
            stat_name = child.get("name")
            if stat_name in stat_mappings:
                value = stat_mappings[stat_name]
                if value is not None:
                    child.set("value", str(value))
                    debug_print(f"Updated {component_id} {stat_name} to {value}")


def extractStatsValues(stats, numCores):
    """Extract statistics values that may vary based on number of cores"""
    
    # Initialize totals
    totalInstructions = 0
    totalBranches = 0
    totalCycles = 0
    totalIdleCycles = 0
    totalIntAluAccesses = 0
    totalFpAluAccesses = 0
    totalLoadInstructions = 0
    totalStoreInstructions = 0
    branchMispredictions = 0
    totalcommittedInstructions = 0
    total_committed_int_instructions = 0
    total_committed_fp_instructions = 0
    totalInstWindowReads = 0
    totalInstWindowWrites = 0 
    totalInstWindowWakeups = 0
    totalFpInstWindowReads = 0
    totalFpInstWindowWrites = 0
    totalFpInstWindowWakeups = 0
    totalIntRegfileReads = 0
    totalIntRegfileWrites = 0
    totalFunctionCalls = 0
    totalIaluAccesses = 0
    totalFpuAccesses = 0


    # Extract functional unit information
    intalu_count = 0
    intmult_count = 0
    fpu_count = 0
    
    try:
        cpu_config = config["system"]["cpu"]
        if isinstance(cpu_config, list):
            cpu = cpu_config[0]
        else:
            cpu = cpu_config
        
        fu_pool = cpu.get("fuPool", {})
        fu_list = fu_pool.get("FUList", [])
        
        for fu_desc in fu_list:
            fu_count = fu_desc.get("count", 0)
            op_list = fu_desc.get("opList", [])
            
            for op_desc in op_list:
                op_class = op_desc.get("opClass", "")
                
                if op_class == "IntAlu":
                    intalu_count = fu_count
                elif op_class == "IntMult":
                    intmult_count = fu_count
                elif op_class in ["FloatAdd", "FloatMult", "FloatDiv"]:
                    fpu_count = fu_count
                    
    except (KeyError, TypeError, IndexError):
        pass  # Use defaults
    
    if numCores == 1:
        # Single core - use existing keys without core index
        totalInstructions = int(float(stats.get("simInsts")))
        totalBranches = int(float(stats.get("system.cpu.numBranches")))
        totalCycles = int(float(stats.get("system.cpu.numCycles")))
        totalIdleCycles = int(float(stats.get("system.cpu.idleCycles")))
        totalIntAluAccesses = int(float(stats.get("system.cpu.intAluAccesses")))
        totalFpAluAccesses = int(float(stats.get("system.cpu.fpAluAccesses")))
        totalLoadInstructions = int(float(stats.get("system.cpu.numLoadInsts")))
        totalStoreInstructions = int(float(stats.get("system.cpu.numStoreInsts")))
        branchMispredictions = int(float(stats.get("system.cpu.branchPred.condIncorrect")))
        totalcommittedInstructions = int(float(stats.get("system.cpu.committedInsts")))
        total_committed_int_instructions = int(float(stats.get("system.cpu.commit.integer")))
        total_committed_fp_instructions = int(float(stats.get("system.cpu.commit.floating")))
        totalInstWindowReads = int(float(stats.get("system.cpu.intInstQueueReads")))
        totalInstWindowWrites = int(float(stats.get("system.cpu.intInstQueueWrites")))
        totalInstWindowWakeups = int(float(stats.get("system.cpu.intInstQueueWakeupAccesses")))
        totalFpInstWindowReads = int(float(stats.get("system.cpu.fpInstQueueReads")))
        totalFpInstWindowWrites = int(float(stats.get("system.cpu.fpInstQueueWrites")))
        totalFpInstWindowWakeups = int(float(stats.get("system.cpu.fpInstQueueWakeupAccesses")))
        totalIntRegfileReads = int(float(stats.get("system.cpu.intRegfileReads")))
        totalIntRegfileWrites = int(float(stats.get("system.cpu.intRegfileWrites")))
        totalFunctionCalls = int(float(stats.get("system.cpu.commit.functionCalls")))
        totalIaluAccesses = int(float(stats.get("system.cpu.intAluAccesses")))
        totalFpuAccesses = int(float(stats.get("system.cpu.fpAluAccesses")))

        
    else:
        # Multi-core - loop through each core and sum values
        import re
        
        # Create regex patterns for multi-core stats
        patterns = {
            'branches': r'system\.cpu(\d+)\.numBranches', 
            'cycles': r'system\.cpu(\d+)\.numCycles',
            'idle_cycles': r'system\.cpu(\d+)\.idleCycles',
            'int_alu': r'system\.cpu(\d+)\.intAluAccesses',
            'fp_alu': r'system\.cpu(\d+)\.fpAluAccesses',
            'load_insts': r'system\.cpu(\d+)\.numLoadInsts',
            'store_insts': r'system\.cpu(\d+)\.numStoreInsts',
            'branch_mispredictions': r'system\.cpu(\d+)\.branchPred\.condIncorrect',
            'committed_instructions': r'system\.cpu(\d+)\.committedInsts',
            'committed_int_instructions': r'system\.cpu(\d+)\.commit\.integer',
            'committed_fp_instructions': r'system\.cpu(\d+)\.commit\.floating',
            'intInstQueueReads': r'system\.cpu(\d+)\.intInstQueueReads',
            'intInstQueueWrites': r'system\.cpu(\d+)\.intInstQueueWrites',
            'intInstQueueWakeupAccesses': r'system\.cpu(\d+)\.intInstQueueWakeupAccesses',
            'fpInstQueueReads': r'system\.cpu(\d+)\.fpInstQueueReads',
            'fpInstQueueWrites': r'system\.cpu(\d+)\.fpInstQueueWrites',
            'fpInstQueueWakeupAccesses': r'system\.cpu(\d+)\.fpInstQueueWakeupAccesses',
            'intRegfileReads': r'system\.cpu(\d+)\.intRegfileReads',
            'intRegfileWrites': r'system\.cpu(\d+)\.intRegfileWrites',
            'function_calls': r'system\.cpu(\d+)\.commit\.functionCalls',
            'intalu_accesses': r'system\.cpu(\d+)\.intAluAccesses',
            'fpu_accesses': r'system\.cpu(\d+)\.fpAluAccesses'
        }
        
        # Use simInsts for total instructions (global stat)
        totalInstructions = int(float(stats.get("simInsts", 0)))
        
        # Sum up per-core statistics
        for stat_key, stat_value in stats.items():
            for pattern_name, pattern in patterns.items():
                match = re.match(pattern, stat_key)
                if match:
                    if pattern_name == 'branches':
                        totalBranches += int(float(stat_value))
                    elif pattern_name == 'cycles':
                        totalCycles += int(float(stat_value))
                    elif pattern_name == 'idle_cycles':
                        totalIdleCycles += int(float(stat_value))
                    elif pattern_name == 'int_alu':
                        totalIntAluAccesses += int(float(stat_value))
                    elif pattern_name == 'fp_alu':
                        totalFpAluAccesses += int(float(stat_value))
                    elif pattern_name == 'load_insts':
                        totalLoadInstructions += int(float(stat_value))
                    elif pattern_name == 'store_insts':
                        totalStoreInstructions += int(float(stat_value))
                    elif pattern_name == 'branch_mispredictions':
                        branchMispredictions += int(float(stat_value))
                    elif pattern_name == 'committed_instructions':  # Fix typo
                        totalcommittedInstructions += int(float(stat_value))
                    elif pattern_name == 'committed_int_instructions':
                        total_committed_int_instructions += int(float(stat_value))
                    elif pattern_name == 'committed_fp_instructions':
                        total_committed_fp_instructions += int(float(stat_value))
                    elif pattern_name == 'intInstQueueReads':
                        totalInstWindowReads += int(float(stat_value))
                    elif pattern_name == 'intInstQueueWrites':
                        totalInstWindowWrites += int(float(stat_value))
                    elif pattern_name == 'intInstQueueWakeupAccesses':
                        totalInstWindowWakeups += int(float(stat_value))
                    elif pattern_name == 'fpInstQueueReads':
                        totalFpInstWindowReads += int(float(stat_value))
                    elif pattern_name == 'fpInstQueueWrites':
                        totalFpInstWindowWrites += int(float(stat_value))
                    elif pattern_name == 'fpInstQueueWakeupAccesses':
                        totalFpInstWindowWakeups += int(float(stat_value))
                    elif pattern_name == 'intRegfileReads':
                        totalIntRegfileReads += int(float(stat_value))
                    elif pattern_name == 'intRegfileWrites':
                        totalIntRegfileWrites += int(float(stat_value))
                    elif pattern_name == 'function_calls':
                        totalFunctionCalls += int(float(stat_value))
                    elif pattern_name == 'intalu_accesses':
                        totalIaluAccesses += int(float(stat_value))
                    elif pattern_name == 'fpu_accesses':
                        totalFpuAccesses += int(float(stat_value))
    
    # Calculate derived values
    totalBusyCycles = totalCycles - totalIdleCycles
    
    # Return dictionary with all extracted values INCLUDING FU counts
    return {
        'totalInstructions': totalInstructions,
        'totalBranches': totalBranches, 
        'totalCycles': totalCycles,
        'totalIdleCycles': totalIdleCycles,
        'totalBusyCycles': totalBusyCycles,
        'totalIntAluAccesses': totalIntAluAccesses,
        'totalFpAluAccesses': totalFpAluAccesses,
        'totalLoadInstructions': totalLoadInstructions,
        'totalStoreInstructions': totalStoreInstructions,
        'branchMispredictions': branchMispredictions,
        'intalu_count': intalu_count,
        'intmult_count': intmult_count,
        'fpu_count': fpu_count,
        'totalcommittedInstructions': totalcommittedInstructions,
        'total_committed_int_instructions': total_committed_int_instructions,
        'total_committed_fp_instructions': total_committed_fp_instructions,
        'totalInstWindowReads': totalInstWindowReads,
        'totalInstWindowWrites': totalInstWindowWrites,
        'totalInstWindowWakeups': totalInstWindowWakeups,
        'totalFpInstWindowReads': totalFpInstWindowReads,
        'totalFpInstWindowWrites': totalFpInstWindowWrites,
        'totalFpInstWindowWakeups': totalFpInstWindowWakeups,
        'totalIntRegfileReads': totalIntRegfileReads,
        'totalIntRegfileWrites': totalIntRegfileWrites,
        'totalFunctionCalls': totalFunctionCalls,
        'totalIaluAccesses': totalIaluAccesses,
        'totalFpuAccesses': totalFpuAccesses
    }

def prepareTemplate(outputFile):
    # Extract basic configuration
    numCores = len(config["system"]["cpu"])
    numL2 = 1 if 'l2cache' in config["system"] else 0
    numL3 = 1 if 'l3cache' in config["system"] else 0
    
    # Calculate clock rate in MHz
    simFreq = float(stats["simFreq"])
    clockPeriod = float(stats["system.clk_domain.clock"])
    clockRate = simFreq / clockPeriod / 1e6
    
    numCacheLevels = 1 + numL2 + numL3

    # Extract all statistics and configurations
    stats_values = extractStatsValues(stats, numCores)
    
    # Extract CPU configuration parameters
    try:
        cpu_config = config["system"]["cpu"]
        system = config["system"]
        if isinstance(cpu_config, list):
            cpu = cpu_config[0]
        else:
            cpu = cpu_config
            
        cpu_params = {
            'numHardwareThreads': cpu.get("numThreads", 1),
            'fetchWidth': cpu.get("fetchWidth", 8),
            'decodeWidth': cpu.get("decodeWidth", 8),
            'issueWidth': cpu.get("issueWidth", 8),
            'commitWidth': cpu.get("commitWidth", 8),
            'instructionBufferSize': cpu.get("fetchBufferSize", 64),
            'ROBSize': cpu.get("numROBEntries", 192),
            'instructionWindowSize': cpu.get("numIQEntries", 64),
            'global_predictor_entries': cpu.get("branchPred", {}).get("globalPredictorSize", 8192),
            'global_predictor_bits': cpu.get("branchPred", {}).get("globalCtrBits", 2),
            'chooser_predictor_entries': cpu.get("branchPred", {}).get("choicePredictorSize", 8192),
            'chooser_predictor_bits': cpu.get("branchPred", {}).get("choiceCtrBits", 2),
            'local_predictor_entries': cpu.get("branchPred", {}).get("localPredictorSize", 2048),
            'local_predictor_size': cpu.get("branchPred", {}).get("localPredictorSize", 2048),
            
        }

        icache_params = {
            'number_entries': cpu.get("mmu", {}).get("itb", {}).get("size", 64),
            # Extract individual icache parameters - Fixed paths
            'size': cpu.get("icache", {}).get("size", 32768),  
            'associativity': cpu.get("icache", {}).get("assoc", 2),  
            'block_size': cpu.get("icache", {}).get("tags", {}).get("block_size", 64),  
            'response_latency': cpu.get("icache", {}).get("response_latency", 2),  
            'icache_config': None,  # Just needs instantiation
            'mshrs': cpu.get("icache", {}).get("mshrs", 4),
            'write_buffers': cpu.get("icache", {}).get("write_buffers", 8),
            'buffer_sizes': None
        }

        # Build the icache_config string after extracting individual values
        iSize = icache_params.get('size', 32768)  # Default to 32768 if not found
        i_block_size = icache_params.get('block_size', 64)  # Default to 64 if not found  
        iAssociativity = icache_params.get('associativity', 2)  # Default to 2 if not found
        i_response_latency = icache_params.get('response_latency', 2)  # Default to 2 if not found
        icache_params['icache_config'] = f"{iSize},{i_block_size},{iAssociativity},1,10,{i_response_latency},{i_block_size},0"

        i_write_buffers = icache_params.get('write_buffers', 8)  # Default to 8 if not found
        i_mshr_size = icache_params.get('mshrs', 8) 
        icache_params['buffer_sizes'] = f"{i_write_buffers},{i_write_buffers},{i_write_buffers},{i_mshr_size}"

        dcache_params = {
            'number_entries': cpu.get("mmu", {}).get("dtb", {}).get("size", 64),
            # Extract individual dcache parameters - Fixed paths
            'size': cpu.get("dcache", {}).get("size", 32768),  
            'associativity': cpu.get("dcache", {}).get("assoc", 2),  
            'block_size': cpu.get("dcache", {}).get("tags", {}).get("block_size", 64),  
            'response_latency': cpu.get("dcache", {}).get("response_latency", 2),  
            'dcache_config': None,  # Just needs instantiation
            'mshrs': cpu.get("dcache", {}).get("mshrs", 4),
            'write_buffers': cpu.get("dcache", {}).get("write_buffers", 8),
            'buffer_sizes': None
        }
        # Build the icache_config string after extracting individual values
        dSize = dcache_params.get('size', 32768)  # Default to 32768 if not found
        d_block_size = dcache_params.get('block_size', 64)  # Default to 64 if not found  
        dAssociativity = dcache_params.get('assoc', 2)  # Default to 2 if not found
        d_response_latency = dcache_params.get('response_latency', 2)  # Default to 2 if not found
        dcache_params['dcache_config'] = f"{dSize},{d_block_size},{dAssociativity},1,10,{d_response_latency},{d_block_size},0"

        d_write_buffers = dcache_params.get('write_buffers', 8)  # Default to 8 if not found
        d_mshr_size = dcache_params.get('mshrs', 8) 
        dcache_params['buffer_sizes'] = f"{d_write_buffers},{d_write_buffers},{d_write_buffers},{d_mshr_size}"

        btb_params = {
            'BTBConfig': None,
            'btbEntries': cpu.get('branchPred', {}).get('BTBEntries', 4096),
            'btbTagSize': cpu.get('branchPred', {}).get('BTBTagSize', 16)
        }
        btbEntries = btb_params.get('btbEntries', 1024)  # Default to 1024 if not found
        btbTagSize = btb_params.get('btbTagSize', 10)
        btb_params['BTBConfig'] = f"{btbEntries},{btbTagSize},2,1,1"

        l2_params = {
            'clockrate': int(clockRate),
            'size': system.get("l2cache", {}).get("size", 1048576),
            'block_size': system.get("l2cache", {}).get("tags", {}).get("block_size", 64),
            'assoc': system.get("l2cache", {}).get("assoc", 8),
            'latency': system.get("l2cache", {}).get("response_latency", 20),
            'l2_config': None,

            'mshrs': system.get("l2cache", {}).get("mshrs", 20),
            'write_buffer': system.get("l2cache", {}).get("write_buffers", 8),
            'buffer_sizes': None
        }
        l2_size = l2_params.get('size', 262144)  # Default to 262144 if not found
        l2_block_size = l2_params.get('block_size', 64)  # Default to 64 if not found
        l2_assoc = l2_params.get('assoc', 8)  # Default to 8 if not found
        l2_latency = l2_params.get('latency', 10)  # Default to 10 if not found
        l2_params['l2_config'] = f"{l2_size},{l2_block_size},{l2_assoc},8,8,{l2_latency},{l2_block_size},0"

        l2_mshrs = l2_params.get('mshrs', 8)  # Default to 8 if not found
        l2_write_buffer = l2_params.get('write_buffer', 8)  #
        l2_params['buffer_sizes'] = f"{l2_write_buffer},{l2_write_buffer},{l2_write_buffer},{l2_mshrs}"


        l3_params = {
            'clockrate': int(clockRate),
            'size': system.get("l3cache", {}).get("size", 16777216),
            'block_size': system.get("l3cache", {}).get("tags", {}).get("block_size", 64),
            'assoc': system.get("l3cache", {}).get("assoc", 16),
            'latency': system.get("l3cache", {}).get("response_latency", 100),
            'l3_config': None,

            'mshrs': system.get("l3cache", {}).get("mshrs", 16),
            'write_buffer': system.get("l3cache", {}).get("write_buffers", 16),
            'buffer_sizes': None
        }
        l3_size = l3_params.get('size', 262144)  # Default to 262144 if not found
        l3_block_size = l3_params.get('block_size', 64)  # Default to 64 if not found
        l3_assoc = l3_params.get('assoc', 8)  # Default to 8 if not found
        l3_latency = l3_params.get('latency', 10)  # Default to 10 if not found
        l3_params['l3_config'] = f"{l3_size},{l3_block_size},{l3_assoc},8,8,{l3_latency},{l3_block_size},0"

        l3_mshrs = l3_params.get('mshrs', 8)  # Default to 8 if not found
        l3_write_buffer = l3_params.get('write_buffer', 8)  #
        l3_params['buffer_sizes'] = f"{l3_write_buffer},{l3_write_buffer},{l3_write_buffer},{l3_mshrs}"

        # Memory controller configuration - simplified to always use 1 MC
        try:
            # Get memory controller (single controller assumed)
            mem_ctrl = system.get("mem_ctrl", {})
            
            # Count memory channels per controller
            memory_channels_per_mc = 1  # Default to 1 channel
            
            if mem_ctrl:
                # Method 1: Check if dram is a list (multiple channels)
                if isinstance(mem_ctrl.get("dram"), list):
                    memory_channels_per_mc = len(mem_ctrl["dram"])
                # Method 2: Check if dram is a single object (single channel)
                elif "dram" in mem_ctrl:
                    memory_channels_per_mc = 1
                # Method 3: Look for numbered dram interfaces (dram0, dram1, etc.)
                else:
                    dram_index = 0
                    while f"dram{dram_index}" in mem_ctrl:
                        dram_index += 1
                    memory_channels_per_mc = dram_index if dram_index > 0 else 1
            
            # Get DRAM parameters from first available DRAM interface
            dram_interface = None
            if mem_ctrl:
                if isinstance(mem_ctrl.get("dram"), list):
                    dram_interface = mem_ctrl["dram"][0]
                elif "dram" in mem_ctrl:
                    dram_interface = mem_ctrl["dram"]
                elif "dram0" in mem_ctrl:
                    dram_interface = mem_ctrl["dram0"]
            
            # Extract memory controller parameters
            mc_params = {
                'tCK': dram_interface.get("tCK", 1250) if dram_interface else 1250,  # Default DDR3-1600 timing
                'block_size': system.get("cache_line_size", 64), 
                'number_mcs': 1,  # Always 1 MC as requested
                'mc_clock': None,
                'memory_channels_per_mc': memory_channels_per_mc,
                'number_ranks': dram_interface.get("ranks_per_channel", 2) if dram_interface else 2,
                'req_window_size_per_channel': dram_interface.get("read_buffer_size", 32) if dram_interface else 32,
                'IO_buffer_size_per_channel': dram_interface.get("write_buffer_size", 64) if dram_interface else 64,
                'device_bus_width': dram_interface.get("device_bus_width", 8) if dram_interface else 8,
                'devices_per_rank': dram_interface.get("devices_per_rank", 8) if dram_interface else 8,
                'databus_width': None
            }
            
            print(f"DEBUG: Using 1 memory controller with {mc_params['memory_channels_per_mc']} channel(s)")
            
        except (KeyError, TypeError, IndexError) as e:
            print(f"WARNING: Could not parse memory controller configuration: {e}")
            # Fallback to defaults
            mc_params = {
                'tCK': 1250,  # Default DDR3-1600 timing
                'block_size': 64, 
                'number_mcs': 1,
                'mc_clock': None,
                'memory_channels_per_mc': 1,
                'number_ranks': 2,
                'req_window_size_per_channel': 32,
                'IO_buffer_size_per_channel': 64,
                'device_bus_width': 8,
                'devices_per_rank': 8,
                'databus_width': None
            }
        
        # Calculate derived values
        tCK = mc_params.get('tCK', 1250)  # in picoseconds
        mc_params['mc_clock'] = int(1000000 / (tCK / 1000))  # Convert ps to MHz
        mc_params['databus_width'] = mc_params['device_bus_width'] * mc_params['devices_per_rank']

        

    except (KeyError, TypeError, IndexError):
        cpu_params = {}
        icache_params = {}
        dcache_params = {}  
        btb_params = {}
        l2_params = {}
        l3_params = {}
        mc_params = {}

    # Get the root element
    root = templateMcpat.getroot()
    
    # Define all component configurations with their mappings
    components_config = {
        "system": {
            "xpath": ".//component[@id='system']",
            "params": {
                "number_of_cores": numCores,
                "number_of_L2s": numL2,
                "number_of_L3s": numL3,
                "number_cache_levels": numCacheLevels,
                "target_core_clockrate": int(clockRate)
            },
            "stats": {
                "total_cycles": stats_values['totalCycles'],
                "idle_cycles": stats_values['totalIdleCycles'],
                "busy_cycles": stats_values['totalBusyCycles']
            }
        },
        
        "core0": {
            "xpath": ".//component[@id='system.core0']",
            "params": {
                "clock_rate": int(clockRate),
                "ALU_per_core": stats_values['intalu_count'],
                "MUL_per_core": stats_values['intmult_count'],
                "FPU_per_core": stats_values['fpu_count'],
                "number_of_BTB": numCores,  # Dynamic: 1 BTB per core
                **{k: v for k, v in cpu_params.items() if v is not None and k not in ['global_predictor_entries', 'global_predictor_bits', 'chooser_predictor_entries', 'chooser_predictor_bits', 'local_predictor_entries']}
            },
            "stats": {
                "total_instructions": stats_values['totalInstructions'],
                "branch_instructions": stats_values['totalBranches'],
                "branch_mispredictions": stats_values['branchMispredictions'],
                "load_instructions": stats_values['totalLoadInstructions'],
                "store_instructions": stats_values['totalStoreInstructions'],
                "ialu_accesses": stats_values['totalIntAluAccesses'],
                "fpu_accesses": stats_values['totalFpAluAccesses'],
                "committed_instructions": stats_values['totalcommittedInstructions'],
                "committed_int_instructions": stats_values['total_committed_int_instructions'],
                "committed_fp_instructions": stats_values['total_committed_fp_instructions'],
                "inst_window_reads": stats_values['totalInstWindowReads'],
                "inst_window_writes": stats_values['totalInstWindowWrites'],
                "inst_window_wakeups": stats_values['totalInstWindowWakeups'],
                "fp_inst_window_reads": stats_values['totalFpInstWindowReads'],
                "fp_inst_window_writes": stats_values['totalFpInstWindowWrites'],
                "fp_inst_window_wakeups": stats_values['totalFpInstWindowWakeups'],
                "int_regfile_reads": stats_values['totalIntRegfileReads'],
                "int_regfile_writes": stats_values['totalIntRegfileWrites'],
                "function_calls": stats_values['totalFunctionCalls'],
                "ialu_accesses": stats_values['totalIaluAccesses'],
                "fpu_accesses": stats_values['totalFpuAccesses']
            }
        },
        
        # ADD THIS NEW SECTION FOR PREDICTOR
        "core0.predictor": {
            "xpath": ".//component[@id='system.core0.predictor']",
            "params": {
                "local_predictor_entries": cpu_params.get('local_predictor_entries'),
                "global_predictor_entries": cpu_params.get('global_predictor_entries'),
                "global_predictor_bits": cpu_params.get('global_predictor_bits'),
                "chooser_predictor_entries": cpu_params.get('chooser_predictor_entries'),
                "chooser_predictor_bits": cpu_params.get('chooser_predictor_bits')
            }
        },

        "core0.itlb":{
            "xpath": ".//component[@id='system.core0.itlb']",
            "params": {
                "number_entries": icache_params.get('number_entries')
            }
        },

       "core0.icache":{
            "xpath": ".//component[@id='system.core0.icache']",
            "params": {
                "icache_config": icache_params.get('icache_config'),
                "buffer_sizes": icache_params.get('buffer_sizes')
            },
            "stats": {
                "read_accesses": int(float(stats.get("system.cpu.icache.demandAccesses::total", 0))),
                "read_misses": int(float(stats.get("system.cpu.icache.demandMisses::total", 0)))
            }
        },

         "core0.dtlb":{
            "xpath": ".//component[@id='system.core0.dtlb']",
            "params": {
                "number_entries": dcache_params.get('number_entries')
            }
        },

         "core0.dcache":{
            "xpath": ".//component[@id='system.core0.dcache']",
            "params": {
                "dcache_config": dcache_params.get('dcache_config'),
                "buffer_sizes": dcache_params.get('buffer_sizes')
            },
            "stats": {
                "read_accesses": int(float(stats.get("system.cpu.dcache.demandAccesses::total", 0))),
                "write_accesses": int(float(stats.get("system.cpu.dcache.WriteReq.accesses::total", 0))),
                "read_misses": int(float(stats.get("system.cpu.dcache.ReadReq.misses::total", 0))),
                "write_misses": int(float(stats.get("system.cpu.dcache.WriteReq.misses::total", 0)))
            }
        },
        "core0.BTB":{
            "xpath": ".//component[@id='system.core0.BTB']",
            "params": {
                "BTB_config": btb_params.get('BTBConfig')
            },
            "stats": {
               "read accesses": int(float(stats.get("system.cpu.branchPred.BTBLookups", 0))),  # ✅ Fixed
            }
        },
        
        "L1Directory0": {
            "xpath": ".//component[@id='system.L1Directory0']",
            "params": {"clockrate": int(clockRate)}
        },
        
        "L2Directory0": {
            "xpath": ".//component[@id='system.L2Directory0']",
            "params": {"clockrate": int(clockRate)}
        },
        
        "L20": {
            "xpath": ".//component[@id='system.L20']",
            "condition": numL2 > 0,
            "params": {
                "clockrate": int(clockRate),
                "L2_config": l2_params.get('l2_config'),
                "buffer_sizes": l2_params.get('buffer_sizes')
            },
            "stats": {
                'read_accesses': int(float(stats.get("system.l2cache.demandAccesses::total", 0))),
                'write_accesses': int(float(stats.get("system.l2cache.ReadExReq.accesses::total", 0))),
                'read_misses': int(float(stats.get("system.l2cache.demandMisses::total", 0))),
                'write_misses': int(float(stats.get("system.l2cache.ReadExReq.misses::total", 0)))
            }
        },  
        "L30": {
            "xpath": ".//component[@id='system.L30']",
            "condition": numL3 > 0,
            "params": {
                "clockrate": int(clockRate),
                "L3_config": l3_params.get('l3_config'),
                "buffer_sizes": l3_params.get('buffer_sizes')},

            "stats": {
                'read_accesses': int(float(stats.get("system.l3cache.demandAccesses::total", 0))),
                'write_accesses': int(float(stats.get("system.l3cache.ReadExReq.accesses::total", 0))),
                'read_misses': int(float(stats.get("system.l3cache.demandMisses::total", 0))),
                'write_misses': int(float(stats.get("system.l3cache.ReadExReq.misses::total", 0)))
            }
        },
        
        "mc": {
            "xpath": ".//component[@id='system.mc']",
            "params": {
                "mc_clock": mc_params.get('mc_clock'),
                "block_size": mc_params.get('block_size'),
                "number_mcs": mc_params.get('number_mcs'),
                "memory_channels_per_mc": mc_params.get('memory_channels_per_mc'),
                "number_ranks": mc_params.get('number_ranks'),
                "req_window_size_per_channel": mc_params.get('req_window_size_per_channel'),
                "IO_buffer_size_per_channel": mc_params.get('IO_buffer_size_per_channel'),
                "databus_width": mc_params.get('device_bus_width')
                }
        }
    }  
    
    # Process all components
    processed_updates = set()  # Track what we've already updated
    
    for comp_name, comp_config in components_config.items():
        # Check condition if specified (default to True)
        if comp_config.get("condition", True):
            component = root.find(comp_config["xpath"])
            if component is not None:
                # Process parameters
                for child in component:
                    if child.tag == 'param':
                        param_name = child.get("name")
                        if param_name in comp_config.get("params", {}):
                            update_key = f"{comp_name}.{param_name}"
                            if update_key not in processed_updates:
                                value = comp_config["params"][param_name]
                                child.set("value", str(value))
                                print(f"Updated {comp_name} {param_name} to {value}")
                                processed_updates.add(update_key)
                    
                    elif child.tag == 'stat':
                        stat_name = child.get("name")
                        if stat_name in comp_config.get("stats", {}):
                            update_key = f"{comp_name}.{stat_name}"
                            if update_key not in processed_updates:
                                value = comp_config["stats"][stat_name]
                                child.set("value", str(value))
                                print(f"Updated {comp_name} {stat_name} to {value}")
                                processed_updates.add(update_key)

def createMcpatInput():
    """Create the final McPAT input XML file"""
    global templateMcpat, args
    
    try:
        root = templateMcpat.getroot()
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="\t", newl="\n")
        
        # Clean up the output
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        
        final_xml = '\n'.join(lines)
        
        args.output.write('<?xml version="1.0" ?>\n')
        args.output.write(final_xml)
        args.output.close()
        
        print(f"McPAT input file created: {args.output.name}")
        
    except Exception as e:
        print(f"Error creating McPAT input file: {e}")
        sys.exit(1)

def main():
    global args, DEBUG
    parser = create_parser()
    args = parser.parse_args()
    
    DEBUG = args.verbose
    
    readStatsFile(args.stats)
    readConfigFile(args.config)
    readMcpatFile(args.template)

    prepareTemplate(args.output)
    createMcpatInput()

if __name__ == '__main__':
    main()
