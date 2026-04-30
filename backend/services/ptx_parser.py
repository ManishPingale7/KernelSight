import re
from typing import Dict, List

class PTXParser:
    """
    Parses PTX assembly files and extracts basic instruction counts and metadata.
    """
    def __init__(self):
        # Regular expressions for different PTX instruction categories
        self.patterns = {
            'memory_load': re.compile(r'^\s*(@\w+\s+)?ld\.(global|shared|local|const)\.'),
            'memory_store': re.compile(r'^\s*(@\w+\s+)?st\.(global|shared|local)\.'),
            'compute': re.compile(r'^\s*(@\w+\s+)?(add|sub|mul|mad|div|fma|setp)\.'),
            'branch': re.compile(r'^\s*(@\w+\s+)?(bra|call|ret)'),
            'register': re.compile(r'^\s*\.reg\s+\.\w+\s+%[a-zA-Z0-9_]+<(\d+)>'),
            'sync': re.compile(r'^\s*(@\w+\s+)?bar\.sync')
        }

    def parse_file(self, filepath: str) -> Dict[str, int]:
        """Reads a PTX file and counts instruction types."""
        with open(filepath, 'r') as f:
            content = f.read()
        return self.parse_text(content)

    def parse_text(self, ptx_text: str) -> Dict[str, int]:
        """Parses PTX text and returns counts for each instruction category."""
        counts = {
            'memory_load': 0,
            'memory_store': 0,
            'compute': 0,
            'branch': 0,
            'register_count': 0,
            'sync': 0,
            'total_instructions': 0
        }

        for line in ptx_text.splitlines():
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Simple heuristic for actual instructions (lines ending with semicolon)
            # Not all directives are instructions but registers are declarations
            is_instr_or_decl = line.endswith(';')
            
            if is_instr_or_decl:
                # registers are a bit different, looking for formats like .reg .f32 %f<10>;
                reg_match = self.patterns['register'].search(line)
                if reg_match:
                    counts['register_count'] += int(reg_match.group(1))
                    continue # registers are declarations, not typical instructions to count as ops
                    
                counts['total_instructions'] += 1

                if self.patterns['memory_load'].search(line):
                    counts['memory_load'] += 1
                elif self.patterns['memory_store'].search(line):
                    counts['memory_store'] += 1
                elif self.patterns['compute'].search(line):
                    counts['compute'] += 1
                elif self.patterns['branch'].search(line):
                    counts['branch'] += 1
                elif self.patterns['sync'].search(line):
                    counts['sync'] += 1

        return counts
