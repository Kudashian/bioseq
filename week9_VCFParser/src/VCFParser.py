from dataclasses import dataclass, fields
from random import sample
from typing import Optional, List

from matplotlib.pylab import sample


@dataclass
class Variant:
    CHROM: str
    POS: int
    ID: str
    REF: str
    ALT: str
    QUAL: Optional[float]
    FILTER: str
    INFO: dict
    GENOTYPES: dict  # sample_id → Genotype

@dataclass
class Genotype:
    GT: Optional[str] = None
    AD: Optional[str] = None
    DP: Optional[float] = None
    GQ: Optional[float] = None
    PL: Optional[str] = None

class VCFParser:    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header = []
        self.samples = []

    def __enter__(self):
        self.file = open(self.filepath, 'r')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def __iter__(self):
        for line in self.file:
            line = line.strip()
            if line.startswith('##'):
                self.header.append(line)
            elif line.startswith('#'):
                self.samples = line.split('\t')[9:]  # Sample names start from 10th column
            else:
                fields = line.split('\t')
                chrom, pos, id_, ref, alt, qual, filter_, info_str = fields[:8]
                info = dict(item.split('=') for item in info_str.split(';') if '=' in item)
                variant = Variant(CHROM=chrom, POS=int(pos), ID=id_, REF=ref, ALT=alt,
                                  QUAL=float(qual) if qual != '.' else None,
                                  FILTER=filter_, INFO=info, GENOTYPES={})
                if len(fields) > 8:
                    format_keys = fields[8].split(':')  # GT:AD:DP:GQ:PL
                    for sample, sample_data in zip(self.samples, fields[9:]):
                        values = sample_data.split(':')
                        # zip format_keys with values → dict
                        if len(format_keys) != len(values):
                            values = values + [None] * (len(format_keys) - len(values))
                        genotype_data = dict(zip(format_keys, values))
                        # build Genotype object from that dict
                        known_fields = {'GT', 'AD', 'DP', 'GQ', 'PL'}
                        filtered_data = {k: v for k, v in genotype_data.items() 
                                        if k in known_fields}
                        variant.GENOTYPES[  sample] = Genotype(**filtered_data)
                yield variant
