query_pharmgkb_schema = {
    "name": "query_pharmgkb",
    "description": "Query PharmGKB for gene-drug annotations. Use when the variant affects a known pharmacogene or AF < 0.01.",
    "input_schema": {
        "type": "object",
        "properties": {
            "gene_symbol": {
                "type": "string",
                "description": "HGNC gene symbol e.g. CYP2C19"
            }
        },
        "required": ["gene_symbol"]
    }
}

query_clinvar_schema = {
    "name": "query_clinvar",
    "description": "Query ClinVar for variant pathogenicity. Use when AF >= 0.01 or not a pharmacogene.",
    "input_schema": {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "ClinVar esearch query e.g. 'CYP2C19[gene] AND pathogenic[clinsig]'"
            }
        },
        "required": ["term"]
    }
}