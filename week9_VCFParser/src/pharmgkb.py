import requests

def get_pharmgkb_info(parsed_ensembl_info):
    '''Retrieves PharmGKB information for a given variant.

    Parameters: variant (Variant) - a dictionary containing a variant and Ensembl information.
                parsed_ensembl_info (dict) - a dictionary containing parsed Ensembl information for a variant.

    Returns: json - a JSON object containing the PharmGKB information for the specified variant.'''
    
    params = {"location.genes.symbol": parsed_ensembl_info['gene_symbol']}
    url = f"https://api.clinpgx.org/v1/data/variantAnnotation?"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, params=params, headers=headers)
    raw_pharmgkb    = response.json()

    return raw_pharmgkb

def parse_pharmgkb_response(raw_pharmgkb):
    '''Parses the PharmGKB response to extract relevant annotations for the variant.
    Parameters: raw_pharmgkb (json) - a JSON object containing the raw PharmGKB response for a variant.
    
    Returns: list - a list of dictionaries, each containing relevant annotations for the variant from PharmGKB.'''
    parsed_pharmgkb_info = []

    for annotation in raw_pharmgkb.get('data', []):
        parsed_pharmgkb_info.append({
        "id": annotation['id'],
        "drug": [chem.get('name') for chem in annotation.get('relatedChemicals', [])],
        "isAssociated": annotation.get('isAssociated'),
        "significance": annotation.get('significance'),
        "evidence": annotation.get('literature', {}).get('id')
    })

    return parsed_pharmgkb_info