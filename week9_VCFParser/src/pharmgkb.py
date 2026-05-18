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
    response = response.json()

    #Now we need to filter the response to find the relevant annotation for our variant using position
    variantannotations = []

    for annotation in response.get('data', []):
        variantannotations.append({
        "id": annotation['id'],
        "drug": [chem.get('name') for chem in annotation.get('relatedChemicals', [])],
        "isAssociated": annotation.get('isAssociated'),
        "significance": annotation.get('significance'),
        "evidence": annotation.get('literature', {}).get('id')
    })

    return variantannotations