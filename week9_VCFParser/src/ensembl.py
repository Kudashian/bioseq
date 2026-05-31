import requests

def get_ensembl_info(variant):
    """
    Retrieves Ensembl information for a given variant.

    Parameters:
    variant (Variant): A variant object containing the necessary information to query Ensembl.

    Returns:
    json: A JSON object containing the Ensembl information for the specified variant.
    """
    #reformat the variant information to match Ensembl's API requirements
    request_data = f"{variant.CHROM.lstrip('chr')}:{variant.POS}-{variant.POS}:1/{variant.ALT[0]}"
    url = f"https://rest.ensembl.org/vep/human/region/{request_data}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    return response.json()

def parse_ensembl_response(response):
    """
    Parses the Ensembl API response to extract relevant information.

    Parameters:
    response (json): The JSON response from the Ensembl API.

    Returns:
    dict: A dictionary containing the parsed information from the Ensembl response.
    """
    if not response:
        return {}
    
    transcript_consequences = response[0].get("transcript_consequences", [{}])
    # Extract relevant information from the response
    parsed_info = {
        "consequence": response[0].get("most_severe_consequence", "N/A"),
        'gene_symbol': transcript_consequences[0].get("gene_symbol", "N/A"),
        "impact": transcript_consequences[0].get("impact", "N/A"),
        "amino_acid_change": transcript_consequences[0].get("amino_acids", "N/A"),
        "protein_position": transcript_consequences[0].get("protein_position", "N/A")
    }
    
    return parsed_info