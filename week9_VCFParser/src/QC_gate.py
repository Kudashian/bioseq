def requires_pass_qc(func):
    def wrapper(variant, *args, **kwargs):
        if variant.FILTER != 'PASS':
            return None  # or raise an exception, depending on your needs
        for sample, genotype in list(variant.GENOTYPES.items()): #Good practice to use list() here since we might modify GENOTYPES during iteration
            if genotype.DP is None or float(genotype.DP) < 10:
                del variant.GENOTYPES[sample]  # Remove sample from GENOTYPES if it fails DP QC
                continue
            if genotype.GQ is None or float(genotype.GQ) < 20:
                del variant.GENOTYPES[sample]  # Remove sample from GENOTYPES if it fails GQ QC
                continue
        if not variant.GENOTYPES:  # If all samples were removed, skip this variant
            print(f"Variant {variant.ID} at {variant.CHROM}:{variant.POS} failed QC and was skipped.")
            return None
        # if passes → call func(variant, *args, **kwargs)
        return func(variant, *args, **kwargs)
    return wrapper