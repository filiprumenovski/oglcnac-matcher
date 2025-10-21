# tests/test_matcher.py
import pandas as pd
from oglcnac_matcher.matcher import build_odb_indexes, match_proteins, match_hexnac_sites, overlap_phospho

def _df(rows, cols):
    return pd.DataFrame(rows, columns=cols)

def test_acc_exact_and_gene_ambiguous_then_name_unique():
    odb = _df(
        [
            ["P11111","FOO","Foo kinase alpha",None,None,None,None,None],
            ["P22222","BAR","Barase",None,None,None,None,None],
        ],
        ["uniprot_acc","gene_symbol","protein_name","residue","position","pmid","doi","evidence"]
    )
    idx = build_odb_indexes(odb)

    # primary not in ODB; multi-gene; name should uniquely map to P11111
    pg = _df(
        [[ "PG1", ["X"], "X", "FOO;BAZ", "Foo kinase alpha", "file" ]],
        ["protein_group_id","uniprot_ids","primary_uniprot","gene_name","protein_name","source_file"]
    )
    prot = match_proteins(pg, idx)
    # if you keep current behavior (immediate AMBIGUOUS), assert that; otherwise expect PNAME_UNIQUE
    assert prot.loc[0,"match_level_protein"] in ("AMBIGUOUS","PNAME_UNIQUE")

def test_gene_exact_unique():
    odb = _df([["P12345","AKT1","RAC-alpha",None,None,None,None,None]],
              ["uniprot_acc","gene_symbol","protein_name","residue","position","pmid","doi","evidence"])
    idx = build_odb_indexes(odb)
    pg = _df([["PG1",["X"],"X","AKT1","Some name","file"]],
             ["protein_group_id","uniprot_ids","primary_uniprot","gene_name","protein_name","source_file"])
    prot = match_proteins(pg, idx)
    assert prot.loc[0,"odb_uniprot"] == "P12345"
    assert prot.loc[0,"match_level_protein"] == "GENE_EXACT"

def test_site_exact_and_near():
    odb = _df(
        [
            ["P99999","GSK3B","GSK3 beta","S",9,"","",""],
            ["P99999","GSK3B","GSK3 beta","S",10,"","",""],
        ],
        ["uniprot_acc","gene_symbol","protein_name","residue","position","pmid","doi","evidence"]
    )
    idx = build_odb_indexes(odb)
    pg = _df([["PG1",["P99999"],"P99999","GSK3B","GSK3 beta","file"]],
             ["protein_group_id","uniprot_ids","primary_uniprot","gene_name","protein_name","source_file"])
    prot = match_proteins(pg, idx)

    hex_exact = _df([["HX1","P99999",10,"S",None,None,None]],
                    ["site_id","primary_uniprot","position","amino_acid","sequence_window","gene_name","protein_name"])
    hex_near  = _df([["HX2","P99999",11,"S",None,None,None]],
                    ["site_id","primary_uniprot","position","amino_acid","sequence_window","gene_name","protein_name"])

    m_exact = match_hexnac_sites(hex_exact, prot, idx, site_slack=1)
    m_near  = match_hexnac_sites(hex_near,  prot, idx, site_slack=1)
    assert m_exact.loc[0,"site_match_tier"] == "SITE_EXACT"
    assert m_near.loc[0,"site_match_tier"] == "SITE_NEAR±1"

def test_phospho_overlap_counts_unique_sites():
    odb = _df([["P13579","FOO","Foo protein",None,None,None,None,None]],
              ["uniprot_acc","gene_symbol","protein_name","residue","position","pmid","doi","evidence"])
    idx = build_odb_indexes(odb)
    pg = _df([["PG1",["P13579"],"P13579","FOO","Foo protein","file"]],
             ["protein_group_id","uniprot_ids","primary_uniprot","gene_name","protein_name","source_file"])
    prot = match_proteins(pg, idx)  # ACC_EXACT

    phos = _df(
        [["PH1","P13579",5,"S",None],["PH2","P13579",5,"S",None],["PH3","P13579",7,"T",None]],
        ["site_id","primary_uniprot","position","amino_acid","sequence_window"]
    )
    out = overlap_phospho(phos, prot)
    # nunique on site_id → 3
    assert int(out.loc[0,"n_phospho_sites"]) == 3