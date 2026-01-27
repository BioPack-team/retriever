from retriever.types.trapi import AttributeConstraintDict, OperatorEnum

# sample
# {
#   "id": "biolink:publications",
#   "name": "Must have 3+ publications",
#   "operator": ">",
#   "value": [2,3],
# }


base_constraint: AttributeConstraintDict ={
    "id": "biolink:has_total",
    "name": "total value must be greater than 2",
    "operator": ">",
    "value": 2
}

base_negation_constraint: AttributeConstraintDict ={
    "id": "biolink:has_total",
    "name": "total value must not be greater than 4",
    "operator": ">",
    "value": 4,
    "not": True
}

base_node_constraint: AttributeConstraintDict ={
    "id": "biolink:chembl_natural_product",
    "name": "Ensure natural product",
    "operator": OperatorEnum.STRICT_EQUAL,
    "value": True,
}

base_node_negation_constraint: AttributeConstraintDict ={
    "id": "biolink:chembl_prodrug",
    "name": "Ensure not prod drug",
    "operator": OperatorEnum.STRICT_EQUAL,
    "value": True,
    "not": True
}


# Valid patterns (Should pass)
valid_regex_patterns = [
    "GO:0055085",
    "GO:00.*",
    "DOID:2236",
    "MESH:D017092",
    "HP:0001993",
    "UniProtKB:Q9Y3N9",
    "GO:0072.*",
    "MESH:D0.*",
    "RGD:13.*55",
    "abc.*def",
    "HP:0002[0-9]{2}",
    "A[0-9]+",
    "abc{2}",
    "abc{2,3}",
    "(foo|bar)",
    r"foo\.",
    r"\(text\)",
    "(GO:0072).*"
]

# Invalid patterns (Should raise ValueError)
invalid_regex_patterns = [
    # --- Empty / Non-string ---
    None,
    "",
    "   ",

    # --- Length > 50 ---
    "a" * 51,

    # --- Leading Wildcard / Leading Expansion ---
    ".*Q9Y3N9",
    ".+HP:0001993",
    ".?MESH:D017092",
    "[A-Z]*:Q9Y3N9",
    "(abc)+def",  # Leading group with quantifier
    "[A-Za-z0-9_:]+",  # Leading class with quantifier

    # --- Nested Quantifiers ---
    "(a+)+",
    "(AB*)+",
    "prefix(.+)?",

    # --- Lazy Quantifiers ---
    "GO:0072.*?",
    "HP:0001+?",
    "A{2,5}?",

    # --- Anchors ---
    "^GO:0072354",
    "HP:0001993$",

    # --- Lookaround ---
    "Q9(?=3)Y3N9",
    "(?!HP)",

    # --- Unsupported Escapes ---
    r"UniProtKB:\d+",
    r"GO:\w+",
    r"HP:\s+",

    # --- Unsupported ES Features (Valid in Python, Invalid/Dangerous in ES) ---
    r"(a)\1",  # Backreference
    r"(?P<id>\d+)",  # Named Group

    # --- Invalid Regex Syntax (or Syntax Mismatch) ---
    "GO:00{",  # Unescaped { not followed by valid quantifier
    "(",  # Unclosed group
    "HP:[0-9",  # Unclosed class
]

def make_regex_constraint(pattern: str, field="original_subject", name="a regex test") -> AttributeConstraintDict:
    """Wrapper for attribute constraint with regex query"""
    return {
        "id": f"biolink:{field}",
        "name": f"{name}",
        "operator": "matches",
        "value": pattern,
    }

VALID_REGEX_CONSTRAINTS = [make_regex_constraint(pattern) for pattern in valid_regex_patterns]
INVALID_REGEX_CONSTRAINTS = [make_regex_constraint(pattern) for pattern in invalid_regex_patterns]

ATTRIBUTE_CONSTRAINTS = [base_constraint, base_negation_constraint]
NODE_CONSTRAINTS = [base_node_constraint, base_node_negation_constraint]
