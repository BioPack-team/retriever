from retriever.data_tiers.tier_1.elasticsearch.constraints.types.attribute_types import (
    AttrFieldMeta,
)

keyword_field = AttrFieldMeta(container="scalar", value_type="keyword", curie=False)

curie_keyword_field = AttrFieldMeta(
    container="scalar",
    value_type="keyword",
    curie=True,
)

text_field = AttrFieldMeta(container="scalar", value_type="text", curie=False)

curie_keyword_array_field = AttrFieldMeta(
    container="array",
    value_type="keyword",
    curie=True,
)

number_field = AttrFieldMeta(container="scalar", value_type="num", curie=False)

datetime_field = AttrFieldMeta(container="scalar", value_type="date", curie=False)

ATTR_META = {
    ## fields below are EDGE attributes
    "knowledge_level": keyword_field,
    "agent_type": keyword_field,
    "original_object": curie_keyword_field,
    "original_predicate": curie_keyword_field,
    "original_subject": curie_keyword_field,
    "allelic_requirement": curie_keyword_field,
    ##### array $$$$$
    "publications": curie_keyword_array_field,
    "has_evidence": curie_keyword_array_field,
    # ================
    "update_date": datetime_field,
    "z_score": number_field,
    "has_confidence_score": number_field,
    "has_count": number_field,
    "has_percentage": number_field,
    "has_quotient": number_field,
    "has_total": number_field,
    ## fields below are NODE attributes
    "information_content": number_field,
    "description": text_field,
    "in_taxon": curie_keyword_field,
    ##### array $$$$$
    "equivalent_identifiers": curie_keyword_array_field,
    # ================
    # no record with these fields, skipped for now
    "provided_by": None,
    "inheritance": None,
}
