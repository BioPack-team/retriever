from retriever.data_tiers.tier_1.elasticsearch.attribute_types import AttrFieldMeta, ESRegexQuery, RegexTerm


def validate_regex(regex: str) -> str:
    # todo
    pass

def handle_match(
        field_meta_info: AttrFieldMeta,
        raw_value,
        target_field_name: str
) -> ESRegexQuery:
    """Generate ES query term for match-regex constraint."""
    # `matches`

    # 0. rule out meaningless queries
    if field_meta_info["value_type"] not in ["keyword", "text"]:
        '''
        !!! be very careful with regex against text field.
        While it's allowed, A regex term `GO:` will not match the text `We have GO:123` because of standard tokenizer used by ES
        '''
        raise TypeError(f"{field_meta_info["value_type"]} fields does not support RegEx query")

    # 1. check if valid / efficient regex
    regex_term = validate_regex(raw_value)

    # 2. generate ES query


    '''
    example 
    {
        "regexp": {
            agent_type: {
                "value": "manual_*",
                "case_insensitive": True
            }
        }
    }
    '''

    query_term = ESRegexQuery(
        regexp={
            target_field_name: RegexTerm(
                value=regex_term,
                case_sensitive=True
            )
        }
    )

    return query_term