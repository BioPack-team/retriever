# split drug IDs with CHEMBL by /
(.data | .[] | select(.drug.id != null) | select(.drug.id | contains("CHEMBL")) | .drug.id) |= (split("/") | last)
