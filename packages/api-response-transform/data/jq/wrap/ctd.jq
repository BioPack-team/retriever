# Split pubMedIDs by |
((.[] | select(.PubMedIDs != null) | .PubMedIDs) |= split("|")) | 
((.[] | select(.PubMedIds != null) | .PubMedIds) |= split("|")) | 
# Split Direct Evidence by |
((.[] | select(.DirectEvidence != null) | .DirectEvidence) |= split("|")) | 
{data: .}
