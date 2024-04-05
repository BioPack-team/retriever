const API_BATCH_SIZE = [
  // pending APIs with template-based querying (POST / batch-query IDs)
  {
    id: '1d288b3a3caf75d541ffaae3aab386c8',
    name: 'BioThings SEMMEDDB API',
    max: 50,
  },
  {
    id: '978fe380a147a8641caf72320862697b',
    name: 'Text Mining Targeted Association API',
    max: 50,
  },
  {
    id: '02af7d098ab304e80d6f4806c3527027',
    name: 'Multiomics Wellness KP API',
    max: 100,
  },
  {
    id: 'adf20dd6ff23dfe18e8e012bde686e31',
    name: 'Multiomics BigGIM-DrugResponse KP API',
    max: 100,
  },
  {
    id: '0212611d1c670f9107baf00b77f0889a',
    name: 'CTD API',
    max: 80,
  },
];

// max node IDs an edge with no other IDs can have
const ENTITY_MAX = 1000;

const EDGE_ATTRIBUTES_USED_IN_RECORD_HASH = [
  // not sure which APIs these are from:
  // perhaps Multiomics and the attribute-type-ids have changed
  'biolink:has_disease_context',
  'biolink:GeneToDrugAssociation',
  // for multiomics wellness 2023-05-25: may later change?
  'MeSH:D005260', // gender female
  'MeSH:D008297', // gender male
  'UMLS CUI:C0001948', // alcohol consumption?
  'UMLS CUI:C0005680', // black population?
  'UMLS CUI:C0043157', // population white
  'UMLS CUI:C0086409', // hispanic population?
  'UMLS CUI:C0425379', // other race?
  'UMLS CUI:C0453995', // tobacco use and exposure?
  'UMLS CUI:C1515945', // American Indian or Alaska Native?
  'UMLS CUI:C1519427', // south asian people?
  'UMLS CUI:C2229974', // children
  'UMLS CUI:C2698217', // middle eastern?
  'UMLS CUI:C4316909', // Marijuana Use?
  'UMLS CUI:C5205795', // east asian people
  'UMLS CUI:C5418925', // study age range
  // bonferroni p-value: would maybe work? but Gwênlyn said not needed
  //"NCIT:C61594",
  // Multiomics BigGIM Drug-Response 2023-05-31: may later change?
  'biolink:context_qualifier',
  // commenting it out since I haven't tested if it works for this KP or
  //   if it'll cause bugs when processing other KPs.
  //   It is only needed to differentiate records from some operations
  // "biolink:publications",
  // for multiomics ehr risk 2023-06-01: may later change?
  'biolink:p_value',
  'STATO:0000209', // auroc
  'biolink:log_odds_ratio',
  'biolink:total_sample_size',
];

// based on https://github.com/biolink/biolink-model/blob/master/infores_catalog.yaml
const text_mining_api_infores = [
  'infores:biothings-semmeddb',
  'infores:scibite',
  'infores:semmeddb',
  'infores:text-mining-provider-cooccurrence',
  'infores:text-mining-provider-targeted',
];

export { API_BATCH_SIZE, ENTITY_MAX, EDGE_ATTRIBUTES_USED_IN_RECORD_HASH, text_mining_api_infores };
