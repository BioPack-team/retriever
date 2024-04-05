exports.ONTOLOGIES = {
  doid: {
    name: "doid",
    url: "http://purl.obolibrary.org/obo/doid.owl",
  },
  go: {
    name: "go",
    url: "http://purl.obolibrary.org/obo/go.owl",
  },
  mondo: {
    name: "mondo",
    url: "http://purl.obolibrary.org/obo/mondo.owl",
  },
  hp: {
    name: "hp",
    url: "http://purl.obolibrary.org/obo/hp.owl",
  },
  chebi: {
    // not sure why but downloading chebi.owl using curl or node js fails
    name: "chebi",
    url: "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl",
  },
  umls: {
    // manually processed from MRREL.REF file in UMLS dump
    name: "umls",
  },
};

exports.CHILD_RELATIONS = {
  is_a: true,
  // 'http://purl.obolibrary.org/obo/BFO_0000050': true, // part of
};

exports.PARENT_RELATIONS = {
  "http://purl.obolibrary.org/obo/BFO_0000051": true, // has part
};
