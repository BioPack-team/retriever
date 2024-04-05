[![Test Codecov](https://github.com/biothings/biomedical_id_resolver.js/actions/workflows/test_cov.yml/badge.svg)](https://github.com/biothings/biomedical_id_resolver.js/actions/workflows/test_cov.yml)
[![codecov](https://codecov.io/github/biothings/biomedical_id_resolver.js/branch/main/graph/badge.svg?token=E2LDAGC7K9)](https://codecov.io/github/biothings/biomedical_id_resolver.js)
![npm](https://img.shields.io/npm/dw/biomedical_id_resolver)
![GitHub issues](https://img.shields.io/github/issues/biothings/biomedical_id_resolver.js)
![NPM](https://img.shields.io/npm/l/biomedical_id_resolver)
![npm](https://img.shields.io/npm/v/biomedical_id_resolver?style=plastic)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/biothings/biomedical_id_resolver.js)



# biomedical_id_resolver.js
js library for resolving biological ids to their equivalent ids in batch

## Install

```
$ pnpm i biomedical_id_resolver
```

## Usage

```js
const resolve = require('biomedical_id_resolver');

// input should be an object, with semantic type as the key, and array of CURIEs as value
let input = {
    "Gene": ["NCBIGene:1017", "NCBIGene:1018", "HGNC:1177"],
    "SmallMolecule": ["CHEBI:15377"],
    "Disease": ["MONDO:0004976"],
    "Cell": ["CL:0002372"]
  };

(async () => {
  const resolver = new resolve();
	console.log(await resolver.resolve(input);
	//=> {'NCBIGene:1017': {...}, 'NCBIGene:1018': {...}, 'HGNC:1177': {...}, 'CHEBI:15377': {...}, 'MONDO:0004976': {...}, 'CL:0002372': {...}}
})();

```

## Output Schema

- Output is a javascript Object

- The root keys are CURIES (e.g. NCBIGene:1017) which are passed in as input

- The values represents resolved identifiers

- Each CURIE will have 4 required fields

  - id: the primary id (selected based on the ranking described in the next section) and label

  - curies: an array, each element represents a resolved id in CURIE format

  - type: the semantic type of the identifier

  - db_ids: original ids from source database, could be curies or non-curies.


- if an ID can not be resolved using the package, it will have an additional field called "flag", with value equal to "failed"

- Example Output

```json
{
  "NCBIGene:1017": {
    "id": {
      "label": "cyclin dependent kinase 2",
      "identifier": "NCBIGene:1017"
    },
    "db_ids": {
      "NCBIGene": [
        "1017"
      ],
      "ENSEMBL": [
        "ENSG00000123374"
      ],
      "HGNC": [
        "1771"
      ],
      "SYMBOL": [
        "CDK2"
      ],
      "UMLS": [
        "C1332733",
        "C0108855"
      ],
      "name": [
        "cyclin dependent kinase 2"
      ]
    },
    "type": "Gene",
    "curies": [
      "NCBIGene:1017",
      "ENSEMBL:ENSG00000123374",
      "HGNC:1771",
      "SYMBOL:CDK2",
      "UMLS:C1332733",
      "UMLS:C0108855"
    ]
  }
}
```

## Query Using [SRI node normalizer](https://nodenormalization-sri-dev.renci.org/1.1/docs#/)
### Usage
```js
const resolver = require('biomedical_id_resolver');

// input must be an object, with semantic type as the key, and array of CURIEs as value

let input = {
    "Gene": ["NCBIGene:1017", "NCBIGene:1018", "HGNC:1177"],
    "SmallMolecule": ["CHEBI:15377"],
    "Disease": ["MONDO:0004976"],
    "Cell": ["CL:0002372"]
};

// SRI resolver will figure out the semantic type if the input type is 'unknown', 'undefined', or 'NamedThing'
let input = {
    "unknown": ["NCBIGene:1017", "MONDO:0004976"],
};

(async () => {
  let res = await resolver.resolveSRI(input);
  console.log(res);
})();
```

### Example Output
The output contains `id` and `equivalent_identifiers` straight from SRI as well as the same fields as the base resolver to make it backwards compatible with it. If the SRI resolved semantic type doesn't agree with the input semantic type, there will be 2 entries in the array for the curie.
```json
{
  "NCBIGene:1017": [
    {
      "id": {
        "identifier": "NCBIGene:1017",
        "label": "CDK2"
      },
      "equivalent_identifiers": [
        {
          "identifier": "NCBIGene:1017",
          "label": "CDK2"
        },
        {
          "identifier": "ENSEMBL:ENSG00000123374"
        },
        {
          "identifier": "HGNC:1771",
          "label": "CDK2"
        },
        {
          "identifier": "OMIM:116953"
        },
        {
          "identifier": "UMLS:C1332733",
          "label": "CDK2 gene"
        }
      ],
      "type": [
        "biolink:Gene",
        "biolink:GeneOrGeneProduct",
        "biolink:BiologicalEntity",
        "biolink:NamedThing",
        "biolink:Entity",
        "biolink:MacromolecularMachineMixin"
      ],
      "primaryID": "NCBIGene:1017",
      "label": "CDK2",
      "attributes": {},
      "semanticType": "Gene",
      "semanticTypes": [
        "biolink:Gene",
        "biolink:GeneOrGeneProduct",
        "biolink:BiologicalEntity",
        "biolink:NamedThing",
        "biolink:Entity",
        "biolink:MacromolecularMachineMixin"
      ],
      "dbIDs": {
        "NCBIGene": [
          "1017"
        ],
        "ENSEMBL": [
          "ENSG00000123374"
        ],
        "HGNC": [
          "1771"
        ],
        "OMIM": [
          "116953"
        ],
        "UMLS": [
          "C1332733"
        ],
        "name": [
          "CDK2",
          "CDK2 gene"
        ]
      },
      "curies": [
        "NCBIGene:1017",
        "ENSEMBL:ENSG00000123374",
        "HGNC:1771",
        "OMIM:116953",
        "UMLS:C1332733"
      ]
    }
  ]
}
```
## Available Semantic Types & prefixes

> Gene, Transcript, Protein ID resolution is done through MyGene.info API

- Gene
  1. NCBIGene
  2. ENSEMBL
  3. HGNC
  4. MGI
  5. OMIM
  6. UMLS
  7. SYMBOL
  8. UniProtKB
  9. name

- Transcript
  1. ENSEMBL
  2. SYMBOL
  3. name

- Protein
  1. UniProtKB
  2. ENSEMBL
  3. UMLS
  4. SYMBOL
  5. name

> Variant ID resolution is done through MyVariant.info API

- SequenceVariant
  1. CLINVAR
  2. DBSNP
  3. HGVS
  4. MYVARIANT_HG19

> SmallMolecule, Drug ID resolution is done through MyChem.info API

- SmallMolecule
    1. PUBCHEM.COMPOUND
    2. CHEMBL.COMPOUND
    3. UNII
    4. CHEBI
    5. DRUGBANK
    6. MESH
    7. CAS
    8. HMDB
    9. KEGG.COMPOUND
    10. INCHI
    11. INCHIKEY
    12. UMLS
    13. LINCS
    14. name

- Drug
    1. RXCUI
    2. NDC
    3. DRUGBANK
    4. PUBCHEM.COMPOUND
    5. CHEMBL.COMPOUND
    6. UNII
    7. CHEBI
    8. MESH
    9. CAS
    10. HMDB
    11. KEGG.COMPOUND
    12. INCHI
    13. INCHIKEY
    14. UMLS
    15. LINCS
    16. name

> Disease, ClinicalFinding ID Resolution is done through MyDisease.info API

- Disease

  1. MONDO
  2. DOID
  3. OMIM
  4. ORPHANET
  5. EFO
  6. UMLS
  7. MESH
  8. MEDDRA
  9. NCIT
  10. SNOMEDCT
  11. HP
  12. GARD
  13. name

- ClinicalFinding
  1. LOINC
  2. NCIT
  3. EFO
  4. name

> Pathway ID Resolution is done through biothings.ncats.io/geneset API

- Pathway
  1. GO
  2. REACT
  3. KEGG
  4. SMPDB
  5. PHARMGKB.PATHWAYS
  6. WIKIPATHWAYS
  7. BIOCARTA
  8. name

> MolecularActivity ID Resolution is done through BioThings Gene Ontology Molecular Activity API

- MolecularActivity
  1. GO
  2. REACT
  3. RHEA
  4. MetaCyc
  5. KEGG.REACTION
  6. name

> CellularComponent ID Resolution is done through BioThings Gene Ontology Cellular Component API

- CellularComponent
  1. GO
  2. MetaCyc
  3. name

> BiologicalProcess ID Resolution is done through BioThings Gene Ontology Biological Process API

- BiologicalProcess

  1. GO
  2. REACT
  3. MetaCyc
  4. KEGG
  5. name

> AnatomicalEntity ID Resolution is done through BioThings UBERON API

- AnatomicalEntity
  1. UBERON
  2. UMLS
  3. MESH
  4. NCIT
  5. name

> PhenotypicFeature ID Resolution is done through BioThings HPO API

- PhenotypicFeature
  1. HP
  2. EFO
  3. NCIT
  4. UMLS
  5. MEDDRA
  6. MP
  7. SNOMEDCT
  8. MESH
  9. name

> Cell ID Resolution is done through Biothings Cell Ontology API

- Cell
  1. CL
  2. NCIT
  3. MESH
  4. EFO
  5. name

## Development

1. Install Node 12 or later. You can use the [package manager](https://www.npmjs.com/) of your choice.
   Tests need to pass in Node 12 and 14.
2. Clone this repository.
3. Run `npm ci` to install the dependencies.
4. scripts are stored in `/src` folder
5. Add test to `/__tests__` folder
6. run `npm run release` to bump version and generate change log
7. run `npx depcheck` to check for unused packages in package.json

## CHANGELOG

See [CHANGELOG.md](https://github.com/biothings/biomedical_id_resolver.js/blob/main/CHANGELOG.md)
