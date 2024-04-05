import Entity from '../../src/object/entity_object';

describe("Test EntityObject class", () => {
  test("If input entity is Gene, should add additonal ID prefixes in addtion to provided ones", () => {
    const entity = new Entity("Gene", {
      "id_prefixes": ["NCBIGene"]
    })
    expect(entity.id_prefixes).toContain("NCBIGene");
    expect(entity.id_prefixes).toContain("UMLS");
    expect(entity.id_prefixes).toContain("SYMBOL");
    expect(entity.id_prefixes).toContain("OMIM");
  })

  test("If input entity is SmallMolecule, should add additonal ID prefixes in addtion to provided ones", () => {
    const entity = new Entity("SmallMolecule", {
      "id_prefixes": ["CHEMBL.COMPOUND"]
    })
    expect(entity.id_prefixes).toContain("CHEMBL.COMPOUND");
    expect(entity.id_prefixes).toContain("UMLS");
  })

  test("If input entity is Disease, should add additonal ID prefixes in addition to provided ones", () => {
    const entity = new Entity("Disease", {
      "id_prefixes": ["KEGG"]
    })
    expect(entity.id_prefixes).toHaveLength(2);
    expect(entity.id_prefixes).toContain("KEGG");
    expect(entity.id_prefixes).toContain("GARD");
  })

  test("If input entity is not Gene or SmallMolecule or Disease, should just return provided ones", () => {
    const entity = new Entity("PhenotypicFeature", {
      "id_prefixes": ["HP"]
    })
    expect(entity.id_prefixes).toHaveLength(1);
    expect(entity.id_prefixes).toContain("HP");
  })
})
