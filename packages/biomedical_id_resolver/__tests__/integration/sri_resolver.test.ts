import { resolveSRI } from '../../src/index';
import { SRIBioEntity } from '../../src/common/types';

describe("Test SRI Resolver", () => {
  test("Test old format", async () => {
    let input = {
      "Gene": ["NCBIGene:1017", "NCBIGene:1018", "HGNC:1177"],
      "SmallMolecule": ["CHEBI:15377"],
      "Disease": ["MONDO:0004976"],
      "Cell": ["CL:0002372"]
    };
    const res = await resolveSRI(input);

    expect(res["NCBIGene:1017"].primaryID).toEqual("NCBIGene:1017");
    expect(res["NCBIGene:1017"].label).toEqual("CDK2");
    expect(res["NCBIGene:1017"].primaryTypes).toEqual(["Gene"]);
    expect(res["NCBIGene:1017"].semanticTypes).toEqual(expect.any(Array));
    expect(res["NCBIGene:1017"].labelAliases).toEqual(expect.any(Array));
    expect(res["NCBIGene:1017"].equivalentIDs).toEqual(expect.any(Array));
  });

  test("Test unresolvable curie/bad input", async () => {
    let input = {
      "Gene": ["NCBIGene:ABCD", "NCBIGene:GENE:1017"],
    };
    const res = await resolveSRI(input);

    expect(res["NCBIGene:ABCD"].primaryTypes).toEqual(["Gene"]);
    expect(res["NCBIGene:ABCD"].primaryID).toEqual("NCBIGene:ABCD");
    expect(res["NCBIGene:ABCD"].label).toEqual("NCBIGene:ABCD");
    expect(res["NCBIGene:ABCD"].labelAliases).toEqual(expect.any(Array));

    expect(res["NCBIGene:GENE:1017"].primaryTypes).toEqual(["Gene"]);
    expect(res["NCBIGene:GENE:1017"].primaryID).toEqual("NCBIGene:GENE:1017");
    expect(res["NCBIGene:GENE:1017"].label).toEqual("NCBIGene:GENE:1017");
    expect(res["NCBIGene:GENE:1017"].labelAliases).toEqual(expect.any(Array));
  });

  test("Test SRI Semantic type resolver with unknown", async () => {
    let input = {
      unknown: ["NCBIGene:3778"],
    };
    const res = await resolveSRI(input);

    expect(res["NCBIGene:3778"].primaryTypes).toEqual(["Gene"]);
  })

  test("Test SRI Semantic type resolver with undefined", async () => {
    let input = {
      undefined: ["NCBIGene:3778"],
    };
    const res = await resolveSRI(input);

    expect(res["NCBIGene:3778"].primaryTypes).toEqual(["Gene"]);
  })

  test("Test SRI Semantic type resolver with NamedThing", async () => {
    let input = {
      NamedThing: ["NCBIGene:3778"],
    };
    const res = await resolveSRI(input);

    expect(res["NCBIGene:3778"].primaryTypes).toEqual(["Gene"]);
  })

  test("Test Same ID different semantic types", async () => {
    let input = {
      "Gene": ["NCBIGene:1017"],
      "Disease": ["NCBIGene:1017"]
    };
    const res = await resolveSRI(input);

    expect(res["NCBIGene:1017"].primaryTypes.length).toBe(2);
    expect(res["NCBIGene:1017"].primaryTypes).toEqual(["Gene", "Disease"])
  });

  test("Test using SRI to get semantic types", async () => {
    let input = {
      unknown: ["NCBIGene:1017"]
    };
    const res = await resolveSRI(input);

    expect(res).toHaveProperty("NCBIGene:1017");
    expect(res["NCBIGene:1017"].primaryTypes).toContain("Gene");
  });

  test("Test handling semantic type conflicts", async () => {
    let input = {
      "SmallMolecule": ["PUBCHEM.COMPOUND:23680530"]
    };
    const res = await resolveSRI(input);

    expect(res["PUBCHEM.COMPOUND:23680530"].primaryTypes.length).toBe(2);
    expect(res["PUBCHEM.COMPOUND:23680530"].primaryTypes).toEqual(["MolecularMixture", "SmallMolecule"])
  });

  test("Test large batch of inputs should be correctly resolved and should not give an error", async () => {
    const fakeNCBIGeneInputs = [...Array(15000).keys()].map(item => 'NCBIGene:' + item.toString());
    let input = {
      Gene: fakeNCBIGeneInputs,
    };
    const res = await resolveSRI(input);

    expect(Object.keys(res)).toHaveLength(fakeNCBIGeneInputs.length);
  })

});
