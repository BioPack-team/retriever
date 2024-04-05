import BioLinkBasedResolver from '../../src/resolve/biolink_based_resolver';
import { ResolvableBioEntity } from '../../src/bioentity/valid_bioentity';
import { IrresolvableBioEntity } from '../../src/bioentity/irresolvable_bioentity';

jest.setTimeout(30000)
describe.skip("Test ID Resolver", () => {
    test("Test valid inputs should be corretly resolved", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "Gene": ["NCBIGene:1017"] });
        expect(res).toHaveProperty("NCBIGene:1017");
        expect(res['NCBIGene:1017']).toHaveLength(1);
        expect(res['NCBIGene:1017'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['NCBIGene:1017'][0].primaryID).toEqual("NCBIGene:1017");
        expect(res['NCBIGene:1017'][0].label).toEqual("CDK2")
    })

    test("Test BioThings output include integer should be converted to string", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "SmallMolecule": ["CHEMBL.COMPOUND:CHEMBL744"] });
        expect(res['CHEMBL.COMPOUND:CHEMBL744'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['CHEMBL.COMPOUND:CHEMBL744'][0].dbIDs["PUBCHEM.COMPOUND"]).toEqual(["5070"]);
    })

    test("Test valid inputs from multiple semantic types should be corretly resolved", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "Gene": ["NCBIGene:1017"], "SmallMolecule": ["DRUGBANK:DB01609"] });
        expect(res).toHaveProperty("NCBIGene:1017");
        expect(res['NCBIGene:1017']).toHaveLength(1);
        expect(res['NCBIGene:1017'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['NCBIGene:1017'][0].primaryID).toEqual("NCBIGene:1017");
        expect(res['NCBIGene:1017'][0].label).toEqual("CDK2");
        expect(res).toHaveProperty("DRUGBANK:DB01609");
        expect(res["DRUGBANK:DB01609"]).toHaveLength(1);
        expect(res['DRUGBANK:DB01609'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['DRUGBANK:DB01609'][0].label.toUpperCase()).toEqual("DEFERASIROX");
    })

    test("Test Irresolvable inputs should be part of the result", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "Gene": ["NCBIGene:1017", "kkk:123"] });
        expect(res).toHaveProperty("kkk:123");
        expect(res['kkk:123']).toHaveLength(1);
        expect(res['kkk:123'][0]).toBeInstanceOf(IrresolvableBioEntity);
        expect(res['kkk:123'][0].primaryID).toEqual('kkk:123');
        expect(res['kkk:123'][0].label).toEqual('kkk:123')
    })

    // uses old ID resolver and has issues right now...
    test.skip("test large batch of inputs should be correctly resolved", async () => {
        const fakeNCBIGeneInputs = [...Array(1990).keys()].map(item => 'NCBIGene:' + item.toString());
        const fakeOMIMGeneInputs = [...Array(2300).keys()].map(item => "OMIM:" + item.toString());
        const fakeDrugbankInputs = [...Array(3500).keys()].map(item => "DRUGBANK:DB00" + item.toString());
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({
            Gene: [...fakeNCBIGeneInputs, ...fakeOMIMGeneInputs],
            SmallMolecule: fakeDrugbankInputs
        })
        expect(Object.keys(res)).toHaveLength(fakeDrugbankInputs.length + fakeNCBIGeneInputs.length + fakeOMIMGeneInputs.length);
        expect(res['OMIM:0'][0]).toBeInstanceOf(IrresolvableBioEntity)

    })

    test("Test inputs with NamedThing semanticType should be corretly resolved", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "NamedThing": ["NCBIGene:1017"] });
        expect(res).toHaveProperty("NCBIGene:1017");
        expect(res["NCBIGene:1017"]).toHaveLength(2);
        expect(res['NCBIGene:1017'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['NCBIGene:1017'][0].primaryID).toEqual("NCBIGene:1017");
        expect(res['NCBIGene:1017'][0].label).toEqual("CDK2");
        expect(res['NCBIGene:1017'][0].semanticType).toEqual("Gene");
        expect(res["NCBIGene:1017"][0].semanticTypes).toContain("NamedThing");
    })

    test("Test inputs with NamedThing semanticType and could be mapped to multiple semantictypes should be corretly resolved", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "NamedThing": ["UMLS:C0008780"] });
        expect(res).toHaveProperty("UMLS:C0008780");
        const valid = res["UMLS:C0008780"].filter(rec => rec instanceof ResolvableBioEntity);
        const validTypes = valid.map(item => item.semanticType);
        expect(validTypes).toContain("PhenotypicFeature");
        expect(validTypes).toContain("Disease");
        expect(valid).toHaveLength(2);
        expect(valid[0].semanticTypes).toContain("NamedThing");
    })

    test("Test inputs with NamedThing semanticType and could be mapped to multiple semantictypes using OMIM ID as example should be corretly resolved", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "NamedThing": ["OMIM:116953"] });
        expect(res).toHaveProperty("OMIM:116953");
        const valid = res["OMIM:116953"].filter(item => item instanceof ResolvableBioEntity);
        expect(valid).toHaveLength(1);
        expect(valid[0]).toBeInstanceOf(ResolvableBioEntity);
        expect(valid[0].primaryID).toEqual("NCBIGene:1017");
        expect(valid[0].label).toEqual("CDK2");
        expect(valid[0].semanticType).toEqual("Gene");
        expect(valid[0].semanticTypes).toContain("NamedThing");
        expect(valid[0].semanticTypes).toContain("BiologicalEntity");
    })

    test("Test inputs with undefined semanticType and could not be mapped to any semantic type should return Irresolvable", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "undefined": ["OMIM1:116953"] });
        expect(res).toHaveProperty("OMIM1:116953");
        expect(res['OMIM1:116953'][0]).toBeInstanceOf(IrresolvableBioEntity);
        expect(res['OMIM1:116953'][0].semanticType).toEqual("undefined");
    })

    test("Test Irresolvable inputs should not overwrite the result of a valid input", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "Gene": ["NCBIGene:1017"], "Disease": ["NCBIGene:1017"] });
        expect(res).toHaveProperty("NCBIGene:1017");
        expect(res["NCBIGene:1017"]).toHaveLength(2);
        expect(res['NCBIGene:1017'][0]).toBeInstanceOf(ResolvableBioEntity);
        expect(res['NCBIGene:1017'][0].primaryID).toEqual("NCBIGene:1017");
        expect(res['NCBIGene:1017'][0].label).toEqual("CDK2");
        expect(res['NCBIGene:1017'][1]).toBeInstanceOf(IrresolvableBioEntity);
    })

    test("Test input with multiple colons in it should just be classified as irresolvable but not cause an error", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "Gene": ["NCBIGene:GENE:1017"] });
        expect(res).toHaveProperty("NCBIGene:GENE:1017");
        expect(res["NCBIGene:GENE:1017"]).toHaveLength(1);
        expect(res['NCBIGene:GENE:1017'][0]).toBeInstanceOf(IrresolvableBioEntity);
        expect(res['NCBIGene:GENE:1017'][0].primaryID).toEqual("NCBIGene:GENE:1017");
    })

    test("Test input with space in it should be correctly resolved", async () => {
        const resolver = new BioLinkBasedResolver();
        const res = await resolver.resolve({ "SmallMolecule": ["name:Regorafenib", "name:Sunitinib", "name:Imatinib", "name:Ponatinib", "name:Dasatinib", "name:Bosutinib", "name:Imatinib Mesylate"] });
        expect(res).toHaveProperty("name:Imatinib Mesylate");
        expect(res).toHaveProperty("name:Regorafenib");
        expect(res["name:Imatinib Mesylate"]).toHaveLength(1);
        expect(res["name:Imatinib Mesylate"][0]).toBeInstanceOf(ResolvableBioEntity);
    })

})