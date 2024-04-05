import { getAttributes } from '../../src/index';


describe("Test Attribute Getter", () => {
    
    test("getting supported semantic type attributes", async () => {

        let ids = {
        "SmallMolecule":["PUBCHEM.COMPOUND:6230","PUBCHEM.COMPOUND:18142","PUBCHEM.COMPOUND:6231",
        "PUBCHEM.COMPOUND:6279","PUBCHEM.COMPOUND:9051","PUBCHEM.COMPOUND:11683",
        "PUBCHEM.COMPOUND:6917715","PUBCHEM.COMPOUND:40973","PUBCHEM.COMPOUND:68873",
        "PUBCHEM.COMPOUND:28417","PUBCHEM.COMPOUND:6540478","PUBCHEM.COMPOUND:13109",
        "PUBCHEM.COMPOUND:6238","PUBCHEM.COMPOUND:9568628","PUBCHEM.COMPOUND:9270"]} 
        const res = await getAttributes(ids);
        expect(Object.keys(res)).toContain('PUBCHEM.COMPOUND:40973');
    });

    test("getting expected attribute from semantic type", async () => {

        let ids = {
        "SmallMolecule":["PUBCHEM.COMPOUND:10631"]} 
        const res = await getAttributes(ids);
        expect(res["PUBCHEM.COMPOUND:10631"]).toHaveProperty('biolink:drug_regulatory_status_world_wide');
    });

    test("getting no attributes for unsupported semantic type", async () => {

        let ids = {
        "Gene":["UMLS:C1366500","UMLS:C0812258","UMLS:C1421286","UniProtKB:P24385","NCBIGene:23221"]} 
        const res = await getAttributes(ids);
        expect(res).toStrictEqual({});
    });

});