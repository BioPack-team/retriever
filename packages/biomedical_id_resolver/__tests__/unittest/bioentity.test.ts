import { ResolvableBioEntity } from '../../src/bioentity/valid_bioentity';
import { IrresolvableBioEntity } from '../../src/bioentity/irresolvable_bioentity';

const CDK2_DB_IDs = {
    "NCBIGene": ["1017"],
    "HGNC": ["1771"],
    "SYMBOL": ["CDK2"],
    "name": ["cyclin dependent kinase 2"],
}

const RILUZOLE_DB_IDS = {
    "CHEMBL.COMPOUND": ["CHEMBL744"],
    "name": ["Riluzole", "RILUZOLE"],
    "PUBCHEM.COMPOUND": ["5070"],
}

const DB_ID_WITH_NO_PRIMARY = {
    kk: ["kkk"]
}

const DISEASE_DB_IDS = {
    "MONDO": ["MONDO:12345"]
}

const CHEMBL7512_DB_IDS = {
    "CHEMBL.COMPOUND": ["CHEMBL7512"],
    "PUBCHEM.COMPOUND": ["53428"]
}

describe("Test ResolvableBioEntity Class", () => {
    test.skip("return semanticType when called semanticType property", () => {
        const entity = new ResolvableBioEntity("Gene", CDK2_DB_IDs, {});
        const res = entity.semanticType;
        expect(res).toEqual("Gene");
    })
    describe("Test getPrimaryID function", () => {
        test.skip("db ids with prefixes defined in metadata should return the primary id", () => {
            const entity = new ResolvableBioEntity("Gene", CDK2_DB_IDs, {});
            const primary_id = entity.primaryID;
            expect(primary_id).toBe("NCBIGene:1017")
        })

        test.skip("db ids always prefixed should return itself", () => {
            const entity = new ResolvableBioEntity("Disease", DISEASE_DB_IDS, {});
            const primary_id = entity.primaryID;
            expect(primary_id).toBe("MONDO:12345")
        })

        test.skip("db ids without prefixes defined in metadata should return undefined", () => {
            const entity = new ResolvableBioEntity("Gene", DB_ID_WITH_NO_PRIMARY, {});
            const primary_id = entity.primaryID;
            expect(primary_id).toBeUndefined();
        })
    })

    describe("Test getLabel function", () => {
        test.skip("if SYMBOL provided in db ids, should return symbol", () => {
            const entity = new ResolvableBioEntity("Gene", CDK2_DB_IDs, {});
            const label = entity.label;
            expect(label).toBe("CDK2");
        })

        test("if SYMBOL not provided in db ids and name is provided, should return name", () => {
            const entity = new ResolvableBioEntity("ChemicalEntity", RILUZOLE_DB_IDS, {});
            const label = entity.label;
            expect(label).toBe("Riluzole");
        })

        test("if both SYMBOL and name are not provided in db ids, should return primary id", () => {
            const entity = new ResolvableBioEntity("ChemicalEntity", CHEMBL7512_DB_IDS, {});
            const label = entity.label;
            expect(label).toBe("PUBCHEM.COMPOUND:53428");
        })
    });

    describe("Test getCuries function", () => {
        test("test getCuries", () => {
            const entity = new ResolvableBioEntity("ChemicalEntity", CHEMBL7512_DB_IDS, {});
            const curies = entity.curies;
            expect(curies).toContain("CHEMBL.COMPOUND:CHEMBL7512");
            expect(curies).toHaveLength(2);
        })

    })

    describe("Test getDBIDs function", () => {
        test("test getDBIDs", () => {
            const entity = new ResolvableBioEntity("ChemicalEntity", CHEMBL7512_DB_IDS, {});
            const dbIDs = entity.dbIDs;
            expect(dbIDs).toEqual(CHEMBL7512_DB_IDS);
        })

    })

})

describe("Test IrresolvableBioEntity Class", () => {
    test("return semanticType when called semanticType property", () => {
        const entity = new IrresolvableBioEntity("Gene", "KK:123");
        const res = entity.semanticType;
        expect(res).toEqual("Gene");
    })
    describe("Test getPrimaryID function", () => {
        test("should return the input curie", () => {
            const entity = new IrresolvableBioEntity("Gene", "KK:123");
            const primary_id = entity.primaryID;
            expect(primary_id).toBe("KK:123")
        })
    })

    describe("Test getLabel function", () => {
        test("should return the input curie", () => {
            const entity = new IrresolvableBioEntity("Gene", "KK:123");
            const label = entity.label;
            expect(label).toBe("KK:123");
        })
    });

    describe("Test getCuries function", () => {
        test("test getCuries", () => {
            const entity = new IrresolvableBioEntity("ChemicalEntity", "KK:123");
            const curies = entity.curies;
            expect(curies).toEqual(["KK:123"]);
        })

    })

    describe("Test getDBIDs function", () => {
        test("test getDBIDs", () => {
            const entity = new IrresolvableBioEntity("ChemicalEntity", "KK:123");
            const dbIDs = entity.dbIDs;
            expect(dbIDs).toEqual(
                {
                    KK: ["KK:123"]
                }
            );
        })

    })

    describe("Test getAttributes function", () => {
        test("test getAttributes", () => {
            const entity = new IrresolvableBioEntity("ChemicalEntity", "KK:123");
            const attributes = entity.attributes;
            expect(attributes).toEqual(
                {

                }
            );
        })

    })

})