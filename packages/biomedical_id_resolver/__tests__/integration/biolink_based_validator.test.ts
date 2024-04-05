import BioLinkBasedValidator from '../../src/validate/biolink_based_validator';

describe.skip("Integration test for BioLink Based Validator", () => {
    test("Semantic Type and Prefix defined in BioLink and config should appear in resolvable", () => {
        const input = {
            Disease: ["MONDO:1234"]
        }
        const validator = new BioLinkBasedValidator(input);
        validator.validate();
        expect(validator.resolvable).toHaveProperty('Disease', ["MONDO:1234"]);
        expect(validator.irresolvable).toEqual({});
    })

    test("Semantic Type but not Prefix defined in BioLink and config should appear in irresolvable", () => {
        const input = {
            Disease: ["MONDO1:1234"]
        }
        const validator = new BioLinkBasedValidator(input);
        validator.validate();
        expect(validator.irresolvable).toHaveProperty('Disease', ["MONDO1:1234"]);
        expect(validator.resolvable).toEqual({});
    })

    test("Semantic Type but not Prefix defined in BioLink and config should appear in irresolvable using Gene Or Gene Product as example", () => {
        const input = {
            GeneOrGeneProduct: ["CHEMBL.TARGET:CHEBML123"]
        }
        const validator = new BioLinkBasedValidator(input);
        validator.validate();
        expect(validator.irresolvable).toHaveProperty('GeneOrGeneProduct', ["CHEMBL.TARGET:CHEBML123"]);
        expect(validator.irresolvable).not.toHaveProperty('Gene');
        expect(validator.resolvable).toEqual({});
    })

    test("Semantic Type and Prefix that is not defined in BioLink or config should appear in irresolvable", () => {
        const input = {
            Disease1: ["MONDO:1234"]
        }
        const validator = new BioLinkBasedValidator(input);
        validator.validate();
        expect(validator.irresolvable).toHaveProperty('Disease1', ["MONDO:1234"]);
        expect(validator.resolvable).toEqual({});
    })

    test("Descendant Semantic Type and Prefix defined in BioLink and config should appear in resolvable", () => {
        const input = {
            DiseaseOrPhenotypicFeature: ["UMLS:1234"]
        }
        const validator = new BioLinkBasedValidator(input);
        validator.validate();
        expect(validator.resolvable).toHaveProperty('Disease', ["UMLS:1234"]);
        expect(validator.resolvable).toHaveProperty('PhenotypicFeature', ["UMLS:1234"]);
        expect(validator.irresolvable).toEqual({});
    })

    test("Descendant Semantic Type and Prefix defined in BioLink and config should appear in resolvable using NamedThing as example", () => {
        const input = {
            NamedThing: ["UMLS:1234"]
        }
        const validator = new BioLinkBasedValidator(input);
        validator.validate();
        expect(validator.resolvable).toHaveProperty('Disease', ["UMLS:1234"]);
        expect(validator.resolvable).toHaveProperty('PhenotypicFeature', ["UMLS:1234"]);
        expect(validator.resolvable).toHaveProperty('AnatomicalEntity', ["UMLS:1234"]);
        expect(validator.irresolvable).toHaveProperty('Cell', ["UMLS:1234"]);
        expect(validator.irresolvable).toHaveProperty('CellularComponent', ["UMLS:1234"]);
    })

    test("Descendant Semantic Type But not Prefix defined in BioLink and config should appear in irresolvable", () => {
        const input = {
            NamedThing: ["KEGG.GENE:1234"]
        }
        const validator = new BioLinkBasedValidator(input);
        validator.validate();
        expect(validator.resolvable).toEqual({});
        expect(validator.irresolvable).toHaveProperty('Gene', ["KEGG.GENE:1234"]);
        expect(validator.irresolvable).toHaveProperty('NamedThing', ["KEGG.GENE:1234"]);
    })

    test("Item with comma in it should be counted as irresolvable", () => {
        const input = {
            SmallMolecule: ["KEGG:1,2"]
        }
        const validator = new BioLinkBasedValidator(input);
        validator.validate();
        expect(validator.resolvable).toEqual({});
        expect(validator.irresolvable).toHaveProperty('SmallMolecule', ["KEGG:1,2"]);
    })
})