import DefaultValidator from '../src/validate/default_validator';
import IrresolvableIDResolverInputError from '../src/common/exceptions';

describe("Test Validator Class", () => {
    describe("Test validateIfInputIsObject function", () => {
        test("raise exception if input is an array", () => {
            const test_data = ["123"];
            const vd = new DefaultValidator(test_data);
            expect(() => vd.validate()).toThrow(IrresolvableIDResolverInputError);
            expect(() => vd.validate()).toThrow("Your Input to ID Resolver is Irresolvable. It should be a plain object!");
        })

        test("raise exception if input is a string", () => {
            const test_data = "123";
            const vd = new DefaultValidator(test_data);
            expect(() => vd.validate()).toThrow(IrresolvableIDResolverInputError);
            expect(() => vd.validate()).toThrow("Your Input to ID Resolver is Irresolvable. It should be a plain object!");
        })

        test("not raise exception if input is an object", () => {
            const test_data = {};
            const vd = new DefaultValidator(test_data);
            expect(() => vd.validate()).not.toThrow(IrresolvableIDResolverInputError);
        })
    })

    describe("Test validateIfValuesOfInputIsArray function", () => {
        test("raise exception if input is values of input is string", () => {
            const test_data = { "Gene": "123" };
            const vd = new DefaultValidator(test_data);
            expect(() => vd.validate()).toThrow(IrresolvableIDResolverInputError);
            expect(() => vd.validate()).toThrow("Your Input to ID Resolver is Irresolvable. All values of your input dictionary should be a list!");
        })

        test("raise exception if input is values of input is an object", () => {
            const test_data = { "Gene": { "Protein": ["PR:001"] } };
            const vd = new DefaultValidator(test_data);
            expect(() => vd.validate()).toThrow(IrresolvableIDResolverInputError);
            expect(() => vd.validate()).toThrow("Your Input to ID Resolver is Irresolvable. All values of your input dictionary should be a list!");
        })

        test("not raise exception if values of input is array", () => {
            const test_data = { "Gene": ["NCBIGene:1017"] };
            const vd = new DefaultValidator(test_data);
            expect(() => vd.validate()).not.toThrow(IrresolvableIDResolverInputError);
        })
    })

    describe("Test validateIfEachItemInInputValuesIsCurie function", () => {
        test("raise exception if values of input contains non-string type", () => {
            const test_data = { "Gene": [123] };
            const vd = new DefaultValidator(test_data);
            expect(() => vd.validate()).toThrow(IrresolvableIDResolverInputError);
            expect(() => vd.validate()).toThrow("Your Input to ID Resolver is Irresolvable. Each item in the values of your input dictionary should be a curie. Spotted 123 is not a curie");
        })

        test("raise exception if values of input contains non-curie type", () => {
            const test_data = { "Gene": ["1234"] };
            const vd = new DefaultValidator(test_data);
            expect(() => vd.validate()).toThrow(IrresolvableIDResolverInputError);
            expect(() => vd.validate()).toThrow("Your Input to ID Resolver is Irresolvable. Each item in the values of your input dictionary should be a curie. Spotted 1234 is not a curie");
        })

        test("not raise exception if values of input are all curies", () => {
            const test_data = { "Gene": ["NCBIGene:1234", "NCBIGene:1345"] };
            const vd = new DefaultValidator(test_data);
            expect(() => vd.validate()).not.toThrow(IrresolvableIDResolverInputError);
        })
    })

    describe("Test checkIfSemanticTypeCanBeResolved function", () => {
        test("semantic types not appear in APIMeta file should be classified as Irresolvable", () => {
            const test_data = { "Gene": ["NCBIGene:1017"], "Gene1": ["NCBIGene:1018"] };
            const vd = new DefaultValidator(test_data);
            vd.validate();
            expect(vd.irresolvable).toHaveProperty("Gene1");
        })
    })

    describe("Test checkIfPrefixCanBeResolved function", () => {
        test("id prefixes not appear in APIMeta file should be classified as Irresolvable", () => {
            const test_data = { "Gene": ["NCBIGene:1017", "kkk:1323"] };
            const vd = new DefaultValidator(test_data);
            vd.validate();
            expect(vd.irresolvable).toHaveProperty("Gene");
            expect(vd.irresolvable.Gene).toContain("kkk:1323");
        })

        test("check if Irresolvable object are correctly initialized", () => {
            const test_data = { "Gene": ["NCBIGene:1017", "kkk:1323", "kkk:12345"] };
            const vd = new DefaultValidator(test_data);
            vd.validate();
            expect(vd.irresolvable).toHaveProperty("Gene");
            expect(vd.irresolvable.Gene).toContain("kkk:1323");
            expect(vd.irresolvable.Gene).toContain("kkk:12345");
        })
    })

    describe.skip("Test handleUndefinedIDs function", () => {
        test("id appear in config should be mapped to the correct semantic type", () => {
            const test_data = { "undefined": ["NCBIGene:1017", "kkk:1323"] };
            const vd = new DefaultValidator(test_data);
            vd.validate();
            expect(vd.resolvable.Gene).toContain("NCBIGene:1017");
            expect(vd.irresolvable.undefined).not.toContain("NCBIGene:1017");
            expect(vd.irresolvable.undefined).toContain("kkk:1323");
        })

        test("id that can be mapped to mulitple semantic types are correcty mapped", () => {
            const test_data = { "undefined": ["NCBIGene:1017", "OMIM:123"] };
            const vd = new DefaultValidator(test_data);
            vd.validate();
            expect(vd.resolvable.Gene).toContain("OMIM:123");
            expect(vd.resolvable.Disease).toContain("OMIM:123");
        })
    })

    describe.skip("Test validate function", () => {
        test("valid answers can be retrieved through valid property of the class", () => {
            const test_data = { "Gene": ["NCBIGene:1017", "kkk:1323"], "SmallMolecule": ["DRUGBANK:DB0001"] };
            const vd = new DefaultValidator(test_data);
            vd.validate();
            expect(vd.resolvable).toHaveProperty("Gene");
            expect(vd.resolvable.Gene).toEqual(['NCBIGene:1017']);
            expect(vd.resolvable).toHaveProperty("SmallMolecule");
            expect(vd.resolvable.SmallMolecule).toEqual(["DRUGBANK:DB0001"])
        })
    })
})
