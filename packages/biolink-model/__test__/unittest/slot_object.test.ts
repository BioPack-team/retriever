import Slot from '../../src/object/slot_object';

describe("Test Slot Object", () => {
    describe("Test inverse method", () => {
        test("inverse property is correctly retreived if provided", () => {
            const s = new Slot("negatively_regulates", { inverse: "negatively regulated by" });
            expect(s.inverse).toEqual("negatively_regulated_by");
        })

        test("inverse property is undefined if not provided", () => {
            const s = new Slot("negatively_regulates", {});
            expect(s.inverse).toBeUndefined;
        })
    })

    describe("Test symmetric property", () => {
        test("symmetric property is correctly retreived if provided", () => {
            const s = new Slot("correlated_with", { symmetric: true });
            expect(s.symmetric).toEqual(true);
        })

        test("symmetric property is false if not provided", () => {
            const s = new Slot("negatively_regulates", {});
            expect(s.symmetric).toEqual(false);
        })
    })

    describe("Test description method", () => {
        test("description property is correctly retreived if provided", () => {
            const s = new Slot("affects_uptake_of", { description: "holds between two molecular entities where the action or effect of one impacts the rate of uptake of the other into of a cell, gland, or organ" });
            expect(s.description).toEqual("holds between two molecular entities where the action or effect of one impacts the rate of uptake of the other into of a cell, gland, or organ");
        })

        test("description property is undefined if not provided", () => {
            const s = new Slot("negatively_regulates", {});
            expect(s.description).toBeUndefined;
        })
    })

    describe("Test domain method", () => {
        test("domain property is correctly retreived if provided", () => {
            const s = new Slot("regulates", { domain: "physical essence or occurrent" });
            expect(s.domain).toEqual("PhysicalEssenceOrOccurrent");
        })

        test("domain property is undefined if not provided", () => {
            const s = new Slot("negatively_regulates", {});
            expect(s.domain).toBeUndefined;
        })
    })

    describe("Test range method", () => {
        test("range property is correctly retreived if provided", () => {
            const s = new Slot("regulates", { range: "physical essence or occurrent" });
            expect(s.range).toEqual("PhysicalEssenceOrOccurrent");
        })

        test("range property is undefined if not provided", () => {
            const s = new Slot("negatively_regulates", {});
            expect(s.range).toBeUndefined;
        })
    })

    describe("Test exact_mapping method", () => {
        test("exact_mapping property is correctly retreived if provided", () => {
            const s = new Slot("regulates", { exact_mappings: ["GO:regulates"] });
            expect(s.exact_mapping).toEqual(["GO:regulates"]);
        })

        test("exact_mapping property is undefined if not provided", () => {
            const s = new Slot("negatively_regulates", {});
            expect(s.exact_mapping).toBeUndefined;
        })
    })

    describe("Test close_mapping method", () => {
        test("close_mapping property is correctly retreived if provided", () => {
            const s = new Slot("regulates", { close_mappings: ["GO:regulated_by", "RO:0002334"] });
            expect(s.close_mapping).toEqual(["GO:regulated_by", "RO:0002334"]);
        })

        test("close_mapping property is undefined if not provided", () => {
            const s = new Slot("negatively_regulates", {});
            expect(s.close_mapping).toBeUndefined;
        })
    })

    describe("Test narrow_mapping method", () => {
        test("narrow_mapping property is correctly retreived if provided", () => {
            const s = new Slot("regulates", { narrow_mappings: ["WIKIDATA_PROPERTY:P128", "CHEMBL.MECHANISM:modulator"] });
            expect(s.narrow_mapping).toEqual(["WIKIDATA_PROPERTY:P128", "CHEMBL.MECHANISM:modulator"]);
        })

        test("narrow_mapping property is undefined if not provided", () => {
            const s = new Slot("negatively_regulates", {});
            expect(s.narrow_mapping).toBeUndefined;
        })
    })

    describe("Test add_child method", () => {
        test("add child function is correctly performed", () => {
            const s = new Slot("regulates", { narrow_mappings: ["WIKIDATA_PROPERTY:P128", "CHEMBL.MECHANISM:modulator"] });
            s.addChild("negatively regulates")
            expect(s.children).toEqual(["negatively_regulates"]);
        })
    })
})