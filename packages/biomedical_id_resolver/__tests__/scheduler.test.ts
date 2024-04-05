import { Scheduler } from '../src/query/scheduler';


describe("Test Scheduler Class", () => {
    describe("Test schedule function", () => {
        test.skip("Promises with from one API with less than 1000 ids should return one promise", () => {
            const scheduler = new Scheduler({ "Gene": ["NCBIGene:1017"] });
            scheduler.schedule();
            expect(scheduler.buckets[0]).toHaveLength(1);
        })

        test("Promises with from multiple API with less than 1000 ids should return one promise", () => {
            const scheduler = new Scheduler({ "ChemicalEntity": ["DRUGBANK:DB0001"] });
            scheduler.schedule();
            expect(scheduler.buckets[0]).toHaveLength(1);
        })

        test.skip("Promises with from multiple API with more than 3000 ids should return one promise", () => {
            const fakeNCBIGeneInputs = [...Array(1990).keys()].map(item => 'NCBIGene:' + item.toString());
            const fakeOMIMGeneInputs = [...Array(2300).keys()].map(item => "OMIM:" + item.toString());
            const fakeDrugbankInputs = [...Array(3500).keys()].map(item => "DRUGBANK:DB00" + item.toString());
            const scheduler = new Scheduler({ "Gene": [...fakeNCBIGeneInputs, ...fakeOMIMGeneInputs], "SmallMolecule": fakeDrugbankInputs });
            scheduler.schedule();
            expect(scheduler.buckets[0]).toHaveLength(6);
            expect(scheduler.buckets[1]).toHaveLength(3);
        })
    })
})