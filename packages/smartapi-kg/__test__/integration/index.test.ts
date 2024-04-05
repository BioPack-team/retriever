import MetaKG from '../../src/index';
import FailToLoadSpecError from '../../src/exceptions/fail_to_load_spec'
import path from "path";


describe('Test constructMetaKG from remote', () => {
    test("Test construct meta-kg based with default parameters", async () => {
        const meta_kg = new MetaKG();
        await meta_kg.constructMetaKG();
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.filter(op => op.association.api_name === "MyGene.info API").length).toBeGreaterThan(0);
    });
    test("Test construct meta-kg based without options", async () => {
        const meta_kg = new MetaKG();
        await meta_kg.constructMetaKG(false, {});
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.filter(op => op.association.api_name === "MyGene.info API").length).toBeGreaterThan(0);
    });
    test("Test construct meta-kg based on team name", async () => {
        const meta_kg = new MetaKG();
        await meta_kg.constructMetaKG(false, { teamName: "Text Mining Provider" });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        expect(meta_kg.ops[0].association["x-translator"].team).toContain("Text Mining Provider");
    });

    test("Test filter function", async () => {
        const meta_kg = new MetaKG();
        await meta_kg.constructMetaKG(false, { teamName: "Text Mining Provider" });
        const res = meta_kg.filter({ input_type: "Disease" });
        expect(res.length).toBeGreaterThan(0);
        expect(res.filter(op => op.association.input_type === "Disease").length).toEqual(res.length);
    });

    test("Test construct meta-kg with tag equal to biothings", async () => {
        const meta_kg = new MetaKG();
        await meta_kg.constructMetaKG(false, { tag: "biothings" });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        expect(meta_kg.ops[0].tags).toContain("biothings");
    });

    test("Test construct meta-kg with not existing tag", async () => {
        const meta_kg = new MetaKG();
        await meta_kg.constructMetaKG(false, { tag: "fakj;kjkj" });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toEqual(0);
    });

    test("Test construct meta-kg with smartapi id", async () => {
        const meta_kg = new MetaKG();
        await meta_kg.constructMetaKG(false, { smartAPIID: "671b45c0301c8624abbd26ae78449ca2" });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        expect(meta_kg.ops[0].association.smartapi.id).toEqual("671b45c0301c8624abbd26ae78449ca2");
        expect(meta_kg.ops[0].association.api_name).toEqual("MyDisease.info API");
    });

    test("Test construct meta-kg with invalid smartapi id should throw an error", async () => {
        const meta_kg = new MetaKG();
        await expect(meta_kg.constructMetaKG(false, { smartAPIID: "invalid" })).rejects.toThrowError(new FailToLoadSpecError("Query to https://smart-api.info/api/metadata/invalid failed with status code 404"));;
    });

    test("Test construct meta-kg with component name", async () => {
        const meta_kg = new MetaKG();
        await meta_kg.constructMetaKG(false, { component: "KP" });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        expect(meta_kg.ops[0].association["x-translator"].component).toEqual("KP");
        expect(meta_kg.ops[4].association["x-translator"].component).toEqual("KP");
    });
    // 2021-09-09: Marco identified an issue where async isn't waiting for task to finish. This might be the problem with this test...
    test.skip("Test construct meta-kg including reasoner tags", async () => {
        const meta_kg = new MetaKG();
        await meta_kg.constructMetaKG(true, { smartAPIID: "912372f46127b79fb387cd2397203709" });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        expect(meta_kg.ops[0].association.input_type).toEqual("ChemicalSubstance");
        const reasoner_ops = meta_kg.ops.filter(op => 'bte-trapi' in op.tags);
        reasoner_ops.map(op => {
            expect(op.query_operation.path).toEqual('/query');
            expect(op.query_operation.method).toEqual('post');
        })

    });

    test("Test construct meta-kg including reasoner tags with no restrictions", async () => {
        const meta_kg = new MetaKG();
        await meta_kg.constructMetaKG(true, {});
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
    });


});

describe('Test constructMetaKG from local stored specs', () => {
    test("Test construct meta-kg from local provided file path is successful", () => {
        const file_path = path.resolve(__dirname, '../data/smartapi_multiomics_kp_query.json');
        let meta_kg = new MetaKG(file_path);
        meta_kg.constructMetaKGSync(false, {});
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
    });

    test("Test construct meta-kg from local provided file path not from /query endpoint", () => {
        const file_path = path.resolve(__dirname, '../data/mygene.json');
        let meta_kg = new MetaKG(file_path);
        meta_kg.constructMetaKGSync(false, {});
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
    });

    test("Test construct meta-kg from default file stored with the package", () => {
        let meta_kg = new MetaKG();
        meta_kg.constructMetaKGSync(false, {});
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
    });

    test("Test construct meta-kg with default parameters", () => {
        let meta_kg = new MetaKG();
        meta_kg.constructMetaKGSync();
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
    });

    test("Test construct meta-kg sync including reasoner", () => {
        let meta_kg_without = new MetaKG();
        meta_kg_without.constructMetaKGSync();
        let meta_kg = new MetaKG();
        meta_kg.constructMetaKGSync(true);
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(meta_kg_without.ops.length);
    });

    test("Test construct meta-kg with tag equal to biothings", () => {
        const meta_kg = new MetaKG();
        meta_kg.constructMetaKGSync(false, { tag: "biothings" });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        expect(meta_kg.ops[0].tags).toContain("biothings");
    });

    test("Test construct meta-kg with team equal to Text Mining Provider", () => {
        const meta_kg = new MetaKG();
        meta_kg.constructMetaKGSync(false, { teamName: "Text Mining Provider" });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        expect(meta_kg.ops[0].association["x-translator"].team).toContain("Text Mining Provider");
    });

    test("Test construct meta-kg with smartapi id", async () => {
        const meta_kg = new MetaKG();
        meta_kg.constructMetaKGSync(false, { smartAPIID: "671b45c0301c8624abbd26ae78449ca2" });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        expect(meta_kg.ops[0].association.smartapi.id).toEqual("671b45c0301c8624abbd26ae78449ca2");
        expect(meta_kg.ops[0].association.api_name).toEqual("MyDisease.info API");
    });

    test("Test construct meta-kg with component name", async () => {
        const meta_kg = new MetaKG();
        meta_kg.constructMetaKGSync(false, { component: "KP" });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        expect(meta_kg.ops[0].association["x-translator"].component).toEqual("KP");
        expect(meta_kg.ops[4].association["x-translator"].component).toEqual("KP");
    });

    test("Test construct meta-kg with a constrained list of apis", async () => {
        const meta_kg = new MetaKG();
        meta_kg.constructMetaKGSync(false, {
          apiList: {
            include: [
              {
                id: "59dce17363dce279d389100834e43648",
                name: "MyGene.info API",
              },
              {
                id: "09c8782d9f4027712e65b95424adba79",
                name: "MyVariant.info API",
              },
            ],
            exclude: [],
          },
        });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        const apis = Array.from(new Set(meta_kg.ops.map(op => op.association.api_name)));
        expect(apis).toHaveLength(2);
        expect(apis).toContain("MyGene.info API");
        expect(apis).toContain("MyVariant.info API");
    });

    test("Test construct meta-kg with a constrained list of apis and enable reasoner", async () => {
        const meta_kg = new MetaKG();
        meta_kg.constructMetaKGSync(true, {
          apiList: {
              include: [
                {
                  id: "59dce17363dce279d389100834e43648",
                  name: "MyGene.info API",
                },
                {
                  id: "09c8782d9f4027712e65b95424adba79",
                  name: "MyVariant.info API",
                },
              ],
              exclude: []
          }
        });
        expect(meta_kg.ops).toBeInstanceOf(Array);
        expect(meta_kg.ops.length).toBeGreaterThan(0);
        const apis = Array.from(new Set(meta_kg.ops.map(op => op.association.api_name)));
        expect(apis).toHaveLength(2);
        expect(apis).toContain("MyGene.info API");
        expect(apis).toContain("MyVariant.info API");
    });

});
