import DefaultLoader from '../../src/loader/default_loader';

describe("Test Default Loader", () => {
    test("BioLink yaml should be read from data/ folder and convert to json", async () => {
        const loader = new DefaultLoader();
        const res = await loader.load();
        expect(res).toHaveProperty('id', 'https://w3id.org/biolink/biolink-model');
        expect(res).toHaveProperty('name', 'Biolink-Model')
    })
})