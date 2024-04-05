import FileLoader from '../../src/loader/file_loader';
import path from 'path';

describe("Test File Loader", () => {
    test("BioLink yaml should be read from data/ folder and convert to json", async () => {
        const loader = new FileLoader();
        const res = await loader.load(path.resolve(__dirname, '../../data/biolink.yaml'));
        expect(res).toHaveProperty('id', 'https://w3id.org/biolink/biolink-model');
        expect(res).toHaveProperty('name', 'Biolink-Model')
    })
})