import SyncFileLoader from '../../src/loader/sync_file_loader';
import path from 'path';

describe("Test File Loader", () => {
    test("BioLink yaml should be read from file path given and convert to json", () => {
        const loader = new SyncFileLoader();
        const res = loader.load(path.resolve(__dirname, '../../data/biolink.yaml'));
        expect(res).toHaveProperty('id', 'https://w3id.org/biolink/biolink-model');
        expect(res).toHaveProperty('name', 'Biolink-Model')
    })
})