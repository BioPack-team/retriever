import SyncFileLoader from '../../src/loader/sync_file_loader';
import SyncDefaultLoader from '../../src/loader/sync_default_loader';
import loader from '../../src/loader/sync_loader_factory';
import path from 'path';

describe("Test loader entry point", () => {
    test("When input is undefined, return sync default loader", () => {
        const res = loader();
        expect(res).toBeInstanceOf(SyncDefaultLoader);
    })

    test("When input is a valid file path, return sync file loader", () => {
        const file_path = path.resolve(__dirname, '../../data/biolink.yaml');
        const res = loader(file_path);
        expect(res).toBeInstanceOf(SyncFileLoader);
    })

    test("When input is a invalid file path, raise an execption", () => {
        const file_path = path.resolve(__dirname, '../../data/biolink1.yaml');
        expect(() => {
            loader(file_path)
        }).toThrow();
    })
})
