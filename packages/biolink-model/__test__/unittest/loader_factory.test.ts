import URLLoader from '../../src/loader/url_loader';
import FileLoader from '../../src/loader/file_loader';
import DefaultLoader from '../../src/loader/default_loader';
import loader from '../../src/loader/loader_factory';
import path from 'path';

describe("Test loader entry point", () => {
    test("When input is undefined, return default loader", () => {
        const res = loader();
        expect(res).toBeInstanceOf(DefaultLoader);
    })

    test("When input is url, return url loader", () => {
        const res = loader('http://123');
        expect(res).toBeInstanceOf(URLLoader);
        const res1 = loader('https://123');
        expect(res1).toBeInstanceOf(URLLoader);
    })

    test("When input is a valid file path, return path loader", () => {
        const file_path = path.resolve(__dirname, '../../data/biolink.yaml');
        const res = loader(file_path);
        expect(res).toBeInstanceOf(FileLoader);
    })

    test("When input is a invalid file path, raise an execption", () => {
        const file_path = path.resolve(__dirname, '../../data/biolink1.yaml');
        expect(() => {
            loader(file_path)
        }).toThrow();
    })
})
