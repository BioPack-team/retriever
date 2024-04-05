import URLLoader from '../../src/loader/url_loader';
import FileLoadingError from '../../src/exceptions/file_loading_error';
import axios from 'axios';
import yaml from 'js-yaml';
jest.mock('js-yaml')
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;
const mockedYaml = yaml as jest.Mocked<typeof yaml>;

describe("Test URL Loader", () => {
    test("Axios query returning status code > 200 should raise an exception", async () => {
        mockedAxios.get.mockImplementation(async () => { return { status: 400 } })
        const loader = new URLLoader();
        await expect(loader.load('hello')).rejects.toThrowError(new FileLoadingError("Failed to load BioLink Model. Query to hello returns 400 status code."));
    })

    test("Axios query returning non string data should raise an exception", async () => {
        mockedAxios.get.mockImplementation(async () => { return { status: 200, data: { "wrong": "data" } } })
        const loader = new URLLoader();
        await expect(loader.load('hello')).rejects.toThrowError(new FileLoadingError("Failed to load BioLink Model. Unable to load BioLink yaml as a string."));
    })

    test("YAML2JSON parser that raise an error should raise an exception", async () => {
        mockedAxios.get.mockImplementation(async () => { return { status: 200, data: 'string' } });
        mockedYaml.load.mockImplementation(() => { throw new Error() })
        const loader = new URLLoader();
        await expect(loader.load('hello')).rejects.toThrowError(new FileLoadingError("Failed to load BioLink Model. Unable to convert the yaml file into json."));
    })

    test("Axios query that raise an axios exception should raise an exception", async () => {
        mockedAxios.get.mockImplementation(async () => { throw new Error(); })
        const loader = new URLLoader();
        await expect(loader.load('hello')).rejects.toThrowError(new FileLoadingError("Failed to load BioLink Model. Query to hello raise an exception."));
    })
})