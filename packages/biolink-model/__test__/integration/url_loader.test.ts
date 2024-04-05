import axios from 'axios';
import URLLoader from '../../src/loader/url_loader';
import path from 'path';

describe("Test URL Loader", () => {
    test("BioLink yaml should be read from data/ folder and convert to json", async () => {
        axios.defaults.adapter = require('axios/lib/adapters/http');
        const loader = new URLLoader();
        const res = await loader.load('https://raw.githubusercontent.com/biolink/biolink-model/master/biolink-model.yaml');
        expect(res).toHaveProperty('id', 'https://w3id.org/biolink/biolink-model');
        expect(res).toHaveProperty('name', 'Biolink-Model')
    })
})
