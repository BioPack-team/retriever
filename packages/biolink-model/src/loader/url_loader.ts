import axios from 'axios';
import Loader from './base_loader';
import { BioLinkJSON } from '../types/types';
import FileLoadingError from '../exceptions/file_loading_error';

export default class URLLoader extends Loader {
  async load(input) {
    let res;
    try {
      const userAgent = `BTE/${process.env.NODE_ENV === 'production' ? 'prod' : 'dev'} Node/${process.version} ${
        process.platform
      }`;
      res = await axios.get(input, {
        responseType: 'text',
        headers: { 'User-Agent': userAgent },
      });
    } catch (err) {
      throw new FileLoadingError(`Failed to load BioLink Model. Query to ${input} raise an exception.`);
    }
    if (res.status !== 200) {
      throw new FileLoadingError(`Failed to load BioLink Model. Query to ${input} returns ${res.status} status code.`);
    }
    if (typeof res.data === 'string') {
      return this.yaml2json(res.data) as BioLinkJSON;
    } else {
      throw new FileLoadingError('Failed to load BioLink Model. Unable to load BioLink yaml as a string.');
    }
  }
}
