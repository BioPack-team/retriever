import Loader from './base_loader';
import util from 'util';
import fs from 'fs';
const readFile = util.promisify(fs.readFile);
import { BioLinkJSON } from '../types/types';

export default class FileLoader extends Loader {
  async load(input) {
    const file = await readFile(input, { encoding: 'utf8' });
    return this.yaml2json(file) as BioLinkJSON;
  }
}
