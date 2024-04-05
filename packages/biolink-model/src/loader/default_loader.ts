import Loader from './base_loader';
import util from 'util';
import path from 'path';
import fs from 'fs';
const readFile = util.promisify(fs.readFile);
import { BioLinkJSON } from '../types/types';

export default class DefaultLoader extends Loader {
  async load() {
    const biolink_path = process.env.BIOLINK_FILE
      ? path.resolve(process.env.BIOLINK_FILE)
      : path.resolve(__dirname, '../../data/biolink.yaml');
    const file = await readFile(biolink_path, { encoding: 'utf8' });
    return (
      ['.yaml', '.yml'].includes(path.extname(biolink_path)) ? this.yaml2json(file) : JSON.parse(file)
    ) as BioLinkJSON;
  }
}
