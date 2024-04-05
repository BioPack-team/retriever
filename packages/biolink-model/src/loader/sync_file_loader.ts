import Loader from './base_loader';
import fs from 'fs';
import { BioLinkJSON } from '../types/types';

export default class SyncFileLoader extends Loader {
  load(input) {
    const file = fs.readFileSync(input, { encoding: 'utf8' });
    return this.yaml2json(file) as BioLinkJSON;
  }
}
