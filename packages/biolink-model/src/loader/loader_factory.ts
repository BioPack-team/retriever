import DefaultLoader from './default_loader';
import FileLoader from './file_loader';
import UrlLoader from './url_loader';
import fs from 'fs';
import { LoaderClass } from '../types/types';

const loader = (source: string = undefined): LoaderClass => {
  if (typeof source === 'undefined') {
    return new DefaultLoader();
  }
  if (source.startsWith('http://') || source.startsWith('https://')) {
    return new UrlLoader();
  }
  try {
    if (fs.lstatSync(source).isFile()) {
      return new FileLoader();
    }
  } catch (e) {
    throw new Error('Your input is invalid. It should be undefined or a valid url or a valid file path.');
  }
};

export default loader;
