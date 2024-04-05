import SyncDefaultLoader from './sync_default_loader';
import SyncFileLoader from './sync_file_loader';
import fs from 'fs';

const syncLoader = (source: string = undefined) => {
  if (typeof source === 'undefined') {
    return new SyncDefaultLoader();
  }
  try {
    if (fs.lstatSync(source).isFile()) {
      return new SyncFileLoader();
    }
  } catch (e) {
    throw new Error('Your input is invalid. It should be undefined or a valid url or a valid file path.');
  }
};

export default syncLoader;
