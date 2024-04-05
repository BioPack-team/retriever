import crypto from 'crypto';
// import biolink from './biolink';
// import config from './config.js';

export default class QueryGraphHelper {
  static _generateHash(stringToBeHashed: string): string {
    return crypto.createHash('md5').update(stringToBeHashed).digest('hex');
  }
}
