import _ from 'lodash';

import { BioThingsQueryBuilder } from './builder/biothings_builder';
import { DBIdsObject, Buckets } from '../common/types';
import { MAX_CONCURRENT_QUERIES } from '../config';
import Debug from 'debug';
const debug = Debug('retriever:biomedical-id-resolver:Scheduler');

export class Scheduler {
  private validInput: DBIdsObject;
  private _buckets: Buckets;

  constructor(validInput: DBIdsObject) {
    this.validInput = validInput;
    this._buckets = {};
  }

  get buckets() {
    return this._buckets;
  }

  schedule() {
    debug(`Max number of concurrent queries sent to one API is: ${MAX_CONCURRENT_QUERIES}`);
    Object.keys(this.validInput).map((semanticType) => {
      const builder = new BioThingsQueryBuilder(semanticType, this.validInput[semanticType]);
      const promises = builder.build();
      debug(`Number of API queries made for semantic type ${semanticType} is ${promises.length}`);
      promises.map((p, i) => {
        const bucketIndex = Math.floor(i / MAX_CONCURRENT_QUERIES);
        if (!(bucketIndex in this._buckets)) {
          this._buckets[bucketIndex] = [];
        }
        this._buckets[bucketIndex].push(p);
      });
    });
    debug(`Total number of API Query Buckets created is ${Object.keys(this.buckets).length}`);
    return this.buckets;
  }
}
