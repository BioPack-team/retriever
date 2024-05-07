import BaseResolver from './base_resolver';
import BioLinkBasedValidator from '../validate/biolink_based_validator';
import query from '../query/index';
import { ResolverOutput, IndividualResolverOutput, DBIdsObject } from '../common/types';
import BioLinkHandlerInstance from '../biolink';
import Debug from 'debug';
const debug = Debug('retriever:biomedical-id-resolver:resolver');

export default class BioLinkBasedResolver extends BaseResolver {
  private _biolink: typeof BioLinkHandlerInstance;

  constructor() {
    super();
    this._biolink = BioLinkHandlerInstance;
  }

  private constructPrefixTypeDict(input: DBIdsObject): DBIdsObject {
    const res = {};
    for (const type in input) {
      for (const curie of input[type]) {
        if (!(curie in res)) {
          res[curie] = [];
        }
        if (!res[curie].includes(type)) {
          res[curie].push(type);
        }
      }
    }
    return res;
  }

  private getPath(downstreamType: string, upstreamTypes: string[]): string[] {
    let paths = upstreamTypes.slice();
    upstreamTypes.map((type) => {
      const path = this._biolink.classTree.getPath(downstreamType, type).map((entity) => entity.name);
      paths = [...paths, ...path];
    });
    paths.push(downstreamType);
    return Array.from(new Set(paths));
  }

  private organizeResolvedOutputs(resolved: IndividualResolverOutput[], resolvable: DBIdsObject): ResolverOutput {
    const result = {};
    const prefixTypeDict = this.constructPrefixTypeDict(resolvable);
    resolved.map((recs) => {
      for (const curie of Object.keys(recs)) {
        if (!(curie in result)) {
          result[curie] = [];
        }
        recs[curie].semanticTypes = this.getPath(recs[curie].semanticType, prefixTypeDict[curie]);
        result[curie].push(recs[curie]);
      }
    });
    return result;
  }

  async resolve(userInput: unknown) {
    const validator = new BioLinkBasedValidator(userInput);
    validator.validate();
    const queryResult = await query(validator.resolvable);
    let result = this.organizeResolvedOutputs(queryResult, validator.valid);
    result = this.annotateUnresolvedInputs(validator.irresolvable, result);
    debug(`Total number of results returned are: ${Object.keys(result).length}`);
    return result;
  }
}
