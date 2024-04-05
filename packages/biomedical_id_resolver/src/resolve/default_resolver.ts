import BaseResolver from './base_resolver';
import { ResolverOutput, IndividualResolverOutput } from '../common/types';
import query from '../query/index';
import DefaultValidator from '../validate/default_validator';
import Debug from 'debug';
const debug = Debug('biomedical-id-resolver:resolver');

export default class DefaultResolver extends BaseResolver {
  private organizeResolvedOutputs(resolved: IndividualResolverOutput[]): ResolverOutput {
    const result = {};
    resolved.map((recs) => {
      for (const curie of Object.keys(recs)) {
        if (!(curie in result)) {
          result[curie] = [];
        }
        result[curie].push(recs[curie]);
      }
    });
    return result;
  }

  async resolve(userInput: unknown): Promise<ResolverOutput> {
    const validator = new DefaultValidator(userInput);
    validator.validate();
    const queryResult = await query(validator.resolvable);
    let result = this.organizeResolvedOutputs(queryResult);
    result = this.annotateUnresolvedInputs(validator.irresolvable, result);
    debug(`Total number of results returned are: ${Object.keys(result).length}`);
    return result;
  }
}
