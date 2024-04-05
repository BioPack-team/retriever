import { IResolver, ResolverOutput, DBIdsObject } from '../common/types';
import { IrresolvableBioEntity } from '../bioentity/irresolvable_bioentity';
import Debug from 'debug';
const debug = Debug('biomedical-id-resolver:resolver');

export default abstract class BaseResolver implements IResolver {
  protected annotateUnresolvedInputs(unresolvedInputs: DBIdsObject, currentResult: ResolverOutput): ResolverOutput {
    Object.keys(unresolvedInputs).map((semanticType) => {
      for (const curie of unresolvedInputs[semanticType]) {
        if (!(curie in currentResult)) {
          currentResult[curie] = [];
        }
        currentResult[curie].push(new IrresolvableBioEntity(semanticType, curie));
      }
    });
    debug(`Total number of Irresolvable curies are: ${Object.keys(unresolvedInputs).length}`);
    return currentResult;
  }

  abstract resolve(usrInput: unknown): Promise<ResolverOutput>;
}
