import BioLinkBasedValidator from './validate/biolink_based_validator';
import { DBIdsObject, ResolverOutput } from './common/types';
import { IrresolvableBioEntity } from './bioentity/irresolvable_bioentity';

const annotateUnresolvedInputs = (unresolvedInputs: DBIdsObject, currentResult: ResolverOutput): ResolverOutput => {
  Object.keys(unresolvedInputs).map((semanticType) => {
    for (const curie of unresolvedInputs[semanticType]) {
      if (!(curie in currentResult)) {
        currentResult[curie] = [];
      }
      currentResult[curie].push(new IrresolvableBioEntity(semanticType, curie));
    }
  });
  return currentResult;
};
const generateInvalid = (userInput: any) => {
  const validator = new BioLinkBasedValidator(userInput);
  validator.validate();
  const result = annotateUnresolvedInputs(validator.valid, {});
  return result;
};

export default generateInvalid;
