import BaseValidator from './base_validator';
import { DBIdsObject } from '../common/types';
import { APIMETA } from '../config';
import { getPrefixFromCurie, generateIDTypeDict } from '../utils';

export default class DefaultValidator extends BaseValidator {
  private handleUndefinedIDs(userInput: DBIdsObject): DBIdsObject {
    if (!('undefined' in userInput)) {
      return userInput;
    }
    const idTypeDict = generateIDTypeDict();
    const identified = [];
    for (const curie of userInput.undefined) {
      const prefix = getPrefixFromCurie(curie);
      const possibleSemanticTypes = idTypeDict[prefix];
      if (typeof possibleSemanticTypes === 'undefined') {
        continue;
      }
      for (const semanticType of possibleSemanticTypes) {
        if (!(semanticType in userInput)) {
          userInput[semanticType] = [];
        }
        userInput[semanticType].push(curie);
      }
      identified.push(curie);
    }
    userInput.undefined = userInput.undefined.filter((item) => !identified.includes(item));
    return userInput;
  }

  private checkIfSemanticTypeCanBeResolved(userInput: DBIdsObject) {
    const DBIDsWithCorrectSemanticTypes = {} as DBIdsObject;
    for (const semanticType in userInput) {
      if (Object.keys(APIMETA).includes(semanticType)) {
        DBIDsWithCorrectSemanticTypes[semanticType] = userInput[semanticType];
      } else {
        this._irresolvable[semanticType] = userInput[semanticType];
      }
    }
    return DBIDsWithCorrectSemanticTypes;
  }

  private checkIfPrefixCanBeResolved(userInput: DBIdsObject) {
    Object.keys(userInput).map((semanticType) => {
      for (const item of userInput[semanticType]) {
        if (!(getPrefixFromCurie(item) in APIMETA[semanticType].mapping)) {
          if (!(semanticType in this._irresolvable)) {
            this._irresolvable[semanticType] = [];
          }
          this._irresolvable[semanticType].push(item);
        } else {
          if (!(semanticType in this._resolvable)) {
            this._resolvable[semanticType] = [];
          }
          this._resolvable[semanticType].push(item);
        }
      }
    });
  }

  validate(): void {
    this.validateInputStructure();
    const restructuredInput = this.handleUndefinedIDs(this.userInput);
    const tmpResolvableRes = this.checkIfSemanticTypeCanBeResolved(restructuredInput);
    this.checkIfPrefixCanBeResolved(tmpResolvableRes);
  }
}
