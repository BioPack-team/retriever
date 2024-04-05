/**
 * Traverse BioLink class hierarchy when considering inputs.
 */

import BaseValidator from './base_validator';
import BioLinkHandlerInstance from '../biolink';
import { DBIdsObject, BioLinkHandlerClass, DBIdsObjects, BioLinkBasedValidatorObject } from '../common/types';
import { getPrefixFromCurie } from '../utils';
import { APIMETA } from '../config';

export default class BioLinkBasedValidator extends BaseValidator implements BioLinkBasedValidator {
  private _biolink: BioLinkHandlerClass;
  private _valid: DBIdsObject;

  constructor(userInput: unknown) {
    super(userInput);
    this._biolink = BioLinkHandlerInstance;
    this._valid = {} as DBIdsObject;
    this._resolvable = {} as DBIdsObject;
    this._irresolvable = {} as DBIdsObject;
  }

  get valid(): DBIdsObject {
    return this._valid;
  }

  private checkIfTypeDefinedInBioLink(userInput: DBIdsObject): DBIdsObject {
    const tmp = {};
    for (const semanticType in userInput) {
      if (semanticType in this._biolink.classTree.objects) {
        tmp[semanticType] = userInput[semanticType];
      } else {
        this._irresolvable[semanticType] = userInput[semanticType];
      }
    }
    return tmp;
  }

  private groupDBIDsByPrefix(userInput: DBIdsObject): DBIdsObjects {
    const tmp = {};
    for (const semanticType in userInput) {
      if (!(semanticType in tmp)) {
        tmp[semanticType] = {};
      }
      for (const curie of userInput[semanticType]) {
        const prefix = getPrefixFromCurie(curie);
        if (!(prefix in tmp[semanticType])) {
          tmp[semanticType][prefix] = [];
        }
        tmp[semanticType][prefix].push(curie);
      }
    }
    return tmp;
  }

  private checkIfSemanticTypeAndPrefixDefinedInConfig(semanticType: string, prefix: string) {
    if (semanticType in APIMETA && prefix in APIMETA[semanticType].mapping) {
      return true;
    }
    return false;
  }

  private addValidInputToResolvable(semanticType: string, prefix: string, curies: string[]) {
    if (!(semanticType in this._resolvable)) {
      this._resolvable[semanticType] = [];
    }
    this._resolvable[semanticType] = [...this._resolvable[semanticType], ...curies];
  }

  private addValidInputToIRResolvable(semanticType: string, prefix: string, curies: string[]) {
    if (!(semanticType in this._irresolvable)) {
      this._irresolvable[semanticType] = [];
    }
    this._irresolvable[semanticType] = [...this._irresolvable[semanticType], ...curies];
  }

  private getDescendantClassesWithGivenIDPrefix(semanticType: string, prefix: string) {
    const descendants = this._biolink.classTree.getDescendants(semanticType);
    const validDescendants = descendants.filter((d) => d.id_prefixes && d.id_prefixes.includes(prefix));
    return validDescendants.map((d) => d.name);
  }

  private classifyASemanticTypeAndPrefixPair(semanticType: string, prefix: string, curies: string[]) {
    let canBeResolved = false;
    if (this.checkIfSemanticTypeAndPrefixDefinedInConfig(semanticType, prefix)) {
      this.addValidInputToResolvable(semanticType, prefix, curies);
      canBeResolved = true;
    } else {
      this.addValidInputToIRResolvable(semanticType, prefix, curies);
    }
    return canBeResolved;
  }

  private classify(userInput: DBIdsObject) {
    const input = this.groupDBIDsByPrefix(userInput);
    for (const semanticType in input) {
      for (const prefix in input[semanticType]) {
        let canBeResolved = false;
        if (this.checkIfSemanticTypeAndPrefixDefinedInConfig(semanticType, prefix)) {
          this.addValidInputToResolvable(semanticType, prefix, input[semanticType][prefix]);
          canBeResolved = true;
        }
        const validDescendants = this.getDescendantClassesWithGivenIDPrefix(semanticType, prefix);
        validDescendants.map((d) => {
          const resolvable = this.classifyASemanticTypeAndPrefixPair(d, prefix, input[semanticType][prefix]);
          if (resolvable === true) {
            canBeResolved = true;
          }
        });
        if (canBeResolved === false) {
          this.addValidInputToIRResolvable(semanticType, prefix, input[semanticType][prefix]);
        }
      }
    }
  }

  validate() {
    this.validateInputStructure();
    this._valid = this.checkIfTypeDefinedInBioLink(this.userInput);
    this.classify(this._valid);
  }
}
